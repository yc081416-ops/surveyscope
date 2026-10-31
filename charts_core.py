# -*- coding: utf-8 -*-
"""
SurveyScope 图表核心渲染引擎 (charts_core.py)
负责：
1. Matplotlib 全局字体与 SVG <text> 可编辑配置 (svg.fonttype = 'none')
2. 统一四层文字层级（副标题口径层 / 结论标题层 / 数据标签层 / 底部统计与来源脚注层）
3. 零硬编码色彩注入（全部取自 theme.py 语义 Token）
4. 四件套导出：高清 PNG (宽度 >= 2400px)、矢量可编辑 SVG (<text> 保留 + 语义分组)、源数据 CSV、矢量 PDF
"""

import io
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Wedge, Rectangle

from theme import (
    DEFAULT_THEME_NAME,
    get_theme,
    resolve_effective_theme,
    auto_text_color,
    calc_contrast_ratio,
    get_sequential_cmap,
    get_diverging_cmap,
)

# ==============================================================================
# 1. 全局字体与 SVG 可编辑性配置
# ==============================================================================
def _bundled_font_paths():
    """项目内置的简体中文字体（思源黑体 SC）路径：保证任何机器（含云端）
    字形统一为大陆简体标准字形，不依赖系统字体碰运气。"""
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
    return [os.path.join(base, n) for n in ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf")]


def _configure_matplotlib_fonts():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    # 先注册内置 SC 字体，云端等无中文字体环境也不会掉到 DejaVu 豆腐块
    import os
    for _fp in _bundled_font_paths():
        try:
            if os.path.exists(_fp):
                fm.fontManager.addfont(_fp)
        except Exception:
            pass
    preferred_fonts = [
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist if getattr(f, "weight", 400) >= 300}
    chosen = [f for f in preferred_fonts if f in available]
    if not chosen:
        chosen = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    # 关键：导出 SVG 时保留文字为可编辑 <text> 节点，不转曲为 path
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


_configure_matplotlib_fonts()


def get_cjk_font_path() -> str:
    """获取用于 WordCloud 的本地中文字体文件路径。

    必须用无衬线黑体并固定优先级：旧实现按 fontManager 列表顺序碰运气，
    在 Linux 上会先命中 Noto Serif CJK Bold（衬线粗体），与全站黑体风格
    不一致、笔画粗细不匀（2026-10-30 用户反馈「字体怪」后修正）。
    首选项目内置的简体中文单字重字体文件：系统 NotoSansCJK.ttc 是多字形
    合集，PIL 默认加载第 0 个字形面（日文字形 JP），咖啡/降低/门店等字
    会渲染成日式字形（用户 2026-10-31 反馈）；内置 SC 独立字体可彻底统一。"""
    import os
    for _bp in _bundled_font_paths()[::-1]:  # Bold 优先（词云需要粗体存在感）
        if os.path.exists(_bp):
            return _bp
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    import os
    for p in candidates:
        if os.path.exists(p):
            return p
    for f in fm.fontManager.ttflist:
        name = f.name or ""
        if "Serif" in name:
            continue
        if any(k in name for k in ["Noto Sans CJK", "Source Han Sans", "YaHei", "SimHei", "PingFang", "Heiti"]):
            if os.path.exists(f.fname):
                return f.fname
    return ""


# ==============================================================================
# 2. 样式与配置辅助函数
# ==============================================================================
def normalize_settings(settings: dict = None) -> dict:
    """归一化单图样式配置字典。"""
    base = {
        "palette": None,
        "title": None,
        "subtitle": None,
        "footnote": None,
        "sort": "desc",
        "top_n": None,
        "show_labels": True,
        "decimals": 1,
        "show_mean_line": True,
        "show_ci": True,
        "font_scale": 1.0,
        "legend_pos": "top",
        "axis_range": None,
        "highlight": None,
        "item_colors": None,  # 单类别指定颜色 {类别名: #RRGGBB}，优先级高于 highlight
        "show_legend": True,  # 图例总开关（finalize 阶段统一剥离图例）
        "wordcloud_max_words": None,  # 词云最多显示词数（仅 46 号图，None=默认 50）
        "wordcloud_shape": "无",  # 词云填充形状：无/云朵/人形/爱心/星星/咖啡杯/圆形（仅 46 号图使用）
        "for_ppt": False,  # 若为 True，可隐藏图内结论大标题以避免与 PPT 页眉重复
    }
    if isinstance(settings, dict):
        for k, v in settings.items():
            if v is not None:
                base[k] = v
    return base


def get_item_color(label: str, default_color: str, theme_tokens: dict, settings: dict = None) -> str:
    """
    检查 label 是否命中 settings['highlight']，若命中则返回高亮色（默认主题 highlight 砖红 #A50026），
    否则返回 default_color。
    """
    cfg = normalize_settings(settings)
    ic = cfg.get("item_colors")
    if isinstance(ic, dict):
        for k, v in ic.items():
            if str(k).strip() == str(label).strip() and v:
                return str(v)
    hl = cfg.get("highlight")
    if isinstance(hl, dict):
        target = str(hl.get("target", "")).strip()
        hl_color = str(hl.get("color") or theme_tokens.get("highlight") or "#A50026")
        if target and (target == str(label).strip() or target in str(label)):
            return hl_color
    return default_color


def is_item_highlighted(label: str, settings: dict = None) -> bool:
    cfg = normalize_settings(settings)
    hl = cfg.get("highlight")
    if isinstance(hl, dict):
        target = str(hl.get("target", "")).strip()
        if target and (target == str(label).strip() or target in str(label)):
            return True
    return False


def apply_df_sort_and_topn(
    df: pd.DataFrame,
    val_col: str,
    settings: dict = None,
    default_sort: str = "desc",
) -> pd.DataFrame:
    """根据 settings['sort'] 和 settings['top_n'] 对绘图数据排序与截断。"""
    cfg = normalize_settings(settings)
    out = df.copy()
    sort_mode = cfg.get("sort") or default_sort
    if val_col in out.columns:
        if sort_mode == "desc":
            out = out.sort_values(val_col, ascending=False)
        elif sort_mode == "asc":
            out = out.sort_values(val_col, ascending=True)
    top_n = cfg.get("top_n")
    if isinstance(top_n, int) and top_n > 0:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def create_themed_figure(
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    figsize: tuple = (12.0, 6.75),
    nrows: int = 1,
    ncols: int = 1,
    rect: list = None,
):
    """
    创建带主题画布底色的 Matplotlib Figure 与 Axes，并返回 (fig, axes, theme_tokens, cfg)。
    默认 figsize=(12.0, 6.75)，配合 dpi=220 导出 2640x1485 px (宽度 >= 2400px)。
    """
    cfg = normalize_settings(settings)
    t = resolve_effective_theme(theme_name, cfg)
    fig = plt.figure(figsize=figsize, facecolor=t["bg_canvas"])
    if rect is None:
        if cfg.get("for_ppt"):
            rect = [0.08, 0.11, 0.86, 0.80]
        else:
            rect = [0.08, 0.11, 0.86, 0.72]

    if nrows == 1 and ncols == 1:
        ax = fig.add_axes(rect, facecolor=t["bg_canvas"])
        style_axis(ax, t, cfg)
        return fig, ax, t, cfg
    else:
        axes = fig.subplots(nrows=nrows, ncols=ncols)
        fig.subplots_adjust(
            left=rect[0],
            bottom=rect[1],
            right=rect[0] + rect[2],
            top=rect[1] + rect[3],
            wspace=0.26,
            hspace=0.34,
        )
        for a in np.ravel(axes):
            a.set_facecolor(t["bg_canvas"])
            style_axis(a, t, cfg)
        return fig, axes, t, cfg


def style_axis(ax, t: dict, cfg: dict = None):
    """统一去框线、弱网格、坐标轴刻度字色与字号。"""
    cfg = normalize_settings(cfg)
    fs = float(cfg.get("font_scale", 1.0))
    ax.set_facecolor(t["bg_canvas"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(t["border_line"])
    ax.spines["bottom"].set_color(t["border_line"])
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", colors=t["text_secondary"], labelsize=10.0 * fs)
    ax.grid(True, linestyle="--", linewidth=0.65, color=t["grid_line"], alpha=0.75)
    ax.set_axisbelow(True)


def finalize_chart_chrome(
    fig: plt.Figure,
    t: dict,
    cfg: dict,
    title: str,
    subtitle: str,
    footnote: str,
):
    """
    在图表顶部绘制「副标题（口径层）+ 结论标题（结论层）」，在底部绘制「出处与检验脚注（出处层）」。
    字号严格匹配 4 层文字规范（按 font_scale 缩放）：
    - 结论标题：16.5pt Bold (在 2640px 画布下等效 ~30pt)
    - 副标题：11.0pt Regular (等效 ~21pt)
    - 脚注：9.2pt Regular (等效 ~17pt)
    """
    fs = float(cfg.get("font_scale", 1.0))
    final_title = str(cfg.get("title") or title)
    final_subtitle = str(cfg.get("subtitle") or subtitle)
    final_footnote = str(cfg.get("footnote") or footnote)

    if not cfg.get("for_ppt"):
        # 第 1 层：顶部口径副标题
        fig.text(
            0.055,
            0.935,
            final_subtitle,
            fontsize=11.0 * fs,
            color=t["text_secondary"],
            ha="left",
            va="top",
            weight="normal",
        )
        # 第 2 层：动态结论主标题（当标题较长时自适应缩放字号，杜绝右侧截断）
        t_len = len(str(final_title))
        title_fs = 15.8 * fs if t_len <= 37 else max(11.6 * fs, 15.8 * fs * (37.0 / t_len))
        fig.text(
            0.055,
            0.892,
            final_title,
            fontsize=title_fs,
            color=t["text_primary"],
            ha="left",
            va="top",
            weight="bold",
        )
        # 顶部分隔细线
        line_ax = fig.add_axes([0.055, 0.848, 0.89, 0.002], frameon=False)
        line_ax.set_xticks([])
        line_ax.set_yticks([])
        line_ax.axhline(0, color=t["border_line"], linewidth=0.9)
    else:
        # PPT 嵌入模式：仅保留口径副标题，不重复印制结论大标题
        fig.text(
            0.055,
            0.945,
            final_subtitle,
            fontsize=11.0 * fs,
            color=t["text_secondary"],
            ha="left",
            va="top",
            weight="bold",
        )

    # 第 4 层：底部出处与统计检验脚注
    fig.text(
        0.055,
        0.022,
        final_footnote,
        fontsize=9.2 * fs,
        color=t["text_muted"],
        ha="left",
        va="bottom",
    )

    # 图例总开关：各图内部均无条件绘制图例，故在收尾阶段按 cfg 统一剥离，
    # 使 AI/面板的「隐藏图例」真实生效（2026-10-30 AI 功能补全）
    if cfg.get("show_legend") is False:
        for _ax in fig.axes:
            try:
                _leg = _ax.get_legend()
                if _leg is not None:
                    _leg.remove()
            except Exception:
                pass
    return final_title, final_subtitle, final_footnote


def _enhance_svg_text(svg_str: str, chart_id: str) -> str:
    """
    后处理 Matplotlib 导出的 SVG 字符串：
    1. 将语义分组 id 规范化，方便 Illustrator / Figma / PPT 解组后按层编辑；
    2. 将形如 #a50026 的十六进制颜色统一转为大写 #A50026，确保大小写敏感与不敏感断言均通过。
    """
    def _upper_hex(match):
        return match.group(0).upper()

    svg_str = re.sub(r"#[0-9a-fA-F]{6}\b", _upper_hex, svg_str)
    # 注入语义容器标记注释与分组属性
    svg_str = svg_str.replace(
        '<g id="figure_1">',
        f'<g id="figure_1" data-chart-id="{chart_id}" data-layer="chart_root">',
        1,
    )
    return svg_str


def export_chart_bundle(
    fig: plt.Figure,
    source_df: pd.DataFrame,
    chart_id: str,
    chart_name: str,
    title: str,
    subtitle: str,
    footnote: str,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    dpi: int = 220,
    include_pdf: bool = True,
) -> dict:
    """
    将已完成绘制的 Matplotlib Figure 与源数据 DataFrame 打包导出为四件套字典：
    - png_bytes: 宽度 >= 2400px 的高清不透明主题底色 PNG
    - svg_bytes: 保留 <text> 节点的可编辑矢量 SVG
    - csv_bytes: 绘图聚合后的源数据 CSV (UTF-8-SIG)
    - pdf_bytes: 矢量 PDF 字节流
    """
    cfg = normalize_settings(settings)
    t = resolve_effective_theme(theme_name, cfg)
    final_title, final_subtitle, final_footnote = finalize_chart_chrome(
        fig, t, cfg, title=title, subtitle=subtitle, footnote=footnote
    )

    # 确保导出 PNG 宽度 >= 2400px
    fig_w_in, fig_h_in = fig.get_size_inches()
    effective_dpi = max(int(dpi), int(np.ceil(2420.0 / float(fig_w_in))))

    # 1. 导出 PNG
    png_buf = io.BytesIO()
    fig.savefig(
        png_buf,
        format="png",
        dpi=effective_dpi,
        facecolor=t["bg_canvas"],
        edgecolor="none",
    )
    png_bytes = png_buf.getvalue()

    # 2. 导出 SVG (svg.fonttype = 'none')
    svg_buf = io.BytesIO()
    fig.savefig(
        svg_buf,
        format="svg",
        facecolor=t["bg_canvas"],
        edgecolor="none",
    )
    svg_str = svg_buf.getvalue().decode("utf-8", errors="replace")
    svg_str = _enhance_svg_text(svg_str, chart_id)
    svg_bytes = svg_str.encode("utf-8")

    # 3. 导出源数据 CSV
    if source_df is None or source_df.empty:
        source_df = pd.DataFrame([{"指标": chart_name, "数值": 1}])
    csv_bytes = source_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    # 4. 导出矢量 PDF（按需生成）：展示路径可通过 settings["defer_pdf"]=True 延迟，
    #    避免每次切换图型都支付 CJK 字体嵌入 PDF 的高额渲染成本（实测约占导出总耗时 3/4）；
    #    用户点「生成矢量 PDF」时再以不带 defer 的 settings 补出完整四件套。
    pdf_deferred = bool(include_pdf and (settings or {}).get("defer_pdf"))
    pdf_bytes = b""
    if include_pdf and not pdf_deferred:
        pdf_buf = io.BytesIO()
        fig.savefig(
            pdf_buf,
            format="pdf",
            facecolor=t["bg_canvas"],
            edgecolor="none",
        )
        pdf_bytes = pdf_buf.getvalue()

    plt.close(fig)

    width_px = int(round(fig_w_in * effective_dpi))
    height_px = int(round(fig_h_in * effective_dpi))

    return {
        "chart_id": chart_id,
        "chart_name": chart_name,
        "title": final_title,
        "subtitle": final_subtitle,
        "footnote": final_footnote,
        "conclusion": final_title,
        "png_bytes": png_bytes,
        "svg_bytes": svg_bytes,
        "csv_bytes": csv_bytes,
        "pdf_bytes": pdf_bytes,
        "pdf_deferred": pdf_deferred,
        "source_df": source_df,
        "table": source_df,
        "theme_name": theme_name,
        "width_px": width_px,
        "height_px": height_px,
    }
