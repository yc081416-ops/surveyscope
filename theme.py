# -*- coding: utf-8 -*-
"""
SurveyScope 多主题设计系统令牌与全站样式引擎 (theme.py)
提供 5 套完整主题（暖纸杂志、墨蓝学术、纯白极简、深空夜读、薄荷清新）：
1. 涵盖画布底色、侧边栏次级底色、卡片底色、主墨色、次级文字、描边网格、主强调色及分类/发散/序渐色阶；
2. 全部 5 套主题的主文字与次级文字相对画布底色、侧栏底色、卡片底色均严格满足 WCAG AA 对比度 >= 4.5:1；
3. 提供 build_streamlit_theme_css(theme_name) 全站级样式表，统一接管页面容器、侧边栏、数据表（含 Canvas DataGrid）、
   输入控件、下拉菜单浮层、上传框、按钮、指标卡与 SaaS 业务组件，严格遵循 8pt 间距栅格、四级字阶与 8px 单一圆角规范。
"""

import contextvars
import matplotlib.colors as mcolors

# 逐图配色上下文：chart_utils 渲染单图前设定当前 chart_id，
# resolve_effective_theme 据此叠加 chart_palettes 注册表中的专属色板。
_ACTIVE_CHART_ID = contextvars.ContextVar("surveyscope_active_chart_id", default=None)
_USE_CHART_PALETTES = True


def set_active_chart_id(chart_id):
    return _ACTIVE_CHART_ID.set(chart_id)


def reset_active_chart_id(token):
    _ACTIVE_CHART_ID.reset(token)


def set_use_chart_palettes(enabled: bool):
    """控制「暖纸杂志」主题下是否启用 chart_palettes.py 逐图专属色板（默认开启）。"""
    global _USE_CHART_PALETTES
    _USE_CHART_PALETTES = bool(enabled)


def get_use_chart_palettes() -> bool:
    return bool(_USE_CHART_PALETTES)


try:
    import cmcrameri.cm as cmc
    BATLOW_CMAP = cmc.batlow
except Exception:
    import matplotlib.pyplot as plt
    BATLOW_CMAP = plt.get_cmap("viridis")


FONT_FAMILY_NAME = "Noto Sans CJK SC"
FONT_FALLBACK_LIST = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "sans-serif",
]


THEMES = {
    "暖纸杂志": {
        "key": "warm_paper",
        "name": "暖纸杂志",
        "is_dark": False,
        "description": "暖纸画布与松绿主色，默认启用逐图艺术/期刊色板，兼具编辑部质感与高可读性",
        "bg_canvas": "#F7F2EA",
        "bg_secondary": "#EFE9DC",
        "bg_card": "#FBF9F5",
        "ink": "#22312B",
        "muted_text": "#544E46",
        "grid": "#D8CEBC",
        "insignificant": "#9E9484",
        "main": "#2E6B57",
        "main_dark": "#1E4D3D",
        "main_light": "#6FA08A",
        "main_pale": "#DCE8E1",
        "accent": "#A64B2A",
        "highlight": "#A50026",
        "categorical": [
            "#88CCEE", "#CC6677", "#44AA99", "#332288", "#DDCC77",
            "#882255", "#999933", "#AA4499", "#117733",
        ],
        "diverging": [
            "#543005", "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",
            "#F5F5F5", "#C7EAE5", "#80CDC1", "#35978F", "#01665E", "#003C30",
        ],
        "sequential_6": [
            "#EFE9DC", "#D8E2D5", "#A8C3B2", "#6FA08A", "#2E6B57", "#1E4D3D",
        ],
        "continuous_name": "batlow",
    },
    "墨蓝学术": {
        "key": "academic_blue",
        "name": "墨蓝学术",
        "is_dark": False,
        "description": "象牙白底与深墨蓝强调色，全站统一应用 Paul Tol 学术期刊色阶",
        "bg_canvas": "#FAF9F5",
        "bg_secondary": "#EEF1F6",
        "bg_card": "#FFFFFF",
        "ink": "#182638",
        "muted_text": "#475566",
        "grid": "#D5DCE6",
        "insignificant": "#94A3B8",
        "main": "#1F4E79",
        "main_dark": "#123252",
        "main_light": "#4B7BA8",
        "main_pale": "#DCE7F3",
        "accent": "#A84A28",
        "highlight": "#9E1B32",
        "categorical": [
            "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
            "#AA3377", "#335588", "#BB7733", "#2B6E68",
        ],
        "diverging": [
            "#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
            "#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061",
        ],
        "sequential_6": [
            "#EEF1F6", "#D0DDF0", "#9BBAD9", "#5C8AB8", "#2B5C8F", "#123252",
        ],
        "continuous_name": "batlow",
    },
    "纯白极简": {
        "key": "minimal_white",
        "name": "纯白极简",
        "is_dark": False,
        "description": "纯白画布与石板灰中性阶，单一钴蓝强调色，适合正式商业汇报与打印",
        "bg_canvas": "#FFFFFF",
        "bg_secondary": "#F3F5F8",
        "bg_card": "#F8FAFC",
        "ink": "#111827",
        "muted_text": "#475569",
        "grid": "#E2E8F0",
        "insignificant": "#94A3B8",
        "main": "#1D4ED8",
        "main_dark": "#1E3A8A",
        "main_light": "#60A5FA",
        "main_pale": "#DBEAFE",
        "accent": "#B45309",
        "highlight": "#BE123C",
        "categorical": [
            "#2563EB", "#0D9488", "#D97706", "#7C3AED", "#DB2777",
            "#4B5563", "#059669", "#EA580C", "#4338CA",
        ],
        "diverging": [
            "#7C2D12", "#9A3412", "#C2410C", "#FB923C", "#FED7AA",
            "#F8FAFC", "#BAE6FD", "#38BDF8", "#0284C7", "#0369A1", "#0C4A6E",
        ],
        "sequential_6": [
            "#F1F5F9", "#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A8A",
        ],
        "continuous_name": "batlow",
    },
    "深空夜读": {
        "key": "dark_night",
        "name": "深空夜读",
        "is_dark": True,
        "description": "全站原生深色工作台，侧栏、表格与控件同步暗色，搭配 Okabe-Ito 高对比无障碍色板",
        "bg_canvas": "#1E242E",
        "bg_secondary": "#171C24",
        "bg_card": "#242B37",
        "ink": "#ECF2F8",
        "muted_text": "#B4C2D4",
        "grid": "#364152",
        "insignificant": "#6E7F96",
        "main": "#4CC38A",
        "main_dark": "#2E9968",
        "main_light": "#7CE0AE",
        "main_pale": "#243D35",
        "accent": "#F08C5B",
        "highlight": "#FF5C77",
        "categorical": [
            "#56B4E9", "#E69F00", "#009E73", "#F0E442", "#CC79A7",
            "#78A9FF", "#FF8389", "#42BE65", "#BE95FF",
        ],
        "diverging": [
            "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3", "#4A4E58",
            "#2E3542", "#528E87", "#66C2A5", "#35978F", "#28B5A8", "#80EDE3",
        ],
        "sequential_6": [
            "#28303D", "#2E4745", "#31685A", "#388E72", "#4CC38A", "#89F0BC",
        ],
        "continuous_name": "batlow",
    },
    "薄荷清新": {
        "key": "mint_fresh",
        "name": "薄荷清新",
        "is_dark": False,
        "description": "浅青瓷底色与深青绿主色，清爽柔和，长时间检视数据不易视觉疲劳",
        "bg_canvas": "#F0F7F4",
        "bg_secondary": "#E1EFEA",
        "bg_card": "#F8FCFA",
        "ink": "#16302B",
        "muted_text": "#415C55",
        "grid": "#C7DDD5",
        "insignificant": "#8FA9A0",
        "main": "#0E6E64",
        "main_dark": "#0B5750",
        "main_light": "#4EB0A6",
        "main_pale": "#D3ECE8",
        "accent": "#A63D26",
        "highlight": "#B81D33",
        "categorical": [
            "#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311",
            "#EE3377", "#6699CC", "#44AA77", "#AA6688",
        ],
        "diverging": [
            "#662506", "#993404", "#CC4C02", "#FE9929", "#FEE391",
            "#F5FAF8", "#C7EAE5", "#80CDC1", "#35978F", "#01665E", "#004438",
        ],
        "sequential_6": [
            "#E1EFEA", "#BFE3DC", "#8CCBC1", "#45A89C", "#148277", "#0B5750",
        ],
        "continuous_name": "batlow",
    },
}

DEFAULT_THEME_NAME = "暖纸杂志"


def _enrich_theme_aliases(t: dict) -> dict:
    """补充语义别名键，兼容各图表、前端组件与 PPT 模块的统一访问。"""
    out = t.copy()
    out.setdefault("text_primary", out["ink"])
    out.setdefault("text_secondary", out["muted_text"])
    out.setdefault("text_muted", out["muted_text"])
    out.setdefault("grid_line", out["grid"])
    out.setdefault("border_line", out["grid"])
    out.setdefault("accent_primary", out["main"])
    out.setdefault("accent_secondary", out["accent"])
    out.setdefault("map_empty_fill", out["bg_secondary"])
    out.setdefault("map_border", out["grid"])
    return out


def get_theme(theme_name: str = None, apply_chart_palette: bool = True) -> dict:
    """获取指定名称的主题令牌字典，未指定或不存在时返回默认「暖纸杂志」主题。"""
    if isinstance(theme_name, dict):
        return _enrich_theme_aliases(theme_name)
    if not theme_name or theme_name not in THEMES:
        t = _enrich_theme_aliases(THEMES[DEFAULT_THEME_NAME])
    else:
        t = _enrich_theme_aliases(THEMES[theme_name])
    if apply_chart_palette and _USE_CHART_PALETTES:
        try:
            from chart_palettes import get_chart_palette_override
            _ov = get_chart_palette_override(_ACTIVE_CHART_ID.get(), theme_name if isinstance(theme_name, str) else None)
            if _ov:
                t.update(_ov)
        except Exception:
            pass
    return t


def hex_to_rgb01(hex_str: str) -> tuple:
    """将 #RRGGBB 转换为 (r, g, b) ∈ [0, 1]^3。"""
    return mcolors.to_rgb(hex_str)


def hex_to_rgb255(hex_str: str) -> tuple:
    """将 #RRGGBB 转换为 (R, G, B) ∈ [0, 255]^3 整数元组。"""
    r, g, b = mcolors.to_rgb(hex_str)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def relative_luminance(hex_str: str) -> float:
    """按 WCAG 2.1 标准计算颜色的相对明度 Y ∈ [0, 1]。"""
    rgb = mcolors.to_rgb(hex_str)
    lin = []
    for c in rgb:
        if c <= 0.03928:
            lin.append(c / 12.92)
        else:
            lin.append(((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def calc_contrast_ratio(hex1: str, hex2: str) -> float:
    """计算两个颜色之间的 WCAG 对比度 (>= 4.5:1 为 AA 级可读标准)。"""
    l1 = relative_luminance(hex1)
    l2 = relative_luminance(hex2)
    bright = max(l1, l2)
    dark = min(l1, l2)
    return round((bright + 0.05) / (dark + 0.05), 2)


def auto_text_color(bg_hex: str, theme: dict = None) -> str:
    """按底色明度自动切换深色字或浅色反白字，保证数据标签与按钮文字清晰可读。"""
    lum = relative_luminance(bg_hex)
    if isinstance(theme, str):
        if lum < 0.36:
            return "#FFFFFF"
        return theme if relative_luminance(theme) < 0.35 else "#22312B"
    if lum < 0.36:
        return "#FFFFFF" if (theme is None) else (theme["bg_canvas"] if relative_luminance(theme["bg_canvas"]) > 0.7 else "#FFFFFF")
    return "#22312B" if (theme is None) else (theme["ink"] if relative_luminance(theme["ink"]) < 0.3 else "#182638")


def get_continuous_cmap(theme=None):
    """获取连续色标（默认 Crameri batlow）。"""
    return BATLOW_CMAP


def get_diverging_cmap(theme=None):
    """根据当前主题的 11 级发散色板构建线性插值 Colormap。"""
    t = get_theme(theme)
    return mcolors.LinearSegmentedColormap.from_list("theme_diverging", t["diverging"], N=256)


def get_sequential_cmap(theme=None):
    """根据当前主题的 6 档序渐色构建连续 Colormap。"""
    t = get_theme(theme)
    return mcolors.LinearSegmentedColormap.from_list("theme_sequential", t["sequential_6"], N=256)


def resolve_effective_theme(theme_name: str = None, settings: dict = None) -> dict:
    """合并主题默认令牌、逐图色板（若开启）与格式面板显式设置。"""
    t = get_theme(theme_name)

    if _USE_CHART_PALETTES:
        try:
            from chart_palettes import get_chart_palette_override
            _ov = get_chart_palette_override(_ACTIVE_CHART_ID.get(), theme_name if isinstance(theme_name, str) else None)
            if _ov:
                t.update(_ov)
        except Exception:
            pass

    if not settings:
        return t

    scheme = settings.get("color_scheme") or settings.get("palette")
    if scheme in ("松绿序渐", "sequential"):
        t["categorical"] = list(reversed(t["sequential_6"][1:])) + t["categorical"][:4]
    elif scheme in ("BrBG发散", "diverging"):
        t["categorical"] = [
            t["diverging"][9], t["diverging"][1], t["diverging"][8], t["diverging"][2],
            t["diverging"][7], t["diverging"][3], t["main"], t["accent"], t["highlight"],
        ]
    elif scheme in ("Tol Muted", "muted"):
        t["categorical"] = [
            "#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
            "#CC6677", "#AA4499", "#882255", "#999999",
        ]
    elif scheme:
        # 命名配色注册表（期刊色 / 色盲友好 / MetBrewer 艺术系列），AI 与手动面板共用
        try:
            from chart_palettes import NAMED_COLOR_SCHEMES

            if scheme in NAMED_COLOR_SCHEMES:
                t["categorical"] = list(NAMED_COLOR_SCHEMES[scheme])
        except Exception:
            pass

    custom_main = settings.get("custom_main_color")
    if custom_main and isinstance(custom_main, str) and custom_main.startswith("#") and len(custom_main) == 7:
        t["main"] = custom_main
        t["accent_primary"] = custom_main

    return _enrich_theme_aliases(t)


def check_all_themes_contrast() -> list:
    """自检：计算全部 5 套主题主文字与次级文字在各背景层上的 WCAG 对比度（均须 >= 4.5:1）。"""
    results = []
    for name, t in THEMES.items():
        ratio = calc_contrast_ratio(t["ink"], t["bg_canvas"])
        sec_ratio = calc_contrast_ratio(t["muted_text"], t["bg_canvas"])
        sec_sidebar = calc_contrast_ratio(t["muted_text"], t["bg_secondary"])
        sec_card = calc_contrast_ratio(t["muted_text"], t["bg_card"])
        min_all = min(
            ratio,
            calc_contrast_ratio(t["ink"], t["bg_secondary"]),
            calc_contrast_ratio(t["ink"], t["bg_card"]),
            sec_ratio,
            sec_sidebar,
            sec_card,
        )
        results.append({
            "theme_name": name,
            "bg_canvas": t["bg_canvas"],
            "ink": t["ink"],
            "contrast_ratio": ratio,
            "muted_contrast": sec_ratio,
            "min_all_surfaces": min_all,
            "pass_aa": min_all >= 4.5,
        })
    return results


def build_streamlit_theme_css(theme_name: str = None) -> str:
    """
    构建全站级 SaaS 设计系统样式表（5 主题全量适配）：
    - 8pt 间距栅格（8 / 16 / 24 / 32 / 48px）与全站统一版心对齐；
    - 严格四级字阶（页标题 24px / 区块标题 16px / 正文 14px / 注释 12px + 指标数字 26px）；
    - 全站统一单一圆角（8px）、1px 细腻描边与极浅单层阴影；
    - 页面层严格限制为「1 个主强调色 + 中性色阶 + 当前主题底色」；
    - 完整接管侧边栏、表格（含 Glide DataGrid Canvas 变量）、输入框、选择器浮层、上传框、按钮、Tab 与折叠面板。
    """
    t = get_theme(theme_name, apply_chart_palette=False)
    is_dark = bool(t.get("is_dark", False))
    btn_fg = auto_text_color(t["main"], t)

    if is_dark:
        shadow_sm = "0 1px 2px rgba(0, 0, 0, 0.32)"
        hover_bg = "rgba(255, 255, 255, 0.05)"
        focus_ring = f"rgba({hex_to_rgb255(t['main'])[0]}, {hex_to_rgb255(t['main'])[1]}, {hex_to_rgb255(t['main'])[2]}, 0.28)"
        ok_bg, ok_bd, ok_fg = "#193229", "#2D6A52", "#9CE8C2"
        warn_bg, warn_bd, warn_fg = "#352A18", "#785B2A", "#F6D592"
        err_bg, err_bd, err_fg = "#381D24", "#823544", "#FCA5B3"
    else:
        shadow_sm = "0 1px 2px rgba(15, 23, 42, 0.04)"
        hover_bg = "rgba(15, 23, 42, 0.035)"
        focus_ring = f"rgba({hex_to_rgb255(t['main'])[0]}, {hex_to_rgb255(t['main'])[1]}, {hex_to_rgb255(t['main'])[2]}, 0.18)"
        ok_bg, ok_bd, ok_fg = "#EDF7F2", "#B7DEC9", "#18543C"
        warn_bg, warn_bd, warn_fg = "#FDF6EA", "#E6CFA0", "#734B10"
        err_bg, err_bd, err_fg = "#FDF1F2", "#EABBC1", "#8A1C2B"

    return f"""
    <style>
    :root, .stApp {{
        --primary-color: {t['main']};
        --background-color: {t['bg_canvas']};
        --secondary-background-color: {t['bg_secondary']};
        --text-color: {t['ink']};

        --ss-bg-canvas: {t['bg_canvas']};
        --ss-bg-secondary: {t['bg_secondary']};
        --ss-bg-card: {t['bg_card']};
        --ss-ink: {t['ink']};
        --ss-muted: {t['muted_text']};
        --ss-border: {t['grid']};
        --ss-accent: {t['main']};
        --ss-accent-dark: {t['main_dark']};
        --ss-accent-pale: {t['main_pale']};
        --ss-accent-fg: {btn_fg};
        --ss-hover: {hover_bg};
        --ss-focus: {focus_ring};
        --ss-shadow: {shadow_sm};
        --ss-radius: 8px;

        --ss-ok-bg: {ok_bg};
        --ss-ok-bd: {ok_bd};
        --ss-ok-fg: {ok_fg};
        --ss-warn-bg: {warn_bg};
        --ss-warn-bd: {warn_bd};
        --ss-warn-fg: {warn_fg};
        --ss-err-bg: {err_bg};
        --ss-err-bd: {err_bd};
        --ss-err-fg: {err_fg};

        /* Glide DataGrid (st.dataframe Canvas) 主题变量覆盖 */
        --gdg-bg-cell: {t['bg_card']};
        --gdg-bg-cell-medium: {t['bg_secondary']};
        --gdg-bg-header: {t['bg_secondary']};
        --gdg-bg-header-has-focus: {t['bg_secondary']};
        --gdg-bg-header-hovered: {t['bg_secondary']};
        --gdg-bg-bubble: {t['bg_secondary']};
        --gdg-bg-bubble-selected: {t['main_pale']};
        --gdg-bg-search-result: {t['main_pale']};
        --gdg-text-dark: {t['ink']};
        --gdg-text-medium: {t['muted_text']};
        --gdg-text-light: {t['muted_text']};
        --gdg-text-bubble: {t['ink']};
        --gdg-text-header: {t['ink']};
        --gdg-text-header-selected: {t['ink']};
        --gdg-border-color: {t['grid']};
        --gdg-horizontal-border-color: {t['grid']};
        --gdg-accent-color: {t['main']};
        --gdg-accent-fg: {btn_fg};
        --gdg-accent-light: {t['main_pale']};
        --gdg-link-color: {t['main']};
        color-scheme: {'dark' if is_dark else 'light'};
    }}

    /* ==================== 全局容器与顶栏 ==================== */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: var(--ss-bg-canvas) !important;
        color: var(--ss-ink) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif !important;
        font-size: 14px;
        line-height: 1.6;
    }}

    header[data-testid="stHeader"] {{
        background-color: var(--ss-bg-canvas) !important;
        border-bottom: 1px solid var(--ss-border) !important;
        height: 48px !important;
    }}

    header[data-testid="stHeader"] * {{
        color: var(--ss-muted) !important;
    }}

    .main .block-container, [data-testid="stMainBlockContainer"] {{
        max-width: 1280px !important;
        padding-top: 24px !important;
        padding-bottom: 48px !important;
        padding-left: 32px !important;
        padding-right: 32px !important;
    }}

    /* ==================== 侧边栏全量主题化 ==================== */
    section[data-testid="stSidebar"] {{
        background-color: var(--ss-bg-secondary) !important;
        border-right: 1px solid var(--ss-border) !important;
    }}

    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        background-color: var(--ss-bg-secondary) !important;
        color: var(--ss-ink) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: var(--ss-ink) !important;
    }}

    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
        color: var(--ss-muted) !important;
        opacity: 1 !important;
        font-size: 12px !important;
    }}

    /* ==================== 四级字阶严格收口 ==================== */
    h1, .ss-h1 {{
        font-size: 24px !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
        letter-spacing: -0.015em !important;
        color: var(--ss-ink) !important;
        margin: 0 0 8px 0 !important;
        padding: 0 !important;
    }}

    h2, h3, h4, .ss-h2 {{
        font-size: 16px !important;
        font-weight: 600 !important;
        line-height: 1.45 !important;
        color: var(--ss-ink) !important;
        margin: 0 0 8px 0 !important;
        padding: 0 !important;
    }}

    p, li, .ss-body, [data-testid="stMarkdownContainer"] p {{
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
        color: var(--ss-ink) !important;
    }}

    small, .ss-caption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        font-size: 12px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        color: var(--ss-muted) !important;
        opacity: 1 !important;
    }}

    hr {{
        border: none !important;
        border-top: 1px solid var(--ss-border) !important;
        margin: 24px 0 !important;
    }}

    /* ==================== 控件统一主题化（杜绝裸奔灰控件） ==================== */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--ss-muted) !important;
        margin-bottom: 4px !important;
    }}

    /* 输入框、数字框、文本域、下拉选择器（同时兼容 BaseWeb 与 React Aria 双代组件） */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"],
    div[data-baseweb="select"] > div,
    [data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
    [data-testid="stMultiSelect"] [data-rac][role="group"] {{
        background-color: var(--ss-bg-card) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
        color: var(--ss-ink) !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}

    div[data-baseweb="select"],
    [data-testid="stSelectbox"] .react-aria-ComboBox {{
        border: none !important;
        background-color: transparent !important;
    }}

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    [data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"]:focus-within,
    [data-testid="stMultiSelect"] [data-rac][role="group"]:focus-within {{
        border-color: var(--ss-accent) !important;
        box-shadow: 0 0 0 3px var(--ss-focus) !important;
    }}

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    div[data-baseweb="textarea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] input[role="combobox"],
    [data-testid="stMultiSelect"] input {{
        color: var(--ss-ink) !important;
        background-color: transparent !important;
        font-size: 14px !important;
        -webkit-text-fill-color: var(--ss-ink) !important;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {{
        color: var(--ss-ink) !important;
        font-size: 14px !important;
        -webkit-text-fill-color: var(--ss-ink) !important;
    }}

    div[data-baseweb="select"] > div div {{
        color: var(--ss-ink) !important;
        background-color: transparent !important;
    }}

    [data-testid="stSelectbox"] .react-aria-ComboBox button {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--ss-muted) !important;
    }}

    div[data-baseweb="select"] svg,
    [data-testid="stSelectbox"] .react-aria-ComboBox svg,
    [data-testid="stMultiSelect"] svg {{
        fill: var(--ss-muted) !important;
        color: var(--ss-muted) !important;
    }}

    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {{
        color: var(--ss-muted) !important;
        opacity: 0.75 !important;
        -webkit-text-fill-color: var(--ss-muted) !important;
    }}

    [data-testid="stNumberInput"] button {{
        background-color: var(--ss-bg-secondary) !important;
        color: var(--ss-ink) !important;
        border-left: 1px solid var(--ss-border) !important;
    }}

    /* 多选 Tag 胶囊 */
    span[data-baseweb="tag"] {{
        background-color: var(--ss-accent-pale) !important;
        color: var(--ss-ink) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: 6px !important;
    }}

    span[data-baseweb="tag"] span {{
        color: var(--ss-ink) !important;
    }}

    /* 下拉浮层菜单（Portal 挂在 body 下，同时兼容 BaseWeb 与 React Aria） */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"],
    .react-aria-Popover,
    .react-aria-ListBox {{
        background-color: var(--ss-bg-card) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
        box-shadow: var(--ss-shadow) !important;
        color: var(--ss-ink) !important;
    }}

    li[role="option"],
    div[role="option"],
    .react-aria-ListBoxItem,
    div[data-baseweb="menu"] li {{
        background-color: transparent !important;
        color: var(--ss-ink) !important;
        font-size: 14px !important;
    }}

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][data-focused],
    div[role="option"][data-hovered],
    div[role="option"][aria-selected="true"],
    .react-aria-ListBoxItem[data-focused],
    .react-aria-ListBoxItem[data-hovered],
    .react-aria-ListBoxItem[data-selected] {{
        background-color: var(--ss-bg-secondary) !important;
        color: var(--ss-accent) !important;
    }}

    /* 单选组与复选框（侧边栏模块导航与面板选项） */
    [data-testid="stRadio"] > div[role="radiogroup"] {{
        gap: 4px !important;
    }}

    label[data-baseweb="radio"] > div:first-child {{
        background-color: var(--ss-bg-card) !important;
        border: 1.5px solid var(--ss-muted) !important;
    }}

    label[data-baseweb="radio"] > div:first-child > div {{
        background-color: var(--ss-bg-card) !important;
    }}

    label[data-baseweb="radio"]:has(input:checked) > div:first-child {{
        background-color: var(--ss-accent) !important;
        border-color: var(--ss-accent) !important;
    }}

    label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {{
        background-color: var(--ss-accent-fg) !important;
    }}

    label[data-baseweb="checkbox"] > span:first-child {{
        background-color: var(--ss-bg-card) !important;
        border: 1.5px solid var(--ss-muted) !important;
    }}

    label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {{
        background-color: var(--ss-accent) !important;
        border-color: var(--ss-accent) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {{
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: var(--ss-radius) !important;
        padding: 8px 12px !important;
        margin: 0 !important;
        width: 100% !important;
        transition: background-color 0.14s ease, border-color 0.14s ease;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
        background-color: var(--ss-hover) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
        background-color: var(--ss-bg-card) !important;
        border-color: var(--ss-border) !important;
        box-shadow: var(--ss-shadow) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {{
        color: var(--ss-accent) !important;
        font-weight: 600 !important;
    }}

    /* 按钮与下载按钮 */
    .stButton > button,
    .stDownloadButton > button {{
        background-color: var(--ss-bg-card) !important;
        color: var(--ss-ink) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
        padding: 8px 16px !important;
        min-height: 38px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        box-shadow: var(--ss-shadow) !important;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: var(--ss-accent) !important;
        color: var(--ss-accent) !important;
        background-color: var(--ss-bg-secondary) !important;
    }}

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {{
        background-color: var(--ss-accent) !important;
        color: var(--ss-accent-fg) !important;
        border-color: var(--ss-accent) !important;
        font-weight: 600 !important;
    }}

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stDownloadButton > button[kind="primary"] p,
    .stDownloadButton > button[kind="primary"] span {{
        color: var(--ss-accent-fg) !important;
    }}

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {{
        background-color: var(--ss-accent-dark) !important;
        border-color: var(--ss-accent-dark) !important;
        color: var(--ss-accent-fg) !important;
    }}

    /* 文件上传拖拽框 */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {{
        background-color: var(--ss-bg-card) !important;
        border: 1px dashed var(--ss-accent) !important;
        border-radius: var(--ss-radius) !important;
        padding: 24px 16px !important;
        color: var(--ss-ink) !important;
    }}

    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div {{
        color: var(--ss-muted) !important;
    }}

    [data-testid="stFileUploaderDropzone"] button {{
        background-color: var(--ss-accent) !important;
        color: var(--ss-accent-fg) !important;
        border: 1px solid var(--ss-accent) !important;
        border-radius: var(--ss-radius) !important;
        font-weight: 600 !important;
    }}

    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stFileUploaderDropzone"] button p {{
        color: var(--ss-accent-fg) !important;
    }}

    [data-testid="stFileUploaderFile"] {{
        background-color: var(--ss-bg-secondary) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
        color: var(--ss-ink) !important;
    }}

    /* 指标卡 stMetric */
    div[data-testid="stMetric"] {{
        background-color: var(--ss-bg-card) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
        padding: 16px !important;
        min-height: 88px !important;
        box-shadow: var(--ss-shadow) !important;
    }}

    div[data-testid="stMetricLabel"] p {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--ss-muted) !important;
    }}

    div[data-testid="stMetricValue"] div {{
        font-size: 26px !important;
        font-weight: 700 !important;
        color: var(--ss-ink) !important;
        font-variant-numeric: tabular-nums !important;
    }}

    div[data-testid="stMetricDelta"] div {{
        font-size: 12px !important;
        color: var(--ss-muted) !important;
    }}

    /* 数据表格 stDataFrame 与 HTML Table */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background-color: var(--ss-bg-card) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
        overflow: hidden !important;
    }}

    [data-testid="stDataFrame"] > div {{
        background-color: var(--ss-bg-card) !important;
        border-radius: var(--ss-radius) !important;
        overflow: hidden !important;
        {'filter: invert(0.895) hue-rotate(180deg) contrast(1.03);' if is_dark else ''}
    }}

    [data-testid="stTable"] table,
    .ss-table {{
        width: 100%;
        border-collapse: collapse;
        background-color: var(--ss-bg-card);
        color: var(--ss-ink);
        font-size: 14px;
    }}

    [data-testid="stTable"] th,
    .ss-table th {{
        background-color: var(--ss-bg-secondary) !important;
        color: var(--ss-ink) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-align: left;
        padding: 10px 14px;
        border-bottom: 1px solid var(--ss-border) !important;
    }}

    [data-testid="stTable"] td,
    .ss-table td {{
        color: var(--ss-ink) !important;
        font-size: 14px !important;
        padding: 10px 14px;
        border-bottom: 1px solid var(--ss-border) !important;
    }}

    /* 折叠面板 stExpander 与选项卡 stTabs */
    [data-testid="stExpander"],
    [data-testid="stExpander"] details {{
        background-color: var(--ss-bg-card) !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: var(--ss-radius) !important;
    }}

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {{
        color: var(--ss-ink) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 16px !important;
        border-bottom: 1px solid var(--ss-border) !important;
        background-color: transparent !important;
    }}

    [data-testid="stTabs"] button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: var(--ss-muted) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 4px 10px 4px !important;
        border: none !important;
    }}

    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--ss-accent) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--ss-accent) !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background-color: var(--ss-accent) !important;
    }}

    /* ==================== SaaS 产品级专用结构组件 ==================== */
    .ss-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        padding: 16px 24px;
        margin-bottom: 24px;
        background-color: var(--ss-bg-card);
        border: 1px solid var(--ss-border);
        border-radius: var(--ss-radius);
        box-shadow: var(--ss-shadow);
    }}

    .ss-brand-group {{
        display: flex;
        align-items: baseline;
        gap: 12px;
        flex-wrap: wrap;
    }}

    .ss-wordmark {{
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: var(--ss-ink);
    }}

    .ss-wordmark-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--ss-accent);
        margin-right: 8px;
        vertical-align: middle;
    }}

    .ss-tagline {{
        font-size: 12px;
        color: var(--ss-muted);
        padding-left: 12px;
        border-left: 1px solid var(--ss-border);
    }}

    .ss-status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: var(--ss-radius);
        background-color: var(--ss-bg-secondary);
        border: 1px solid var(--ss-border);
        font-size: 12px;
        font-weight: 500;
        color: var(--ss-ink);
        font-variant-numeric: tabular-nums;
    }}

    .ss-status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--ss-accent);
    }}

    .ss-status-dot.idle {{
        background-color: var(--ss-muted);
    }}

    .ss-page-header {{
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--ss-border);
    }}

    .ss-hero {{
        padding: 32px;
        margin-bottom: 24px;
        background-color: var(--ss-bg-card);
        border: 1px solid var(--ss-border);
        border-radius: var(--ss-radius);
        box-shadow: var(--ss-shadow);
    }}

    .ss-eyebrow {{
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ss-accent);
        margin-bottom: 8px;
    }}

    .ss-hero-title {{
        font-size: 24px;
        font-weight: 700;
        line-height: 1.35;
        color: var(--ss-ink);
        margin-bottom: 8px;
    }}

    .ss-hero-desc {{
        font-size: 14px;
        line-height: 1.65;
        color: var(--ss-muted);
        max-width: 760px;
        margin-bottom: 0;
    }}

    .ss-grid-3 {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}

    @media (max-width: 900px) {{
        .ss-grid-3 {{
            grid-template-columns: 1fr;
        }}
    }}

    .ss-card {{
        background-color: var(--ss-bg-card);
        border: 1px solid var(--ss-border);
        border-radius: var(--ss-radius);
        padding: 24px;
        box-shadow: var(--ss-shadow);
    }}

    .ss-card-compact {{
        background-color: var(--ss-bg-card);
        border: 1px solid var(--ss-border);
        border-radius: var(--ss-radius);
        padding: 16px;
        box-shadow: var(--ss-shadow);
    }}

    .ss-icon-box {{
        width: 32px;
        height: 32px;
        border-radius: var(--ss-radius);
        background-color: var(--ss-bg-secondary);
        border: 1px solid var(--ss-border);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--ss-accent);
        margin-bottom: 16px;
    }}

    .ss-step-num {{
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        color: var(--ss-accent);
        background-color: var(--ss-bg-secondary);
        border: 1px solid var(--ss-border);
        border-radius: var(--ss-radius);
        padding: 2px 8px;
        margin-bottom: 12px;
        font-variant-numeric: tabular-nums;
    }}

    .ss-conclusion {{
        background-color: var(--ss-bg-card);
        border: 1px solid var(--ss-border);
        border-left: 3px solid var(--ss-accent);
        border-radius: var(--ss-radius);
        padding: 12px 16px;
        margin: 12px 0 16px 0;
        font-size: 14px;
        line-height: 1.6;
        color: var(--ss-ink);
    }}

    .ss-state-box {{
        background-color: var(--ss-bg-card);
        border: 1px solid var(--ss-border);
        border-radius: var(--ss-radius);
        padding: 24px;
        margin: 16px 0;
    }}

    .ss-state-ok {{
        background-color: var(--ss-ok-bg);
        border: 1px solid var(--ss-ok-bd);
        color: var(--ss-ok-fg);
        border-radius: var(--ss-radius);
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 14px;
    }}

    .ss-state-warn {{
        background-color: var(--ss-warn-bg);
        border: 1px solid var(--ss-warn-bd);
        color: var(--ss-warn-fg);
        border-radius: var(--ss-radius);
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 14px;
    }}

    .ss-state-err {{
        background-color: var(--ss-err-bg);
        border: 1px solid var(--ss-err-bd);
        color: var(--ss-err-fg);
        border-radius: var(--ss-radius);
        padding: 16px;
        margin: 12px 0;
        font-size: 14px;
    }}

    .ss-cap-row {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--ss-border);
        font-size: 12px;
    }}

    .ss-cap-row:last-child {{
        border-bottom: none;
    }}

    .ss-badge-on {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: var(--ss-radius);
        background-color: var(--ss-ok-bg);
        border: 1px solid var(--ss-ok-bd);
        color: var(--ss-ok-fg);
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }}

    .ss-badge-off {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: var(--ss-radius);
        background-color: var(--ss-bg-secondary);
        border: 1px solid var(--ss-border);
        color: var(--ss-muted);
        font-size: 12px;
        font-weight: 500;
        white-space: nowrap;
    }}

    .ss-footer {{
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid var(--ss-border);
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 12px;
        color: var(--ss-muted);
    }}
    </style>
    """
