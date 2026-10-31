# -*- coding: utf-8 -*-
"""
SurveyScope — 自适应问卷统计分析与出版级可视化平台 (app.py)
架构原则：
1. 冷启动无内置默认数据，展示正式产品级落地页；
2. 上传 CSV / Excel 原始答卷后，自动识别题型、过滤系统元数据列并按需执行数据质控；
3. 按数据真实列结构动态解锁单题分析、交叉检验、量表建模、市调专项模型、文本挖掘与加权校准；
4. 全站 5 套主题（含深空夜读原生暗色）统一接管容器、侧栏、表格、控件与出版级报告导出。
"""

import html
import io
import numpy as np
import pandas as pd
import streamlit as st

import stats_utils as su
import chart_utils as cu
import ppt_report as pr
from theme import (
    THEMES,
    DEFAULT_THEME_NAME,
    get_theme,
    build_streamlit_theme_css,
    set_use_chart_palettes,
    get_use_chart_palettes,
)
from ai_edit import (
    apply_ai_instruction,
    push_settings_history,
    undo_settings,
    reset_settings_to_default,
)

st.set_page_config(
    page_title="SurveyScope · 问卷数据一键分析平台",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 线性单色 SVG 图标集（严格统一 18x18 / 1.6px 描边 / 跟随主题强调色）
# ==============================================================================
SVG_ICONS = {
    "detect": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1.5"/><path d="M14 17.5h7M17.5 14v7"/></svg>'
    ),
    "dist": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 6h16M4 12h11M4 18h7"/><circle cx="20" cy="6" r="1.5"/>'
        '<circle cx="15" cy="12" r="1.5"/><circle cx="11" cy="18" r="1.5"/></svg>'
    ),
    "cross": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>'
    ),
    "scale": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="6" cy="7" r="2.5"/><circle cx="6" cy="17" r="2.5"/><circle cx="18" cy="12" r="3"/>'
        '<path d="M8.5 8l6.5 3M8.5 16l6.5-3"/></svg>'
    ),
    "market": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg>'
    ),
    "report": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/><path d="M8 13h8M8 17h5"/></svg>'
    ),
}


# ==============================================================================
# Session State 初始化（严禁预载任何内置默认数据）
# ==============================================================================
def _init_session_state():
    defaults = {
        "raw_df": None,
        "dataset_name": "",
        "upload_sig": None,
        "upload_error": "",
        "theme_name": DEFAULT_THEME_NAME,
        "use_chart_palettes": True,
        "qtypes_override": {},
        "qc_params": {
            "check_dup": True,
            "check_straight": True,
            "check_speed": True,
            "min_seconds": 30,
            "remove_flagged": True,
        },
        "chart_settings": {},
        "chart_history": {},
        "session_charts": {},
        "ai_config": {
            "base_url": "https://api.deepseek.com",
            "api_key": "",
            "model": "deepseek-flash",
        },
        "active_module": "数据概览",
        "last_ppt_bytes": None,
        "last_word_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    try:
        qp_theme = st.query_params.get("theme")
        if qp_theme in THEMES and st.session_state.get("_last_qp_theme") != qp_theme:
            st.session_state["theme_name"] = qp_theme
            st.session_state["_last_qp_theme"] = qp_theme
    except Exception:
        pass


_init_session_state()
set_use_chart_palettes(st.session_state["use_chart_palettes"])


# ==============================================================================
# 性能缓存层（2026-10-31）：Streamlit 每次交互都重跑整个脚本，以下纯计算按
# 参数内容哈希缓存，避免每次点击都重算题型识别 / 质控 / 能力探测 / 图表渲染。
# 带下划线前缀的参数不参与哈希、原样透传，保证缓存键稳定且语义不变。
# ==============================================================================
def _freeze(obj):
    """把 dict/list/DataFrame 等参数冻结为可哈希键（仅作缓存键，不改变真实入参）。"""
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        import hashlib
        h = pd.util.hash_pandas_object(obj, index=True).values
        return ("__df__", obj.shape, hashlib.md5(h.tobytes()).hexdigest())
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_auto_qtypes(df: pd.DataFrame):
    return su.auto_detect_question_types(df)


@st.cache_data(max_entries=16, show_spinner=False)
def _cached_caps(df: pd.DataFrame, qtypes_key, _qtypes):
    return su.detect_dataset_capabilities(df, _qtypes)


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_qc(df, check_dup, check_straight, straight_cols, check_speed,
               duration_col, min_seconds, remove_flagged):
    return su.run_quality_control(
        df,
        check_dup=check_dup,
        check_straight=check_straight,
        straight_cols=list(straight_cols),
        check_speed=check_speed,
        duration_col=duration_col,
        min_seconds=min_seconds,
        remove_flagged=remove_flagged,
    )


@st.cache_data(max_entries=32, show_spinner=False)
def _cached_chart_bundle(chart_id, df_input, theme_name, use_palettes,
                         settings_key, kwargs_key, _settings, _kwargs):
    return cu.render_chart_by_id(
        chart_id,
        df_input,
        theme_name=theme_name,
        settings=_settings,
        **_kwargs,
    )


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_theme_css(theme_name):
    return build_streamlit_theme_css(theme_name)


# ==============================================================================
# 文件解析与状态管理辅助函数（带可读中文错误诊断，杜绝暴露原始异常栈）
# ==============================================================================
def _friendly_parse_error(exc: Exception, filename: str) -> str:
    msg = str(exc)
    ext = (filename or "").lower().rsplit(".", 1)[-1] if "." in (filename or "") else ""
    if ext not in ("csv", "xlsx", "xls"):
        return f"文件格式不受支持（当前后缀：.{ext or '未知'}）。请上传标准 .csv、.xlsx 或 .xls 格式的原始答卷表。"
    if "empty" in msg.lower() or "no columns" in msg.lower() or "无有效" in msg:
        return "上传的文件内容为空或未检测到表头列。请确认导出的是「按行排列的受访者原始答卷明细」，而非空白模板。"
    if "codec" in msg.lower() or "decode" in msg.lower() or "encoding" in msg.lower():
        return "CSV 字符编码无法识别。建议在 Excel 或问卷平台中重新导出为 UTF-8（带 BOM）或标准 .xlsx 文件后再上传。"
    if "zip" in msg.lower() or "ole2" in msg.lower() or "corrupt" in msg.lower() or "badzipfile" in msg.lower():
        return "Excel 工作簿结构损坏，或实际为 HTML/文本伪装的表格文件。请在本地用 Excel 打开并另存为标准 .xlsx 格式后重试。"
    return f"未能完成数据表解析（原因：{msg[:120]}）。请检查文件首行是否为题目名称、且至少包含 2 行有效受访者记录。"


def _handle_uploaded_file(uploaded_file) -> bool:
    if uploaded_file is None:
        return False
    raw_bytes = uploaded_file.getvalue()
    sig = f"{uploaded_file.name}:{len(raw_bytes)}:{hash(raw_bytes[:4096])}"
    if sig == st.session_state.get("upload_sig") and isinstance(st.session_state.get("raw_df"), pd.DataFrame):
        return False

    with st.spinner("正在解析问卷编码、过滤系统元数据列并识别题型结构..."):
        try:
            df = su.load_survey_file(raw_bytes, filename=uploaded_file.name)
            if not isinstance(df, pd.DataFrame) or df.empty:
                st.session_state["upload_error"] = "上传的数据表不含任何有效答卷记录（0 行）。请导出完整答卷明细后重新上传。"
                return False
            if df.shape[1] < 2:
                st.session_state["upload_error"] = (
                    f"仅识别到 {df.shape[1]} 列数据。若上传的为 CSV 文件，可能是分隔符异常（请使用标准英文逗号分隔或改传 .xlsx 文件）。"
                )
                return False

            st.session_state["raw_df"] = df
            st.session_state["dataset_name"] = uploaded_file.name
            st.session_state["upload_sig"] = sig
            st.session_state["upload_error"] = ""
            st.session_state["qtypes_override"] = {}
            st.session_state["session_charts"] = {}
            st.session_state["last_ppt_bytes"] = None
            st.session_state["last_word_bytes"] = None
            st.session_state["active_module"] = "数据概览"
            return True
        except Exception as exc:
            st.session_state["upload_error"] = _friendly_parse_error(exc, uploaded_file.name)
            return False


def _clear_current_dataset():
    st.session_state["raw_df"] = None
    st.session_state["dataset_name"] = ""
    st.session_state["upload_sig"] = None
    st.session_state["upload_error"] = ""
    st.session_state["qtypes_override"] = {}
    st.session_state["session_charts"] = {}
    st.session_state["last_ppt_bytes"] = None
    st.session_state["last_word_bytes"] = None
    st.session_state["active_module"] = "数据概览"


# ==============================================================================
# 统一三态组件与顶部产品栏
# ==============================================================================
def render_top_product_bar(raw_df: pd.DataFrame = None, qc_res: dict = None):
    theme_nm = html.escape(st.session_state["theme_name"])
    if raw_df is None:
        status_html = (
            '<div class="ss-status-pill">'
            '<span class="ss-status-dot idle"></span>'
            f'<span>未上传数据 · 主题：{theme_nm}</span>'
            '</div>'
        )
    else:
        fname = html.escape(st.session_state.get("dataset_name") or "已加载问卷")
        n_raw = len(raw_df)
        n_val = qc_res["valid_n"] if qc_res else n_raw
        v_rate = qc_res["valid_rate"] if qc_res else 100.0
        status_html = (
            '<div class="ss-status-pill">'
            '<span class="ss-status-dot"></span>'
            f'<span>{fname} · 有效样本 n={n_val}/{n_raw} ({v_rate:.1f}%) · 主题：{theme_nm}</span>'
            '</div>'
        )

    st.markdown(
        f"""
        <div class="ss-topbar">
            <div class="ss-brand-group">
                <span class="ss-wordmark"><span class="ss-wordmark-dot"></span>SurveyScope</span>
                <span class="ss-tagline">问卷数据一键分析平台</span>
            </div>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="ss-page-header">
            <div class="ss-h1">{html.escape(title)}</div>
            <div class="ss-caption">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, hint: str):
    st.markdown(
        f"""
        <div class="ss-state-box">
            <div class="ss-h2">{html.escape(title)}</div>
            <div class="ss-caption" style="margin-top:4px;">{html.escape(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state(title: str, reason: str, suggestion: str = ""):
    sug_html = f'<div style="margin-top:6px;font-size:12px;opacity:0.9;">建议操作：{html.escape(suggestion)}</div>' if suggestion else ""
    st.markdown(
        f"""
        <div class="ss-state-err">
            <div style="font-weight:600;">{html.escape(title)}</div>
            <div style="margin-top:4px;">{html.escape(reason)}</div>
            {sug_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_conclusion_box(label: str, text: str):
    st.markdown(
        f'<div class="ss-conclusion"><strong>{html.escape(label)}：</strong>{html.escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# 统一交互式图表容器（四件套导出 + 手动格式面板 + 17 动作 AI 样式助手）
# ==============================================================================
@st.fragment
def render_interactive_chart_card(
    chart_id: str,
    df_input: pd.DataFrame,
    instance_key: str = None,
    section_label: str = None,
    **chart_kwargs,
):
    slot_key = instance_key or chart_id
    curr_settings = st.session_state["chart_settings"].get(slot_key, {})
    # 展示渲染延迟生成矢量 PDF（最慢的一件），点「生成矢量 PDF」时再补齐
    disp_settings = dict(curr_settings)
    disp_settings["defer_pdf"] = True

    try:
        with st.spinner("正在按当前主题绘制出版级矢量图表..."):
            bundle = _cached_chart_bundle(
                chart_id,
                df_input,
                st.session_state["theme_name"],
                bool(st.session_state["use_chart_palettes"]),
                _freeze(disp_settings),
                _freeze(chart_kwargs),
                disp_settings,
                chart_kwargs,
            )
    except Exception as exc:
        render_error_state(
            f"图表未能生成（{chart_id}）",
            str(exc),
            "请检查所选题目的有效非空样本量是否充足，或切换其他适配图型。",
        )
        return

    # 自动记录本次会话实际生成过的图表，供报告导出模块按需勾选
    rec_title = section_label or f"{chart_id} · {bundle['title']}"
    st.session_state["session_charts"][slot_key] = {
        "title": rec_title,
        "conclusion": bundle["conclusion"],
        "table": bundle["source_df"],
        "source_df": bundle["source_df"],
        "png_bytes": bundle["png_bytes"],
        "chart_id": chart_id,
    }

    st.markdown(
        f'<div class="ss-card-compact" style="margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;padding:8px 16px;">'
        f'<span class="ss-caption" style="font-weight:600;color:var(--ss-ink);">图型：{html.escape(chart_id)}</span>'
        f'<span class="ss-caption">输出规格 {bundle["width_px"]}×{bundle["height_px"]} px · 支持 PNG / 矢量 SVG / 源数据 CSV / PDF 导出</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _preview_svg = bundle.get("preview_svg_bytes") or b""
    if _preview_svg and len(_preview_svg) <= 900_000:
        # 矢量预览：任意缩放与高分屏下不发虚（下载仍提供 PNG/SVG/CSV/PDF 四件套）。
        # 注意：st.image 识别 SVG 靠字符串入参，传 bytes 会被当位图走 PIL 而报错。
        st.image(_preview_svg.decode("utf-8", errors="replace"), use_container_width=True)
    else:
        st.image(bundle["png_bytes"], use_container_width=True)
    render_conclusion_box("统计发现", bundle["conclusion"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            f"下载 PNG ({bundle['width_px']}px)",
            data=bundle["png_bytes"],
            file_name=f"{slot_key}.png",
            mime="image/png",
            key=f"dl_png_{slot_key}",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "下载矢量 SVG",
            data=bundle["svg_bytes"],
            file_name=f"{slot_key}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{slot_key}",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "下载源数据 CSV",
            data=bundle["csv_bytes"],
            file_name=f"{slot_key}_source.csv",
            mime="text/csv",
            key=f"dl_csv_{slot_key}",
            use_container_width=True,
        )
    with c4:
        pdf_sig = repr((
            st.session_state.get("upload_sig"),
            chart_id,
            st.session_state["theme_name"],
            _freeze(curr_settings),
            _freeze(chart_kwargs),
        ))
        pdf_stash = st.session_state.get("pdf_stash", {}).get(slot_key)
        pdf_ready = bundle.get("pdf_bytes") or (
            pdf_stash[1] if pdf_stash and pdf_stash[0] == pdf_sig else None
        )
        if pdf_ready:
            st.download_button(
                "下载矢量 PDF",
                data=pdf_ready,
                file_name=f"{slot_key}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{slot_key}",
                use_container_width=True,
            )
        elif st.button(
            "生成矢量 PDF",
            key=f"gen_pdf_{slot_key}",
            use_container_width=True,
        ):
            with st.spinner("正在生成矢量 PDF（单次生成，后续可直接下载）..."):
                full_bundle = _cached_chart_bundle(
                    chart_id,
                    df_input,
                    st.session_state["theme_name"],
                    bool(st.session_state["use_chart_palettes"]),
                    _freeze(curr_settings),
                    _freeze(chart_kwargs),
                    curr_settings,
                    chart_kwargs,
                )
            st.session_state.setdefault("pdf_stash", {})[slot_key] = (
                pdf_sig,
                full_bundle["pdf_bytes"],
            )
            st.rerun()

    with st.expander(f"图表格式微调、AI 自然语言样式助手与绘图源数据 — {chart_id}", expanded=False):
        tab_fmt, tab_ai, tab_data = st.tabs(["格式调整", "AI 样式改图", "源数据表"])

        with tab_fmt:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(
                    '<div style="font-size:13px;font-weight:700;color:var(--ss-ink);margin:2px 0 6px 0;">标题与配色</div>',
                    unsafe_allow_html=True,
                )
                new_title = st.text_input(
                    "自定义主标题（留空使用自动结论标题）",
                    value=curr_settings.get("title") or "",
                    key=f"fmt_title_{slot_key}",
                )
                new_sub = st.text_input(
                    "自定义副标题",
                    value=curr_settings.get("subtitle") or "",
                    key=f"fmt_sub_{slot_key}",
                )
                try:
                    from chart_palettes import COLOR_SCHEME_OPTIONS as _pal_opts_all

                    pal_opts = list(_pal_opts_all)
                except Exception:
                    pal_opts = ["主题默认", "松绿序渐", "BrBG发散", "Tol Muted"]
                curr_pal = curr_settings.get("color_scheme") or "主题默认"
                new_palette = st.selectbox(
                    "单图配色覆盖",
                    pal_opts,
                    index=pal_opts.index(curr_pal) if curr_pal in pal_opts else 0,
                    key=f"fmt_pal_{slot_key}",
                )
                # 色卡预览：先看色再选，不用记配色名
                try:
                    from theme import resolve_effective_theme as _ret2
                    from chart_palettes import NAMED_SCHEME_SOURCES as _pal_src

                    _eff = _ret2(
                        st.session_state["theme_name"],
                        {"color_scheme": None if new_palette == "主题默认" else new_palette},
                    )
                    _swatches = "".join(
                        f'<span style="display:inline-block;width:22px;height:14px;border-radius:3px;'
                        f'margin-right:3px;border:1px solid rgba(0,0,0,0.12);background:{_c};"></span>'
                        for _c in _eff["categorical"][:12]
                    )
                    _src_note = _pal_src.get(new_palette, "")
                    if not _src_note and new_palette == "主题默认":
                        try:
                            from chart_palettes import CHART_PALETTES as _cp_all

                            _own_src = _cp_all.get(chart_id, {}).get("source", "")
                            if _own_src:
                                _src_note = f"本图专属色板 · {_own_src}"
                        except Exception:
                            pass
                    st.markdown(
                        f'<div style="margin:2px 0 2px 0;">{_swatches}</div>'
                        + (f'<div style="font-size:12px;opacity:0.75;">出处：{_src_note}</div>' if _src_note else ""),
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass
            with fc2:
                st.markdown(
                    '<div style="font-size:13px;font-weight:700;color:var(--ss-ink);margin:2px 0 6px 0;">数据呈现</div>',
                    unsafe_allow_html=True,
                )
                sort_opts = ["desc", "asc", "original"]
                sort_labels = {"desc": "按数值降序", "asc": "按数值升序", "original": "保持原始顺序"}
                curr_sort = curr_settings.get("sort") or "desc"
                new_sort = st.selectbox(
                    "类别排序方式",
                    sort_opts,
                    index=sort_opts.index(curr_sort) if curr_sort in sort_opts else 0,
                    format_func=lambda k: sort_labels.get(k, k),
                    key=f"fmt_sort_{slot_key}",
                )
                new_topn = st.number_input(
                    "仅显示前 N 项 (0 表示全部)",
                    min_value=0,
                    max_value=30,
                    value=int(curr_settings.get("top_n") or 0),
                    key=f"fmt_topn_{slot_key}",
                )
                new_decimals = st.selectbox(
                    "数据标签小数位",
                    [0, 1, 2],
                    index=int(curr_settings.get("decimals", 1)) if int(curr_settings.get("decimals", 1)) in (0, 1, 2) else 1,
                    key=f"fmt_dec_{slot_key}",
                )
                new_show_lbl = st.checkbox(
                    "显示数据标签",
                    value=bool(curr_settings.get("show_labels", True)),
                    key=f"fmt_lbl_{slot_key}",
                )
                new_show_mean = st.checkbox(
                    "显示均值参考线",
                    value=bool(curr_settings.get("show_mean_line", True)),
                    key=f"fmt_mean_{slot_key}",
                )
            with fc3:
                st.markdown(
                    '<div style="font-size:13px;font-weight:700;color:var(--ss-ink);margin:2px 0 6px 0;">标注与高亮</div>',
                    unsafe_allow_html=True,
                )
                new_fscale = st.slider(
                    "字号缩放倍率",
                    0.80,
                    1.30,
                    float(curr_settings.get("font_scale", 1.0)),
                    0.05,
                    key=f"fmt_fs_{slot_key}",
                )
                hl_target = st.text_input(
                    "指定高亮类别名称",
                    value=(curr_settings.get("highlight") or {}).get("target", ""),
                    key=f"fmt_hlt_{slot_key}",
                )
                _hc1, _hc2 = st.columns([1, 2])
                with _hc1:
                    hl_color = st.color_picker(
                        "高亮颜色",
                        value=(curr_settings.get("highlight") or {}).get("color", "#A50026"),
                        key=f"fmt_hlc_{slot_key}",
                    )
                if "词云" in chart_id:
                    shape_opts = ["无", "云朵", "人形", "爱心", "星星", "咖啡杯", "圆形"]
                    curr_sh = curr_settings.get("wordcloud_shape") or "无"
                    new_wc_shape = st.selectbox(
                        "词云轮廓形状",
                        shape_opts,
                        index=shape_opts.index(curr_sh) if curr_sh in shape_opts else 0,
                        key=f"fmt_shape_{slot_key}",
                    )
                else:
                    new_wc_shape = curr_settings.get("wordcloud_shape") or "无"

            b_apply, b_reset = st.columns(2)
            with b_apply:
                if st.button("应用样式设置", type="primary", key=f"btn_apply_{slot_key}", use_container_width=True):
                    push_settings_history(st.session_state["chart_history"], slot_key, curr_settings)
                    updated = dict(curr_settings)
                    updated.update({
                        "title": new_title.strip() if new_title.strip() else None,
                        "subtitle": new_sub.strip() if new_sub.strip() else None,
                        "color_scheme": None if new_palette == "主题默认" else new_palette,
                        "sort": new_sort,
                        "top_n": int(new_topn) if new_topn > 0 else None,
                        "decimals": int(new_decimals),
                        "show_labels": new_show_lbl,
                        "show_mean_line": new_show_mean,
                        "font_scale": float(new_fscale),
                        "highlight": {"target": hl_target.strip(), "color": hl_color} if hl_target.strip() else None,
                        "wordcloud_shape": new_wc_shape,
                    })
                    st.session_state["chart_settings"][slot_key] = updated
                    st.rerun()
            with b_reset:
                if st.button("恢复默认样式", key=f"btn_reset_{slot_key}", use_container_width=True):
                    st.session_state["chart_settings"][slot_key] = reset_settings_to_default()
                    st.rerun()

        with tab_ai:
            st.caption(
                "支持 17 项受控样式动作（修改主副标题/脚注、类别高亮与单色指定、切换色板、开关标签/图例/均值线/误差棒、"
                "升降序、前 N 项筛选、小数位、字号缩放、词云形状与词数）。严格隔离原始问卷数据，自动拦截任何篡改统计数值请求。"
            )
            src_cats = []
            if isinstance(bundle.get("source_df"), pd.DataFrame) and not bundle["source_df"].empty:
                src_cats = bundle["source_df"].iloc[:, 0].astype(str).tolist()[:30]

            user_prompt = st.text_input(
                "输入自然语言样式指令",
                placeholder="例如：把首位选项标成深红色 / 按升序排列 / 保留两位小数 / 隐藏图例",
                key=f"ai_in_{slot_key}",
            )
            ac1, ac2 = st.columns(2)
            with ac1:
                if st.button("执行样式指令", type="primary", key=f"btn_ai_{slot_key}", use_container_width=True):
                    if user_prompt.strip():
                        with st.spinner("正在解析样式指令，请稍候…"):
                            res_ai = apply_ai_instruction(
                                user_instruction=user_prompt.strip(),
                                current_settings=curr_settings,
                                api_config=st.session_state["ai_config"],
                                chart_type=chart_id,
                                categories=src_cats,
                            )
                        if res_ai["ok"]:
                            push_settings_history(st.session_state["chart_history"], slot_key, curr_settings)
                            st.session_state["chart_settings"][slot_key] = res_ai["new_settings"]
                            st.rerun()
                        else:
                            render_error_state("样式指令未执行", res_ai["message"])
            with ac2:
                if st.button("撤销上一步修改 (最多 5 步)", key=f"btn_undo_{slot_key}", use_container_width=True):
                    prev_s = undo_settings(st.session_state["chart_history"], slot_key)
                    if prev_s is not None:
                        st.session_state["chart_settings"][slot_key] = prev_s
                        st.rerun()
                    else:
                        st.caption("当前没有可撤销的历史记录。")

        with tab_data:
            st.dataframe(bundle["source_df"], use_container_width=True, hide_index=True)


# ==============================================================================
# 侧边栏：主题控制器（全状态常驻）+ 自适应数据与模块导航
# ==============================================================================
with st.sidebar:
    st.markdown(
        '<div style="padding:4px 0 12px 0;border-bottom:1px solid var(--ss-border);margin-bottom:16px;">'
        '<div style="font-size:16px;font-weight:700;letter-spacing:0.02em;">SurveyScope</div>'
        '<div style="font-size:12px;color:var(--ss-muted);">自适应问卷统计分析工作台</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    theme_options = list(THEMES.keys())
    curr_theme_idx = theme_options.index(st.session_state["theme_name"]) if st.session_state["theme_name"] in theme_options else 0
    selected_theme = st.selectbox(
        "主题切换",
        options=theme_options,
        index=curr_theme_idx,
        key="sb_theme_selector",
    )
    if selected_theme != st.session_state["theme_name"]:
        st.session_state["theme_name"] = selected_theme
        st.session_state["last_ppt_bytes"] = None
        st.session_state["last_word_bytes"] = None

    active_t = get_theme(st.session_state["theme_name"], apply_chart_palette=False)
    st.markdown(
        f'<div class="ss-caption" style="margin:2px 0 8px 0;">{html.escape(active_t["description"])}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["theme_name"] == "暖纸杂志":
        pal_toggle = st.checkbox(
            "启用逐图专属色板（MetBrewer / 期刊色）",
            value=st.session_state["use_chart_palettes"],
            key="sb_use_chart_palettes",
        )
        if pal_toggle != st.session_state["use_chart_palettes"]:
            st.session_state["use_chart_palettes"] = pal_toggle
            set_use_chart_palettes(pal_toggle)
        st.caption("配色规则说明：「暖纸杂志」默认按图型匹配独立艺术/期刊色板；取消勾选或切换至其余 4 套主题时，全站图表自动切换为当前主题统一色板。")
    else:
        st.caption(
            f"配色规则说明：当前为「{st.session_state['theme_name']}」主题，已自动停用逐图独立色板，全站图表与报告统一采用本主题无障碍色阶（对比度 ≥ 4.5:1）。"
        )

    st.markdown("---")

    # 若尚未上传数据：侧边栏仅显示置灰引导，不渲染任何模块导航
    if st.session_state["raw_df"] is None:
        st.markdown(
            '<div class="ss-card-compact">'
            '<div style="font-size:12px;font-weight:600;margin-bottom:4px;">分析模块待解锁</div>'
            '<div class="ss-caption">请先在右侧主页面上传 .csv 或 .xlsx/.xls 原始问卷文件。系统识别真实列结构后将自动生成对应分析导航。</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div style="font-size:12px;font-weight:600;color:var(--ss-muted);margin-bottom:8px;">当前数据集管理</div>', unsafe_allow_html=True)
        with st.expander("更换问卷文件", expanded=False):
            sb_up = st.file_uploader(
                "选择新的 .csv / .xlsx / .xls 文件",
                type=["csv", "xlsx", "xls"],
                key="sb_uploader",
            )
            if sb_up is not None and _handle_uploaded_file(sb_up):
                st.rerun()

        if st.button("清空当前数据并返回首页", key="sb_clear_data", use_container_width=True):
            _clear_current_dataset()
            st.rerun()


# 注入全站级主题 CSS（覆盖主画布、侧边栏、数据表、控件与业务组件）
st.markdown(_cached_theme_css(st.session_state["theme_name"]), unsafe_allow_html=True)


# ==============================================================================
# 分支一：无数据状态 — 正式 SaaS 落地页（零假数字、零预置图表）
# ==============================================================================
if st.session_state["raw_df"] is None:
    render_top_product_bar(None, None)

    st.markdown(
        """
        <div class="ss-hero">
            <div class="ss-eyebrow">Adaptive Survey Intelligence</div>
            <div class="ss-hero-title">上传原始问卷数据，按真实题目与题型自动完成严谨统计与报告交付</div>
            <p class="ss-hero-desc">
                SurveyScope 拒绝千篇一律的模板套用与数字拼凑。上传 CSV 或 Excel 答卷明细后，系统自动剥离问卷平台元数据列、
                逐题判定统计类型、按数据实际支持的维度解锁显著性检验与市调专项模型，并输出原生可编辑的出版级 PPTX 与 Word 报告。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 核心上传交互区（三态齐全：上传引导 / Spinner 解析中 / 人话报错卡片）
    up_left, up_right = st.columns([1.45, 1.0], gap="large")
    with up_left:
        st.markdown('<div class="ss-h2">上传原始答卷文件</div>', unsafe_allow_html=True)
        st.caption("支持标准 .csv（自动识别 UTF-8 / GBK / GB18030 编码）与 Excel 工作簿（.xlsx / .xls），每行代表一位受访者，首行为题目名称。")
        landing_file = st.file_uploader(
            "拖拽问卷数据文件至此处，或点击浏览本地文件",
            type=["csv", "xlsx", "xls"],
            key="landing_file_uploader",
        )
        if landing_file is not None:
            if _handle_uploaded_file(landing_file):
                st.rerun()
        if st.session_state.get("upload_error"):
            render_error_state(
                "文件解析未通过",
                st.session_state["upload_error"],
                "请导出问卷星「按文本数据」或「按分数数据」的原始明细表（.xlsx 或 .csv），确认首行为列名后重新上传。",
            )

    with up_right:
        st.markdown(
            """
            <div class="ss-card" style="height:100%;">
                <div class="ss-h2">问卷星 / 腾讯问卷导出格式说明</div>
                <div class="ss-caption" style="margin-top:8px;line-height:1.75;">
                    1. <strong>推荐导出格式</strong>：在问卷平台后台选择「查看下载答卷 → 下载原始答卷 → 按文本数据导出 (.xlsx)」；若包含量表打分题，亦可导出数值编码表。<br/>
                    2. <strong>系统列自动过滤</strong>：平台自带的「序号、提交答卷时间、所用时间、来源、来源详情、来自IP」会自动归为系统元信息，不混入业务题目统计。<br/>
                    3. <strong>多选题格式兼容</strong>：支持分号（；/;）、竖线（|）、顿号（、）分隔的合并多选列，自动区分受访者渗透率与响应人次占比。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="ss-h2" style="margin-bottom:16px;">核心分析能力体系</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ss-grid-3">
            <div class="ss-card">
                <div class="ss-icon-box">{SVG_ICONS['detect']}</div>
                <div class="ss-h2">题型自动识别与自适应质控</div>
                <div class="ss-caption">自动甄别单选、多选、量表、数值、NPS 与开放文本题，按需执行重复样本、量表直线作答与过快答题清洗，缺失列规则自动跳过并注明。</div>
            </div>
            <div class="ss-card">
                <div class="ss-icon-box">{SVG_ICONS['dist']}</div>
                <div class="ss-h2">按题驱动的单题深度分布</div>
                <div class="ss-caption">选中任意题目即刻匹配专属统计表与适配图型（棒棒糖、帕累托、多选渗透率、Likert 发散堆叠、核密度与箱线图），拒绝无关图表堆砌。</div>
            </div>
            <div class="ss-card">
                <div class="ss-icon-box">{SVG_ICONS['cross']}</div>
                <div class="ss-h2">交叉显著性检验与效应量</div>
                <div class="ss-caption">按需指定行×列变量，自动执行卡方检验、Bonferroni 列比例 z 检验字母标注、Welch t / 单因素方差分析及 Cramér's V、Cohen's d、η² 评估。</div>
            </div>
            <div class="ss-card">
                <div class="ss-icon-box">{SVG_ICONS['scale']}</div>
                <div class="ss-h2">信效度诊断与多变量建模</div>
                <div class="ss-caption">自动聚类潜变量维度输出 Cronbach's α 与因子载荷矩阵（EFA），衔接多元线性回归、Logistic ROC 曲线、结构方程路径图与树模型重要度。</div>
            </div>
            <div class="ss-card">
                <div class="ss-icon-box">{SVG_ICONS['market']}</div>
                <div class="ss-h2">市调专项决策模型矩阵</div>
                <div class="ss-caption">依据上传列结构动态启用对应分析（CA）、IPA 四象限、PSM 价格敏感度、KANO 需求分层、联合分析、转化漏斗、K-Means 画像与省级地图。</div>
            </div>
            <div class="ss-card">
                <div class="ss-icon-box">{SVG_ICONS['report']}</div>
                <div class="ss-h2">出版级可编辑 PPTX 与 Word</div>
                <div class="ss-caption">仅纳入当前数据真实支持的分析章节，输出内嵌原生 XML 图表、svgBlip 双层矢量图形与备注栏 Markdown 源数据表的 16:9 演示报告。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ss-h2" style="margin-bottom:16px;">三步标准化研究工作流</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ss-grid-3">
            <div class="ss-card">
                <span class="ss-step-num">STEP 01</span>
                <div class="ss-h2">上传原始答卷宽表</div>
                <div class="ss-caption">上传 CSV 或 Excel 原始调查数据，系统自动完成编码识别、空白行列剥离与样本完整性扫描。</div>
            </div>
            <div class="ss-card">
                <span class="ss-step-num">STEP 02</span>
                <div class="ss-h2">自动识别题型与能力边界</div>
                <div class="ss-caption">第一屏呈现样本规模、题型识别清单与质控结果，并按数据列特征动态构建左侧可用分析模块菜单。</div>
            </div>
            <div class="ss-card">
                <span class="ss-step-num">STEP 03</span>
                <div class="ss-h2">逐题深入检验与勾选导出</div>
                <div class="ss-caption">按需执行单题、交叉与高阶建模分析，支持自然语言微调图表样式，勾选内容清单后一键导出正式报告。</div>
            </div>
        </div>
        <div class="ss-footer">
            <span>SurveyScope v4.0 · 自适应问卷统计分析与出版级报告平台</span>
            <span>统计准则：双侧显著性水平 α = 0.05 (95% CI) · 全站 WCAG AA 对比度认证</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ==============================================================================
# 分支二：已上传数据 — 自适应质控、能力探测与正式分析工作台
# ==============================================================================
raw_df = st.session_state["raw_df"]
auto_qtypes = _cached_auto_qtypes(raw_df)
effective_qtypes = dict(auto_qtypes)
for col_k, override_t in st.session_state["qtypes_override"].items():
    if col_k in effective_qtypes:
        effective_qtypes[col_k] = override_t

raw_caps = _cached_caps(raw_df, _freeze(effective_qtypes), effective_qtypes)
dur_col = raw_caps["duration_col"]
straight_cands = raw_caps["scale_cols"]

qcp = st.session_state["qc_params"]
qc_res = _cached_qc(
    raw_df,
    bool(qcp["check_dup"]),
    bool(qcp["check_straight"]),
    tuple(straight_cands),
    bool(qcp["check_speed"]),
    dur_col,
    float(qcp["min_seconds"]),
    bool(qcp["remove_flagged"]),
)
valid_df = qc_res["valid_df"].copy()
if valid_df.empty:
    valid_df = raw_df.copy()

caps = _cached_caps(valid_df, _freeze(effective_qtypes), effective_qtypes)
sm = caps["special_models"]
manual_ipa_available = (not sm["ipa"]["enabled"]) and (len(caps["scale_cols"]) >= 4)
has_special_nav = bool(caps["modules"]["special_models"] or manual_ipa_available)

# 动态组装侧边栏模块导航（不满足数据条件的模块绝不出现在主导航中）
NAV_MODULES = ["数据概览", "数据质控"]
if caps["modules"]["single_analysis"]:
    NAV_MODULES.append("单题分析")
if caps["modules"]["cross_analysis"]:
    NAV_MODULES.append("交叉分析")
if caps["modules"]["scale_modeling"]:
    NAV_MODULES.append("量表与建模")
if has_special_nav:
    NAV_MODULES.append("专项模型")
if caps["modules"]["text_analysis"]:
    NAV_MODULES.append("文本分析")
if caps["modules"]["weighting"]:
    NAV_MODULES.append("加权校准")
NAV_MODULES.append("报告导出")

if st.session_state["active_module"] not in NAV_MODULES:
    st.session_state["active_module"] = "数据概览"

with st.sidebar:
    st.markdown("---")
    selected_mod = st.radio(
        "分析模块导航",
        options=NAV_MODULES,
        index=NAV_MODULES.index(st.session_state["active_module"]),
        key="sb_nav_radio",
    )
    st.session_state["active_module"] = selected_mod

    st.markdown("---")
    with st.expander("全部能力与数据要求说明", expanded=False):
        st.caption("系统已根据当前上传数据的列结构自动启停专项能力，不满足数据条件的模块自动隐藏：")
        cap_items = [
            ("区域分布地图", sm["region_map"]["enabled"], sm["region_map"]["requirement"]),
            ("品牌转化漏斗", sm["funnel"]["enabled"], sm["funnel"]["requirement"]),
            ("PSM 价格敏感度", sm["psm"]["enabled"], sm["psm"]["requirement"]),
            ("KANO 需求分层", sm["kano"]["enabled"], sm["kano"]["requirement"]),
            ("联合分析 (Conjoint)", sm["conjoint"]["enabled"], sm["conjoint"]["requirement"]),
            ("SEM 结构方程路径", sm["sem"]["enabled"], sm["sem"]["requirement"]),
            ("对应分析 (CA)", sm["correspondence"]["enabled"], sm["correspondence"]["requirement"]),
            ("消费者聚类画像", sm["clustering"]["enabled"], sm["clustering"]["requirement"]),
            ("IPA 四象限矩阵", bool(sm["ipa"]["enabled"] or manual_ipa_available), "成对重要性×满意度列（或 ≥4 道量表题手动配对）"),
            ("流转留存桑基图", sm["brand_switching"]["enabled"], sm["brand_switching"]["requirement"]),
            ("开放题文本挖掘", caps["modules"]["text_analysis"], "至少 1 道非结构化开放文本题（满 10 条出图）"),
            ("事后分层加权", caps["modules"]["weighting"], "包含类别数 2–10 的人口学或分组单选列"),
        ]
        rows_html = []
        for nm, is_on, req_txt in cap_items:
            badge = '<span class="ss-badge-on">已启用</span>' if is_on else '<span class="ss-badge-off">未满足</span>'
            rows_html.append(
                f'<div class="ss-cap-row">'
                f'<div><div style="font-weight:600;color:var(--ss-ink);">{html.escape(nm)}</div>'
                f'<div style="color:var(--ss-muted);margin-top:2px;">{html.escape(req_txt)}</div></div>'
                f'{badge}</div>'
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)

    with st.expander("AI 样式助手接口设置", expanded=False):
        st.caption("未填写 API Key 时默认使用本地确定性规则引擎解析 17 种改图指令；填写后调用兼容大模型接口。")
        st.session_state["ai_config"]["base_url"] = st.text_input("接口地址 (Base URL)", value=st.session_state["ai_config"]["base_url"])
        st.session_state["ai_config"]["api_key"] = st.text_input("接口密钥 (API Key)", value=st.session_state["ai_config"]["api_key"], type="password")
        st.session_state["ai_config"]["model"] = st.text_input("模型名称 (Model)", value=st.session_state["ai_config"]["model"])


# 渲染全站统一顶部产品状态栏
render_top_product_bar(raw_df, qc_res)
current_module = st.session_state["active_module"]


# ==============================================================================
# 模块 1：数据概览（上传后默认第一屏）
# ==============================================================================
if current_module == "数据概览":
    render_page_header(
        "数据概览与题型自动识别",
        "自动盘点样本规模、业务题目类型、缺失率分布及数据质控初筛摘要",
    )
    ov = su.get_dataset_overview(raw_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("原始上传样本量", f"{ov['n_samples']} 份")
    meta_suffix = f"（另剥离元信息 {len(caps['meta_cols'])} 列）" if caps["meta_cols"] else ""
    m2.metric("业务分析题目数", f"{len(caps['analysis_cols'])} 道")
    m3.metric("质控后有效样本", f"{qc_res['valid_n']} 份")
    m4.metric("问卷整体有效率", f"{qc_res['valid_rate']:.1f}%")

    render_conclusion_box("质控摘要", f"{qc_res['conclusion']}{meta_suffix}")
    if qc_res.get("skipped_rules"):
        st.markdown(
            f'<div class="ss-state-warn"><strong>自动跳过的质控规则说明：</strong>{"；".join(html.escape(x) for x in qc_res["skipped_rules"])}。</div>',
            unsafe_allow_html=True,
        )

    # 合并题型与缺失率，构造题目 × 题型 × 缺失率识别总表（保留 Round 3 字典安全构造）
    miss_map_cnt = dict(zip(ov["missing_df"]["题目名称"], ov["missing_df"]["缺失样本数"]))
    miss_map_pct = dict(zip(ov["missing_df"]["题目名称"], ov["missing_df"]["缺失率(%)"]))
    qtype_table = pd.DataFrame({
        "题目": list(effective_qtypes.keys()),
        "自动识别题型": list(effective_qtypes.values()),
        "有效样本数": [int(len(raw_df) - miss_map_cnt.get(c, 0)) for c in effective_qtypes.keys()],
        "缺失数": [int(miss_map_cnt.get(c, 0)) for c in effective_qtypes.keys()],
        "缺失率(%)": [float(miss_map_pct.get(c, 0.0)) for c in effective_qtypes.keys()],
    })

    st.markdown('<div class="ss-h2" style="margin-top:16px;">题目类型与缺失率识别清单</div>', unsafe_allow_html=True)
    st.dataframe(qtype_table, use_container_width=True, hide_index=True)

    with st.expander("手动修正个别题目的识别题型（可选）", expanded=False):
        st.caption("若个别数值编码题希望改按单选题或量表题统计，可在此指定覆盖并立即同步全站模块：")
        adj_c1, adj_c2, adj_c3 = st.columns([2, 1.2, 1])
        with adj_c1:
            target_q_adj = st.selectbox("选择需调整的列名", list(effective_qtypes.keys()), key="ov_adj_col")
        with adj_c2:
            all_t_opts = ["单选题", "多选题", "量表题", "数值题", "NPS题", "开放文本题", "元信息"]
            curr_t_val = effective_qtypes.get(target_q_adj, "单选题")
            new_t_val = st.selectbox(
                "指定目标题型",
                all_t_opts,
                index=all_t_opts.index(curr_t_val) if curr_t_val in all_t_opts else 0,
                key="ov_adj_type",
            )
        with adj_c3:
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            if st.button("应用题型修正", type="primary", key="ov_adj_btn", use_container_width=True):
                st.session_state["qtypes_override"][target_q_adj] = new_t_val
                st.rerun()

    st.markdown('<div class="ss-h2" style="margin-top:24px;">原始问卷数据前 20 行预览</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True)


# ==============================================================================
# 模块 2：数据质控与异常清洗
# ==============================================================================
elif current_module == "数据质控":
    render_page_header(
        "数据质控与异常样本筛查",
        "依据当前问卷具备的列特征执行重复答卷、量表直线作答与作答时长筛查",
    )

    with st.container():
        st.markdown('<div class="ss-h2">质控规则参数配置</div>', unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            c_dup = st.checkbox("启用完全重复答卷筛查", value=bool(qcp["check_dup"]), key="qc_ui_dup")
        with q2:
            has_st_cols = len(straight_cands) >= 2
            c_st = st.checkbox(
                f"启用量表直线作答筛查 ({len(straight_cands)}题)" if has_st_cols else "直线作答筛查（无量表列自动跳过）",
                value=bool(qcp["check_straight"] and has_st_cols),
                disabled=not has_st_cols,
                key="qc_ui_straight",
            )
        with q3:
            has_dur = dur_col is not None
            c_sp = st.checkbox(
                f"启用过快作答筛查（{dur_col}）" if has_dur else "过快作答筛查（无时长列自动跳过）",
                value=bool(qcp["check_speed"] and has_dur),
                disabled=not has_dur,
                key="qc_ui_speed",
            )
        with q4:
            c_rm = st.checkbox("从后续分析中剔除异常样本", value=bool(qcp["remove_flagged"]), key="qc_ui_rm")

        if has_dur and c_sp:
            min_sec_val = st.slider("最短有效作答时长阈值（秒）", 10, 300, int(qcp["min_seconds"]), 5, key="qc_ui_sec")
        else:
            min_sec_val = int(qcp["min_seconds"])

        if (
            c_dup != qcp["check_dup"]
            or (has_st_cols and c_st != qcp["check_straight"])
            or (has_dur and c_sp != qcp["check_speed"])
            or c_rm != qcp["remove_flagged"]
            or min_sec_val != qcp["min_seconds"]
        ):
            st.session_state["qc_params"] = {
                "check_dup": c_dup,
                "check_straight": c_st if has_st_cols else False,
                "check_speed": c_sp if has_dur else False,
                "min_seconds": min_sec_val,
                "remove_flagged": c_rm,
            }
            st.rerun()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("重复答卷标记", f"{qc_res['dup_n']} 份")
    k2.metric("直线作答标记", f"{qc_res['straight_n']} 份" if len(straight_cands) >= 2 else "已跳过")
    k3.metric("过快作答标记", f"{qc_res['speed_n']} 份" if dur_col else "已跳过")
    k4.metric("异常样本合计", f"{qc_res['flagged_n']} 份")
    k5.metric("保留有效样本", f"{qc_res['valid_n']} 份", f"有效率 {qc_res['valid_rate']:.1f}%")

    render_conclusion_box("质控执行结论", qc_res["conclusion"])

    st.markdown('<div class="ss-h2" style="margin-top:24px;">异常样本明细核查</div>', unsafe_allow_html=True)
    flagged_mask = qc_res["marked_df"]["是否异常样本"]
    if int(flagged_mask.sum()) == 0:
        render_empty_state(
            "当前质控条件下未检出异常样本",
            f"全部 {qc_res['total_n']} 份原始答卷均通过核验，已全量纳入有效分析样本集。",
        )
    else:
        flagged_detail = pd.concat(
            [qc_res["marked_df"].loc[flagged_mask, ["重复作答", "直线作答", "作答过快"]], raw_df.loc[flagged_mask]],
            axis=1,
        )
        st.dataframe(flagged_detail, use_container_width=True)


# ==============================================================================
# 模块 3：单题分析（严格按题型分组 → 选一题 → 仅渲染该题适配统计与图表）
# ==============================================================================
elif current_module == "单题分析":
    render_page_header(
        "单题深度分布分析",
        "按自动识别题型选择具体题目，系统仅生成与该题测量尺度匹配的统计指标与图表",
    )

    type_groups = []
    for grp_label, col_list in [
        ("单选题", caps["single_cols"]),
        ("多选题", caps["multi_cols"]),
        ("量表题", caps["scale_cols"]),
        ("数值题", caps["numeric_cols"]),
        ("NPS题", caps["nps_cols"]),
        ("开放文本题", caps["text_cols"]),
    ]:
        if col_list:
            type_groups.append((grp_label, col_list))

    if not type_groups:
        render_empty_state("未检测到可分析的业务题目", "请在「数据概览」中核对各列的自动识别题型设置。")
    else:
        s_col1, s_col2 = st.columns([1.0, 2.0])
        with s_col1:
            grp_names = [f"{g}（{len(cs)} 题）" for g, cs in type_groups]
            sel_grp_idx = st.selectbox("第一步：选择题型分组", range(len(type_groups)), format_func=lambda i: grp_names[i], key="sq_grp_sel")
            active_qtype, active_qlist = type_groups[sel_grp_idx]
        with s_col2:
            sel_q = st.selectbox("第二步：选择要分析的题目", active_qlist, key=f"sq_col_sel_{active_qtype}")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ---- 3.1 单选题 ----
        if active_qtype == "单选题":
            res_cat = su.analyze_categorical_question(valid_df[sel_q])
            freq_df = res_cat["table"]
            n_cats = len(freq_df)
            top_row = freq_df.iloc[0] if not freq_df.empty else None

            m1, m2, m3 = st.columns(3)
            m1.metric("有效作答样本", f"{res_cat['n_valid']} 份")
            m2.metric("有效选项类别数", f"{n_cats} 项")
            m3.metric("首位众数选项", f"{top_row['选项']} ({top_row['占比(%)']:.1f}%)" if top_row is not None else "—")

            render_conclusion_box(f"「{sel_q}」统计摘要", res_cat["conclusion"])
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

            chart_choices = [
                ("01-棒棒糖图", "棒棒糖图（占比与频数排序）"),
                ("02-水平条形图", "水平条形图（标准占比对比）"),
                ("03-帕累托图", "帕累托图（累计集中度曲线）"),
            ]
            if n_cats <= 6:
                chart_choices.append(("05-甜甜圈图", "甜甜圈图（类别 ≤6 构成占比）"))
                chart_choices.append(("04-华夫饼图", "华夫饼百格图（类别 ≤6 网格占比）"))

            sel_cid = st.radio(
                "切换适配图表样式",
                options=[c[0] for c in chart_choices],
                format_func=lambda cid: dict(chart_choices)[cid],
                horizontal=True,
                key=f"sq_cat_chart_{sel_q}",
            )
            render_interactive_chart_card(
                sel_cid,
                valid_df,
                instance_key=f"{sel_cid}__{sel_q}",
                section_label=f"单题分析：{sel_q}",
                col=sel_q,
            )

        # ---- 3.2 多选题 ----
        elif active_qtype == "多选题":
            res_mc = su.split_multiple_choice(valid_df[sel_q])
            mc_df = res_mc["summary_df"]
            m1, m2, m3 = st.columns(3)
            m1.metric("有效作答人数 (Cases)", f"{res_mc['n_cases']} 人")
            m2.metric("总勾选响应人次 (Responses)", f"{res_mc['total_responses']} 人次")
            avg_pick = res_mc["total_responses"] / max(res_mc["n_cases"], 1)
            m3.metric("人均勾选选项数", f"{avg_pick:.2f} 项")

            render_conclusion_box(
                f"「{sel_q}」多选渗透率摘要",
                f"{res_mc['conclusion']}（统计口径说明：受访者普及率 % of Cases 以有效作答人数 n={res_mc['n_cases']} 为分母，各选项之和大于 100% 属正常多选特征。）",
            )
            st.dataframe(mc_df, use_container_width=True, hide_index=True)
            render_interactive_chart_card(
                "06-多选题条形图",
                valid_df,
                instance_key=f"06__{sel_q}",
                section_label=f"多选题分析：{sel_q}",
                col=sel_q,
            )

        # ---- 3.3 量表题 ----
        elif active_qtype == "量表题":
            res_num = su.analyze_numeric_question(valid_df[sel_q])
            dims_map = caps["scale_dimensions"]
            matched_dim_name = None
            matched_dim_cols = [sel_q]
            for d_nm, d_cols in dims_map.items():
                if sel_q in d_cols:
                    matched_dim_name = d_nm
                    matched_dim_cols = d_cols
                    break
            if matched_dim_name is None and len(caps["scale_cols"]) >= 2:
                matched_dim_name = "全部量表题项"
                matched_dim_cols = caps["scale_cols"][:10]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("本题均值 ± 标准差", f"{res_num['mean']:.2f} ± {res_num['sd']:.2f}")
            m2.metric("中位数 (IQR)", f"{res_num['median']:.1f} ({res_num['iqr']:.1f})")
            if len(matched_dim_cols) >= 2:
                w_rel = valid_df.copy()
                rel_cols_clean = []
                for c_it in matched_dim_cols:
                    if "反向" in str(c_it):
                        pos_nm = f"{c_it}_正向折算"
                        w_rel[pos_nm] = su.reverse_score(w_rel[c_it], max_scale=5)
                        rel_cols_clean.append(pos_nm)
                    else:
                        rel_cols_clean.append(c_it)
                rel_info = su.calc_reliability(w_rel, rel_cols_clean)
                m3.metric(f"所属维度（{matched_dim_name}）题数", f"{len(matched_dim_cols)} 题")
                m4.metric("同维度 Cronbach's α", f"{rel_info['alpha']:.3f}")
                render_conclusion_box(f"「{sel_q}」及所属维度统计结论", f"{res_num['conclusion']} {rel_info['conclusion']}")
                with st.expander(f"查看「{matched_dim_name}」维度内部一致性信度明细表", expanded=False):
                    st.dataframe(rel_info["item_table"], use_container_width=True, hide_index=True)
            else:
                m3.metric("有效样本量", f"{res_num['n']} 份")
                m4.metric("95% 均值置信区间", f"[{res_num['ci_low']:.2f}, {res_num['ci_high']:.2f}]")
                render_conclusion_box(f"「{sel_q}」统计摘要", res_num["conclusion"])

            plot_cols = matched_dim_cols[:10] if len(matched_dim_cols) >= 2 else [sel_q]
            sc_view = st.radio(
                "选择量表可视化视图",
                ["12-均值置信区间图", "11-发散堆叠条形图"],
                format_func=lambda c: "均值棒棒糖与 95% 置信区间图（同维度对比）" if c.startswith("12") else "Likert 态度发散堆叠条形图（正负向占比分布）",
                horizontal=True,
                key=f"sq_scale_view_{sel_q}",
            )
            render_interactive_chart_card(
                sc_view,
                valid_df,
                instance_key=f"{sc_view}__{matched_dim_name or sel_q}",
                section_label=f"量表分布：{matched_dim_name or sel_q}",
                cols=plot_cols,
            )

        # ---- 3.4 数值题 ----
        elif active_qtype == "数值题":
            res_num = su.analyze_numeric_question(valid_df[sel_q])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("均值 (Mean)", f"{res_num['mean']:.2f}")
            m2.metric("标准差 (SD)", f"{res_num['sd']:.2f}")
            m3.metric("中位数 (Median)", f"{res_num['median']:.2f}")
            m4.metric("四分位距 (IQR)", f"{res_num['iqr']:.2f}")

            render_conclusion_box(f"「{sel_q}」描述统计与正态性诊断", res_num["conclusion"])
            desc_tbl = pd.DataFrame([{
                "题目": sel_q,
                "有效样本(n)": res_num["n"],
                "均值": round(res_num["mean"], 3),
                "标准差": round(res_num["sd"], 3),
                "最小值": round(res_num["min"], 3),
                "P25": round(res_num["q25"], 3),
                "中位数": round(res_num["median"], 3),
                "P75": round(res_num["q75"], 3),
                "最大值": round(res_num["max"], 3),
                "偏度": round(res_num["skew"], 3),
                "峰度": round(res_num["kurt"], 3),
            }])
            st.dataframe(desc_tbl, use_container_width=True, hide_index=True)

            num_chart = st.radio(
                "切换数值分布图型",
                ["07-直方图与核密度", "08a-箱线图", "08b-小提琴图", "10-经验累积分布图"],
                horizontal=True,
                key=f"sq_num_chart_{sel_q}",
            )
            render_interactive_chart_card(
                num_chart,
                valid_df,
                instance_key=f"{num_chart}__{sel_q}",
                section_label=f"数值分布：{sel_q}",
                col=sel_q,
            )

        # ---- 3.5 NPS题 ----
        elif active_qtype == "NPS题":
            res_nps = su.analyze_nps_question(valid_df[sel_q])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("净推荐值 (NPS)", f"{res_nps['nps']:+.1f}")
            m2.metric("推荐者 (9–10分)", f"{res_nps['promoter_pct']:.1f}% ({res_nps['promoters']}人)")
            m3.metric("中立者 (7–8分)", f"{res_nps['passive_pct']:.1f}% ({res_nps['passives']}人)")
            m4.metric("贬损者 (0–6分)", f"{res_nps['detractor_pct']:.1f}% ({res_nps['detractors']}人)")

            render_conclusion_box(f"「{sel_q}」净推荐值诊断", res_nps["conclusion"])
            st.dataframe(res_nps["table"], use_container_width=True, hide_index=True)
            render_interactive_chart_card(
                "13-NPS仪表盘",
                valid_df,
                instance_key=f"13__{sel_q}",
                section_label=f"NPS 净推荐值：{sel_q}",
                col=sel_q,
            )

        # ---- 3.6 开放文本题（严格遵守有效文本 <10 条不出图原则） ----
        elif active_qtype == "开放文本题":
            n_txt = caps["text_valid_counts"].get(sel_q, 0)
            clean_txt_s = valid_df[sel_q].dropna().astype(str).str.strip()
            clean_txt_s = clean_txt_s[~clean_txt_s.isin(["", "nan", "None", "无", "没有", "略"])]

            if n_txt < 10:
                render_empty_state(
                    f"有效非空文本样本不足（当前仅 {n_txt} 条，需至少 10 条）",
                    "为避免极少样本生成失真的词云与共现关系，系统已自动停用图形渲染，以下直接展示原始回答记录：",
                )
                st.dataframe(pd.DataFrame({"原始开放回答": clean_txt_s.tolist()}), use_container_width=True, hide_index=True)
            else:
                txt_res = su.analyze_open_text(valid_df[sel_q], top_n=20)
                render_conclusion_box(f"「{sel_q}」高频词统计摘要", txt_res["conclusion"])
                st.dataframe(txt_res["freq_df"], use_container_width=True, hide_index=True)

                txt_cid = st.radio(
                    "选择文本可视化图型",
                    ["46-词云与词频组合图", "49-高频词共现网络图"],
                    horizontal=True,
                    key=f"sq_txt_chart_{sel_q}",
                )
                render_interactive_chart_card(
                    txt_cid,
                    valid_df,
                    instance_key=f"{txt_cid}__{sel_q}",
                    section_label=f"开放题分析：{sel_q}",
                    col=sel_q,
                )


# ==============================================================================
# 模块 4：交叉分析（选行题 × 列题后才执行计算；未选择时只展示选择器与方法说明）
# ==============================================================================
elif current_module == "交叉分析":
    render_page_header(
        "双变量交叉分析与显著性检验",
        "选定分组行变量与对比列变量后实时匹配卡方检验、参数均值检验或双数值相关回归",
    )

    placeholder = "— 请选择题目 —"
    cross_pool = caps["single_cols"] + caps["multi_cols"] + caps["scale_cols"] + caps["numeric_cols"] + caps["nps_cols"]

    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        row_var = st.selectbox(
            "行变量（分组题 / 自变量 X）",
            options=[placeholder] + cross_pool,
            index=0,
            key="cross_row_var",
        )
    with c_sel2:
        col_cands = [c for c in cross_pool if c != row_var]
        col_var = st.selectbox(
            "列变量（目标题 / 因变量 Y）",
            options=[placeholder] + col_cands,
            index=0,
            key="cross_col_var",
        )

    if row_var == placeholder or col_var == placeholder:
        render_empty_state(
            "请选择行变量与列变量以启动交叉检验",
            "自动检验匹配规则：① 分类题 × 分类题 → Pearson 卡方检验、Cramér's V 效应量与 Bonferroni 列比例 z 检验字母标注；"
            "② 分类题 × 数值/量表/NPS题 → Welch 独立样本 t 检验（Cohen's d）或单因素方差分析 ANOVA（η² 与 Tukey 事后检验）；"
            "③ 数值/量表题 × 数值/量表题 → Pearson / Spearman 相关系数与 OLS 线性拟合置信带。",
        )
    else:
        t_row = effective_qtypes.get(row_var, "单选题")
        t_col = effective_qtypes.get(col_var, "单选题")
        cat_types = {"单选题", "多选题"}
        quant_types = {"量表题", "数值题", "NPS题"}

        st.markdown("<hr/>", unsafe_allow_html=True)

        # 情形 A：分类 × 分类
        if t_row in cat_types and t_col in cat_types:
            work_ct = valid_df.copy()
            r_use, c_use = row_var, col_var
            if t_row == "多选题":
                work_ct[r_use] = work_ct[r_use].astype(str).str.split(r"[;；|｜、,]").str[0].str.strip()
                st.caption(f"注：行变量「{row_var}」为多选题，列联卡方检验按受访者首项主选类别归组。")
            if t_col == "多选题":
                work_ct[c_use] = work_ct[c_use].astype(str).str.split(r"[;；|｜、,]").str[0].str.strip()
                st.caption(f"注：列变量「{col_var}」为多选题，列联卡方检验按受访者首项主选类别归组。")

            ct_res = su.crosstab_chi2_analysis(work_ct, r_use, c_use)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pearson χ² 统计量", f"{ct_res['chi2']:.3f}")
            m2.metric("显著性 p 值", f"{ct_res['p_value']:.4f}")
            m3.metric("效应量 Cramér's V", f"{ct_res['cramers_v']:.3f}")
            m4.metric("自由度 (df)", f"{ct_res['dof']}")

            render_conclusion_box(f"「{row_var} × {col_var}」卡方检验与列比例比较结论", ct_res["conclusion"])

            t_pct, t_cnt, t_res = st.tabs([
                "列百分比与 Bonferroni z 检验标注",
                "观测频数交叉表",
                "调整后标准化残差矩阵",
            ])
            with t_pct:
                st.caption("表头字母 [A], [B], [C]... 代表列代号；单元格内附带字母表示该列占比在 α=0.05（Bonferroni 校正）水平下显著高于对应字母列。")
                st.dataframe(ct_res["annotated_col_pct_df"], use_container_width=True)
            with t_cnt:
                st.dataframe(ct_res["obs_df"], use_container_width=True)
            with t_res:
                st.caption("绝对值 > 1.96 表示该单元格观测频数与独立性期望频数存在显著正/负偏离。")
                st.dataframe(ct_res["std_resid_df"], use_container_width=True)

            cat_cross_charts = [
                "15-100%堆叠条形图",
                "16-分组簇状条形图",
                "18-卡方残差热力图",
                "17-马赛克图",
            ]
            if work_ct[c_use].dropna().nunique() == 2:
                cat_cross_charts.append("47-蝴蝶对比图")

            chosen_cc = st.radio("选择交叉可视化图型", cat_cross_charts, horizontal=True, key=f"cc_chart_{r_use}_{c_use}")
            kw = {"group_col": c_use, "target_col": r_use} if chosen_cc == "47-蝴蝶对比图" else {"row_col": r_use, "col_col": c_use}
            render_interactive_chart_card(
                chosen_cc,
                work_ct,
                instance_key=f"{chosen_cc}__{r_use}__{c_use}",
                section_label=f"交叉分析：{row_var} × {col_var}",
                **kw,
            )

        # 情形 B：分类 × 数值/量表（若顺序相反则自动将分类题作为分组变量）
        elif (t_row in cat_types and t_col in quant_types) or (t_row in quant_types and t_col in cat_types):
            g_col = row_var if t_row in cat_types else col_var
            n_col = col_var if t_row in cat_types else row_var
            work_gm = valid_df.copy()
            if effective_qtypes.get(g_col) == "多选题":
                work_gm[g_col] = work_gm[g_col].astype(str).str.split(r"[;；|｜、,]").str[0].str.strip()

            gm_res = su.group_mean_comparison(work_gm, g_col, n_col)
            if "stat_val" not in gm_res:
                render_empty_state("有效分组不足 2 组", gm_res.get("conclusion", "请更换包含至少 2 个非空类别的分组题目。"))
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("检验方法", gm_res["test_name"])
                m2.metric("检验统计量", f"{gm_res['stat_val']:.3f}")
                m3.metric("显著性 p 值", f"{gm_res['p_value']:.4f}")
                m4.metric(f"效应量 ({gm_res['effect_name']})", f"{gm_res['effect_size']:.3f}")

                render_conclusion_box(f"「{g_col} × {n_col}」分组差异检验结论", gm_res["conclusion"])
                disp_cols = [c for c in ["分组", "组别代号", "样本量(n)", "均值", "标准差(SD)", "95%CI下限", "95%CI上限", "显著高于组(α=0.05)"] if c in gm_res["table"].columns]
                st.dataframe(gm_res["table"][disp_cols], use_container_width=True, hide_index=True)

                if isinstance(gm_res.get("tukey_df"), pd.DataFrame) and not gm_res["tukey_df"].empty:
                    with st.expander("查看 Tukey HSD 事后两两多重比较明细表", expanded=False):
                        st.dataframe(gm_res["tukey_df"], use_container_width=True, hide_index=True)

                cn_charts = ["09-雨云图", "22-均值标准误条形图", "21-分组小提琴箱线图", "23-山脊图"]
                chosen_cn = st.radio("选择分组分布图型", cn_charts, horizontal=True, key=f"cn_chart_{g_col}_{n_col}")
                render_interactive_chart_card(
                    chosen_cn,
                    work_gm,
                    instance_key=f"{chosen_cn}__{g_col}__{n_col}",
                    section_label=f"分组对比：{g_col} × {n_col}",
                    group_col=g_col,
                    num_col=n_col,
                )

        # 情形 C：数值/量表 × 数值/量表
        else:
            cp_res = su.correlation_pair_analysis(valid_df[row_var], valid_df[col_var])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pearson 相关系数 r", f"{cp_res['pearson_r']:.3f}")
            m2.metric("显著性 p 值", f"{cp_res['pearson_p']:.4f}")
            m3.metric("Spearman 秩相关 ρ", f"{cp_res['spearman_r']:.3f}")
            m4.metric("决定系数 R²", f"{cp_res['r_sq']:.3f}")

            render_conclusion_box(f"「{row_var} × {col_var}」双变量相关分析结论", cp_res["conclusion"])
            render_interactive_chart_card(
                "25-散点回归图",
                valid_df,
                instance_key=f"25__{row_var}__{col_var}",
                section_label=f"散点回归：{row_var} × {col_var}",
                x_col=row_var,
                y_col=col_var,
            )


# ==============================================================================
# 模块 5：量表信效度与驱动力建模
# ==============================================================================
elif current_module == "量表与建模":
    render_page_header(
        "量表信效度诊断与多变量预测建模",
        "依据当前问卷的量表与连续变量结构，按需执行信度、因子分析、回归森林图及机器学习评估",
    )

    quant_all = list(dict.fromkeys(caps["scale_cols"] + caps["numeric_cols"] + caps["nps_cols"]))
    model_tabs = []
    if len(caps["scale_cols"]) >= 2:
        model_tabs.append("信度与探索性因子分析")
    if len(quant_all) >= 2:
        model_tabs.append("多变量相关性矩阵")
    if len(quant_all) >= 2 and len(valid_df) >= 20:
        model_tabs.append("回归与机器学习重要度")
    if sm["sem"]["enabled"]:
        model_tabs.append("结构方程路径模型 (SEM)")

    active_m_tab = st.radio("选择分析子模块", model_tabs, horizontal=True, key="mod_sub_radio")
    st.markdown("<hr/>", unsafe_allow_html=True)

    if active_m_tab == "信度与探索性因子分析":
        sel_scale_items = st.multiselect(
            "选择纳入信效度检验的量表题项（至少 2 题；含「反向」字样题项支持自动反向折算）",
            options=caps["scale_cols"],
            default=caps["scale_cols"][: min(8, len(caps["scale_cols"]))],
            key="rel_scale_sel",
        )
        auto_rev = st.checkbox("对题名含「反向」的题项自动执行反向计分折算", value=True, key="rel_auto_rev")

        if len(sel_scale_items) < 2:
            render_empty_state("请至少选择 2 道量表题项", "Cronbach's α 内部一致性信度计算要求题项数 k ≥ 2。")
        else:
            work_sc = valid_df.copy()
            use_items = []
            for c_it in sel_scale_items:
                if auto_rev and "反向" in str(c_it):
                    p_nm = f"{c_it}_正向"
                    work_sc[p_nm] = su.reverse_score(work_sc[c_it], max_scale=5)
                    use_items.append(p_nm)
                else:
                    use_items.append(c_it)

            rel_res = su.calc_reliability(work_sc, use_items)
            can_efa = len(use_items) >= 3 and len(work_sc[use_items].dropna()) >= 15
            if can_efa:
                max_f = min(5, len(use_items) - 1)
                n_fac = st.slider("探索性因子分析提取公因子数", 1, max(1, max_f), min(2, max_f), key="efa_n_fac")
                efa_res = su.calc_validity_efa(work_sc, use_items, n_factors=n_fac)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cronbach's α 系数", f"{rel_res['alpha']:.3f}")
                m2.metric("KMO 抽样适切性", f"{efa_res['kmo']:.3f}")
                m3.metric("Bartlett 球形检验 p", f"{efa_res['bartlett_p']:.4f}")
                m4.metric("累计方差解释率", f"{efa_res['cum_var_pct']:.1f}%")
                render_conclusion_box("信效度综合诊断", f"{rel_res['conclusion']} {efa_res['conclusion']}")
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown('<div class="ss-h2">题项 CITC 与删题后 α 表</div>', unsafe_allow_html=True)
                    st.dataframe(rel_res["item_table"], use_container_width=True, hide_index=True)
                with c_r:
                    st.markdown('<div class="ss-h2">Varimax 正交旋转因子载荷矩阵</div>', unsafe_allow_html=True)
                    st.dataframe(efa_res["loadings_df"], use_container_width=True)
            else:
                m1, m2 = st.columns(2)
                m1.metric("Cronbach's α 系数", f"{rel_res['alpha']:.3f}")
                m2.metric("有效完整样本", f"{rel_res['n']} 份")
                render_conclusion_box("信度分析结论", rel_res["conclusion"])
                st.dataframe(rel_res["item_table"], use_container_width=True, hide_index=True)

    elif active_m_tab == "多变量相关性矩阵":
        sel_corr_cols = st.multiselect(
            "选择纳入相关性矩阵的定量指标（2–10 列）",
            options=quant_all,
            default=quant_all[: min(7, len(quant_all))],
            key="corr_cols_sel",
        )
        if len(sel_corr_cols) < 2:
            render_empty_state("请至少选择 2 个定量变量", "相关性热力图与气泡矩阵需要至少 2 列数值或量表指标。")
        else:
            corr_chart = st.radio(
                "选择相关性呈现样式",
                ["26-相关性热力图", "27-气泡矩阵图"],
                horizontal=True,
                key="corr_chart_type",
            )
            render_interactive_chart_card(
                corr_chart,
                valid_df,
                instance_key=f"{corr_chart}__{len(sel_corr_cols)}",
                section_label="多变量相关性矩阵",
                cols=sel_corr_cols[:10],
            )

    elif active_m_tab == "回归与机器学习重要度":
        rc1, rc2 = st.columns([1.0, 2.0])
        with rc1:
            default_tgt = quant_all[-1]
            y_reg = st.selectbox("选择因变量 / 预测目标 (Y)", quant_all, index=quant_all.index(default_tgt), key="reg_y_sel")
        with rc2:
            x_cands = [c for c in quant_all if c != y_reg]
            x_reg = st.multiselect(
                "选择自变量 / 预测特征 (X)",
                options=x_cands,
                default=x_cands[: min(6, len(x_cands))],
                key=f"reg_x_sel_{y_reg}",
            )

        if len(x_reg) < 1:
            render_empty_state("请至少选择 1 个自变量特征", "选定自变量后将自动拟合多元线性回归与预测模型。")
        else:
            ols_res = su.run_linear_regression(valid_df, y_reg, x_reg)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("决定系数 R²", f"{ols_res['r2']:.3f}")
            m2.metric("调整后 Adj R²", f"{ols_res['adj_r2']:.3f}")
            m3.metric("模型整体 F 值", f"{ols_res['f_value']:.2f} (p={ols_res['f_pvalue']:.4f})")
            m4.metric("最高方差膨胀因子 VIF", f"{ols_res['max_vif']:.2f}")

            render_conclusion_box("多元线性回归结论", ols_res["conclusion"])
            st.dataframe(ols_res["coef_table"], use_container_width=True, hide_index=True)

            ml_choices = ["38-回归系数森林图", "39-ROC曲线图"]
            if len(x_reg) >= 2:
                ml_choices.extend(["40-随机森林重要性图", "41-决策树规则图", "42-混淆矩阵图"])
            sel_ml_cid = st.radio("选择模型评估视图", ml_choices, horizontal=True, key=f"ml_cid_{y_reg}")

            if sel_ml_cid in ("38-回归系数森林图", "39-ROC曲线图"):
                render_interactive_chart_card(
                    sel_ml_cid,
                    valid_df,
                    instance_key=f"{sel_ml_cid}__{y_reg}",
                    section_label=f"回归建模：{y_reg}",
                    y_col=y_reg,
                    x_cols=x_reg,
                )
            else:
                render_interactive_chart_card(
                    sel_ml_cid,
                    valid_df,
                    instance_key=f"{sel_ml_cid}__{y_reg}",
                    section_label=f"树模型评估：{y_reg}",
                    target_col=y_reg,
                    feature_cols=x_reg,
                )

    elif active_m_tab == "结构方程路径模型 (SEM)":
        sem_dims = sm["sem"]["dims"]
        dim_names = list(sem_dims.keys())
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            d1_sel = st.selectbox("外生潜变量维度 1 (ξ1)", dim_names, index=0, key="sem_d1")
        with sc2:
            d2_opts = [d for d in dim_names if d != d1_sel]
            d2_sel = st.selectbox("外生潜变量维度 2 (ξ2)", d2_opts, index=0, key="sem_d2")
        with sc3:
            tgt_opts = sm["sem"]["targets"]
            sem_tgt = st.selectbox("内生结果变量 (η)", tgt_opts, index=0, key="sem_tgt")

        d1_cols_clean = []
        d2_cols_clean = []
        work_sem = valid_df.copy()
        for c_i in sem_dims[d1_sel]:
            if "反向" in str(c_i):
                nm_p = f"{c_i}_正向"
                work_sem[nm_p] = su.reverse_score(work_sem[c_i], max_scale=5)
                d1_cols_clean.append(nm_p)
            else:
                d1_cols_clean.append(c_i)
        for c_i in sem_dims[d2_sel]:
            if "反向" in str(c_i):
                nm_p = f"{c_i}_正向"
                work_sem[nm_p] = su.reverse_score(work_sem[c_i], max_scale=5)
                d2_cols_clean.append(nm_p)
            else:
                d2_cols_clean.append(c_i)

        sem_res = su.calc_sem_path_model(
            work_sem,
            dim1_cols=d1_cols_clean,
            dim2_cols=d2_cols_clean,
            target_col=sem_tgt,
            dim1_name=d1_sel,
            dim2_name=d2_sel,
        )
        render_conclusion_box("结构方程路径估计结论", sem_res["conclusion"])
        st.dataframe(sem_res["path_table"], use_container_width=True, hide_index=True)
        render_interactive_chart_card(
            "34-结构方程路径图",
            work_sem,
            instance_key=f"34__{d1_sel}_{d2_sel}_{sem_tgt}",
            section_label="结构方程路径模型 (SEM)",
            dim1_cols=d1_cols_clean,
            dim2_cols=d2_cols_clean,
            target_col=sem_tgt,
            dim1_name=d1_sel,
            dim2_name=d2_sel,
        )


# ==============================================================================
# 模块 6：市调专项模型（严格仅展示满足当前数据集列条件的模型）
# ==============================================================================
elif current_module == "专项模型":
    render_page_header(
        "市场研究专项模型",
        "以下仅列出当前上传数据集真实满足字段要求的专项模型，绝不使用合成数据填充",
    )

    enabled_spec = []
    if sm["region_map"]["enabled"]:
        enabled_spec.append(("region_map", "区域分布地图"))
    if sm["correspondence"]["enabled"]:
        enabled_spec.append(("correspondence", "对应分析 (CA)"))
    if sm["funnel"]["enabled"]:
        enabled_spec.append(("funnel", "品牌转化漏斗"))
    if sm["brand_switching"]["enabled"]:
        enabled_spec.append(("brand_switching", "流转留存桑基图"))
    if sm["ipa"]["enabled"] or manual_ipa_available:
        enabled_spec.append(("ipa", "IPA 重要性-满意度四象限"))
    if sm["psm"]["enabled"]:
        enabled_spec.append(("psm", "PSM 价格敏感度测试"))
    if sm["kano"]["enabled"]:
        enabled_spec.append(("kano", "KANO 需求分层模型"))
    if sm["conjoint"]["enabled"]:
        enabled_spec.append(("conjoint", "联合分析 (Conjoint)"))
    if sm["clustering"]["enabled"]:
        enabled_spec.append(("clustering", "消费者聚类画像"))

    spec_keys = [k for k, _ in enabled_spec]
    spec_labels = dict(enabled_spec)
    chosen_spec = st.radio(
        "选择已解锁的专项模型",
        options=spec_keys,
        format_func=lambda k: spec_labels[k],
        horizontal=True,
        key="spec_model_sel",
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    if chosen_spec == "region_map":
        p_col = sm["region_map"]["col"]
        pv_res = su.calc_province_distribution(valid_df, col=p_col)
        render_conclusion_box("省级样本分布摘要", pv_res["conclusion"])
        c_map, c_tbl = st.columns([1.45, 1.0], gap="large")
        with c_map:
            render_interactive_chart_card(
                "44-区域分布地图",
                valid_df,
                instance_key=f"44__{p_col}",
                section_label=f"区域分布地图：{p_col}",
                col=p_col,
            )
        with c_tbl:
            st.markdown('<div class="ss-h2">各省级行政区样本明细表</div>', unsafe_allow_html=True)
            st.dataframe(pv_res["table"], use_container_width=True, hide_index=True)

    elif chosen_spec == "correspondence":
        ca_cols = sm["correspondence"]["cols"]
        cc1, cc2 = st.columns(2)
        with cc1:
            ca_r = st.selectbox("行分类变量", ca_cols, index=0, key="ca_r_sel")
        with cc2:
            ca_c_opts = [c for c in ca_cols if c != ca_r]
            ca_c = st.selectbox("列分类/多选变量", ca_c_opts, index=0, key="ca_c_sel")
        is_mc = effective_qtypes.get(ca_c) == "多选题"
        ca_res = su.calc_correspondence_analysis(valid_df, ca_r, ca_c, is_multi_col=is_mc)
        render_conclusion_box("对应分析惯量解释结论", ca_res["conclusion"])
        render_interactive_chart_card(
            "31-对应分析双标图",
            valid_df,
            instance_key=f"31__{ca_r}__{ca_c}",
            section_label=f"对应分析：{ca_r} × {ca_c}",
            row_col=ca_r,
            col_col=ca_c,
            is_multi_col=is_mc,
        )

    elif chosen_spec == "funnel":
        f_cols = sm["funnel"]["cols"]
        fn_res = su.calc_brand_funnel(valid_df, f_cols, stage_names=[str(c) for c in f_cols])
        render_conclusion_box("转化漏斗流失诊断", fn_res["conclusion"])
        st.dataframe(fn_res["table"], use_container_width=True, hide_index=True)
        render_interactive_chart_card(
            "29-漏斗图",
            valid_df,
            instance_key="29__funnel",
            section_label="转化漏斗分析",
            cols=f_cols,
        )

    elif chosen_spec == "brand_switching":
        p_c, cur_c = sm["brand_switching"]["cols"]
        sw_res = su.calc_brand_switching(valid_df, p_c, cur_c)
        render_conclusion_box("前期至当期流转与留存结论", sw_res["conclusion"])
        st.dataframe(sw_res["retention_df"], use_container_width=True, hide_index=True)
        render_interactive_chart_card(
            "30-桑基图",
            valid_df,
            instance_key=f"30__{p_c}__{cur_c}",
            section_label=f"流转桑基图：{p_c} → {cur_c}",
            prev_col=p_c,
            curr_col=cur_c,
        )

    elif chosen_spec == "ipa":
        ipa_info = sm["ipa"]["info"]
        if ipa_info["enabled"]:
            imp_c, sat_c, attr_nms = ipa_info["imp_cols"], ipa_info["sat_cols"], ipa_info["attr_names"]
        else:
            st.caption("未检测到自动命名的「重要性_X / 满意度_X」列，请手动指定等长的两组量表列进行配对分析：")
            ic1, ic2 = st.columns(2)
            half = len(caps["scale_cols"]) // 2
            with ic1:
                imp_c = st.multiselect("选择重要性量表列（≥2 列）", caps["scale_cols"], default=caps["scale_cols"][: max(2, min(4, half))], key="ipa_man_imp")
            with ic2:
                rem_sc = [c for c in caps["scale_cols"] if c not in imp_c]
                sat_c = st.multiselect("选择一一对应的满意度量表列", rem_sc, default=rem_sc[: len(imp_c)], key="ipa_man_sat")
            attr_nms = [str(c) for c in imp_c]

        if len(imp_c) < 2 or len(imp_c) != len(sat_c):
            render_empty_state("请确保重要性列与满意度列数量一致且不少于 2 对", "两组题项按顺序一一配对计算象限归属。")
        else:
            ipa_res = su.calc_ipa_analysis(valid_df, imp_c, sat_c, attr_nms)
            render_conclusion_box("IPA 象限诊断结论", ipa_res["conclusion"])
            st.dataframe(ipa_res["table"], use_container_width=True, hide_index=True)
            render_interactive_chart_card(
                "32-IPA四象限图",
                valid_df,
                instance_key=f"32__{len(imp_c)}",
                section_label="IPA 重要性-满意度四象限",
                imp_cols=imp_c,
                sat_cols=sat_c,
                names=attr_nms,
            )

    elif chosen_spec == "psm":
        pc = sm["psm"]["cols"]
        psm_res = su.calc_van_westendorp(valid_df, pc["too_cheap"], pc["cheap"], pc["expensive"], pc["too_expensive"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最优价格点 (OPP)", f"{psm_res['opp'][0]:.2f} 元")
        m2.metric("无差异价格点 (IPP)", f"{psm_res['ipp'][0]:.2f} 元")
        m3.metric("最低可接受价 (PMC)", f"{psm_res['pmc'][0]:.2f} 元")
        m4.metric("最高可接受价 (PME)", f"{psm_res['pme'][0]:.2f} 元")
        render_conclusion_box("Van Westendorp 价格敏感度结论", psm_res["conclusion"])
        st.dataframe(psm_res["table"], use_container_width=True, hide_index=True)
        render_interactive_chart_card(
            "33-PSM价格敏感度曲线",
            valid_df,
            instance_key="33__psm",
            section_label="PSM 价格敏感度测试",
            cols=pc,
        )

    elif chosen_spec == "kano":
        k_info = sm["kano"]["info"]
        kano_res = su.calc_kano_model(
            valid_df,
            pos_cols=k_info["pos_cols"],
            neg_cols=k_info["neg_cols"],
            feature_names=k_info["feature_names"],
        )
        render_conclusion_box("KANO 需求属性归类结论", kano_res["conclusion"])
        st.dataframe(kano_res["table"], use_container_width=True, hide_index=True)
        render_interactive_chart_card(
            "35-KANO模型图",
            valid_df,
            instance_key="35__kano",
            section_label="KANO 需求分层模型",
            pos_cols=k_info["pos_cols"],
            neg_cols=k_info["neg_cols"],
            feature_names=k_info["feature_names"],
        )

    elif chosen_spec == "conjoint":
        cj_info = sm["conjoint"]["info"]
        cj_res = su.calc_conjoint_analysis(valid_df, attr_cols=cj_info["attr_cols"], rating_col=cj_info["rating_col"])
        render_conclusion_box("联合分析部分价值与相对重要性结论", cj_res["conclusion"])
        c_imp, c_utl = st.columns(2)
        with c_imp:
            st.dataframe(cj_res["importance_df"], use_container_width=True, hide_index=True)
        with c_utl:
            st.dataframe(cj_res["utility_df"], use_container_width=True, hide_index=True)

        cj_view = st.radio("切换联合分析图表", ["36-联合分析重要性图", "37-联合分析效用曲线"], horizontal=True, key="cj_view_sel")
        render_interactive_chart_card(
            cj_view,
            valid_df,
            instance_key=f"{cj_view}__conjoint",
            section_label="联合分析 (Conjoint)",
            attr_cols=cj_info["attr_cols"],
            rating_col=cj_info["rating_col"],
        )

    elif chosen_spec == "clustering":
        cl_pool = sm["clustering"]["cols"]
        kc1, kc2 = st.columns([2.2, 1.0])
        with kc1:
            sel_cl_cols = st.multiselect(
                "选择参与 K-Means 聚类的定量指标（3–6 列）",
                options=cl_pool,
                default=cl_pool[: min(4, len(cl_pool))],
                key="cl_feat_sel",
            )
        with kc2:
            k_num = st.slider("分群聚类簇数 (k)", 2, 5, 3, key="cl_k_sel")

        if len(sel_cl_cols) < 3:
            render_empty_state("请至少选择 3 列数值或量表特征", "多维客群分群画像至少需要 3 个定量特征维度。")
        else:
            km_res = su.run_kmeans_clustering(valid_df, sel_cl_cols[:6], k=k_num, seed=42)
            render_conclusion_box("K-Means 聚类分群结论", km_res["conclusion"])
            st.dataframe(km_res["profile_df"], use_container_width=True, hide_index=True)

            cl_view = st.radio("选择客群可视化视图", ["45-消费者画像卡片", "28-平行坐标图"], horizontal=True, key="cl_view_sel")
            if cl_view == "45-消费者画像卡片":
                render_interactive_chart_card(
                    cl_view,
                    valid_df,
                    instance_key=f"45__k{k_num}_{len(sel_cl_cols)}",
                    section_label="消费者聚类画像卡片",
                    feat_cols=sel_cl_cols[:4],
                    k=k_num,
                )
            else:
                render_interactive_chart_card(
                    cl_view,
                    valid_df,
                    instance_key=f"28__k{k_num}_{len(sel_cl_cols)}",
                    section_label="客群特征平行坐标图",
                    cols=sel_cl_cols[:6],
                )


# ==============================================================================
# 模块 7：开放题文本分析
# ==============================================================================
elif current_module == "文本分析":
    render_page_header(
        "开放题非结构化文本挖掘",
        "中文分词高频词提取、词云轮廓渲染、同句语义共现网络与极性情感拆解",
    )

    tc1, tc2 = st.columns([1.4, 1.6])
    with tc1:
        sel_t_col = st.selectbox("选择开放文本题目", caps["text_cols"], key="txt_mod_col")
    with tc2:
        extra_sw_str = st.text_input("补充自定义停用词（多个词用空格或逗号分隔）", value="", key="txt_mod_sw")

    n_valid_txt = caps["text_valid_counts"].get(sel_t_col, 0)
    clean_s = valid_df[sel_t_col].dropna().astype(str).str.strip()
    clean_s = clean_s[~clean_s.isin(["", "nan", "None", "无", "没有", "略"])]

    if n_valid_txt < 10:
        render_empty_state(
            f"「{sel_t_col}」有效回答仅 {n_valid_txt} 条（低于 10 条绘图阈值）",
            "文本条数过少时不硬性生成词云与共现网络，请直接审阅下方受访者原始作答文本：",
        )
        st.dataframe(pd.DataFrame({"受访者原始文本": clean_s.tolist()}), use_container_width=True, hide_index=True)
    else:
        custom_sw = {w.strip() for w in extra_sw_str.replace("，", ",").replace(",", " ").split() if w.strip()}
        txt_info = su.analyze_open_text(valid_df[sel_t_col], extra_stopwords=custom_sw, top_n=25)
        sent_info = su.calc_sentiment_analysis(valid_df[sel_t_col])

        m1, m2, m3 = st.columns(3)
        m1.metric("有效非空评论数", f"{n_valid_txt} 条")
        m2.metric("首位高频核心词", f"{txt_info['freq_df'].iloc[0]['词语']} ({txt_info['freq_df'].iloc[0]['词频']}次)" if not txt_info["freq_df"].empty else "—")
        pos_p = float(sent_info["overall_df"].loc[sent_info["overall_df"]["情感极性"] == "正向", "占比(%)"].iloc[0])
        m3.metric("正向情感评论占比", f"{pos_p:.1f}%")

        render_conclusion_box("文本挖掘与情感极性综合结论", f"{txt_info['conclusion']} {sent_info['conclusion']}")

        txt_sub_view = st.radio(
            "选择文本分析图表",
            ["46-词云与词频组合图", "49-高频词共现网络图", "50-情感分布图"],
            horizontal=True,
            key=f"txt_mod_view_{sel_t_col}",
        )
        render_interactive_chart_card(
            txt_sub_view,
            valid_df,
            instance_key=f"{txt_sub_view}__mod_{sel_t_col}",
            section_label=f"文本分析：{sel_t_col}",
            col=sel_t_col,
        )


# ==============================================================================
# 模块 8：事后分层加权校准（保留 Round 3 加权键名与对比接口契约）
# ==============================================================================
elif current_module == "加权校准":
    render_page_header(
        "事后分层抽样加权校准",
        "依据目标总体人口学结构设定配比权重，消除样本抽样偏差并对比加权前后关键指标变化",
    )

    w_col = st.selectbox("选择分层基准变量（人口学或核心分组题）", caps["weight_cols"], index=0, key="wt_base_col")
    obs_vc = valid_df[w_col].dropna().astype(str).value_counts(normalize=True) * 100.0
    cats_list = obs_vc.index.tolist()

    st.markdown('<div class="ss-h2" style="margin-top:8px;">设定目标总体各层占比（合计须为 100.0%）</div>', unsafe_allow_html=True)
    target_props = {}
    if len(cats_list) == 2:
        c_a, c_b = cats_list[0], cats_list[1]
        default_a = round(float(obs_vc.iloc[0]), 1)
        val_a = st.slider(f"目标「{c_a}」占比 (%)", 5.0, 95.0, 50.0, 0.5, key=f"wt_sl_{w_col}")
        target_props[c_a] = val_a
        target_props[c_b] = round(100.0 - val_a, 2)
        st.caption(f"样本实测：{c_a} {default_a:.1f}% / {c_b} {100.0 - default_a:.1f}% → 当前目标设定：{c_a} {val_a:.1f}% / {c_b} {100.0 - val_a:.1f}%")
    else:
        eq_share = round(100.0 / len(cats_list), 1)
        cols_ui = st.columns(min(4, len(cats_list)))
        for idx_c, cat_nm in enumerate(cats_list):
            default_v = eq_share if idx_c < len(cats_list) - 1 else round(100.0 - eq_share * (len(cats_list) - 1), 1)
            with cols_ui[idx_c % len(cols_ui)]:
                target_props[cat_nm] = st.number_input(
                    f"「{cat_nm}」目标占比(%)",
                    min_value=0.1,
                    max_value=99.9,
                    value=float(default_v),
                    step=0.5,
                    key=f"wt_num_{w_col}_{cat_nm}",
                )

    prop_sum = sum(target_props.values())
    if abs(prop_sum - 100.0) > 0.5:
        render_error_state(
            f"目标占比总和当前为 {prop_sum:.1f}%（须等于 100.0%）",
            "请调整各层目标百分比，使总和精确等于 100% 后再计算加权结果。",
        )
    else:
        wt_res = su.compute_post_stratification_weights(valid_df[w_col].astype(str), target_props)
        render_conclusion_box("分层权重诊断", wt_res["conclusion"])
        st.dataframe(wt_res["table"], use_container_width=True, hide_index=True)

        weighted_work = valid_df.copy()
        weighted_work["_weight"] = wt_res["weight_series"]

        cmp_pool = [c for c in (caps["single_cols"] + caps["numeric_cols"] + caps["scale_cols"] + caps["nps_cols"]) if c != w_col]
        if cmp_pool:
            st.markdown('<div class="ss-h2" style="margin-top:24px;">加权前后核心指标对比验证</div>', unsafe_allow_html=True)
            sel_cmp_col = st.selectbox("选择要对比验证的业务题目", cmp_pool, index=0, key=f"wt_cmp_{w_col}")
            is_num_cmp = effective_qtypes.get(sel_cmp_col) in ("数值题", "量表题", "NPS题")
            cmp_df = su.compare_weighted_unweighted(weighted_work, sel_cmp_col, "_weight", is_numeric=is_num_cmp)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)


# ==============================================================================
# 模块 9：自适应报告导出（导出前给出内容勾选清单，仅包含真实支持的分析章节）
# ==============================================================================
elif current_module == "报告导出":
    render_page_header(
        "出版级报告与清洗数据导出",
        "严格依据当前数据集实际支持的分析范围组装报告，导出前可自主勾选纳入章节",
    )

    avail_chapters = pr.get_available_report_chapters(valid_df, effective_qtypes)
    default_report_title = st.session_state.get("dataset_name", "").rsplit(".", 1)[0] or "问卷调查统计分析报告"

    st.markdown('<div class="ss-h2">第一步：确认报告标题与纳入章节清单</div>', unsafe_allow_html=True)
    rpt_title = st.text_input("报告封面标题", value=default_report_title, key="exp_rpt_title")

    selected_chapter_keys = []
    for ch in avail_chapters:
        checked = st.checkbox(
            f"{ch['title']} — {ch['summary']}",
            value=True,
            key=f"exp_ch_{ch['key']}",
        )
        if checked:
            selected_chapter_keys.append(ch["key"])

    session_custom_list = list(st.session_state.get("session_charts", {}).values())
    include_session_charts = False
    if session_custom_list:
        include_session_charts = st.checkbox(
            f"同时附入本次会话已交互浏览/微调过的 {len(session_custom_list)} 张自定义图表",
            value=False,
            key="exp_inc_session",
        )

    if not selected_chapter_keys:
        render_error_state("请至少勾选 1 个报告章节", "报告内容清单为空时无法生成文档。")
    else:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="ss-h2" style="margin-bottom:16px;">第二步：生成并下载交付文件</div>', unsafe_allow_html=True)

        ov_exp = su.get_dataset_overview(raw_df)
        e1, e2, e3 = st.columns(3, gap="medium")

        with e1:
            st.markdown(
                '<div class="ss-card-compact" style="margin-bottom:12px;">'
                '<div class="ss-h2">16:9 可编辑演示文稿 (.pptx)</div>'
                '<div class="ss-caption">内嵌原生可编辑图表、svgBlip 矢量图形与备注栏 Markdown 源数据表，配色跟随当前主题。</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("生成 PPT 演示报告", type="primary", key="btn_gen_ppt", use_container_width=True):
                with st.spinner("正在按勾选章节编译可编辑 PPTX 演示报告..."):
                    try:
                        exp_opts = {
                            "report_title": rpt_title,
                            "enabled_chapters": selected_chapter_keys,
                            "qtypes_override": effective_qtypes,
                            "include_custom_sections": include_session_charts,
                        }
                        ppt_bytes = pr.build_ppt_report(
                            ov_exp,
                            qc_res,
                            report_sections=session_custom_list if include_session_charts else None,
                            theme_name=st.session_state["theme_name"],
                            df=valid_df,
                            export_options=exp_opts,
                        )
                        st.session_state["last_ppt_bytes"] = ppt_bytes
                    except Exception as exc:
                        render_error_state("PPT 报告生成失败", str(exc))

            if st.session_state.get("last_ppt_bytes"):
                st.download_button(
                    "下载 SurveyScope_分析报告.pptx",
                    data=st.session_state["last_ppt_bytes"],
                    file_name=f"{rpt_title}_{st.session_state['theme_name']}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="dl_final_ppt",
                    use_container_width=True,
                )

        with e2:
            st.markdown(
                '<div class="ss-card-compact" style="margin-bottom:12px;">'
                '<div class="ss-h2">图文分析与源数据报告 (.docx)</div>'
                '<div class="ss-caption">包含样本概况、质控说明、勾选章节高清图表、统计检验结论与末尾完整源数据表附录。</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("生成 Word 分析报告", type="primary", key="btn_gen_word", use_container_width=True):
                with st.spinner("正在按勾选章节编译 Word 分析报告与数据附录..."):
                    try:
                        word_secs = pr.build_adaptive_word_sections(
                            valid_df,
                            theme_name=st.session_state["theme_name"],
                            enabled_chapters=selected_chapter_keys,
                            qtypes_override=effective_qtypes,
                            custom_sections=session_custom_list if include_session_charts else None,
                        )
                        word_bytes = pr.build_word_report(
                            ov_exp,
                            qc_res,
                            word_secs,
                            theme_name=st.session_state["theme_name"],
                        )
                        st.session_state["last_word_bytes"] = word_bytes
                    except Exception as exc:
                        render_error_state("Word 报告生成失败", str(exc))

            if st.session_state.get("last_word_bytes"):
                st.download_button(
                    "下载 SurveyScope_分析报告.docx",
                    data=st.session_state["last_word_bytes"],
                    file_name=f"{rpt_title}_{st.session_state['theme_name']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_final_word",
                    use_container_width=True,
                )

        with e3:
            st.markdown(
                '<div class="ss-card-compact" style="margin-bottom:12px;">'
                '<div class="ss-h2">质控清洗后有效样本 (.xlsx)</div>'
                '<div class="ss-caption">导出当前质控规则筛查后的干净宽表数据，可直接用于 SPSS、R 或归档备查。</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            excel_bytes = su.export_cleaned_excel(valid_df)
            st.download_button(
                "下载清洗后有效数据 (.xlsx)",
                data=excel_bytes,
                file_name=f"{rpt_title}_清洗后有效样本.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_final_xlsx",
                use_container_width=True,
            )

# redeploy trigger 2026-10-31 (footnote cleanup + vector preview)
