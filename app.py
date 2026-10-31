# -*- coding: utf-8 -*-
"""
SurveyScope — 问卷数据一键分析与出版级可视化平台 (app.py)
支持：
- 5 套出版级视觉主题实时切换（暖纸杂志 / 墨蓝学术 / 纯白极简 / 深空夜读 / 薄荷清新）
- 全部 48 种图型（49 张样板图 + 共现网络/情感分布扩展图）
- 每张图标配：四件套下载（PNG>=2400px / 矢量 SVG / 源数据 CSV / 矢量 PDF）+ 手动格式面板 + AI 自然语言改图助手
- 10 大业务模块，导航顺序与出版级 PPT 报告章节严格对齐
"""

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

import stats_utils as su
import chart_utils as cu
import ppt_report as pr
from theme import THEMES, DEFAULT_THEME_NAME, get_theme, resolve_effective_theme
from ai_edit import (
    apply_ai_instruction,
    push_settings_history,
    undo_settings,
    reset_settings_to_default,
)

st.set_page_config(
    page_title="SurveyScope · 问卷数据一键分析与出版级报告平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Session State 初始化
# ==============================================================================
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = su.generate_demo_coffee_data(seed=42, n=300)
    st.session_state["is_demo"] = True
if "theme_name" not in st.session_state:
    st.session_state["theme_name"] = DEFAULT_THEME_NAME
if "chart_settings" not in st.session_state:
    st.session_state["chart_settings"] = {}
if "chart_history" not in st.session_state:
    st.session_state["chart_history"] = {}
if "ai_config" not in st.session_state:
    st.session_state["ai_config"] = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
    }
if "weight_col" not in st.session_state:
    st.session_state["weight_col"] = None


# ==============================================================================
# 侧边栏：主题切换器、数据源管理、模块导航、AI 接口配置
# ==============================================================================
with st.sidebar:
    st.markdown("## 🎨 视觉主题与全局设置")
    selected_theme = st.selectbox(
        "选择全局图表与报告主题",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state["theme_name"]),
        help="切换后全站图表、页面卡片及导出的 PPT/Word 报告自动同步最新主题配色。",
    )
    st.session_state["theme_name"] = selected_theme
    active_t = get_theme(selected_theme)

    # 主题色板实时预览条
    swatches_html = "".join([
        f'<span style="display:inline-block;width:18px;height:18px;border-radius:4px;margin-right:4px;background:{c};border:1px solid #888;"></span>'
        for c in [active_t["bg_canvas"], active_t["accent_primary"], active_t["accent_secondary"], active_t["highlight"]] + active_t["categorical"][:4]
    ])
    st.markdown(f'<div style="margin-bottom:6px;">{swatches_html}</div><small style="color:{active_t["text_secondary"]};">{active_t["description"]}</small>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📂 问卷数据源")
    data_mode = st.radio("数据模式", ["内置咖啡消费演示数据 (n=300)", "上传本地问卷文件 (CSV/Excel)"], index=0 if st.session_state["is_demo"] else 1)
    if data_mode.startswith("内置"):
        if not st.session_state["is_demo"] or st.button("🔄 重新生成演示数据"):
            st.session_state["raw_df"] = su.generate_demo_coffee_data(seed=42, n=300)
            st.session_state["is_demo"] = True
            st.session_state["weight_col"] = None
    else:
        up_file = st.file_uploader("上传 .csv 或 .xlsx/.xls 文件", type=["csv", "xlsx", "xls"])
        if up_file is not None:
            st.session_state["raw_df"] = su.load_survey_file(up_file.getvalue(), up_file.name)
            st.session_state["is_demo"] = False
            st.session_state["weight_col"] = None

    st.markdown("---")
    st.markdown("## 🧭 业务分析模块导航")
    MODULES = [
        "01 概览与题型识别",
        "02 数据质控与清洗",
        "03 变量加工与合成",
        "04 单题分析 (01–14图型)",
        "05 交叉分析 (15–25, 47图型)",
        "06 量表与高阶建模 (26–28, 34, 38–42)",
        "07 市调特色模型 (29–33, 35–37, 43–45, 48)",
        "08 开放题文本分析 (46, 49, 50)",
        "09 事后分层加权",
        "10 报告导出与49图全览",
    ]
    current_module = st.radio("按 PPT 报告叙事顺序排列：", MODULES, index=0)

    st.markdown("---")
    with st.expander("🤖 AI 改图助手 API 配置（仅存内存）", expanded=False):
        st.caption("留空 API Key 时自动启用本地智能规则引擎解析改图指令；填写后可调用 OpenAI 兼容大模型。")
        st.session_state["ai_config"]["base_url"] = st.text_input("Base URL", value=st.session_state["ai_config"]["base_url"])
        st.session_state["ai_config"]["api_key"] = st.text_input("API Key", value=st.session_state["ai_config"]["api_key"], type="password")
        st.session_state["ai_config"]["model"] = st.text_input("Model 名称", value=st.session_state["ai_config"]["model"])


# 动态注入当前主题 CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {active_t['bg_canvas']};
        color: {active_t['text_primary']};
    }}
    div[data-testid="stMetric"] {{
        background-color: {active_t['bg_card']};
        border: 1px solid {active_t['border_line']};
        padding: 12px 16px;
        border-radius: 8px;
    }}
    .conclusion-banner {{
        background-color: {active_t['bg_card']};
        border-left: 5px solid {active_t['accent_primary']};
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0 14px 0;
        color: {active_t['text_primary']};
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 全局质控与有效样本准备
# ==============================================================================
raw_df = st.session_state["raw_df"]
default_straight_cols = [c for c in ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_品质不稳定_反向"] if c in raw_df.columns]
dur_col = "答题时长（秒）" if "答题时长（秒）" in raw_df.columns else None

qc_res = su.run_quality_control(
    raw_df,
    check_dup=True,
    check_straight=bool(default_straight_cols),
    straight_cols=default_straight_cols,
    check_speed=bool(dur_col),
    duration_col=dur_col,
    min_seconds=30,
)
valid_df = qc_res["valid_df"].copy()
if "品质量表3_品质不稳定_反向" in valid_df.columns and "品质量表3_正向" not in valid_df.columns:
    valid_df["品质量表3_正向"] = su.reverse_score(valid_df["品质量表3_品质不稳定_反向"], max_scale=5)
if all(c in valid_df.columns for c in ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"]):
    valid_df["价格感知均分"] = su.compute_scale_composite(valid_df, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"])
if all(c in valid_df.columns for c in ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"]):
    valid_df["品质感知均分"] = su.compute_scale_composite(valid_df, ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"])

# 顶部全局状态横幅
if st.session_state["is_demo"]:
    st.info(f"💡 **当前处于内置演示模式**（咖啡消费调查模拟数据，原始样本 n={len(raw_df)}，质控后有效样本 n={len(valid_df)}）| 当前主题：**{selected_theme}**")
else:
    st.success(f"✅ **已加载实测问卷数据**（原始样本 n={len(raw_df)}，质控后有效样本 n={len(valid_df)}）| 当前主题：**{selected_theme}**")


# ==============================================================================
# 通用交互式图表卡片组件（四件套下载 + 手动格式面板 + AI 自然语言改图）
# ==============================================================================
def render_interactive_chart_card(chart_id: str, df_input: pd.DataFrame = None):
    """渲染指定 chart_id 的高清图表，并提供四件套下载、手动格式面板与 AI 改图助手。"""
    target_df = df_input if isinstance(df_input, pd.DataFrame) else valid_df
    curr_settings = st.session_state["chart_settings"].get(chart_id, {})
    bundle = cu.render_chart_by_id(
        chart_id,
        target_df,
        theme_name=st.session_state["theme_name"],
        settings=curr_settings,
        is_demo=st.session_state["is_demo"],
    )

    st.markdown(f"#### 📊 `{chart_id}` · {bundle['title']}")
    st.image(bundle["png_bytes"], use_container_width=True)
    st.markdown(f'<div class="conclusion-banner">📌 <b>统计结论</b>：{bundle["conclusion"]}</div>', unsafe_allow_html=True)

    # 四件套下载按钮栏
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            f"📥 下载高清 PNG ({bundle['width_px']}px)",
            data=bundle["png_bytes"],
            file_name=f"{chart_id}.png",
            mime="image/png",
            key=f"dl_png_{chart_id}",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📐 下载可编辑矢量 SVG",
            data=bundle["svg_bytes"],
            file_name=f"{chart_id}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{chart_id}",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "📊 下载源数据 CSV",
            data=bundle["csv_bytes"],
            file_name=f"{chart_id}_source.csv",
            mime="text/csv",
            key=f"dl_csv_{chart_id}",
            use_container_width=True,
        )
    with c4:
        st.download_button(
            "📄 下载矢量 PDF",
            data=bundle["pdf_bytes"],
            file_name=f"{chart_id}.pdf",
            mime="application/pdf",
            key=f"dl_pdf_{chart_id}",
            use_container_width=True,
        )

    # 格式调整面板 + AI 改图助手 + 源数据表预览
    with st.expander(f"🎨 格式调整面板、AI 改图助手与源数据表 — {chart_id}", expanded=False):
        tab_fmt, tab_ai, tab_data = st.tabs(["🛠️ 手动格式调整", "✨ AI 自然语言改图", "📋 绘图源数据表"])

        with tab_fmt:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                new_title = st.text_input("自定义结论主标题（留空自动生成）", value=curr_settings.get("title") or "", key=f"fmt_title_{chart_id}")
                new_sub = st.text_input("自定义口径副标题", value=curr_settings.get("subtitle") or "", key=f"fmt_sub_{chart_id}")
                new_palette = st.selectbox("配色方案覆盖", ["主题默认", "松绿序渐", "BrBG发散", "Tol Muted"], index=0, key=f"fmt_pal_{chart_id}")
            with fc2:
                new_sort = st.selectbox("排序方式", ["desc", "asc", "original"], index=0, key=f"fmt_sort_{chart_id}")
                new_topn = st.number_input("仅显示前 N 项 (0=全部)", min_value=0, max_value=20, value=int(curr_settings.get("top_n") or 0), key=f"fmt_topn_{chart_id}")
                new_decimals = st.selectbox("数据标签小数位数", [0, 1, 2], index=int(curr_settings.get("decimals", 1)), key=f"fmt_dec_{chart_id}")
            with fc3:
                new_show_lbl = st.checkbox("显示数据标签", value=bool(curr_settings.get("show_labels", True)), key=f"fmt_lbl_{chart_id}")
                new_show_mean = st.checkbox("显示均值参考线", value=bool(curr_settings.get("show_mean_line", True)), key=f"fmt_mean_{chart_id}")
                new_fscale = st.slider("字号缩放倍率", 0.8, 1.3, float(curr_settings.get("font_scale", 1.0)), 0.05, key=f"fmt_fs_{chart_id}")
                hl_target = st.text_input("高亮目标名称（如：安徽 / 生椰拿铁）", value=(curr_settings.get("highlight") or {}).get("target", ""), key=f"fmt_hlt_{chart_id}")
                hl_color = st.color_picker("高亮颜色", value=(curr_settings.get("highlight") or {}).get("color", "#A50026"), key=f"fmt_hlc_{chart_id}")
                if "词云" in chart_id:
                    _shape_opts = ["无", "云朵", "人形", "爱心", "星星", "咖啡杯", "圆形"]
                    new_wc_shape = st.selectbox("词云填充形状", _shape_opts, index=_shape_opts.index(curr_settings.get("wordcloud_shape") or "无"), key=f"fmt_shape_{chart_id}")
                else:
                    new_wc_shape = curr_settings.get("wordcloud_shape") or "无"

            b_apply, b_reset = st.columns(2)
            with b_apply:
                if st.button("✅ 应用格式修改", key=f"btn_apply_{chart_id}", use_container_width=True):
                    push_settings_history(st.session_state["chart_history"], chart_id, curr_settings)
                    updated = {
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
                    }
                    st.session_state["chart_settings"][chart_id] = updated
                    st.rerun()
            with b_reset:
                if st.button("🔄 恢复默认样式", key=f"btn_reset_{chart_id}", use_container_width=True):
                    st.session_state["chart_settings"][chart_id] = reset_settings_to_default()
                    st.rerun()

        with tab_ai:
            st.caption("支持 17 种白名单样式指令（例如：`把安徽标红`、`把瑞幸改成绿色`、`把标题改成华东区样本突破76%`、`按升序排列`、`只保留前3项`、`隐藏数据标签`、`保留两位小数`、`去掉均值线`、`隐藏图例`、`字大一点`、`把词云做成云朵形状`）。严禁篡改底层统计数值。")
            user_prompt = st.text_input("输入自然语言改图指令：", placeholder="例如：把安徽标红 / 把标题改成自定义结论", key=f"ai_in_{chart_id}")
            ac1, ac2 = st.columns(2)
            with ac1:
                if st.button("✨ 执行 AI 改图", key=f"btn_ai_{chart_id}", use_container_width=True):
                    if user_prompt.strip():
                        res_ai = apply_ai_instruction(
                            user_instruction=user_prompt.strip(),
                            current_settings=curr_settings,
                            api_config=st.session_state["ai_config"],
                            chart_type=chart_id,
                        )
                        if res_ai["ok"]:
                            push_settings_history(st.session_state["chart_history"], chart_id, curr_settings)
                            st.session_state["chart_settings"][chart_id] = res_ai["new_settings"]
                            st.success(res_ai["message"])
                            st.rerun()
                        else:
                            st.warning(res_ai["message"])
            with ac2:
                if st.button("↩️ 撤销上一步 (最多5步)", key=f"btn_undo_{chart_id}", use_container_width=True):
                    prev_s = undo_settings(st.session_state["chart_history"], chart_id)
                    if prev_s is not None:
                        st.session_state["chart_settings"][chart_id] = prev_s
                        st.rerun()
                    else:
                        st.info("已无更多历史步骤可撤销。")

        with tab_data:
            st.dataframe(bundle["source_df"], use_container_width=True)


# ==============================================================================
# 各业务模块页面渲染
# ==============================================================================
if current_module.startswith("01"):
    st.header("01 · 问卷数据概览与题型自动识别")
    ov = su.get_dataset_overview(raw_df)
    qtypes = su.auto_detect_question_types(raw_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("原始上传样本量", f"{ov['n_samples']} 份")
    m2.metric("问卷题目总数", f"{ov['n_questions']} 道")
    m3.metric("质控后有效样本", f"{len(valid_df)} 份")
    m4.metric("整体有效率", f"{qc_res['valid_rate']:.1f}%")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📋 自动识别题型与统计口径一览")
        st.dataframe(pd.DataFrame({"题目": list(qtypes.keys()), "自动识别题型": list(qtypes.values())}), use_container_width=True, hide_index=True)
    with c_right:
        st.subheader("🔍 各题缺失值与完整率统计")
        st.dataframe(ov["missing_df"], use_container_width=True)

    st.subheader("👀 原始问卷数据前 20 行预览")
    st.dataframe(raw_df.head(20), use_container_width=True)

elif current_module.startswith("02"):
    st.header("02 · 数据质控与异常样本清洗")
    st.markdown(f'<div class="conclusion-banner">📌 <b>质控结论</b>：{qc_res["conclusion"]}</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("重复ID样本", f"{qc_res['dup_n']} 份")
    k2.metric("直线作答样本", f"{qc_res['straight_n']} 份")
    k3.metric("过快作答 (<30s)", f"{qc_res['speed_n']} 份")
    k4.metric("异常标记总计", f"{qc_res['flagged_n']} 份")
    k5.metric("保留有效样本", f"{qc_res['valid_n']} 份 ({qc_res['valid_rate']:.1f}%)")

    st.subheader("🚨 被标记为异常的样本明细表")
    flagged_rows = qc_res["marked_df"][qc_res["marked_df"]["是否异常样本"]]
    st.dataframe(flagged_rows, use_container_width=True)

elif current_module.startswith("03"):
    st.header("03 · 变量加工：反向题计分、量表维度合成与数值分箱")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔄 反向题一键重编码 (A13)")
        st.write("已自动将 `品质量表3_品质不稳定_反向` 按 `6 - 原分` 转换为正向指标 `品质量表3_正向`。")
        st.dataframe(valid_df[["品质量表3_品质不稳定_反向", "品质量表3_正向"]].head(10), use_container_width=True)
    with c2:
        st.subheader("📐 量表维度均值合成")
        st.write("已合成 `价格感知均分`（价格1~3题均值）与 `品质感知均分`（品质1~3题均值）。")
        st.dataframe(valid_df[["价格感知均分", "品质感知均分", "NPS推荐度"]].describe().round(2), use_container_width=True)

elif current_module.startswith("04"):
    st.header("04 · 单题分析（涵盖 01–14 共 15 张出版级样板图）")
    single_charts = [cid for cid, _, mod, _ in cu.CHART_REGISTRY_49 if mod == "单题分析"]
    sel_single = st.multiselect("选择要展示的单题图表（默认展示前 4 张，可全选）：", single_charts, default=single_charts[:4])
    for cid in sel_single:
        render_interactive_chart_card(cid, valid_df)
        st.markdown("---")

elif current_module.startswith("05"):
    st.header("05 · 交叉分析（涵盖 15–25、47 共 12 张出版级样板图）")
    cross_charts = [cid for cid, _, mod, _ in cu.CHART_REGISTRY_49 if mod == "交叉分析"]
    sel_cross = st.multiselect("选择要展示的交叉分析图表：", cross_charts, default=cross_charts[:4])
    for cid in sel_cross:
        render_interactive_chart_card(cid, valid_df)
        st.markdown("---")

elif current_module.startswith("06"):
    st.header("06 · 量表信效度与高阶驱动力建模（涵盖 26–28, 34, 38–42）")
    rel_p = su.calc_reliability(valid_df, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"])
    rel_q = su.calc_reliability(valid_df, ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"])
    efa_res = su.calc_validity_efa(valid_df, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"], n_factors=2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("价格量表 Cronbach's α", f"{rel_p['alpha']:.3f}", rel_p["level"])
    m2.metric("品质量表 Cronbach's α", f"{rel_q['alpha']:.3f}", rel_q["level"])
    m3.metric("探索性因子分析 KMO", f"{efa_res['kmo']:.3f}", "Bartlett p < 0.001")
    m4.metric("双因子累计方差解释率", f"{efa_res['cum_var_pct']:.1f}%")

    model_charts = [cid for cid, _, mod, _ in cu.CHART_REGISTRY_49 if mod == "量表与建模"]
    sel_mod = st.multiselect("选择要展示的量表与建模图表：", model_charts, default=["34-结构方程路径图", "26-相关性热力图", "38-回归系数森林图", "40-随机森林重要性图"])
    for cid in sel_mod:
        render_interactive_chart_card(cid, valid_df)
        st.markdown("---")

elif current_module.startswith("07"):
    st.header("07 · 市场研究特色模型（漏斗 / 桑基 / 对应 / IPA / PSM / KANO / 联合 / 地图 / 画像）")
    mkt_charts = [cid for cid, _, mod, _ in cu.CHART_REGISTRY_49 if mod == "市调特色"]
    sel_mkt = st.multiselect("选择要展示的市调特色图表：", mkt_charts, default=["44-区域分布地图", "32-IPA四象限图", "33-PSM价格敏感度曲线", "35-KANO模型图", "36-联合分析重要性图", "37-联合分析效用曲线"])
    for cid in sel_mkt:
        render_interactive_chart_card(cid, valid_df)
        st.markdown("---")

elif current_module.startswith("08"):
    st.header("08 · 开放题文本分析（词云与词频 / 高频词共现网络 / 情感极性分布）")
    text_charts = ["46-词云与词频组合图", "49-高频词共现网络图", "50-情感分布图"]
    for cid in text_charts:
        render_interactive_chart_card(cid, valid_df)
        st.markdown("---")

elif current_module.startswith("09"):
    st.header("09 · 事后分层加权（Post-Stratification Weighting）")
    st.caption("设置人口学目标配比（如男 50% / 女 50%），对比加权前后关键指标变化。")
    c1, c2 = st.columns(2)
    with c1:
        target_m = st.slider("目标男性占比 (%)", 10.0, 90.0, 50.0, 1.0)
        target_f = 100.0 - target_m
        st.write(f"目标女性占比：**{target_f:.1f}%**")
    wt_res = su.compute_post_stratification_weights(valid_df["性别"], {"男": target_m, "女": target_f})
    with c2:
        st.markdown(f'<div class="conclusion-banner">📌 {wt_res["conclusion"]}</div>', unsafe_allow_html=True)
        st.dataframe(wt_res["table"], use_container_width=True)

    valid_df["_weight"] = wt_res["weight_series"]
    cmp_freq = su.compare_weighted_unweighted(valid_df, "消费频次", "_weight", is_numeric=False)
    cmp_spend = su.compare_weighted_unweighted(valid_df, "月消费额", "_weight", is_numeric=True)
    st.subheader("📊 加权前后关键题目对比表")
    st.dataframe(cmp_freq, use_container_width=True)
    st.dataframe(cmp_spend, use_container_width=True)

elif current_module.startswith("10"):
    st.header("10 · 出版级 PPT / Word / Excel 报告一键导出与 49 图全览")
    st.markdown(
        """
        - **出版级 16:9 PPT 报告 (`.pptx`)**：含封面、目录、5 大章节过渡页、研究方法与口径页、轮换版式核心图表页、策略建议与技术附录（共 **24 页**，内嵌 **9 个原生可编辑 PowerPoint 图表** + **14 个可编辑矢量 SVG** + **全页 Markdown 源数据备注表**）。
        - **同步主题 Word 报告 (`.docx`)**：含样本概况、质控小结、图文分析章节与末尾完整的**图表源数据附录**。
        """
    )
    ov = su.get_dataset_overview(raw_df)
    e1, e2, e3 = st.columns(3)
    with e1:
        if st.button("📽️ 生成出版级可编辑 PPT 报告 (.pptx)", use_container_width=True):
            with st.spinner("正在构建 24 页出版级 PPT 报告（含原生图表与矢量 SVG）..."):
                ppt_bytes = pr.build_ppt_report(ov, qc_res, is_demo=st.session_state["is_demo"], theme_name=selected_theme, df=valid_df)
                st.session_state["last_ppt_bytes"] = ppt_bytes
        if "last_ppt_bytes" in st.session_state:
            st.download_button(
                "⬇️ 点击下载 SurveyScope_分析报告.pptx",
                data=st.session_state["last_ppt_bytes"],
                file_name=f"SurveyScope_分析报告_{selected_theme}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )

    with e2:
        if st.button("📝 生成完整 Word 分析报告 (.docx)", use_container_width=True):
            with st.spinner("正在生成含源数据附录的 Word 报告..."):
                sec_list = []
                for cid in ["01-棒棒糖图", "06-多选题条形图", "11-发散堆叠条形图", "13-NPS仪表盘", "32-IPA四象限图", "33-PSM价格敏感度曲线", "34-结构方程路径图", "35-KANO模型图", "36-联合分析重要性图", "44-区域分布地图"]:
                    b = cu.render_chart_by_id(cid, valid_df, theme_name=selected_theme, is_demo=st.session_state["is_demo"])
                    sec_list.append(b)
                word_bytes = pr.build_word_report(ov, qc_res, sec_list, is_demo=st.session_state["is_demo"], theme_name=selected_theme)
                st.session_state["last_word_bytes"] = word_bytes
        if "last_word_bytes" in st.session_state:
            st.download_button(
                "⬇️ 点击下载 SurveyScope_分析报告.docx",
                data=st.session_state["last_word_bytes"],
                file_name=f"SurveyScope_分析报告_{selected_theme}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

    with e3:
        excel_bytes = su.export_cleaned_excel(valid_df)
        st.download_button(
            "📊 下载清洗后有效样本数据 (.xlsx)",
            data=excel_bytes,
            file_name="SurveyScope_清洗后有效数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🖼️ 49 张标准样板图型浏览器")
    all_ids = [cid for cid, _, _, _ in cu.CHART_REGISTRY_49]
    chosen_id = st.selectbox("选择任意一张样板图在线检视、改图或下载四件套：", all_ids, index=44)
    render_interactive_chart_card(chosen_id, valid_df)
