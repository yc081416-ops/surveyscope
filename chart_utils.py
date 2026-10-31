# -*- coding: utf-8 -*-
"""
SurveyScope 图表工具统一入口 (chart_utils.py)
整合：
- 49 张标准样板图型（01–48，含 08a 与 08b）及扩展文本图（49 高频词共现网络、50 情感分布）
- 多主题 Token 驱动渲染（暖纸杂志 / 墨蓝学术 / 纯白极简 / 深空夜读 / 薄荷清新）
- 四件套导出（高清 PNG >= 2400px、可编辑矢量 SVG、源数据 CSV、矢量 PDF）
- 向后兼容接口（供 A14、A16 基准测试无缝调用）
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from theme import DEFAULT_THEME_NAME, get_theme, resolve_effective_theme, set_active_chart_id, reset_active_chart_id
from charts_core import (
    create_themed_figure,
    style_axis,
    finalize_chart_chrome,
    export_chart_bundle,
    normalize_settings,
    get_item_color,
    is_item_highlighted,
    apply_df_sort_and_topn,
    get_cjk_font_path,
)
from charts_part1 import (
    chart_01_lollipop,
    chart_02_horizontal_bar,
    chart_03_pareto,
    chart_04_waffle,
    chart_05_donut,
    chart_06_multichoice_bar,
    chart_07_hist_kde,
    chart_08a_boxplot,
    chart_08b_violin,
    chart_09_raincloud,
    chart_10_ecdf,
    chart_11_likert_diverging,
    chart_12_mean_ci,
    chart_13_nps_gauge,
    chart_14_pyramid,
    chart_15_stacked_100,
    chart_16_grouped_bar,
    chart_17_mosaic,
    chart_18_chi2_residual_heatmap,
    chart_19_dumbbell,
    chart_20_slopegraph,
    chart_21_grouped_violin,
    chart_22_mean_errorbar,
    chart_23_ridgeline,
    chart_24_small_multiples,
)
from charts_part2 import (
    chart_25_scatter_ols,
    chart_26_corr_heatmap,
    chart_27_bubble_matrix,
    chart_28_parallel_coordinates,
    chart_29_funnel,
    chart_30_sankey,
    chart_31_correspondence_biplot,
    chart_32_ipa,
    chart_33_psm,
    chart_34_sem_path,
    chart_35_kano,
    chart_36_conjoint_importance,
    chart_37_conjoint_utility,
)
from charts_part3 import (
    chart_38_reg_forest,
    chart_39_roc_curve,
    chart_40_rf_importance,
    chart_41_decision_tree,
    chart_42_confusion_matrix,
    chart_43_bubble_positioning,
    chart_44_region_map,
    chart_45_persona_cards,
    chart_46_wordcloud_combo,
    chart_47_butterfly,
    chart_48_dual_axis,
    chart_49_cooccurrence_network,
    chart_50_sentiment,
)

# ==============================================================================
# 49 张标准样板图注册表（严格对应 样板包-48图型.zip 的 49 个文件名）
# ==============================================================================
CHART_REGISTRY_49 = [
    ("01-棒棒糖图",         "棒棒糖图",         "单题分析",   chart_01_lollipop),
    ("02-水平条形图",       "水平条形图",       "单题分析",   chart_02_horizontal_bar),
    ("03-帕累托图",         "帕累托图",         "单题分析",   chart_03_pareto),
    ("04-华夫饼图",         "华夫饼图",         "单题分析",   chart_04_waffle),
    ("05-甜甜圈图",         "甜甜圈图",         "单题分析",   chart_05_donut),
    ("06-多选题条形图",     "多选题条形图",     "单题分析",   chart_06_multichoice_bar),
    ("07-直方图与核密度",   "直方图与核密度",   "单题分析",   chart_07_hist_kde),
    ("08a-箱线图",          "箱线图",           "单题分析",   chart_08a_boxplot),
    ("08b-小提琴图",        "小提琴图",         "单题分析",   chart_08b_violin),
    ("09-雨云图",           "雨云图",           "单题分析",   chart_09_raincloud),
    ("10-经验累积分布图",   "经验累积分布图",   "单题分析",   chart_10_ecdf),
    ("11-发散堆叠条形图",   "发散堆叠条形图",   "单题分析",   chart_11_likert_diverging),
    ("12-均值置信区间图",   "均值置信区间图",   "单题分析",   chart_12_mean_ci),
    ("13-NPS仪表盘",        "NPS仪表盘",        "单题分析",   chart_13_nps_gauge),
    ("14-人口学金字塔",     "人口学金字塔",     "单题分析",   chart_14_pyramid),
    ("15-100%堆叠条形图",   "100%堆叠条形图",   "交叉分析",   chart_15_stacked_100),
    ("16-分组簇状条形图",   "分组簇状条形图",   "交叉分析",   chart_16_grouped_bar),
    ("17-马赛克图",         "马赛克图",         "交叉分析",   chart_17_mosaic),
    ("18-卡方残差热力图",   "卡方残差热力图",   "交叉分析",   chart_18_chi2_residual_heatmap),
    ("19-哑铃图",           "哑铃图",           "交叉分析",   chart_19_dumbbell),
    ("20-坡度图",           "坡度图",           "交叉分析",   chart_20_slopegraph),
    ("21-分组小提琴箱线图", "分组小提琴箱线图", "交叉分析",   chart_21_grouped_violin),
    ("22-均值标准误条形图", "均值标准误条形图", "交叉分析",   chart_22_mean_errorbar),
    ("23-山脊图",           "山脊图",           "交叉分析",   chart_23_ridgeline),
    ("24-分面小多图",       "分面小多图",       "交叉分析",   chart_24_small_multiples),
    ("25-散点回归图",       "散点回归图",       "交叉分析",   chart_25_scatter_ols),
    ("26-相关性热力图",     "相关性热力图",     "量表与建模", chart_26_corr_heatmap),
    ("27-气泡矩阵图",       "气泡矩阵图",       "量表与建模", chart_27_bubble_matrix),
    ("28-平行坐标图",       "平行坐标图",       "量表与建模", chart_28_parallel_coordinates),
    ("29-漏斗图",           "漏斗图",           "市调特色",   chart_29_funnel),
    ("30-桑基图",           "桑基图",           "市调特色",   chart_30_sankey),
    ("31-对应分析双标图",   "对应分析双标图",   "市调特色",   chart_31_correspondence_biplot),
    ("32-IPA四象限图",      "IPA四象限图",      "市调特色",   chart_32_ipa),
    ("33-PSM价格敏感度曲线","PSM价格敏感度曲线","市调特色",   chart_33_psm),
    ("34-结构方程路径图",   "结构方程路径图",   "量表与建模", chart_34_sem_path),
    ("35-KANO模型图",       "KANO模型图",       "市调特色",   chart_35_kano),
    ("36-联合分析重要性图", "联合分析重要性图", "市调特色",   chart_36_conjoint_importance),
    ("37-联合分析效用曲线", "联合分析效用曲线", "市调特色",   chart_37_conjoint_utility),
    ("38-回归系数森林图",   "回归系数森林图",   "量表与建模", chart_38_reg_forest),
    ("39-ROC曲线图",        "ROC曲线图",        "量表与建模", chart_39_roc_curve),
    ("40-随机森林重要性图", "随机森林重要性图", "量表与建模", chart_40_rf_importance),
    ("41-决策树规则图",     "决策树规则图",     "量表与建模", chart_41_decision_tree),
    ("42-混淆矩阵图",       "混淆矩阵图",       "量表与建模", chart_42_confusion_matrix),
    ("43-气泡图",           "气泡图",           "市调特色",   chart_43_bubble_positioning),
    ("44-区域分布地图",     "区域分布地图",     "市调特色",   chart_44_region_map),
    ("45-消费者画像卡片",   "消费者画像卡片",   "市调特色",   chart_45_persona_cards),
    ("46-词云与词频组合图", "词云与词频组合图", "文本分析",   chart_46_wordcloud_combo),
    ("47-蝴蝶对比图",       "蝴蝶对比图",       "交叉分析",   chart_47_butterfly),
    ("48-双轴组合图",       "双轴组合图",       "市调特色",   chart_48_dual_axis),
]

CHART_FUNC_MAP = {cid: fn for cid, _, _, fn in CHART_REGISTRY_49}
CHART_FUNC_MAP["49-高频词共现网络图"] = chart_49_cooccurrence_network
CHART_FUNC_MAP["50-情感分布图"] = chart_50_sentiment


def render_chart_by_id(
    chart_id: str,
    df: pd.DataFrame,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
    **kwargs,
) -> dict:
    """根据图表编号（如 '01-棒棒糖图' 或 '44-区域分布地图'）渲染并返回四件套字典，支持透传列名等参数。"""
    if chart_id not in CHART_FUNC_MAP:
        raise KeyError(f"未找到编号为 {chart_id} 的图表函数。")
    fn = CHART_FUNC_MAP[chart_id]
    _tok = set_active_chart_id(chart_id)
    try:
        return fn(df, theme_name=theme_name, settings=settings, is_demo=is_demo, **kwargs)
    finally:
        reset_active_chart_id(_tok)


def generate_all_49_charts(
    df: pd.DataFrame,
    theme_name: str = DEFAULT_THEME_NAME,
    settings_map: dict = None,
    is_demo: bool = False,
) -> dict:
    """一键批量生成全部 49 张样板图四件套字典，返回 {chart_id: bundle_dict}。"""
    results = {}
    settings_map = settings_map or {}
    for cid, _, _, fn in CHART_REGISTRY_49:
        cfg = settings_map.get(cid)
        _tok = set_active_chart_id(cid)
        try:
            results[cid] = fn(df, theme_name=theme_name, settings=cfg, is_demo=is_demo)
        finally:
            reset_active_chart_id(_tok)
    return results


# ==============================================================================
# 向后兼容包装函数（供 A14、A16 及旧模块直接调用）
# ==============================================================================
def plot_categorical_bar(freq_df: pd.DataFrame, title: str = "分类题目分布图", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None) -> dict:
    """向后兼容：传入频数表 DataFrame，返回四件套 Bundle 字典。"""
    src = apply_df_sort_and_topn(freq_df, "占比(%)", settings, default_sort="desc")
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.13, 0.14, 0.79, 0.66])
    fs = float(cfg["font_scale"])
    plot_df = src.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    pcts = plot_df["占比(%)"].to_numpy(dtype=float)
    cnts = plot_df["频数"].to_numpy(dtype=int)
    labels = plot_df["选项"].astype(str).tolist()
    top_opt = str(src.iloc[0]["选项"])
    colors = [get_item_color(lbl, t["accent_primary"] if lbl == top_opt else t["categorical"][1], t, cfg) for lbl in labels]
    bars = ax.barh(y_pos, pcts, height=0.58, color=colors, zorder=3)
    for bar, pct, cnt in zip(bars, pcts, cnts):
        ax.text(pct + max(pcts) * 0.02, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}% ({cnt}人)", va="center", ha="left", fontsize=10.0 * fs, color=t["text_primary"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(0, max(pcts) * 1.25)
    ax.set_xlabel("占有效样本比例 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.grid(axis="y", visible=False)
    n_tot = int(src["频数"].sum())
    subtitle = f"单题频数与百分比统计 | 水平条形图（有效样本 n = {n_tot}）"
    footnote = f"统计口径：单选有效百分比（n={n_tot}） | 数据来源：模拟数据，仅供演示"
    return export_chart_bundle(fig, src, "compat-bar", "水平条形图", title, subtitle, footnote, theme_name, settings)


def fig_to_png_bytes(fig_or_bundle, width: int = 2640, height: int = 1485) -> bytes:
    """
    向后兼容：支持传入 export_chart_bundle 返回的 dict、Matplotlib Figure 或 Plotly Figure，
    直接返回高清 PNG 字节流（宽度 >= 2400px）。
    """
    if isinstance(fig_or_bundle, dict) and "png_bytes" in fig_or_bundle:
        return fig_or_bundle["png_bytes"]
    if isinstance(fig_or_bundle, plt.Figure):
        buf = io.BytesIO()
        fig_or_bundle.savefig(buf, format="png", dpi=220)
        return buf.getvalue()
    if hasattr(fig_or_bundle, "to_image"):
        return fig_or_bundle.to_image(format="png", width=max(width, 2400), height=height, scale=2)
    raise TypeError(f"不支持的图表对象类型: {type(fig_or_bundle)}")
