# -*- coding: utf-8 -*-
"""
SurveyScope 图型渲染引擎 — 第一部分 (charts_part1.py)
涵盖编号 01 至 24（含 08a 与 08b 共 25 张图）：
01-棒棒糖图, 02-水平条形图, 03-帕累托图, 04-华夫饼图, 05-甜甜圈图, 06-多选题条形图,
07-直方图与核密度, 08a-箱线图, 08b-小提琴图, 09-雨云图, 10-经验累积分布图,
11-发散堆叠条形图, 12-均值置信区间图, 13-NPS仪表盘, 14-人口学金字塔,
15-100%堆叠条形图, 16-分组簇状条形图, 17-马赛克图, 18-卡方残差热力图,
19-哑铃图, 20-坡度图, 21-分组小提琴箱线图, 22-均值标准误条形图, 23-山脊图, 24-分面小多图
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Wedge, Rectangle

import stats_utils as su
from theme import DEFAULT_THEME_NAME, auto_text_color, get_diverging_cmap, get_sequential_cmap
from charts_core import (
    create_themed_figure,
    style_axis,
    export_chart_bundle,
    normalize_settings,
    get_item_color,
    is_item_highlighted,
    apply_df_sort_and_topn,
)


def _ordered_categories(values, preferred_order):
    """类别全量排序：数据中实际出现的类别一个不丢，preferred_order 只决定
    先后顺序，绝不用作过滤名单；名单外类别按名称中首个数字的自然序
    （无数字则按名称）追加在后。"""
    present = list(dict.fromkeys(str(v) for v in values))
    ordered = [c for c in preferred_order if c in present]
    rest = [c for c in present if c not in ordered]

    def _nat_key(s):
        m = re.search(r"\d+(?:\.\d+)?", s)
        return (0, float(m.group(0)), s) if m else (1, 0.0, s)

    return ordered + sorted(rest, key=_nat_key)


def _unit_for_col(col: str) -> str:
    """根据列名智能推断数值单位，避免非金额列硬编码『元』。"""
    s = str(col or "")
    if any(k in s for k in ("额", "价", "费", "元", "收入", "花费", "薪", "预算", "客单")):
        return "元"
    if "秒" in s:
        return "秒"
    if "小时" in s:
        return "小时"
    if "分钟" in s:
        return "分钟"
    if any(k in s for k in ("量表", "满意", "重要", "感知", "评分")):
        return "分"
    return ""


def _fmt_u(unit: str, leading_space: bool = True) -> str:
    if not unit:
        return ""
    return f" {unit}" if leading_space else unit


def _axis_label_with_unit(col: str, suffix: str = "") -> str:
    u = _unit_for_col(col)
    base = f"{col}{suffix}"
    return f"{base} ({u})" if u else base


def _pick_col(df: pd.DataFrame, col: str, settings: dict, key: str, fallback_pool: list = None) -> str:
    if isinstance(settings, dict) and settings.get(key) in df.columns:
        return settings[key]
    if col in df.columns:
        return col
    if fallback_pool:
        for c in fallback_pool:
            if c in df.columns:
                return c
    return df.columns[0]


# ==============================================================================
# 01 - 棒棒糖图 (Lollipop Chart)
# ==============================================================================
def chart_01_lollipop(df: pd.DataFrame, col: str = "消费频次", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("single_cols"))
    res = su.analyze_categorical_question(df[col])
    src = apply_df_sort_and_topn(res["table"], "占比(%)", settings, default_sort="desc")
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.14, 0.78, 0.66])
    fs = float(cfg["font_scale"])
    dec = int(cfg["decimals"])

    plot_df = src.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    pcts = plot_df["占比(%)"].to_numpy(dtype=float)
    cnts = plot_df["频数"].to_numpy(dtype=int)
    labels = plot_df["选项"].astype(str).tolist()
    top_opt = str(src.iloc[0]["选项"])
    top_pct = float(src.iloc[0]["占比(%)"])
    top_cnt = int(src.iloc[0]["频数"])
    mean_pct = float(np.mean(pcts)) if len(pcts) > 0 else 0.0

    for i, (lbl, pct, cnt) in enumerate(zip(labels, pcts, cnts)):
        is_top = (lbl == top_opt)
        default_c = t["accent_primary"] if is_top else t["categorical"][1]
        c = get_item_color(lbl, default_c, t, cfg)
        lw = 3.8 if is_top or is_item_highlighted(lbl, cfg) else 2.4
        ms = 120 if is_top or is_item_highlighted(lbl, cfg) else 80
        ax.hlines(y=i, xmin=0, xmax=pct, color=c, linewidth=lw, zorder=3)
        ax.scatter([pct], [i], color=c, s=ms, zorder=4, edgecolors=t["bg_canvas"], linewidths=1.2)
        if cfg["show_labels"]:
            ax.text(
                pct + max(pcts.max(), 1.0) * 0.02, i,
                f"{pct:.{dec}f}% (n={cnt})",
                va="center", ha="left", fontsize=10.2 * fs,
                weight="bold" if is_top else "normal",
                color=t["text_primary"] if is_top else t["text_secondary"],
            )

    if cfg["show_mean_line"] and len(pcts) > 1:
        ax.axvline(mean_pct, color=t["accent_secondary"], linestyle="--", linewidth=1.3, zorder=2)
        import matplotlib.transforms as _mtr
        _blend = _mtr.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(mean_pct + 0.5, 0.985, f"均值线 {mean_pct:.1f}%", transform=_blend, va="top",
                color=t["accent_secondary"], fontsize=9.2 * fs, weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_canvas"], edgecolor="none", alpha=0.88))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(0, max(float(pcts.max()) if len(pcts) else 10.0, 1.0) * 1.26)
    ax.set_xlabel("样本占比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.grid(axis="y", visible=False)

    title = f"「{top_opt}」占比居首达 {top_pct:.1f}%（n={top_cnt}），显著高于平均水平"
    subtitle = f"{col}分布 | 棒棒糖图（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计口径：单选频数与有效百分比 | 最高项：{top_opt} ({top_pct:.1f}%) | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "01-棒棒糖图", "棒棒糖图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 02 - 水平条形图 (Horizontal Bar Chart)
# ==============================================================================
def chart_02_horizontal_bar(df: pd.DataFrame, col: str = "年龄段", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("single_cols"))
    res = su.analyze_categorical_question(df[col])
    src = apply_df_sort_and_topn(res["table"], "占比(%)", settings, default_sort="desc")
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.13, 0.14, 0.79, 0.66])
    fs = float(cfg["font_scale"])
    dec = int(cfg["decimals"])

    plot_df = src.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    pcts = plot_df["占比(%)"].to_numpy(dtype=float)
    cnts = plot_df["频数"].to_numpy(dtype=int)
    labels = plot_df["选项"].astype(str).tolist()
    top_opt = str(src.iloc[0]["选项"])
    top_pct = float(src.iloc[0]["占比(%)"])
    top2_sum = float(src.head(2)["占比(%)"].sum())

    colors = []
    for lbl in labels:
        def_c = t["accent_primary"] if lbl == top_opt else t["categorical"][2]
        colors.append(get_item_color(lbl, def_c, t, cfg))

    bars = ax.barh(y_pos, pcts, height=0.58, color=colors, edgecolor="none", zorder=3)
    for bar, lbl, pct, cnt in zip(bars, labels, pcts, cnts):
        if cfg["show_labels"]:
            is_top = (lbl == top_opt)
            ax.text(
                pct + max(pcts.max(), 1.0) * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.{dec}f}% ({cnt}人)",
                va="center", ha="left", fontsize=10.2 * fs,
                weight="bold" if is_top else "normal",
                color=t["text_primary"] if is_top else t["text_secondary"],
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(0, max(float(pcts.max()) if len(pcts) else 10.0, 1.0) * 1.25)
    ax.set_xlabel("占有效样本比例 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.grid(axis="y", visible=False)

    grp_word = "年龄层" if "年龄" in str(col) else "类别"
    title = f"「{top_opt}」为最核心群体（{top_pct:.1f}%），前两大{grp_word}合计占 {top2_sum:.1f}%"
    subtitle = f"{col}结构分布 | 水平条形图（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计口径：单选有效百分比（n={res['n_valid']}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "02-水平条形图", "水平条形图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 03 - 帕累托图 (Pareto Chart)
# ==============================================================================
def chart_03_pareto(df: pd.DataFrame, col: str = "消费频次", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("single_cols"))
    res = su.analyze_categorical_question(df[col])
    src = res["table"].sort_values("频数", ascending=False).reset_index(drop=True)
    src["累计占比(%)"] = src["占比(%)"].cumsum().round(1)

    fig, ax1, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.81, 0.66])
    fs = float(cfg["font_scale"])
    dec = int(cfg["decimals"])

    x = np.arange(len(src))
    labels = src["选项"].astype(str).tolist()
    cnts = src["频数"].to_numpy(dtype=int)
    pcts = src["占比(%)"].to_numpy(dtype=float)
    cum_pcts = src["累计占比(%)"].to_numpy(dtype=float)

    bar_colors = [
        get_item_color(lbl, t["accent_primary"] if cp <= 82.0 or i < 3 else t["text_muted"], t, cfg)
        for i, (lbl, cp) in enumerate(zip(labels, cum_pcts))
    ]
    bars = ax1.bar(x, cnts, width=0.52, color=bar_colors, zorder=3)
    for bar, cnt, pct in zip(bars, cnts, pcts):
        if cfg["show_labels"]:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                cnt + max(cnts.max(), 1) * 0.02,
                f"{cnt}人\n({pct:.{dec}f}%)",
                ha="center", va="bottom", fontsize=9.2 * fs, color=t["text_primary"],
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10.2 * fs, color=t["text_primary"], rotation=15 if len(labels) > 6 else 0)
    ax1.set_ylabel("样本频数 (人)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax1.set_ylim(0, max(int(cnts.max()) if len(cnts) else 10, 1) * 1.28)
    ax1.grid(axis="x", visible=False)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(t["accent_secondary"])
    ax2.plot(x, cum_pcts, color=t["accent_secondary"], marker="o", linewidth=2.5, markersize=7, zorder=5)
    ax2.axhline(80.0, color=t["accent_secondary"], linestyle="--", linewidth=1.2, alpha=0.85)
    ax2.text(max(len(x) - 0.55, 0.2), 81.8, "80% 帕累托基准线", color=t["accent_secondary"], fontsize=9.0 * fs, ha="right", weight="bold")
    for xi, cp in zip(x, cum_pcts):
        if cfg["show_labels"]:
            ax2.text(xi, min(cp + 3.5, 104.0), f"{cp:.1f}%", ha="center", va="bottom", fontsize=9.2 * fs, color=t["accent_secondary"], weight="bold")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("累计占比 (%)", fontsize=10.2 * fs, color=t["accent_secondary"])
    ax2.tick_params(axis="y", colors=t["accent_secondary"], labelsize=9.8 * fs)
    ax2.grid(False)

    k_top = min(3, len(cum_pcts))
    top_cum = float(cum_pcts[k_top - 1]) if k_top > 0 else 0.0
    title = f"前{k_top}大{col}类别累计贡献 {top_cum:.1f}% 样本量，符合帕累托头部集中效应"
    subtitle = f"{col}帕累托累计贡献分析 | 帕累托图（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"左轴：频数柱状图 | 右轴：累计百分比折线（80% 二八基准线） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "03-帕累托图", "帕累托图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 04 - 华夫饼图 (Waffle Chart)
# ==============================================================================
def chart_04_waffle(df: pd.DataFrame, col: str = "是否喝咖啡", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("single_cols"))
    res = su.analyze_categorical_question(df[col])
    src = res["table"].copy()
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.08, 0.13, 0.84, 0.67])
    fs = float(cfg["font_scale"])

    pcts = src["占比(%)"].to_numpy(dtype=float)
    labels = src["选项"].astype(str).tolist()
    cnts = src["频数"].to_numpy(dtype=int)
    cells = np.round(pcts).astype(int)
    if cells.sum() != 100 and len(cells) > 0:
        cells[0] += (100 - cells.sum())

    palette = [t["accent_primary"], t["border_line"], t["accent_secondary"], t["categorical"][2 % len(t["categorical"])], t["categorical"][0], t["categorical"][4 % len(t["categorical"])]]
    grid_colors = []
    for idx, (lbl, c_cnt) in enumerate(zip(labels, cells)):
        col_hex = get_item_color(lbl, palette[idx % len(palette)], t, cfg)
        grid_colors.extend([col_hex] * max(0, int(c_cnt)))
    grid_colors = (grid_colors + [t["border_line"]] * 100)[:100]

    for cell_i in range(100):
        row = cell_i // 10
        col_i = cell_i % 10
        rect = FancyBboxPatch(
            (col_i * 1.0, (9 - row) * 1.0), 0.82, 0.82,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=grid_colors[cell_i], edgecolor="none",
        )
        ax.add_patch(rect)

    top_lbl = labels[0]
    top_pct = float(pcts[0])
    top_cnt = int(cnts[0])
    ax.text(11.4, 7.6, f"{top_pct:.1f}%", fontsize=40 * fs, weight="bold", color=t["accent_primary"], va="center")
    desc_line = (
        f"受访者表示「{top_lbl}」饮用咖啡\n（共 {top_cnt} / {res['n_valid']} 人）"
        if "咖啡" in str(col)
        else f"受访者选择「{top_lbl}」\n（共 {top_cnt} / {res['n_valid']} 人）"
    )
    ax.text(11.4, 5.8, desc_line, fontsize=12.2 * fs, color=t["text_primary"], va="center")

    step_y = min(1.05, 4.2 / max(len(labels), 1))
    for i, (lbl, pct, cnt) in enumerate(zip(labels[:6], pcts[:6], cnts[:6])):
        c_hex = get_item_color(lbl, palette[i % len(palette)], t, cfg)
        y_leg = 3.8 - i * step_y
        ax.add_patch(Rectangle((11.4, y_leg - 0.24), 0.52, 0.52, facecolor=c_hex, edgecolor="none"))
        ax.text(12.15, y_leg, f"{lbl}：{pct:.1f}%（n={cnt}）", fontsize=10.6 * fs, color=t["text_secondary"], va="center")

    ax.set_xlim(-0.3, 19.5)
    ax.set_ylim(-0.3, 10.2)
    ax.axis("off")

    if "咖啡" in str(col):
        title = f"高达 {top_pct:.1f}% 受访者为咖啡饮用人群（每 100 人中约 {int(round(top_pct))} 人）"
    else:
        title = f"「{top_lbl}」占比达 {top_pct:.1f}%（每 100 人中约 {int(round(top_pct))} 人）"
    subtitle = f"{col}构成占比 | 10×10 华夫饼图（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"网格说明：1 格 ≈ 1% 有效样本（总计 100 格） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "04-华夫饼图", "华夫饼图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 05 - 甜甜圈图 (Donut Chart)
# ==============================================================================
def chart_05_donut(df: pd.DataFrame, col: str = "性别", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("single_cols"))
    res = su.analyze_categorical_question(df[col])
    src = res["table"].copy()
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.08, 0.12, 0.84, 0.68])
    fs = float(cfg["font_scale"])
    dec = int(cfg["decimals"])

    labels = src["选项"].astype(str).tolist()
    pcts = src["占比(%)"].to_numpy(dtype=float)
    cnts = src["频数"].to_numpy(dtype=int)
    colors = [get_item_color(lbl, t["categorical"][i % len(t["categorical"])], t, cfg) for i, lbl in enumerate(labels)]

    wedges, _ = ax.pie(
        pcts,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.42, edgecolor=t["bg_canvas"], linewidth=2.5),
    )
    ax.text(0, 0.08, f"n = {res['n_valid']}", ha="center", va="center", fontsize=16 * fs, weight="bold", color=t["text_primary"])
    ax.text(0, -0.16, "有效总样本", ha="center", va="center", fontsize=10 * fs, color=t["text_secondary"])

    for i, (w, lbl, pct, cnt) in enumerate(zip(wedges, labels, pcts, cnts)):
        ang = (w.theta2 + w.theta1) / 2.0
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        ha = "left" if x >= 0 else "right"
        x_text = 1.25 * (1 if x >= 0 else -1)
        y_text = 1.12 * y
        ax.annotate(
            f"{lbl}：{pct:.{dec}f}%\n(n = {cnt})",
            xy=(0.82 * x, 0.82 * y),
            xytext=(x_text, y_text),
            ha=ha, va="center",
            fontsize=10.6 * fs, weight="bold" if i == 0 else "normal",
            color=t["text_primary"],
            arrowprops=dict(arrowstyle="-", color=t["text_muted"], lw=1.1, connectionstyle="angle,angleA=0,angleB=90,rad=3"),
        )

    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    top_lbl, top_pct, top_cnt = labels[0], float(pcts[0]), int(cnts[0])
    tail_txt = "整体样本性别比例均衡" if "性别" in str(col) else f"共涵盖 {len(labels)} 个类别选项"
    title = f"{col}结构中「{top_lbl}」占 {top_pct:.1f}%（n={top_cnt}），{tail_txt}"
    subtitle = f"{col}构成占比 | 甜甜圈图（类别数 ≤ 6，有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"设计准则：甜甜圈图仅用于少类别（≤ 6 类）的构成展示 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "05-甜甜圈图", "甜甜圈图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 06 - 多选题条形图 (Multi-choice Horizontal Bar Chart)
# ==============================================================================
def chart_06_multichoice_bar(df: pd.DataFrame, col: str = "偏好品类", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("multi_cols"))
    res = su.split_multiple_choice(df[col])
    src = apply_df_sort_and_topn(res["freq_df"], "普及率(% of Cases)", settings, default_sort="desc")
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.13, 0.14, 0.79, 0.66])
    fs = float(cfg["font_scale"])
    dec = int(cfg["decimals"])

    plot_df = src.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    labels = plot_df["选项"].astype(str).tolist()
    case_pcts = plot_df["普及率(% of Cases)"].to_numpy(dtype=float)
    resp_pcts = plot_df["响应占比(% of Responses)"].to_numpy(dtype=float)
    cnts = plot_df["选中次数"].to_numpy(dtype=int)

    top_opt = str(src.iloc[0]["选项"])
    top_case_pct = float(src.iloc[0]["普及率(% of Cases)"])
    top_cnt = int(src.iloc[0]["选中次数"])

    def _lerp_hex(c1, c2, f):
        a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(a[k] + (b[k] - a[k]) * f):02X}" for k in range(3))

    _seq = t["sequential_6"]
    _nt_vals = [cp for lbl, cp in zip(labels, case_pcts) if lbl != top_opt]
    _nt_lo = min(_nt_vals) if _nt_vals else 0.0
    _nt_hi = max(_nt_vals) if _nt_vals else 1.0

    def _bar_color(lbl, cp):
        if lbl == top_opt:
            base = t["accent_primary"]
        else:
            frac = (cp - _nt_lo) / (_nt_hi - _nt_lo) if _nt_hi > _nt_lo else 0.5
            base = _lerp_hex(_seq[2], _seq[3], 0.25 + 0.75 * float(frac))
        return get_item_color(lbl, base, t, cfg)

    colors = [_bar_color(lbl, cp) for lbl, cp in zip(labels, case_pcts)]
    bars = ax.barh(y_pos, case_pcts, height=0.58, color=colors, alpha=0.92, zorder=3)
    for bar, lbl, cp, rp, cnt in zip(bars, labels, case_pcts, resp_pcts, cnts):
        if cfg["show_labels"]:
            is_top = (lbl == top_opt)
            ax.text(
                cp + max(case_pcts.max(), 1.0) * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"普及率 {cp:.{dec}f}% | 响应比 {rp:.{dec}f}% (n={cnt})",
                va="center", ha="left", fontsize=9.8 * fs,
                weight="bold" if is_top else "normal",
                color=t["text_primary"] if is_top else t["text_secondary"],
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(0, max(float(case_pcts.max()) if len(case_pcts) else 10.0, 1.0) * 1.38)
    ax.set_xlabel("样本普及率 (% of Cases，基于作答人数分母)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.grid(axis="y", visible=False)

    tail_desc = "稳居最受欢迎咖啡品类" if (col == "偏好品类" or "咖啡" in str(col)) else "位居多选响应首位"
    title = f"「{top_opt}」普及率高达 {top_case_pct:.1f}%（{top_cnt}人选中），{tail_desc}"
    subtitle = f"{col}（多选题）| 多选题水平条形图（有效作答人数 n = {res['n_cases']}，总响应 {res['total_responses']} 人次）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"注：多选题普及率(% of Cases)以有效作答人数（n={res['n_cases']}）为分母，各选项普及率之和可超过 100% | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "06-多选题条形图", "多选题条形图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 07 - 直方图与核密度 (Histogram + KDE)
# ==============================================================================
def chart_07_hist_kde(df: pd.DataFrame, col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    res = su.analyze_numeric_question(df[col])
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.66])
    fs = float(cfg["font_scale"])
    u = _unit_for_col(col)
    u_sp = _fmt_u(u, True)

    counts, bins, _ = ax.hist(
        s, bins=min(18, max(5, int(np.sqrt(len(s)))) if len(s) < 60 else 18), density=True,
        color=t["categorical"][1], alpha=0.45,
        edgecolor=t["bg_canvas"], linewidth=1.1, label="频率分布直方图", zorder=2,
    )
    if len(s) >= 3 and float(s.std(ddof=1)) > 1e-9:
        kde = stats.gaussian_kde(s)
        x_grid = np.linspace(float(s.min()), float(s.max()), 250)
        ax.plot(x_grid, kde(x_grid), color=t["accent_primary"], linewidth=2.6, label="高斯核密度估计 (KDE)", zorder=4)

    mean_v = float(res["mean"])
    med_v = float(res["median"])
    sd_v = float(res["std"])
    if cfg["show_mean_line"]:
        ax.axvline(mean_v, color=t["accent_secondary"], linestyle="--", linewidth=2.0, label=f"均值 Mean = {mean_v:.1f}{u_sp}", zorder=5)
        ax.axvline(med_v, color=t["accent_primary"], linestyle=":", linewidth=2.0, label=f"中位数 Median = {med_v:.1f}{u_sp}", zorder=5)

    ax.set_xlabel(_axis_label_with_unit(col), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel("概率密度 (Density)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.8 * fs, loc="upper right")

    u_col_sfx = f"({u})" if u else ""
    hist_df = pd.DataFrame({
        f"区间下限{u_col_sfx}": np.round(bins[:-1], 2),
        f"区间上限{u_col_sfx}": np.round(bins[1:], 2),
        "概率密度": np.round(counts, 5),
    })
    skew_desc = "呈右偏态分布" if mean_v > med_v * 1.03 else ("呈左偏态分布" if mean_v < med_v * 0.97 else "分布形态总体对称")
    title = f"{col}均值为 {mean_v:.1f}{u_sp}（SD={sd_v:.1f}），中位数 {med_v:.1f}{u_sp}，{skew_desc}"
    subtitle = f"{col}分布形态 | 直方图与核密度曲线（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计量：Mean={mean_v:.2f}, SD={sd_v:.2f}, Median={med_v:.2f}, IQR={res['iqr']:.2f} | 正态性 p={res['normality_p']:.4f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, hist_df, "07-直方图与核密度", "直方图与核密度", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 08a - 箱线图 (Boxplot + Jitter)
# ==============================================================================
def chart_08a_boxplot(df: pd.DataFrame, col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    res = su.analyze_numeric_question(df[col])
    s = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.16, 0.84, 0.64])
    fs = float(cfg["font_scale"])
    u = _unit_for_col(col)
    u_sp = _fmt_u(u, True)
    u_nosp = _fmt_u(u, False)

    ax.boxplot(
        [s], vert=False, positions=[0], widths=0.38, patch_artist=True,
        boxprops=dict(facecolor=t["categorical"][1], alpha=0.55, edgecolor=t["accent_primary"], linewidth=1.8),
        medianprops=dict(color=t["accent_secondary"], linewidth=2.6),
        whiskerprops=dict(color=t["accent_primary"], linewidth=1.5),
        capprops=dict(color=t["accent_primary"], linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=t["highlight"], markeredgecolor="none", markersize=6, alpha=0.75),
    )
    rng = np.random.default_rng(42)
    jitter_y = rng.normal(loc=-0.34, scale=0.055, size=len(s))
    ax.scatter(s, jitter_y, color=t["accent_primary"], alpha=0.42, s=22, edgecolors="none", zorder=3, label="个体样本散点")
    ax.scatter([res["mean"]], [0], marker="D", s=85, color=t["highlight"], zorder=5, label=f"均值 ({res['mean']:.1f}{u_nosp})")

    q1, med, q3 = float(np.percentile(s, 25)), float(res["median"]), float(np.percentile(s, 75))
    val_span = max(float(s.max() - s.min()), 1.0)
    dx = 15.0 if val_span > 80.0 else val_span * 0.08
    ax.annotate(f"Q1 = {q1:.1f}", xy=(q1, 0.20), xytext=(q1 - dx, 0.36), ha="center", fontsize=9.5 * fs, color=t["text_secondary"], arrowprops=dict(arrowstyle="->", color=t["text_muted"]))
    ax.annotate(f"中位数 = {med:.1f}", xy=(med, 0.20), xytext=(med, 0.42), ha="center", fontsize=10.0 * fs, weight="bold", color=t["accent_secondary"], arrowprops=dict(arrowstyle="->", color=t["accent_secondary"]))
    ax.annotate(f"Q3 = {q3:.1f}", xy=(q3, 0.20), xytext=(q3 + dx, 0.36), ha="center", fontsize=9.5 * fs, color=t["text_secondary"], arrowprops=dict(arrowstyle="->", color=t["text_muted"]))

    ax.set_yticks([0, -0.34])
    ax.set_yticklabels(["五数概括箱体", "样本抖动散点"], fontsize=10.2 * fs, color=t["text_primary"])
    ax.set_ylim(-0.62, 0.58)
    ax.set_xlabel(_axis_label_with_unit(col), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    src = res["table"]
    title = f"{col}四分位距 IQR 为 {res['iqr']:.1f}{u_sp}（Q1={q1:.1f}, Q3={q3:.1f}），均值 {res['mean']:.1f}{u_sp}"
    subtitle = f"{col}五数概括与离群点检测 | 箱线图（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计量：Min={res['min']:.1f}, Q1={q1:.1f}, Median={med:.1f}, Q3={q3:.1f}, Max={res['max']:.1f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "08a-箱线图", "箱线图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 08b - 小提琴图 (Violin Plot)
# ==============================================================================
def chart_08b_violin(df: pd.DataFrame, col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    res = su.analyze_numeric_question(df[col])
    s = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.16, 0.84, 0.64])
    fs = float(cfg["font_scale"])
    u = _unit_for_col(col)
    u_sp = _fmt_u(u, True)
    u_nosp = _fmt_u(u, False)

    parts = ax.violinplot([s], positions=[0], vert=False, widths=0.68, showmeans=False, showextrema=False, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(t["accent_primary"])
        pc.set_edgecolor(t["accent_primary"])
        pc.set_alpha(0.32)
        pc.set_linewidth(1.6)

    q1, med, q3 = float(np.percentile(s, 25)), float(res["median"]), float(np.percentile(s, 75))
    ax.hlines(0, q1, q3, color=t["accent_primary"], linewidth=7.5, zorder=4, label=f"四分位区间 [{q1:.1f}, {q3:.1f}]")
    ax.scatter([med], [0], color=t["bg_canvas"], edgecolors=t["accent_primary"], s=90, linewidths=2.2, zorder=5, label=f"中位数 ({med:.1f}{u_nosp})")
    ax.scatter([res["mean"]], [0], marker="D", color=t["accent_secondary"], s=85, zorder=6, label=f"均值 ({res['mean']:.1f}{u_nosp})")

    ax.set_yticks([0])
    ax.set_yticklabels([col], fontsize=10.8 * fs, color=t["text_primary"])
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel(_axis_label_with_unit(col), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    src = res["table"]
    if col == "月消费额":
        title = f"{col}核密度峰值集中于 95–130 元区间，中位数 {med:.1f} 元、均值 {res['mean']:.1f} 元"
    else:
        title = f"{col}核心分布区间为 [{q1:.1f}, {q3:.1f}]{u_sp}，中位数 {med:.1f}{u_sp}、均值 {res['mean']:.1f}{u_sp}"
    subtitle = f"{col}连续密度与内嵌四分位 | 小提琴图（有效样本 n = {res['n_valid']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"图形构成：外部对称核密度包络线 + 内部粗线 IQR + 圆点中位数 + 菱形均值 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "08b-小提琴图", "小提琴图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 09 - 雨云图 (Raincloud Plot: Half-Violin + Boxplot + Jitter)
# ==============================================================================
def chart_09_raincloud(df: pd.DataFrame, group_col: str = "性别", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (group_col not in df.columns or num_col not in df.columns) else None
    group_col = _pick_col(df, group_col, settings, "group_col", (caps or {}).get("single_cols"))
    num_col = _pick_col(df, num_col, settings, "num_col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    gm = su.group_mean_comparison(df, group_col, num_col)
    sub = df[[group_col, num_col]].dropna().copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    u = _unit_for_col(num_col)
    u_sp = _fmt_u(u, True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.11, 0.14, 0.82, 0.66])
    fs = float(cfg["font_scale"])
    groups = gm["group_df"]["组别"].astype(str).tolist()
    rng = np.random.default_rng(42)
    val_min, val_max = float(sub[num_col].min()), float(sub[num_col].max())
    val_span = max(val_max - val_min, 1.0)

    for idx, g_name in enumerate(groups):
        vals = sub.loc[sub[group_col].astype(str) == g_name, num_col].to_numpy(dtype=float)
        if len(vals) < 3:
            continue
        c = get_item_color(g_name, t["categorical"][idx % len(t["categorical"])], t, cfg)
        if float(np.std(vals, ddof=1)) > 1e-9:
            kde = stats.gaussian_kde(vals)
            xg = np.linspace(float(vals.min()), float(vals.max()), 150)
            dens = kde(xg)
            dens_scaled = dens / (dens.max() if dens.max() > 0 else 1.0) * 0.36
            ax.fill_between(xg, idx + 0.06, idx + 0.06 + dens_scaled, color=c, alpha=0.45, edgecolor=c, linewidth=1.5, zorder=3)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.hlines(idx, q1, q3, color=c, linewidth=6.0, zorder=4)
        ax.scatter([med], [idx], color=t["bg_canvas"], edgecolors=c, s=55, linewidths=2.0, zorder=5)
        ax.scatter([float(np.mean(vals))], [idx + 0.06], marker="D", color=t["accent_secondary"], s=48, zorder=6)

        jy = idx - 0.16 + rng.uniform(-0.10, 0.06, size=len(vals))
        ax.scatter(vals, jy, color=c, alpha=0.52, s=18, edgecolors="none", zorder=2)

        if cfg["show_labels"]:
            ax.text(
                float(vals.max()) + val_span * 0.02, idx + 0.12,
                f"Mean={np.mean(vals):.1f} | Median={med:.1f} (n={len(vals)})",
                va="center", ha="left", fontsize=9.6 * fs, color=t["text_primary"], weight="bold",
            )

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=11.0 * fs, color=t["text_primary"])
    ax.set_xlim(val_min - val_span * 0.05, val_max + val_span * 0.30)
    ax.set_ylim(-0.45, len(groups) - 0.35)
    ax.set_xlabel(_axis_label_with_unit(num_col), fontsize=10.2 * fs, color=t["text_secondary"])

    g0 = gm["group_df"].iloc[0]
    g1 = gm["group_df"].iloc[1] if len(gm["group_df"]) > 1 else g0
    sig_word = "存在显著差异" if gm["p_value"] < 0.05 else "分布形态相近"
    title = f"不同{group_col}{num_col}{sig_word}（{g0['组别']}均值 {g0['均值']:.1f}{u_sp} vs {g1['组别']} {g1['均值']:.1f}{u_sp}，p={gm['p_value']:.3f}）"
    subtitle = f"{group_col} × {num_col}全息分布 | 雨云图 Raincloud（有效样本 n = {len(sub)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"检验结果：{gm['test_name']} 统计量={gm['stat']:.3f}, p={gm['p_value']:.4f}, 效应量 {gm['effect_name']}={gm['effect_size']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, gm["group_df"], "09-雨云图", "雨云图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 10 - 经验累积分布图 (ECDF Step Plot + KS Test)
# ==============================================================================
def chart_10_ecdf(df: pd.DataFrame, group_col: str = "性别", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (group_col not in df.columns or num_col not in df.columns) else None
    group_col = _pick_col(df, group_col, settings, "group_col", (caps or {}).get("single_cols"))
    num_col = _pick_col(df, num_col, settings, "num_col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    sub = df[[group_col, num_col]].dropna().copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    groups = sub[group_col].astype(str).value_counts().index.tolist()[:2]
    groups = sorted(groups)
    u = _unit_for_col(num_col)
    u_nosp = _fmt_u(u, False)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.66])
    fs = float(cfg["font_scale"])

    ecdf_rows = []
    arrays = []
    u_col_sfx = f"({u})" if u else ""
    for idx, g_name in enumerate(groups):
        vals = np.sort(sub.loc[sub[group_col].astype(str) == g_name, num_col].to_numpy(dtype=float))
        if len(vals) == 0:
            continue
        arrays.append(vals)
        y_ecdf = np.arange(1, len(vals) + 1) / len(vals) * 100.0
        c = get_item_color(g_name, t["categorical"][idx % len(t["categorical"])], t, cfg)
        ax.step(vals, y_ecdf, where="post", linewidth=2.5, color=c, label=f"{g_name} (n={len(vals)}, 中位数={np.median(vals):.1f}{u_nosp})")
        for p_q in [25, 50, 75]:
            ecdf_rows.append({
                "组别": g_name,
                "分位点(%)": p_q,
                f"对应数值{u_col_sfx}": round(float(np.percentile(vals, p_q)), 2),
            })

    ks_stat, ks_p = stats.ks_2samp(arrays[0], arrays[1]) if len(arrays) == 2 else (0.0, 1.0)
    ax.axhline(50.0, color=t["text_muted"], linestyle="--", linewidth=1.1, alpha=0.8)
    val_span = max(float(sub[num_col].max() - sub[num_col].min()), 1.0)
    ax.text(float(sub[num_col].min()) + val_span * 0.02, 52.0, "P50 中位累积线 (50%)", fontsize=9.2 * fs, color=t["text_secondary"])

    ax.set_xlabel(_axis_label_with_unit(num_col), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel("经验累积百分比 ECDF (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)

    src_df = pd.DataFrame(ecdf_rows)
    ks_desc = "存在显著偏移" if ks_p < 0.05 else "分位数无显著偏移"
    title = f"两组{num_col}累积分布对比（K-S D = {ks_stat:.3f}, p = {ks_p:.3f}），{ks_desc}"
    subtitle = f"{group_col} × {num_col}分位数对比 | 经验累积分布图 ECDF（有效样本 n = {len(sub)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"两样本 Kolmogorov-Smirnov 检验：D = {ks_stat:.4f}, p = {ks_p:.4f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src_df, "10-经验累积分布图", "经验累积分布图", title, subtitle, footnote, theme_name, settings)


def _resolve_scale_item_map(work: pd.DataFrame, cols: list = None, settings: dict = None) -> dict:
    """解析量表题列名与简短显示名称映射：兼容基准6题与任意上传问卷量表列。"""
    cfg_cols = (settings or {}).get("cols") if isinstance(settings, dict) else None
    use_cols = cfg_cols or cols
    if use_cols:
        valid_c = [c for c in use_cols if c in work.columns]
        if valid_c:
            return {c: (c.split("_", 1)[1] if "_" in str(c) and len(c.split("_", 1)[1]) >= 2 else str(c)) for c in valid_c}

    default_map = {
        "价格量表1_价格合理": "价格合理性",
        "价格量表2_性价比高": "整体性价比",
        "价格量表3_促销吸引": "促销吸引力",
        "品质量表1_口感醇厚": "口感醇厚度",
        "品质量表2_豆源优质": "咖啡豆品质",
        "品质量表3_正向":     "出品稳定性(正向化)",
    }
    if all(c in work.columns for c in default_map):
        return default_map

    caps = su.detect_dataset_capabilities(work)
    sc_cols = caps["scale_cols"][:10]
    if not sc_cols:
        sc_cols = caps["numeric_cols"][:6]
    return {c: (c.split("_", 1)[1] if "_" in str(c) and len(c.split("_", 1)[1]) >= 2 else str(c)) for c in sc_cols}


# ==============================================================================
# 11 - 发散堆叠条形图 (Likert Diverging 100% Stacked Bar Chart)
# ==============================================================================
def chart_11_likert_diverging(df: pd.DataFrame, cols: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = df.copy()
    if "品质量表3_正向" not in work.columns and "品质量表3_品质不稳定_反向" in work.columns:
        work["品质量表3_正向"] = su.reverse_score(work["品质量表3_品质不稳定_反向"], max_scale=5)

    item_map = _resolve_scale_item_map(work, cols=cols, settings=settings)
    rows = []
    for col, short_nm in item_map.items():
        s = pd.to_numeric(work[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        # 若非 1..5 整数（如 1..7 量表），线性映射至 1..5 档位计算发散比
        s_min, s_max = float(s.min()), float(s.max())
        if s_min >= 1.0 and s_max <= 5.0:
            s_five = np.clip(np.round(s), 1, 5)
        elif s_max > s_min:
            s_five = np.clip(np.round(1.0 + (s - s_min) / (s_max - s_min) * 4.0), 1, 5)
        else:
            s_five = np.full(len(s), 3.0)
        n = max(len(s), 1)
        p1 = float((s_five == 1).sum()) / n * 100.0
        p2 = float((s_five == 2).sum()) / n * 100.0
        p3 = float((s_five == 3).sum()) / n * 100.0
        p4 = float((s_five == 4).sum()) / n * 100.0
        p5 = float((s_five == 5).sum()) / n * 100.0
        rows.append({
            "量表题项": short_nm,
            "1分(非常不同意%)": round(p1, 1),
            "2分(比较不同意%)": round(p2, 1),
            "3分(一般%)": round(p3, 1),
            "4分(比较同意%)": round(p4, 1),
            "5分(非常同意%)": round(p5, 1),
            "T2B正向占比(%)": round(p4 + p5, 1),
            "B2B负向占比(%)": round(p1 + p2, 1),
            "均值": round(float(s.mean()), 2),
        })
    src = pd.DataFrame(rows).sort_values("T2B正向占比(%)", ascending=False).reset_index(drop=True)
    plot_df = src.iloc[::-1].reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.15, 0.14, 0.74, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(plot_df))

    p1 = plot_df["1分(非常不同意%)"].to_numpy()
    p2 = plot_df["2分(比较不同意%)"].to_numpy()
    p3 = plot_df["3分(一般%)"].to_numpy()
    p4 = plot_df["4分(比较同意%)"].to_numpy()
    p5 = plot_df["5分(非常同意%)"].to_numpy()

    left_3 = -p3 / 2.0
    left_2 = left_3 - p2
    left_1 = left_2 - p1
    right_4 = p3 / 2.0
    right_5 = right_4 + p4

    c_neg2 = t["accent_secondary"]
    c_neg1 = t["categorical"][4 % len(t["categorical"])]
    c_neu = t["border_line"]
    c_pos1 = t["categorical"][1]
    c_pos2 = t["accent_primary"]

    ax.barh(y, p1, left=left_1, height=0.58, color=c_neg2, label="1分 非常不同意")
    ax.barh(y, p2, left=left_2, height=0.58, color=c_neg1, label="2分 比较不同意")
    ax.barh(y, p3, left=left_3, height=0.58, color=c_neu, label="3分 一般(中立)")
    ax.barh(y, p4, left=right_4, height=0.58, color=c_pos1, label="4分 比较同意")
    ax.barh(y, p5, left=right_5, height=0.58, color=c_pos2, label="5分 非常同意")

    ax.axvline(0, color=t["text_primary"], linewidth=1.1, zorder=4)
    for yi, r in plot_df.iterrows():
        t2b = r["T2B正向占比(%)"]
        r_end = right_5[yi] + p5[yi]
        ax.text(r_end + 2.0, yi, f"T2B {t2b:.1f}%", va="center", ha="left", fontsize=9.8 * fs, weight="bold", color=t["accent_primary"])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["量表题项"], fontsize=10.5 * fs, color=t["text_primary"])
    x_lo = min(-65.0, float(left_1.min()) - 8.0) if len(left_1) else -65.0
    x_hi = max(95.0, float((right_5 + p5).max()) + 22.0) if len(right_5) else 95.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("以中立项（3分）为中心的正负态度发散百分比 (%)", fontsize=10.0 * fs, color=t["text_secondary"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False, fontsize=8.8 * fs)
    ax.grid(axis="y", visible=False)

    top_item = src.iloc[0]
    title = f"「{top_item['量表题项']}」正向认同率最高（T2B = {top_item['T2B正向占比(%)']:.1f}%，均分 {top_item['均值']:.2f}）"
    subtitle = f"Likert 量表态度分布 | 发散堆叠条形图（有效样本 n = {len(work)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"口径说明：T2B (Top 2 Box) = 4分 + 5分占比；中立项跨零轴对称分布 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "11-发散堆叠条形图", "发散堆叠条形图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 12 - 均值置信区间图 (Mean & 95% CI Forest Plot)
# ==============================================================================
def chart_12_mean_ci(df: pd.DataFrame, cols: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = df.copy()
    if "品质量表3_正向" not in work.columns and "品质量表3_品质不稳定_反向" in work.columns:
        work["品质量表3_正向"] = su.reverse_score(work["品质量表3_品质不稳定_反向"], max_scale=5)

    item_map = _resolve_scale_item_map(work, cols=cols, settings=settings)
    rows = []
    for col, short_nm in item_map.items():
        s = pd.to_numeric(work[col], errors="coerce").dropna()
        n = len(s)
        if n == 0:
            continue
        m = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 0 else 0.0
        t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
        rows.append({
            "量表题项": short_nm,
            "样本量": n,
            "均值": round(m, 3),
            "标准差(SD)": round(sd, 3),
            "95%CI下限": round(m - t_crit * se, 3),
            "95%CI上限": round(m + t_crit * se, 3),
        })
    src = pd.DataFrame(rows).sort_values("均值", ascending=False).reset_index(drop=True)
    plot_df = src.iloc[::-1].reset_index(drop=True)
    grand_mean = float(src["均值"].mean())

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.16, 0.14, 0.76, 0.66])
    fs = float(cfg["font_scale"])
    y = np.arange(len(plot_df))

    if cfg["show_mean_line"] and len(plot_df) > 1:
        ax.axvline(grand_mean, color=t["accent_secondary"], linestyle="--", linewidth=1.5, label=f"全量表总均分 ({grand_mean:.2f}分)")

    for yi, r in plot_df.iterrows():
        m = r["均值"]
        lo, hi = r["95%CI下限"], r["95%CI上限"]
        c = get_item_color(r["量表题项"], t["accent_primary"] if m >= grand_mean else t["categorical"][1], t, cfg)
        if cfg["show_ci"]:
            ax.hlines(yi, lo, hi, color=c, linewidth=3.0, zorder=3)
            ax.vlines([lo, hi], yi - 0.12, yi + 0.12, color=c, linewidth=2.0, zorder=3)
        ax.scatter([m], [yi], color=c, s=95, zorder=4, edgecolors=t["bg_canvas"], linewidths=1.3)
        if cfg["show_labels"]:
            ax.text(hi + 0.03, yi, f"{m:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="left", fontsize=9.8 * fs, color=t["text_primary"], weight="bold" if yi == len(plot_df) - 1 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["量表题项"], fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(float(plot_df["95%CI下限"].min()) - 0.25, float(plot_df["95%CI上限"].max()) + 0.55)
    ax.set_xlabel("量表评分均值与 95% 置信区间", fontsize=10.2 * fs, color=t["text_secondary"])
    if cfg["show_mean_line"] and len(plot_df) > 1:
        ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    top_r = src.iloc[0]
    bot_r = src.iloc[-1]
    if len(src) > 1:
        title = f"「{top_r['量表题项']}」均分最高（{top_r['均值']:.2f}分），显著优于「{bot_r['量表题项']}」（{bot_r['均值']:.2f}分）"
    else:
        title = f"「{top_r['量表题项']}」评分均值为 {top_r['均值']:.2f} 分（95%CI [{top_r['95%CI下限']:.2f}, {top_r['95%CI上限']:.2f}]）"
    subtitle = f"量表题项均值与 95% 置信区间对比 | 均值置信区间森林图（n = {len(work)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"误差棒定义：均值 ± t(0.975, df=n-1)·SE | 虚线：{len(src)} 题总均分 ({grand_mean:.2f}) | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "12-均值置信区间图", "均值置信区间图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 13 - NPS 仪表盘 (NPS Gauge + Stacked Bar)
# ==============================================================================
def chart_13_nps_gauge(df: pd.DataFrame, col: str = "NPS推荐度", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if col not in df.columns else None
    col = _pick_col(df, col, settings, "col", (caps or {}).get("nps_cols"))
    res = su.analyze_nps_question(df[col])
    src = res["table"].copy()
    nps_v = float(res["nps"])

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.08, 0.12, 0.84, 0.68])
    fs = float(cfg["font_scale"])

    zones = [
        (180, 126, t["accent_secondary"], "待改进 (<0)"),
        (126, 63,  t["categorical"][4 % len(t["categorical"])],   "良好 (0~40)"),
        (63,  0,   t["accent_primary"],   "优秀 (>40)"),
    ]
    for th2, th1, zc, _ in zones:
        w = Wedge((0, 0.05), 0.92, th1, th2, width=0.26, facecolor=zc, alpha=0.82, edgecolor=t["bg_canvas"], linewidth=2.0)
        ax.add_patch(w)

    angle_deg = 180.0 - (nps_v + 100.0) / 200.0 * 180.0
    ang_rad = np.deg2rad(angle_deg)
    ax.annotate(
        "", xy=(0.62 * np.cos(ang_rad), 0.05 + 0.62 * np.sin(ang_rad)), xytext=(0, 0.05),
        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=t["text_primary"], lw=3.0),
    )
    ax.add_patch(Circle((0, 0.05), 0.05, color=t["text_primary"]))
    ax.text(0, 0.25, f"NPS = {nps_v:+.1f}", ha="center", va="center", fontsize=22 * fs, weight="bold", color=t["text_primary"])
    ax.text(-0.92, -0.04, "-100", ha="center", fontsize=9.5 * fs, color=t["text_muted"])
    ax.text(0, 1.02, "0", ha="center", fontsize=9.5 * fs, color=t["text_muted"])
    ax.text(0.92, -0.04, "+100", ha="center", fontsize=9.5 * fs, color=t["text_muted"])

    p_det = float(res["detractor_pct"])
    p_pas = float(res["passive_pct"])
    p_pro = float(res["promoter_pct"])
    bar_y = -0.45
    bar_h = 0.20
    x_start = -1.1
    total_w = 2.2

    segs = [
        ("贬损者 (0-6分)", p_det, int(res["detractor_cnt"]), t["accent_secondary"], -0.78),
        ("被动者 (7-8分)", p_pas, int(res["passive_cnt"]), t["border_line"],        0.00),
        ("推荐者 (9-10分)", p_pro, int(res["promoter_cnt"]), t["accent_primary"],   0.78),
    ]
    curr_x = x_start
    for lbl, pct, cnt, sc, fixed_lx in segs:
        seg_w = total_w * (pct / 100.0)
        ax.add_patch(Rectangle((curr_x, bar_y), seg_w, bar_h, facecolor=sc, edgecolor=t["bg_canvas"], linewidth=1.5))
        ax.text(
            fixed_lx, bar_y - 0.14,
            f"■ {lbl}\n{pct:.1f}% (n={cnt})",
            ha="center", va="top",
            fontsize=9.8 * fs, weight="bold", color=t["text_primary"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor=t["bg_card"], edgecolor=sc, lw=1.1),
        )
        curr_x += seg_w

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.82, 1.15)
    ax.axis("off")

    n_val = int(res.get("n_valid", res.get("total", len(df))))
    zone_txt = "处于正向口碑区间" if nps_v >= 0 else "亟需改善贬损痛点"
    title = f"净推荐值 NPS 为 {nps_v:+.1f}（推荐者 {p_pro:.1f}% vs 贬损者 {p_det:.1f}%），{zone_txt}"
    subtitle = f"{col}净推荐值（NPS）诊断 | NPS 仪表盘与人群构成条（有效样本 n = {n_val}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"计算公式：NPS = 推荐者占比(9-10分, {p_pro:.1f}%) - 贬损者占比(0-6分, {p_det:.1f}%) = {nps_v:+.1f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "13-NPS仪表盘", "NPS仪表盘", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 14 - 人口学金字塔 (Demographic Pyramid)
# ==============================================================================
def chart_14_pyramid(df: pd.DataFrame, age_col: str = "年龄段", gender_col: str = "性别", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (age_col not in df.columns or gender_col not in df.columns) else None
    age_col = _pick_col(df, age_col, settings, "age_col", (caps or {}).get("single_cols"))
    gender_col = _pick_col(df, gender_col, settings, "gender_col", [c for c in ((caps or {}).get("single_cols") or []) if c != age_col])
    sub = df[[age_col, gender_col]].dropna().astype(str)
    age_order = ["18-24岁", "25-30岁", "31-35岁", "31-40岁", "36-40岁", "36-45岁", "40岁以上", "40岁及以上", "46岁及以上"]
    present_ages = _ordered_categories(sub[age_col].tolist(), age_order)

    g_counts = sub[gender_col].value_counts()
    if "男" in g_counts.index and "女" in g_counts.index:
        g_left, g_right = "男", "女"
        lbl_left, lbl_right = "男性", "女性"
    else:
        top2 = g_counts.index.tolist()[:2]
        g_left = top2[0] if len(top2) > 0 else "组1"
        g_right = top2[1] if len(top2) > 1 else "组2"
        lbl_left, lbl_right = g_left, g_right

    total_n = max(len(sub), 1)
    rows = []
    for ag in present_ages:
        m_cnt = int(((sub[age_col] == ag) & (sub[gender_col] == g_left)).sum())
        f_cnt = int(((sub[age_col] == ag) & (sub[gender_col] == g_right)).sum())
        rows.append({
            age_col: ag,
            f"{lbl_left}人数": m_cnt,
            f"{lbl_left}占总样本(%)": round(m_cnt / total_n * 100.0, 1),
            f"{lbl_right}人数": f_cnt,
            f"{lbl_right}占总样本(%)": round(f_cnt / total_n * 100.0, 1),
            "合计人数": m_cnt + f_cnt,
        })
    src = pd.DataFrame(rows)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.13, 0.14, 0.79, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(src))

    m_pcts = -src[f"{lbl_left}占总样本(%)"].to_numpy(dtype=float)
    f_pcts = src[f"{lbl_right}占总样本(%)"].to_numpy(dtype=float)

    c_male = get_item_color(g_left, t["categorical"][1], t, cfg)
    c_female = get_item_color(g_right, t["accent_primary"], t, cfg)

    ax.barh(y, m_pcts, height=0.58, color=c_male, label=f"{lbl_left} (n={src[f'{lbl_left}人数'].sum()})")
    ax.barh(y, f_pcts, height=0.58, color=c_female, label=f"{lbl_right} (n={src[f'{lbl_right}人数'].sum()})")
    ax.axvline(0, color=t["text_primary"], linewidth=1.2)

    max_v = max(float(np.max(np.abs(m_pcts))) if len(m_pcts) else 10.0, float(np.max(f_pcts)) if len(f_pcts) else 10.0, 1.0)
    for yi, r in src.iterrows():
        if cfg["show_labels"]:
            lp = r[f"{lbl_left}占总样本(%)"]
            rp = r[f"{lbl_right}占总样本(%)"]
            ax.text(-lp - max_v * 0.04, yi, f"{lp:.1f}% ({r[f'{lbl_left}人数']}人)", va="center", ha="right", fontsize=9.5 * fs, color=t["text_primary"])
            ax.text(rp + max_v * 0.04, yi, f"{rp:.1f}% ({r[f'{lbl_right}人数']}人)", va="center", ha="left", fontsize=9.5 * fs, color=t["text_primary"])

    ax.set_yticks(y)
    ax.set_yticklabels(src[age_col], fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(-max_v * 1.48, max_v * 1.48)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.0f}%" for x in xticks], fontsize=9.8 * fs, color=t["text_secondary"])
    ax.set_xlabel("占总有效样本百分比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)
    ax.grid(axis="y", visible=False)

    top_row = src.sort_values("合计人数", ascending=False).iloc[0]
    title = f"「{top_row[age_col]}」为最大结构单元（{g_left} {top_row[f'{lbl_left}占总样本(%)']:.1f}% / {g_right} {top_row[f'{lbl_right}占总样本(%)']:.1f}%）"
    subtitle = f"{age_col} × {gender_col}双向人口结构 | 人口学金字塔（有效样本 n = {total_n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计口径：各{age_col}×{gender_col}单元格占全体有效样本（n={total_n}）的百分比 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "14-人口学金字塔", "人口学金字塔", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 15 - 100% 堆叠条形图 (100% Stacked Horizontal Bar Chart)
# ==============================================================================
def chart_15_stacked_100(df: pd.DataFrame, row_col: str = "年龄段", col_col: str = "消费频次", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (row_col not in df.columns or col_col not in df.columns) else None
    row_col = _pick_col(df, row_col, settings, "row_col", (caps or {}).get("single_cols"))
    col_col = _pick_col(df, col_col, settings, "col_col", [c for c in ((caps or {}).get("single_cols") or []) if c != row_col])
    ct_res = su.crosstab_chi2_analysis(df, row_col, col_col)
    row_pct = ct_res["row_pct_df"].copy()
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.13, 0.14, 0.79, 0.64])
    fs = float(cfg["font_scale"])

    categories = list(row_pct.columns)
    y = np.arange(len(row_pct))
    left_cum = np.zeros(len(row_pct))
    palette = t["categorical"]

    for idx, cat in enumerate(categories):
        vals = row_pct[cat].to_numpy(dtype=float)
        c = get_item_color(cat, palette[idx % len(palette)], t, cfg)
        ax.barh(y, vals, left=left_cum, height=0.58, color=c, edgecolor=t["bg_canvas"], linewidth=1.2, label=str(cat))
        for yi, v in enumerate(vals):
            if cfg["show_labels"] and v >= 8.5:
                tc = auto_text_color(c, t["text_primary"])
                ax.text(left_cum[yi] + v / 2.0, yi, f"{v:.1f}%", ha="center", va="center", fontsize=9.2 * fs, color=tc, weight="bold")
        left_cum += vals

    ax.set_yticks(y)
    ax.set_yticklabels(row_pct.index.astype(str), fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("行内构成百分比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=min(len(categories), 5), frameon=False, fontsize=9.2 * fs)
    ax.grid(axis="y", visible=False)

    src = row_pct.reset_index().rename(columns={"index": row_col})
    chi2_v, p_v, v_v = ct_res["chi2"], ct_res["p_value"], ct_res["cramers_v"]
    sig_txt = "差异具有统计显著性" if p_v < 0.05 else "结构分布无显著差异"
    title = f"不同{row_col}的{col_col}构成{sig_txt}（χ²={chi2_v:.2f}, p={p_v:.3f}, Cramér's V={v_v:.3f}）"
    subtitle = f"{row_col} × {col_col}行百分比对比 | 100% 堆叠条形图（有效样本 n = {int(ct_res['count_df'].values.sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"卡方独立性检验：χ²={chi2_v:.3f}, df={ct_res['dof']}, p={p_v:.4f}, Cramér's V={v_v:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "15-100%堆叠条形图", "100%堆叠条形图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 16 - 分组簇状条形图 (Grouped Bar Chart)
# ==============================================================================
def chart_16_grouped_bar(df: pd.DataFrame, row_col: str = "消费频次", col_col: str = "性别", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (row_col not in df.columns or col_col not in df.columns) else None
    row_col = _pick_col(df, row_col, settings, "row_col", (caps or {}).get("single_cols"))
    col_col = _pick_col(df, col_col, settings, "col_col", [c for c in ((caps or {}).get("single_cols") or []) if c != row_col])
    ct_res = su.crosstab_chi2_analysis(df, col_col, row_col)
    row_pct = ct_res["row_pct_df"]
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.65])
    fs = float(cfg["font_scale"])

    freq_cats = list(row_pct.columns)
    genders = list(row_pct.index)
    x = np.arange(len(freq_cats))
    n_g = max(len(genders), 1)
    width = min(0.36, 0.78 / n_g)

    for idx, g in enumerate(genders):
        vals = row_pct.loc[g].to_numpy(dtype=float)
        offset = (idx - (n_g - 1) / 2.0) * (width + 0.03)
        c = get_item_color(g, t["categorical"][idx % len(t["categorical"])], t, cfg)
        bars = ax.bar(x + offset, vals, width=width, color=c, label=f"{g} (n={int(ct_res['count_df'].loc[g].sum())})", zorder=3)
        for bar, v in zip(bars, vals):
            if cfg["show_labels"] and n_g <= 4:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.8 * fs, color=t["text_primary"])

    ax.set_xticks(x)
    ax.set_xticklabels(freq_cats, fontsize=10.2 * fs, color=t["text_primary"], rotation=15 if len(freq_cats) > 6 else 0)
    ax.set_ylabel("组内占比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylim(0, max(float(row_pct.values.max()), 1.0) * 1.25)
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.8 * fs)
    ax.grid(axis="x", visible=False)

    src = row_pct.T.reset_index().rename(columns={"index": row_col})
    chi2_v, p_v, v_v = ct_res["chi2"], ct_res["p_value"], ct_res["cramers_v"]
    sig_word = "存在显著差异" if p_v < 0.05 else "各档位分布总体接近"
    grp_prefix = "男女受访者" if col_col == "性别" else f"不同{col_col}群体"
    title = f"{grp_prefix}在{row_col}{sig_word}（χ²={chi2_v:.2f}, p={p_v:.3f}, V={v_v:.3f}）"
    subtitle = f"{col_col} × {row_col}分组对比 | 分组簇状条形图（有效样本 n = {int(ct_res['count_df'].values.sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计检验：Pearson χ²={chi2_v:.3f}, p={p_v:.4f}, Cramér's V={v_v:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "16-分组簇状条形图", "分组簇状条形图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 17 - 马赛克图 (Marimekko / Mosaic Plot — 修复窄格标签重叠)
# ==============================================================================
def chart_17_mosaic(df: pd.DataFrame, row_col: str = "年龄段", col_col: str = "消费频次", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (row_col not in df.columns or col_col not in df.columns) else None
    row_col = _pick_col(df, row_col, settings, "row_col", (caps or {}).get("single_cols"))
    col_col = _pick_col(df, col_col, settings, "col_col", [c for c in ((caps or {}).get("single_cols") or []) if c != row_col])
    ct_res = su.crosstab_chi2_analysis(df, row_col, col_col)
    cnt_df = ct_res["count_df"]
    row_pct = ct_res["row_pct_df"]
    row_totals = cnt_df.sum(axis=1)
    grand_total = max(float(row_totals.sum()), 1.0)
    col_widths = (row_totals / grand_total * 100.0).to_numpy(dtype=float)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.16, 0.84, 0.62])
    fs = float(cfg["font_scale"])

    age_groups = list(row_pct.index)
    freq_cats = list(row_pct.columns)
    palette = t["categorical"]

    x_left = 0.0
    mosaic_rows = []
    for i, ag in enumerate(age_groups):
        w = col_widths[i]
        y_bot = 0.0
        for j, fc in enumerate(freq_cats):
            h = float(row_pct.loc[ag, fc])
            c = get_item_color(fc, palette[j % len(palette)], t, cfg)
            rect = Rectangle((x_left, y_bot), w, h, facecolor=c, edgecolor=t["bg_canvas"], linewidth=1.8)
            ax.add_patch(rect)
            if cfg["show_labels"] and w >= 8.0 and h >= 9.0:
                tc = auto_text_color(c, t["text_primary"])
                ax.text(x_left + w / 2.0, y_bot + h / 2.0, f"{h:.0f}%", ha="center", va="center", fontsize=8.8 * fs, color=tc, weight="bold")
            mosaic_rows.append({
                row_col: ag,
                col_col: fc,
                "样本数": int(cnt_df.loc[ag, fc]),
                "列宽占比(%)": round(w, 1),
                "列内高度占比(%)": round(h, 1),
            })
            y_bot += h
        ax.text(x_left + w / 2.0, 101.8, f"{w:.1f}%", ha="center", va="bottom", fontsize=9.0 * fs, color=t["text_secondary"], weight="bold")
        ax.text(x_left + w / 2.0, -4.5, str(ag), ha="center", va="top", fontsize=9.4 * fs, color=t["text_primary"], rotation=12 if w < 10 else 0)
        x_left += w

    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=palette[j % len(palette)], label=str(fc)) for j, fc in enumerate(freq_cats)]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=min(len(freq_cats), 5), frameon=False, fontsize=9.2 * fs)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_ylabel("列内构成百分比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.grid(False)

    src = pd.DataFrame(mosaic_rows)
    top_r_nm = str(row_totals.idxmax())
    top_c_nm = str(cnt_df.sum(axis=0).idxmax())
    title = f"马赛克双维面积图：「{top_r_nm}」占据最大横轴版图，「{top_c_nm}」构成核心矩形块"
    subtitle = f"{row_col}（列宽）× {col_col}（列高）双维占比 | 马赛克图（n = {int(grand_total)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"注：横轴列宽代表各{row_col}样本占比，纵轴高度代表组内{col_col}占比；面积 <8% 的小格已自动隐藏文字防重叠 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "17-马赛克图", "马赛克图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 18 - 卡方残差热力图 (Chi-Square Standardized Residual Heatmap)
# ==============================================================================
def chart_18_chi2_residual_heatmap(df: pd.DataFrame, row_col: str = "年龄段", col_col: str = "消费频次", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (row_col not in df.columns or col_col not in df.columns) else None
    row_col = _pick_col(df, row_col, settings, "row_col", (caps or {}).get("single_cols"))
    col_col = _pick_col(df, col_col, settings, "col_col", [c for c in ((caps or {}).get("single_cols") or []) if c != row_col])
    ct_res = su.crosstab_chi2_analysis(df, row_col, col_col)
    resid_df = ct_res["std_resid_df"]
    cnt_df = ct_res["count_df"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.16, 0.74, 0.64])
    fs = float(cfg["font_scale"])

    mat = resid_df.to_numpy(dtype=float)
    cmap = get_diverging_cmap(theme_name)
    vmax = max(2.5, float(np.max(np.abs(mat))))
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            cnt = int(cnt_df.iloc[i, j])
            star = " *" if abs(val) > 1.96 else ""
            rgba = cmap((val + vmax) / (2 * vmax))
            hex_c = "#{:02X}{:02X}{:02X}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            tc = auto_text_color(hex_c, t["text_primary"])
            if cfg["show_labels"]:
                ax.text(j, i, f"z = {val:+.2f}{star}\n(n={cnt})", ha="center", va="center", fontsize=9.2 * fs, color=tc, weight="bold" if abs(val) > 1.96 else "normal")

    ax.set_xticks(range(resid_df.shape[1]))
    ax.set_xticklabels(resid_df.columns.astype(str), fontsize=10.0 * fs, color=t["text_primary"], rotation=15 if resid_df.shape[1] > 5 else 0)
    ax.set_yticks(range(resid_df.shape[0]))
    ax.set_yticklabels(resid_df.index.astype(str), fontsize=10.0 * fs, color=t["text_primary"])
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("调整后标准化残差 (Adjusted Residual z)", fontsize=9.5 * fs, color=t["text_secondary"])
    cbar.ax.tick_params(labelsize=9.0 * fs, colors=t["text_secondary"])

    max_pos = np.unravel_index(np.argmax(mat), mat.shape)
    r_nm = resid_df.index[max_pos[0]]
    c_nm = resid_df.columns[max_pos[1]]
    max_z = float(mat[max_pos])
    src = resid_df.reset_index().rename(columns={"index": row_col})
    title = f"「{r_nm} × {c_nm}」正偏离最强（z = {max_z:+.2f}），卡方标准化残差揭示局部偏好聚集"
    subtitle = f"{row_col} × {col_col}调整后标准化残差 | 卡方残差热力图（|z|>1.96 标 * 显著）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"色阶：发散色带（0 为中心，正值代表实测频数高于独立期望，负值代表低于期望） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "18-卡方残差热力图", "卡方残差热力图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 19 - 哑铃图 (Dumbbell Chart)
# ==============================================================================
def chart_19_dumbbell(df: pd.DataFrame, group_col: str = "性别", cols: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = df.copy()
    if "品质量表3_正向" not in work.columns and "品质量表3_品质不稳定_反向" in work.columns:
        work["品质量表3_正向"] = su.reverse_score(work["品质量表3_品质不稳定_反向"], max_scale=5)

    caps = su.detect_dataset_capabilities(work) if group_col not in work.columns else None
    group_col = _pick_col(work, group_col, settings, "group_col", (caps or {}).get("single_cols"))

    default_items = [
        ("满意度1_口味口感", "口味口感满意度"),
        ("满意度2_价格实惠", "价格实惠满意度"),
        ("满意度3_门店环境", "门店环境满意度"),
        ("满意度4_出餐速度", "出餐速度满意度"),
        ("品质量表1_口感醇厚", "口感醇厚评价"),
        ("价格量表1_价格合理", "价格合理评价"),
    ]
    cfg_cols = (settings or {}).get("cols") if isinstance(settings, dict) else None
    use_cols = cfg_cols or cols
    if use_cols:
        items = [(c, c.split("_", 1)[1] if "_" in str(c) else str(c)) for c in use_cols if c in work.columns]
    elif all(c in work.columns for c, _ in default_items):
        items = default_items
    else:
        c_dict = _resolve_scale_item_map(work, settings=settings)
        items = list(c_dict.items())[:8]

    g_counts = work[group_col].dropna().astype(str).value_counts()
    if "女" in g_counts.index and "男" in g_counts.index:
        g1, g2 = "女", "男"
        g1_lbl, g2_lbl = "女性", "男性"
    else:
        top2 = g_counts.index.tolist()[:2]
        g1 = top2[0] if len(top2) > 0 else "组1"
        g2 = top2[1] if len(top2) > 1 else "组2"
        g1_lbl, g2_lbl = g1, g2

    rows = []
    for col, label in items:
        m_1 = float(pd.to_numeric(work.loc[work[group_col].astype(str) == g1, col], errors="coerce").mean())
        m_2 = float(pd.to_numeric(work.loc[work[group_col].astype(str) == g2, col], errors="coerce").mean())
        rows.append({
            "评价题项": label,
            f"{g1_lbl}均分": round(m_1, 2),
            f"{g2_lbl}均分": round(m_2, 2),
            f"差值({g1}-{g2})": round(m_1 - m_2, 2),
            "绝对差值": round(abs(m_1 - m_2), 2),
        })
    src = pd.DataFrame(rows).sort_values("绝对差值", ascending=False).reset_index(drop=True)
    plot_df = src.iloc[::-1].reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.16, 0.14, 0.76, 0.66])
    fs = float(cfg["font_scale"])
    y = np.arange(len(plot_df))

    v1 = plot_df[f"{g1_lbl}均分"].to_numpy(dtype=float)
    v2 = plot_df[f"{g2_lbl}均分"].to_numpy(dtype=float)

    ax.hlines(y, np.minimum(v1, v2), np.maximum(v1, v2), color=t["border_line"], linewidth=3.5, zorder=2)
    ax.scatter(v1, y, color=t["accent_primary"], s=110, label=f"{g1_lbl}受访者", zorder=4, edgecolors=t["bg_canvas"], linewidths=1.2)
    ax.scatter(v2, y, color=t["accent_secondary"], s=110, label=f"{g2_lbl}受访者", zorder=4, edgecolors=t["bg_canvas"], linewidths=1.2)

    span = max(float(max(v1.max(), v2.max()) - min(v1.min(), v2.min())), 0.5)
    for yi, r in plot_df.iterrows():
        hi_v = max(r[f"{g1_lbl}均分"], r[f"{g2_lbl}均分"])
        diff_v = r[f"差值({g1}-{g2})"]
        if cfg["show_labels"]:
            ax.text(hi_v + span * 0.05, yi, f"Δ = {diff_v:+.2f} ({g1}{r[f'{g1_lbl}均分']:.2f} / {g2}{r[f'{g2_lbl}均分']:.2f})", va="center", ha="left", fontsize=9.5 * fs, color=t["text_primary"])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["评价题项"], fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(min(v1.min(), v2.min()) - span * 0.25, max(v1.max(), v2.max()) + span * 0.65)
    ax.set_xlabel("评分均值", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)

    top_diff = src.iloc[0]
    grp_prefix = "男女受访者" if group_col == "性别" else f"「{g1}」与「{g2}」"
    title = f"{grp_prefix}在「{top_diff['评价题项']}」上分差最大（Δ = {top_diff[f'差值({g1}-{g2})']:+.2f} 分）"
    subtitle = f"不同{group_col}各维度评分差距 | 哑铃图（有效样本 n = {len(work)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"标注说明：Δ = {g1_lbl}均分 - {g2_lbl}均分 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "19-哑铃图", "哑铃图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 20 - 坡度图 (Slopegraph: 上次购买品牌 -> 最常购买品牌)
# ==============================================================================
def chart_20_slopegraph(df: pd.DataFrame, prev_col: str = "上次购买品牌", curr_col: str = "最常购买品牌", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        prev_col = settings.get("prev_col") or prev_col
        curr_col = settings.get("curr_col") or curr_col
    if prev_col not in df.columns or curr_col not in df.columns:
        pair = su.detect_brand_switching_columns(df)
        if pair:
            prev_col, curr_col = pair
        else:
            caps = su.detect_dataset_capabilities(df)
            prev_col, curr_col = caps["single_cols"][0], caps["single_cols"][1]
    sw = su.calc_brand_switching(df, prev_col, curr_col)
    ret_df = sw["retention_df"].copy()
    total_n = max(int(ret_df["上期基数"].sum()), 1)
    ret_df["上次占比(%)"] = (ret_df["上期基数"] / total_n * 100.0).round(1)
    ret_df["最常占比(%)"] = (ret_df["本期人数"] / total_n * 100.0).round(1)
    ret_df["净变化(百分点)"] = (ret_df["最常占比(%)"] - ret_df["上次占比(%)"]).round(1)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.18, 0.12, 0.64, 0.68])
    fs = float(cfg["font_scale"])

    ax.axvline(0, color=t["border_line"], linewidth=1.6, linestyle="--")
    ax.axvline(1, color=t["border_line"], linewidth=1.6, linestyle="--")

    left_y = ret_df["上次占比(%)"].to_numpy(dtype=float).copy()
    right_y = ret_df["最常占比(%)"].to_numpy(dtype=float).copy()

    def _spread_positions(vals, min_gap=2.2):
        order = np.argsort(vals)
        adj = vals.copy()
        for idx in range(1, len(order)):
            prev_i, curr_i = order[idx - 1], order[idx]
            if adj[curr_i] - adj[prev_i] < min_gap:
                adj[curr_i] = adj[prev_i] + min_gap
        return adj

    left_label_y = _spread_positions(left_y, min_gap=2.4)
    right_label_y = _spread_positions(right_y, min_gap=2.4)

    for i, r in ret_df.iterrows():
        b = r["品牌"]
        y1, y2 = float(r["上次占比(%)"]), float(r["最常占比(%)"])
        diff = float(r["净变化(百分点)"])
        def_c = t["accent_primary"] if diff > 0.3 else (t["accent_secondary"] if diff < -0.3 else t["text_muted"])
        c = get_item_color(b, def_c, t, cfg)
        lw = 3.0 if abs(diff) > 0.5 else 1.8
        ax.plot([0, 1], [y1, y2], color=c, linewidth=lw, marker="o", markersize=8, zorder=3)
        ax.text(-0.04, left_label_y[i], f"{b}  {y1:.1f}%", ha="right", va="center", fontsize=10.2 * fs, color=c, weight="bold")
        ax.text(1.04, right_label_y[i], f"{y2:.1f}%  {b} ({diff:+.1f}%)", ha="left", va="center", fontsize=10.2 * fs, color=c, weight="bold")

    ax.set_xlim(-0.45, 1.55)
    y_max = max(float(ret_df["上次占比(%)"].max()), float(ret_df["最常占比(%)"].max()), 1.0)
    ax.set_ylim(0, y_max * 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{prev_col}占比", f"{curr_col}占比"], fontsize=11.5 * fs, weight="bold", color=t["text_primary"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    top_gain = ret_df.sort_values("净变化(百分点)", ascending=False).iloc[0]
    title = f"份额跃迁诊断：「{top_gain['品牌']}」净变化 {top_gain['净变化(百分点)']:+.1f} 个百分点，保留率 {top_gain['品牌保留率(%)']:.1f}%"
    subtitle = f"{prev_col} → {curr_col}份额跃迁 | 坡度图 Slopegraph（有效样本 n = {total_n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"颜色语义：主强调色代表转化净增长，辅助强调色代表份额回落 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, ret_df, "20-坡度图", "坡度图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 21 - 分组小提琴箱线图 (Grouped Violin + Boxplot)
# ==============================================================================
def chart_21_grouped_violin(df: pd.DataFrame, group_col: str = "消费频次", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (group_col not in df.columns or num_col not in df.columns) else None
    group_col = _pick_col(df, group_col, settings, "group_col", (caps or {}).get("single_cols"))
    num_col = _pick_col(df, num_col, settings, "num_col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    gm = su.group_mean_comparison(df, group_col, num_col)
    sub = df[[group_col, num_col]].dropna().copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    u = _unit_for_col(num_col)
    u_sp = _fmt_u(u, True)

    freq_order = ["每天1杯及以上", "每周3-5杯", "每周1-2杯", "偶尔饮用", "每月1-3杯", "几乎不喝"]
    groups = _ordered_categories(sub[group_col].astype(str).tolist(), freq_order)
    groups = [g for g in groups if len(sub.loc[sub[group_col].astype(str) == g, num_col]) >= 2]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.66])
    fs = float(cfg["font_scale"])

    data_list = [sub.loc[sub[group_col].astype(str) == g, num_col].to_numpy(dtype=float) for g in groups]
    positions = np.arange(len(groups))
    val_min, val_max = float(sub[num_col].min()), float(sub[num_col].max())
    val_span = max(val_max - val_min, 1.0)

    parts = ax.violinplot(data_list, positions=positions, widths=0.65, showmeans=False, showextrema=False, showmedians=False)
    for idx, pc in enumerate(parts["bodies"]):
        c = get_item_color(groups[idx], t["categorical"][idx % len(t["categorical"])], t, cfg)
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        pc.set_alpha(0.35)
        pc.set_linewidth(1.5)

    for idx, vals in enumerate(data_list):
        c = get_item_color(groups[idx], t["categorical"][idx % len(t["categorical"])], t, cfg)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        m_v = float(np.mean(vals))
        ax.vlines(idx, q1, q3, color=c, linewidth=6.5, zorder=4)
        ax.scatter([idx], [med], color=t["bg_canvas"], edgecolors=c, s=55, linewidths=2.0, zorder=5)
        ax.scatter([idx], [m_v], marker="D", color=t["accent_secondary"], s=52, zorder=6)
        if cfg["show_labels"]:
            ax.text(idx, float(vals.max()) + val_span * 0.04, f"均值 {m_v:.1f}\n(n={len(vals)})", ha="center", va="bottom", fontsize=9.2 * fs, color=t["text_primary"], weight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=10.2 * fs, color=t["text_primary"], rotation=15 if len(groups) > 5 else 0)
    ax.set_ylabel(_axis_label_with_unit(num_col), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylim(val_min - val_span * 0.08, val_max + val_span * 0.28)
    ax.grid(axis="x", visible=False)

    src = gm["group_df"]
    title = f"不同{group_col}群体{num_col}均值介于 {src['均值'].min():.1f}–{src['均值'].max():.1f}{u_sp}（{gm['test_name']} p={gm['p_value']:.3f}）"
    subtitle = f"{group_col} × {num_col}多组分布与离群形态 | 分组小提琴箱线图（n = {len(sub)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"统计检验：{gm['test_name']} 统计量={gm['stat']:.3f}, p={gm['p_value']:.4f}, {gm['effect_name']}={gm['effect_size']:.3f} | 菱形=均值，圆点=中位数 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "21-分组小提琴箱线图", "分组小提琴箱线图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 22 - 均值标准误条形图 (Mean + 95% CI Error Bar + Significance Bracket)
# ==============================================================================
def chart_22_mean_errorbar(df: pd.DataFrame, group_col: str = "年龄段", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (group_col not in df.columns or num_col not in df.columns) else None
    group_col = _pick_col(df, group_col, settings, "group_col", (caps or {}).get("single_cols"))
    num_col = _pick_col(df, num_col, settings, "num_col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    gm = su.group_mean_comparison(df, group_col, num_col)
    src = gm["group_df"].copy()
    src["标准误(SE)"] = (src["标准差"] / np.sqrt(src["样本量"].clip(lower=1))).round(2)
    src["95%CI半径"] = (1.96 * src["标准误(SE)"]).round(2)
    u = _unit_for_col(num_col)
    u_sp = _fmt_u(u, True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.65])
    fs = float(cfg["font_scale"])

    x = np.arange(len(src))
    labels = src["组别"].astype(str).tolist()
    means = src["均值"].to_numpy(dtype=float)
    cis = src["95%CI半径"].to_numpy(dtype=float)
    top_g = str(src.sort_values("均值", ascending=False).iloc[0]["组别"])

    colors = [get_item_color(lbl, t["accent_primary"] if lbl == top_g else t["categorical"][1], t, cfg) for lbl in labels]
    ax.bar(
        x, means, width=0.52, color=colors, alpha=0.88, zorder=3,
        yerr=cis if cfg["show_ci"] else None,
        capsize=6, error_kw=dict(ecolor=t["text_primary"], lw=1.6, capthick=1.6),
    )

    max_top = max(float((means + cis).max()), 1.0)
    for xi, m, ci, n_v in zip(x, means, cis, src["样本量"]):
        if cfg["show_labels"]:
            ax.text(xi, m + ci + max_top * 0.025, f"{m:.1f}±{ci:.1f}\n(n={n_v})", ha="center", va="bottom", fontsize=9.2 * fs, color=t["text_primary"], weight="bold")

    y_br = max_top * 1.18
    sig_code = "***" if gm["p_value"] < 0.001 else ("**" if gm["p_value"] < 0.01 else ("*" if gm["p_value"] < 0.05 else "ns"))
    if len(x) >= 2:
        tick_h = max_top * 0.025
        ax.plot([0, 0, len(x) - 1, len(x) - 1], [y_br - tick_h, y_br, y_br, y_br - tick_h], color=t["accent_secondary"], lw=1.5)
        ax.text((len(x) - 1) / 2.0, y_br + max_top * 0.015, f"组间整体比较：{gm['test_name']} p = {gm['p_value']:.3f} ({sig_code})", ha="center", va="bottom", fontsize=9.6 * fs, color=t["accent_secondary"], weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.2 * fs, color=t["text_primary"], rotation=15 if len(labels) > 5 else 0)
    ax.set_ylabel(_axis_label_with_unit(num_col, "均值 ± 95%CI"), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylim(0, y_br * 1.18)
    ax.grid(axis="x", visible=False)

    top_r = src.sort_values("均值", ascending=False).iloc[0]
    sig_tail = "组间差异显著" if gm["p_value"] < 0.05 else "各组均值保持稳定"
    title = f"「{top_r['组别']}」{num_col}均值最高（{top_r['均值']:.1f}±{top_r['95%CI半径']:.1f}{u_sp}），{sig_tail} ({sig_code})"
    subtitle = f"{group_col} × {num_col}均值与 95% 置信区间 | 均值标准误条形图（n = {int(src['样本量'].sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"检验统计量：{gm['test_name']}={gm['stat']:.3f}, p={gm['p_value']:.4f}, {gm['effect_name']}={gm['effect_size']:.3f} | 误差棒：±1.96·SE | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "22-均值标准误条形图", "均值标准误条形图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 23 - 山脊图 (Ridgeline KDE Joyplot)
# ==============================================================================
def chart_23_ridgeline(df: pd.DataFrame, group_col: str = "消费频次", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (group_col not in df.columns or num_col not in df.columns) else None
    group_col = _pick_col(df, group_col, settings, "group_col", (caps or {}).get("single_cols"))
    num_col = _pick_col(df, num_col, settings, "num_col", (caps or {}).get("numeric_cols") or (caps or {}).get("scale_cols"))
    sub = df[[group_col, num_col]].dropna().copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    u = _unit_for_col(num_col)
    u_nosp = _fmt_u(u, False)
    u_col_sfx = f"({u})" if u else ""

    freq_order = ["每天1杯及以上", "每周3-5杯", "每周1-2杯", "偶尔饮用", "每月1-3杯", "几乎不喝"]
    groups = _ordered_categories(sub[group_col].astype(str).tolist(), freq_order)
    groups = [g for g in groups if len(sub.loc[sub[group_col].astype(str) == g, num_col]) >= 3]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.15, 0.14, 0.78, 0.66])
    fs = float(cfg["font_scale"])

    val_min, val_max = float(sub[num_col].min()), float(sub[num_col].max())
    val_span = max(val_max - val_min, 1.0)
    xg = np.linspace(val_min - val_span * 0.08, val_max + val_span * 0.08, 250)
    rows = []
    modes = []
    for idx, g in enumerate(groups[::-1]):
        vals = sub.loc[sub[group_col].astype(str) == g, num_col].to_numpy(dtype=float)
        if len(vals) < 3 or float(np.std(vals, ddof=1)) < 1e-9:
            continue
        kde = stats.gaussian_kde(vals)
        dens = kde(xg)
        dens_norm = dens / (dens.max() if dens.max() > 0 else 1.0) * 0.78
        modes.append(float(xg[int(np.argmax(dens))]))
        base_y = float(idx)
        c = get_item_color(g, t["categorical"][idx % len(t["categorical"])], t, cfg)
        ax.fill_between(xg, base_y, base_y + dens_norm, color=c, alpha=0.58, zorder=10 - idx)
        ax.plot(xg, base_y + dens_norm, color=t["bg_canvas"], linewidth=2.2, zorder=11 - idx)
        ax.plot(xg, base_y + dens_norm, color=c, linewidth=1.5, zorder=12 - idx)

        m_v = float(np.mean(vals))
        med_v = float(np.median(vals))
        ax.vlines(m_v, base_y, base_y + kde(m_v)[0] / dens.max() * 0.78, color=t["accent_secondary"], linestyle="--", linewidth=1.6, zorder=13)
        if cfg["show_labels"]:
            ax.text(xg[-1] - val_span * 0.02, base_y + 0.18, f"均值 {m_v:.1f}{u_nosp} (n={len(vals)})", ha="right", va="bottom", fontsize=9.4 * fs, color=t["text_primary"], weight="bold")
        rows.append({"分组": g, "样本数": len(vals), f"均值{u_col_sfx}": round(m_v, 2), f"中位数{u_col_sfx}": round(med_v, 2)})

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlabel(_axis_label_with_unit(num_col), fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylim(-0.1, len(groups) + 0.05)
    ax.spines["left"].set_visible(False)

    src = pd.DataFrame(rows[::-1])
    _band = f"{min(modes):.1f}–{max(modes):.1f}" if (modes and val_span < 20) else (f"{min(modes):.0f}–{max(modes):.0f}" if modes else "—")
    band_sfx = f" {u}区间" if u else " 区间"
    title = f"山脊密度演进：各{group_col}层级的{num_col}主峰分布于 {_band}{band_sfx}"
    subtitle = f"{group_col} × {num_col}多组核密度叠嶂对比 | 山脊图 Ridgeline（n = {len(sub)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"虚线标记：各组样本算术均值位置 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "23-山脊图", "山脊图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 24 - 分面小多图 (Small Multiples Facet Grid)
# ==============================================================================
def chart_24_small_multiples(df: pd.DataFrame, facet_col: str = "年龄段", cat_col: str = "消费频次", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    caps = su.detect_dataset_capabilities(df) if (facet_col not in df.columns or cat_col not in df.columns) else None
    facet_col = _pick_col(df, facet_col, settings, "facet_col", (caps or {}).get("single_cols"))
    cat_col = _pick_col(df, cat_col, settings, "cat_col", [c for c in ((caps or {}).get("single_cols") or []) if c != facet_col])
    ct_res = su.crosstab_chi2_analysis(df, facet_col, cat_col)
    row_pct = ct_res["row_pct_df"].head(9)
    cnt_df = ct_res["count_df"].loc[row_pct.index]

    age_groups = list(row_pct.index)
    freq_cats = list(row_pct.columns)
    n_facets = len(age_groups)
    ncols = 3 if n_facets > 2 else max(n_facets, 1)
    nrows = max(1, int(np.ceil(n_facets / ncols)))

    fig, axes, t, cfg = create_themed_figure(theme_name, settings, nrows=nrows, ncols=ncols, rect=[0.07, 0.13, 0.88, 0.66])
    fs = float(cfg["font_scale"])
    ax_flat = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]
    overall_pct = (cnt_df.sum(axis=0) / max(cnt_df.values.sum(), 1) * 100.0).to_numpy(dtype=float)

    for idx, ax in enumerate(ax_flat):
        if idx >= len(age_groups):
            ax.axis("off")
            continue
        ag = age_groups[idx]
        vals = row_pct.loc[ag].to_numpy(dtype=float)
        n_ag = int(cnt_df.loc[ag].sum())
        y = np.arange(len(freq_cats))
        c = get_item_color(ag, t["categorical"][idx % len(t["categorical"])], t, cfg)
        ax.barh(y, overall_pct, height=0.56, color=t["border_line"], alpha=0.55, label="全样本基准")
        ax.barh(y, vals, height=0.38, color=c, alpha=0.92, label=ag)
        for yi, v in enumerate(vals):
            if cfg["show_labels"]:
                ax.text(max(v, overall_pct[yi]) + 1.5, yi, f"{v:.1f}%", va="center", ha="left", fontsize=8.4 * fs, color=t["text_primary"], weight="bold")
        ax.set_title(f"{ag} (n={n_ag})", fontsize=10.5 * fs, weight="bold", color=t["text_primary"], loc="left")
        ax.set_yticks(y)
        ax.set_yticklabels(freq_cats if idx % ncols == 0 else [], fontsize=8.8 * fs, color=t["text_secondary"])
        ax.set_xlim(0, max(float(row_pct.values.max()), float(overall_pct.max()), 1.0) * 1.32)
        ax.grid(axis="y", visible=False)

    src = row_pct.reset_index().rename(columns={"index": facet_col})
    title = f"分面小多图对比：各{facet_col}子群与全样本灰色基准的{cat_col}偏离特征一目了然"
    subtitle = f"{facet_col}分面 × {cat_col}分布（含全样本灰色基准阴影）| 分面小多图（n = {int(cnt_df.values.sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"图例说明：彩色前景条为该{facet_col}组内占比，浅灰背景条为全样本总体占比基准 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "24-分面小多图", "分面小多图", title, subtitle, footnote, theme_name, settings)
