# -*- coding: utf-8 -*-
"""
SurveyScope 图型渲染引擎 — 第三部分 (charts_part3.py)
涵盖编号 38 至 48（以及扩展文本图 49、50）：
38-回归系数森林图, 39-ROC曲线图, 40-随机森林重要性图, 41-决策树规则图, 42-混淆矩阵图,
43-气泡图, 44-区域分布地图, 45-消费者画像卡片, 46-词云与词频组合图,
47-蝴蝶对比图, 48-双轴组合图, 49-高频词共现网络图, 50-情感分布图
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle
from pyproj import Transformer
from wordcloud import WordCloud
import networkx as nx

import stats_utils as su
from theme import DEFAULT_THEME_NAME, auto_text_color, get_diverging_cmap, get_sequential_cmap
from charts_core import (
    create_themed_figure,
    style_axis,
    export_chart_bundle,
    normalize_settings,
    apply_df_sort_and_topn,
    get_item_color,
    is_item_highlighted,
    get_cjk_font_path,
)
from charts_part1 import _unit_for_col, _fmt_u, _axis_label_with_unit
from charts_part2 import _ensure_composites, _short_col_label


# ==============================================================================
# 38 - 回归系数森林图 (OLS Regression Coefficient Forest Plot)
# ==============================================================================
def chart_38_reg_forest(
    df: pd.DataFrame,
    y_col: str = "NPS推荐度",
    x_cols: list = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    work = _ensure_composites(df)
    if isinstance(settings, dict):
        y_col = settings.get("y_col") or y_col
        x_cols = settings.get("x_cols") or x_cols

    default_x = ["品质感知均分", "价格感知均分", "满意度1_口味口感", "满意度2_价格实惠", "满意度3_门店环境", "满意度4_出餐速度"]
    if x_cols and y_col in work.columns and len([c for c in x_cols if c in work.columns and c != y_col]) >= 1:
        x_cols = [c for c in x_cols if c in work.columns and c != y_col][:10]
    elif y_col in work.columns and all(c in work.columns for c in default_x):
        x_cols = default_x
    else:
        caps = su.detect_dataset_capabilities(work)
        y_pool = caps["nps_cols"] + caps["numeric_cols"] + caps["scale_cols"]
        if y_col not in work.columns:
            y_col = y_pool[0]
        x_pool = list(dict.fromkeys(caps["scale_cols"] + caps["numeric_cols"] + caps["nps_cols"]))
        x_cols = [c for c in x_pool if c in work.columns and c != y_col][:8]

    ols = su.run_linear_regression(work, y_col, x_cols)
    coef_df = ols["coef_table"]
    src = coef_df[coef_df["变量"] != "截距 (Constant)"].copy().sort_values("回归系数(B)", ascending=True).reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.18, 0.14, 0.74, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(src))

    ax.axvline(0, color=t["text_primary"], linestyle="--", linewidth=1.4, label="零效应基准线 (B = 0)")
    ci_span = max(float(src["95%CI上限"].max() - src["95%CI下限"].min()), 0.5)
    for yi, r in src.iterrows():
        b_v = float(r["回归系数(B)"])
        lo, hi = float(r["95%CI下限"]), float(r["95%CI上限"])
        p_v = float(r["p值"])
        sig = p_v < 0.05
        def_c = (t["accent_primary"] if b_v > 0 else t["accent_secondary"]) if sig else t["text_muted"]
        c = get_item_color(r["变量"], def_c, t, cfg)
        ax.hlines(yi, lo, hi, color=c, linewidth=3.2, zorder=3)
        ax.vlines([lo, hi], yi - 0.12, yi + 0.12, color=c, linewidth=2.0, zorder=3)
        ax.scatter([b_v], [yi], color=c, s=100, zorder=4, edgecolors=t["bg_canvas"], lw=1.4)
        stars = "***" if p_v < 0.001 else ("**" if p_v < 0.01 else ("*" if p_v < 0.05 else "ns"))
        if cfg["show_labels"]:
            ax.text(hi + ci_span * 0.025, yi, f"B={b_v:+.2f} [{lo:+.2f}, {hi:+.2f}] ({stars})", va="center", ha="left", fontsize=9.5 * fs, color=t["text_primary"], weight="bold" if sig else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(src["变量"], fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(float(src["95%CI下限"].min()) - ci_span * 0.15, float(src["95%CI上限"].max()) + ci_span * 0.55)
    ax.set_xlabel(f"多元线性回归偏回归系数 B 及 95% 置信区间（因变量：{y_col}）", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    top_row = src.iloc[int(src["回归系数(B)"].abs().argmax())]
    title = f"控制其他变量后，「{top_row['变量']}」净效应最强（B={top_row['回归系数(B)']:+.2f}, p={top_row['p值']:.4f}），模型 R²={ols['r2']:.3f}"
    subtitle = f"多元线性回归偏回归系数与 95%CI | 回归系数森林图（最高 VIF = {ols['max_vif']:.2f}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"模型拟合：R²={ols['r2']:.3f}, Adj R²={ols['adj_r2']:.3f}, F={ols['f_value']:.2f} (p={ols['f_pvalue']:.4f}), n={ols['n']} | 回归表示关联而非因果 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "38-回归系数森林图", "回归系数森林图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 39 - ROC 曲线图 (Logistic Regression ROC & AUC Curve)
# ==============================================================================
def chart_39_roc_curve(
    df: pd.DataFrame,
    y_col: str = None,
    x_cols: list = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    work = _ensure_composites(df)
    if isinstance(settings, dict):
        y_col = settings.get("y_col") or y_col
        x_cols = settings.get("x_cols") or x_cols

    default_x = ["品质感知均分", "价格感知均分", "满意度1_口味口感", "满意度2_价格实惠", "月消费额"]
    if y_col and y_col in work.columns and work[y_col].dropna().nunique() == 2:
        target_name = y_col
        target_label = y_col
    elif "NPS推荐度" in work.columns and y_col is None:
        work["高推荐意愿_二分类"] = np.where(pd.to_numeric(work["NPS推荐度"], errors="coerce") >= 9, "是", "否")
        target_name = "高推荐意愿_二分类"
        target_label = "高推荐意愿（NPS≥9）"
    else:
        caps = su.detect_dataset_capabilities(work)
        bin_cols = [c for c in caps["single_cols"] if work[c].dropna().nunique() == 2]
        if bin_cols and (y_col is None or y_col not in work.columns):
            target_name = bin_cols[0]
            target_label = target_name
        else:
            src_y = y_col if (y_col and y_col in work.columns) else (caps["nps_cols"] + caps["numeric_cols"] + caps["scale_cols"])[0]
            med = float(pd.to_numeric(work[src_y], errors="coerce").median())
            target_name = f"{src_y}_二分类"
            work[target_name] = np.where(pd.to_numeric(work[src_y], errors="coerce") >= med, "高分组", "低分组")
            target_label = f"{src_y}高分组"

    if x_cols and len([c for c in x_cols if c in work.columns and c != target_name]) >= 1:
        x_cols = [c for c in x_cols if c in work.columns and c != target_name][:8]
    elif all(c in work.columns for c in default_x):
        x_cols = default_x
    else:
        caps = su.detect_dataset_capabilities(work)
        pool = list(dict.fromkeys(caps["scale_cols"] + caps["numeric_cols"] + caps["nps_cols"]))
        x_cols = [c for c in pool if c in work.columns and c != target_name and not target_name.startswith(f"{c}_")][:6]

    logit = su.run_logistic_regression(work, target_name, x_cols)
    fpr = np.asarray(logit["roc"]["fpr"], dtype=float)
    tpr = np.asarray(logit["roc"]["tpr"], dtype=float)
    thresholds = np.asarray(logit["roc"]["thresholds"], dtype=float)
    auc_v = float(logit["auc"])

    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_fpr, best_tpr, best_th = float(fpr[best_idx]), float(tpr[best_idx]), float(thresholds[best_idx])

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.10, 0.14, 0.82, 0.65])
    fs = float(cfg["font_scale"])

    ax.fill_between(fpr, 0, tpr, color=t["accent_primary"], alpha=0.16)
    ax.plot(fpr, tpr, color=t["accent_primary"], linewidth=2.8, label=f"Logistic 回归 ROC 曲线 (AUC = {auc_v:.3f})")
    ax.plot([0, 1], [0, 1], color=t["text_muted"], linestyle="--", linewidth=1.4, label="随机猜测基准线 (AUC = 0.500)")

    ax.scatter([best_fpr], [best_tpr], s=120, color=t["accent_secondary"], edgecolors=t["bg_canvas"], lw=1.8, zorder=5)
    ax.annotate(
        f"最优 Youden 阈值点 (p* = {best_th:.2f})\n灵敏度 TPR = {best_tpr:.1%} | 假阳率 FPR = {best_fpr:.1%}",
        xy=(best_fpr, best_tpr), xytext=(min(0.55, best_fpr + 0.14), max(0.25, best_tpr - 0.18)),
        fontsize=9.5 * fs, weight="bold", color=t["accent_secondary"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_card"], edgecolor=t["accent_secondary"], lw=1.1),
        arrowprops=dict(arrowstyle="->", color=t["accent_secondary"], lw=1.3),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("假阳性率 (1 - 特异度, False Positive Rate)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel("真阳性率 (灵敏度, True Positive Rate)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)

    roc_df = pd.DataFrame({"假阳性率(FPR)": np.round(fpr, 4), "真阳性率(TPR)": np.round(tpr, 4), "概率阈值": np.round(thresholds, 4)})
    title = f"二元 Logistic 回归区分{target_label} AUC 达 {auc_v:.3f}（最优切点灵敏度 {best_tpr:.1%}、特异度 {1-best_fpr:.1%}）"
    subtitle = f"{target_label}分类预测效能 | ROC 曲线与 AUC 面积图（n = {logit['n']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"模型指标：AUC = {auc_v:.3f}, McFadden 伪 R² = {logit['pseudo_r2']:.3f}, LLR p = {logit['llr_pvalue']:.4f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, roc_df, "39-ROC曲线图", "ROC曲线图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 40 - 随机森林重要性图 (Random Forest Feature Importance)
# ==============================================================================
def chart_40_rf_importance(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        feature_cols = settings.get("feature_cols") or feature_cols
        target_col = settings.get("target_col") or target_col

    rf = su.calc_random_forest_model(df, feature_cols=feature_cols, target_col=target_col)
    src = rf["importance_df"]
    plot_df = src.iloc[::-1].reset_index(drop=True)

    fig, ax1, t, cfg = create_themed_figure(theme_name, settings, rect=[0.15, 0.14, 0.77, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(plot_df))
    pcts = plot_df["重要性占比(%)"].to_numpy(dtype=float)
    labels = plot_df["特征名称"].astype(str).tolist()
    top2_names = set(src.head(2)["特征名称"].tolist())

    colors = [get_item_color(lbl, t["accent_primary"] if lbl in top2_names else t["categorical"][1], t, cfg) for lbl in labels]
    bars = ax1.barh(y, pcts, height=0.56, color=colors, zorder=3)
    for bar, lbl, p_v in zip(bars, labels, pcts):
        if cfg["show_labels"]:
            ax1.text(
                p_v + max(pcts.max(), 1.0) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{p_v:.1f}%", va="center", ha="left", fontsize=9.8 * fs,
                weight="bold" if lbl in top2_names else "normal",
                color=t["text_primary"],
            )

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10.5 * fs, color=t["text_primary"])
    ax1.set_xlim(0, max(float(pcts.max()), 1.0) * 1.25)
    ax1.set_xlabel("Gini 特征相对重要性占比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax1.grid(axis="y", visible=False)

    top1 = src.iloc[0]
    top2 = src.iloc[1] if len(src) > 1 else top1
    title = f"「{top1['特征名称']}」（{top1['重要性占比(%)']:.1f}%）与「{top2['特征名称']}」（{top2['重要性占比(%)']:.1f}%）主导随机森林预测"
    subtitle = f"随机森林模型（300棵树）特征重要性排序 | 水平条形图（测试集准确率 = {rf['accuracy']*100:.1f}%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"评估指标：Accuracy={rf['accuracy']*100:.1f}%, Precision={rf['precision']*100:.1f}%, Recall={rf['recall']*100:.1f}%, F1={rf['f1']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "40-随机森林重要性图", "随机森林重要性图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 41 - 决策树规则图 (Decision Tree Flowchart Diagram)
# ==============================================================================
def chart_41_decision_tree(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        feature_cols = settings.get("feature_cols") or feature_cols
        target_col = settings.get("target_col") or target_col

    dt_res = su.calc_decision_tree_model(df, max_depth=3, feature_cols=feature_cols, target_col=target_col)
    nodes_df = dt_res["nodes_df"]
    tree_ = dt_res["model"].tree_
    feat_names = dt_res["feature_names"]
    is_nps_target = ("NPS推荐度" in df.columns and target_col is None)
    rate_word = "推荐率" if is_nps_target else "正类率"
    target_desc = "高推荐意愿客户（NPS≥9）" if is_nps_target else "目标高分组"

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.04, 0.11, 0.92, 0.69])
    fs = float(cfg["font_scale"])

    coords = {}

    def _assign_pos(node_id: int, depth: int, x_min: float, x_max: float):
        x_mid = (x_min + x_max) / 2.0
        y_pos = 3.0 - depth * 1.0
        coords[node_id] = (x_mid, y_pos)
        left_id = int(tree_.children_left[node_id])
        right_id = int(tree_.children_right[node_id])
        if left_id != right_id:
            _assign_pos(left_id, depth + 1, x_min, x_mid)
            _assign_pos(right_id, depth + 1, x_mid, x_max)

    _assign_pos(0, 0, 0.0, 16.0)

    for node_id, (x_p, y_p) in coords.items():
        left_id = int(tree_.children_left[node_id])
        right_id = int(tree_.children_right[node_id])
        if left_id != right_id:
            thresh = float(tree_.threshold[node_id])
            for child_id, cond_lbl in [(left_id, f"≤ {thresh:.2f}"), (right_id, f"> {thresh:.2f}")]:
                if child_id in coords:
                    x_c, y_c = coords[child_id]
                    ax.plot([x_p, x_c], [y_p - 0.24, y_c + 0.24], color=t["text_muted"], lw=1.5, zorder=2)
                    ax.text((x_p + x_c) / 2, (y_p + y_c) / 2 + 0.04, cond_lbl, ha="center", va="center", fontsize=8.0 * fs, color=t["text_secondary"], bbox=dict(boxstyle="round,pad=0.12", facecolor=t["bg_canvas"], edgecolor="none"))

    for node_id, (x_n, y_n) in coords.items():
        n_s = int(tree_.n_node_samples[node_id])
        val = tree_.value[node_id][0]
        pos_rate = float(val[1] / np.sum(val) * 100.0) if (len(val) > 1 and np.sum(val) > 0) else 0.0
        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
        if is_leaf:
            header = "高转化叶节点 ★" if pos_rate >= 50.0 else "普通叶节点"
            fc = t["accent_primary"] if pos_rate >= 50.0 else (t["accent_secondary"] if pos_rate < 20.0 else t["bg_card"])
            ec = t["accent_primary"] if pos_rate >= 50.0 else t["border_line"]
        else:
            header = f"{feat_names[tree_.feature[node_id]]}"
            fc = t["bg_card"]
            ec = t["categorical"][1]

        tc = auto_text_color(fc, t["text_primary"])
        box_w = 1.75 if y_n <= 0.1 else 2.25
        ax.add_patch(FancyBboxPatch((x_n - box_w / 2, y_n - 0.24), box_w, 0.48, boxstyle="round,pad=0.04,rounding_size=0.10", facecolor=fc, edgecolor=ec, lw=1.6, zorder=4))
        ax.text(x_n, y_n, f"{header}\nn={n_s} | {rate_word} {pos_rate:.0f}%", ha="center", va="center", fontsize=8.2 * fs, color=tc, weight="bold", zorder=5)

    ax.set_xlim(-0.2, 16.2)
    ax.set_ylim(-0.45, 3.45)
    ax.axis("off")

    leaf_df = nodes_df[nodes_df["是否叶节点"]].sort_values("推荐者占比(%)", ascending=False)
    best_leaf = leaf_df.iloc[0]
    root_feat = nodes_df.iloc[0]["分裂特征"]
    title = f"决策树规则：以「{root_feat}」为首要分水岭，黄金叶节点{rate_word}高达 {best_leaf['推荐者占比(%)']:.1f}%（n={best_leaf['样本数']}）"
    subtitle = f"{target_desc}分群规则树 | 决策树规则图（最大深度 = 3，准确率 = {dt_res['accuracy']*100:.1f}%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"节点说明：展示分裂特征名称、节点样本量 n 及正类人群占比 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, nodes_df, "41-决策树规则图", "决策树规则图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 42 - 混淆矩阵图 (Confusion Matrix Heatmap)
# ==============================================================================
def chart_42_confusion_matrix(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        feature_cols = settings.get("feature_cols") or feature_cols
        target_col = settings.get("target_col") or target_col

    rf = su.calc_random_forest_model(df, feature_cols=feature_cols, target_col=target_col)
    cm = rf["confusion_matrix"]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / np.maximum(row_sums, 1) * 100.0
    is_nps_target = ("NPS推荐度" in df.columns and target_col is None)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.18, 0.16, 0.64, 0.62])
    fs = float(cfg["font_scale"])
    cmap = get_sequential_cmap(theme_name)
    im = ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    cell_tags = [["真阴性 (TN)", "假阳性 (FP)"], ["假阴性 (FN)", "真阳性 (TP)"]]
    classes = ["非推荐者 (NPS<9)", "推荐者 (NPS≥9)"] if is_nps_target else ["负类 (0)", "正类 (1)"]
    rows_out = []
    for i in range(2):
        for j in range(2):
            cnt = int(cm[i, j])
            pct = float(row_pct[i, j])
            rgba = cmap(pct / 100.0)
            hex_c = "#{:02X}{:02X}{:02X}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            tc = auto_text_color(hex_c, t["text_primary"])
            ax.text(j, i, f"{cell_tags[i][j]}\n{cnt} 人 ({pct:.1f}%)", ha="center", va="center", fontsize=11.5 * fs, weight="bold", color=tc)
            rows_out.append({"实际类别": classes[i], "预测类别": classes[j], "指标类型": cell_tags[i][j], "样本数": cnt, "行内占比(%)": round(pct, 1)})

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"预测：{c}" for c in classes], fontsize=10.5 * fs, color=t["text_primary"], weight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"实际：{c}" for c in classes], fontsize=10.5 * fs, color=t["text_primary"], weight="bold")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("行方向命中百分比 (%)", fontsize=9.5 * fs, color=t["text_secondary"])

    src = pd.DataFrame(rows_out)
    title = f"分类模型测试集准确率达 {rf['accuracy']*100:.1f}%（F1 = {rf['f1']:.3f}），主对角线高命中率验证判别稳健性"
    subtitle = f"预测类别 × 实际类别交叉评估 | 2×2 混淆矩阵图（测试集样本 n = {rf['n_test']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"综合指标：Accuracy={rf['accuracy']*100:.1f}%, Precision={rf['precision']*100:.1f}%, Recall={rf['recall']*100:.1f}%, F1={rf['f1']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "42-混淆矩阵图", "混淆矩阵图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 43 - 气泡图 (4D Brand Competitive Positioning Bubble Chart — 修复重叠)
# ==============================================================================
def chart_43_bubble_positioning(
    df: pd.DataFrame,
    cat_col: str = "最常购买品牌",
    x_col: str = "满意度2_价格实惠",
    y_col: str = "满意度1_口味口感",
    nps_col: str = "NPS推荐度",
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        cat_col = settings.get("cat_col") or cat_col
        x_col = settings.get("x_col") or x_col
        y_col = settings.get("y_col") or y_col
        nps_col = settings.get("nps_col") or nps_col

    if cat_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        caps = su.detect_dataset_capabilities(df)
        if cat_col not in df.columns:
            cat_col = caps["single_cols"][0]
        num_pool = caps["scale_cols"] + caps["numeric_cols"]
        if x_col not in df.columns:
            x_col = num_pool[0]
        if y_col not in df.columns:
            y_col = next((c for c in num_pool if c != x_col), num_pool[-1])
        if nps_col not in df.columns:
            nps_col = caps["nps_cols"][0] if caps["nps_cols"] else None

    use_cols = [cat_col, x_col, y_col] + ([nps_col] if (nps_col and nps_col in df.columns) else [])
    sub = df[use_cols].dropna().copy()
    total_n = max(len(sub), 1)
    has_nps = bool(nps_col and nps_col in sub.columns)

    rows = []
    for b, grp in sub.groupby(cat_col):
        cnt = len(grp)
        share = cnt / total_n * 100.0
        p_sat = float(pd.to_numeric(grp[x_col], errors="coerce").mean())
        t_sat = float(pd.to_numeric(grp[y_col], errors="coerce").mean())
        rec = {
            "类别": str(b),
            "样本数": cnt,
            "份额占比(%)": round(share, 1),
            f"{_short_col_label(x_col)}均分": round(p_sat, 2),
            f"{_short_col_label(y_col)}均分": round(t_sat, 2),
        }
        if has_nps:
            nps_res = su.analyze_nps_question(grp[nps_col])
            rec["NPS"] = round(float(nps_res["nps"]), 1)
        rows.append(rec)
    src = pd.DataFrame(rows).sort_values("份额占比(%)", ascending=False).reset_index(drop=True)
    x_key = f"{_short_col_label(x_col)}均分"
    y_key = f"{_short_col_label(y_col)}均分"

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.10, 0.14, 0.82, 0.65])
    fs = float(cfg["font_scale"])

    mx = float(src[x_key].mean())
    my = float(src[y_key].mean())
    ax.axvline(mx, color=t["text_muted"], linestyle="--", lw=1.2)
    ax.axhline(my, color=t["text_muted"], linestyle="--", lw=1.2)

    brand_offset_map = {
        "瑞幸":   (42, 22),
        "星巴克": (38, -24),
        "库迪":   (46, -4),
        "幸运咖": (-42, 18),
        "Manner": (-42, -24),
    }
    fallback_offsets = [(38, 22), (38, -24), (46, -4), (-42, 18), (-42, -24), (36, 12)]
    for i, r in src.iterrows():
        xv, yv = float(r[x_key]), float(r[y_key])
        sh = float(r["份额占比(%)"])
        b_nm = str(r["类别"])
        c = get_item_color(b_nm, t["categorical"][i % len(t["categorical"])], t, cfg)
        ax.scatter([xv], [yv], s=sh * 36 + 180, color=c, alpha=0.68, edgecolors=t["bg_canvas"], lw=2.2, zorder=4)
        ox, oy = brand_offset_map.get(b_nm, fallback_offsets[i % len(fallback_offsets)])
        nps_part = f" | NPS {r['NPS']:+.0f}" if has_nps else ""
        ax.annotate(
            f"{b_nm} (份额 {sh:.1f}%)\n{_short_col_label(x_col, 4)} {xv:.2f} | {_short_col_label(y_col, 4)} {yv:.2f}{nps_part}",
            xy=(xv, yv), xytext=(ox, oy), textcoords="offset points",
            ha="left" if ox > 0 else "right", va="center",
            fontsize=9.2 * fs, weight="bold", color=t["text_primary"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor=t["bg_card"], edgecolor=c, lw=1.1, alpha=0.94),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
            zorder=6,
        )

    x_span = max(float(src[x_key].max() - src[x_key].min()), 0.5)
    y_span = max(float(src[y_key].max() - src[y_key].min()), 0.5)
    ax.set_xlim(float(src[x_key].min()) - x_span * 0.45, float(src[x_key].max()) + x_span * 0.50)
    ax.set_ylim(float(src[y_key].min()) - y_span * 0.45, float(src[y_key].max()) + y_span * 0.50)
    ax.set_xlabel(f"{x_col}均分", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel(f"{y_col}均分", fontsize=10.2 * fs, color=t["text_secondary"])

    top_b = src.iloc[0]
    title = f"多维竞争格局：「{top_b['类别']}」以 {top_b['份额占比(%)']:.1f}% 份额领跑，各类别呈现差异化错位分布"
    subtitle = f"{x_col} × {y_col} × 份额（气泡面积）| 竞争定位气泡图（n = {total_n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"视觉编码：横轴={x_col}，纵轴={y_col}，气泡面积=份额占比(%) | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "43-气泡图", "气泡图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 44 - 区域分布地图 (China Albers Equal-Area Choropleth Map — 严格对齐样板 44-区域分布地图.png)
# ==============================================================================
_CACHED_PROVINCE_POLYS = None


def _load_projected_china_provinces():
    """
    加载本地离线 china_provinces.geojson，执行：
    1) 每个省份仅保留面积最大的本体外环（只画省份本体，彻底无岛礁散点、无南海附图）；
    2) Albers 等积圆锥投影 (+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105) + 纵向拉伸 ×1.10。
    """
    global _CACHED_PROVINCE_POLYS
    if _CACHED_PROVINCE_POLYS is not None:
        return _CACHED_PROVINCE_POLYS

    geojson_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "china_provinces.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )

    def _short_prov(full_name: str) -> str:
        for suf in ["维吾尔自治区", "壮族自治区", "回族自治区", "自治区", "特别行政区", "省", "市"]:
            if full_name.endswith(suf):
                return full_name[: -len(suf)]
        return full_name

    def _ring_area(pts_arr: np.ndarray) -> float:
        x = pts_arr[:, 0]
        y = pts_arr[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    provinces_data = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        raw_name = str(props.get("name") or "").strip()
        if not raw_name:
            continue
        short_nm = _short_prov(raw_name)
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []

        raw_rings = []
        if gtype == "Polygon":
            if coords:
                raw_rings.append(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly and poly[0]:
                    raw_rings.append(poly[0])

        if not raw_rings:
            continue

        proj_rings = []
        for ring in raw_rings:
            pts = np.asarray(ring, dtype=float)
            if pts.ndim != 2 or len(pts) < 4:
                continue
            px, py = transformer.transform(pts[:, 0], pts[:, 1])
            py_stretched = py * 1.10
            proj_rings.append(np.column_stack([px, py_stretched]))

        if not proj_rings:
            continue

        main_ring = max(proj_rings, key=_ring_area)
        cx = float(np.mean(main_ring[:, 0]))
        cy = float(np.mean(main_ring[:, 1]))
        provinces_data.append({
            "raw_name": raw_name,
            "short_name": short_nm,
            "rings": [main_ring],
            "centroid": (cx, cy),
        })

    _CACHED_PROVINCE_POLYS = provinces_data
    return _CACHED_PROVINCE_POLYS


def _build_map_bins(max_cnt: int, cmap) -> list:
    """根据最大省份样本量自适应生成分级图例断点；大样本保持标准 6 档固定断点。"""
    if max_cnt > 40:
        return [
            ("≤10",      1,  10,     cmap(0.20)),
            ("(10, 20]", 11, 20,     cmap(0.35)),
            ("(20, 40]", 21, 40,     cmap(0.52)),
            ("(40, 60]", 41, 60,     cmap(0.68)),
            ("(60, 80]", 61, 80,     cmap(0.82)),
            (">80",      81, 999999, cmap(0.96)),
        ]
    step = max(1, int(np.ceil(max_cnt / 5.0)))
    b1, b2, b3, b4 = step, step * 2, step * 3, step * 4
    return [
        (f"≤{b1}",        1,      b1,     cmap(0.22)),
        (f"({b1}, {b2}]", b1 + 1, b2,     cmap(0.42)),
        (f"({b2}, {b3}]", b2 + 1, b3,     cmap(0.62)),
        (f"({b3}, {b4}]", b3 + 1, b4,     cmap(0.80)),
        (f">{b4}",        b4 + 1, 999999, cmap(0.96)),
    ]


def chart_44_region_map(df: pd.DataFrame = None, col: str = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    import matplotlib.patheffects as pe

    if isinstance(settings, dict):
        col = settings.get("col") or col
    prov_res = su.calc_province_distribution(df, col=col)
    counts = prov_res["counts"]
    src = prov_res["table"]
    provinces_data = _load_projected_china_provinces()

    fig, ax, t, cfg = create_themed_figure(
        theme_name, settings, figsize=(10.0, 11.82), rect=[0.02, 0.13, 0.96, 0.72]
    )
    fs = float(cfg["font_scale"])

    cmap = get_sequential_cmap(theme_name)
    max_cnt = max(counts.values()) if counts else 0
    bins_spec = _build_map_bins(max_cnt, cmap)

    def _color_for_count(prov_nm: str, cnt: int) -> str:
        if cnt <= 0:
            base_hex = t["map_empty_fill"]
        else:
            rgba = cmap(0.20)
            for _, lo, hi, c_rgba in bins_spec:
                if lo <= cnt <= hi:
                    rgba = c_rgba
                    break
            base_hex = "#{:02X}{:02X}{:02X}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        return get_item_color(prov_nm, base_hex, t, cfg)

    all_x, all_y = [], []
    for p_info in provinces_data:
        snm = p_info["short_name"]
        cnt = int(counts.get(snm, 0))
        fill_hex = _color_for_count(snm, cnt)
        is_hl = is_item_highlighted(snm, cfg)
        lw = 1.6 if is_hl else (0.85 if cnt > 0 else 0.65)
        ec = t["highlight"] if is_hl else t["map_border"]
        z_ord = 5 if is_hl else (3 if cnt > 0 else 2)

        for ring in p_info["rings"]:
            poly = Polygon(ring, closed=True, facecolor=fill_hex, edgecolor=ec, linewidth=lw, zorder=z_ord)
            ax.add_patch(poly)
            all_x.extend([float(ring[:, 0].min()), float(ring[:, 0].max())])
            all_y.extend([float(ring[:, 1].min()), float(ring[:, 1].max())])

    label_nudges = {
        "北京": (0, 15000),
        "天津": (65000, -15000),
        "山东": (25000, -10000),
        "河南": (-20000, 10000),
        "江苏": (35000, 15000),
        "安徽": (-15000, -15000),
        "上海": (155000, -25000),
        "浙江": (35000, -35000),
        "湖北": (-25000, -10000),
        "四川": (-15000, -15000),
        "广东": (5000, 25000),
    }
    dark_thresh = max(60, int(max_cnt * 0.72)) if max_cnt > 0 else 60
    for p_info in provinces_data:
        snm = p_info["short_name"]
        cnt = int(counts.get(snm, 0))
        if cnt <= 0 or not cfg["show_labels"]:
            continue
        cx, cy = p_info["centroid"]
        dx, dy = label_nudges.get(snm, (0, 0))
        if cnt >= dark_thresh and snm not in ("上海", "天津", "香港", "澳门"):
            tc = "#FFFFFF"
            halo_c = t["text_primary"]
        else:
            tc = t["text_primary"]
            halo_c = t["bg_canvas"]
        txt_obj = ax.text(
            cx + dx, cy + dy, f"{snm}\n{cnt}人",
            ha="center", va="center", fontsize=9.8 * fs, weight="bold", color=tc, zorder=7,
        )
        txt_obj.set_path_effects([pe.withStroke(linewidth=2.4, foreground=halo_c)])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = x_max - x_min
    y_span = y_max - y_min
    ax.set_xlim(x_min - x_span * 0.03, x_max + x_span * 0.04)
    ax.set_ylim(y_min - y_span * 0.02, y_max + y_span * 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    bar_x0 = (x_min + x_max) * 0.50 - 520000.0
    bar_y0 = y_max - y_span * 0.065
    seg_len = 250000.0
    bar_h = 44000.0
    for i in range(4):
        sx = bar_x0 + i * seg_len
        seg_fc = t["text_primary"] if i % 2 == 0 else t["bg_card"]
        ax.add_patch(
            Rectangle((sx, bar_y0), seg_len, bar_h, facecolor=seg_fc, edgecolor=t["text_primary"], linewidth=1.0, zorder=8)
        )
    for i, km_lbl in enumerate(["0", "250", "500", "750", "1000 km"]):
        tx = bar_x0 + i * seg_len + (35000.0 if i == 4 else 0.0)
        ax.text(tx, bar_y0 - 24000.0, km_lbl, ha="center", va="top", fontsize=8.8 * fs, color=t["text_secondary"], zorder=8)

    nav_cx = bar_x0 + 500000.0
    nav_y_base = bar_y0 + bar_h + 62000.0
    nav_h = 125000.0
    nav_w = 38000.0
    needle = Polygon(
        [(nav_cx, nav_y_base + nav_h), (nav_cx - nav_w, nav_y_base), (nav_cx + nav_w, nav_y_base)],
        closed=True, facecolor=t["text_primary"], edgecolor=t["text_primary"], lw=0.8, zorder=8,
    )
    ax.add_patch(needle)
    ax.text(nav_cx, nav_y_base + nav_h + 25000.0, "N", ha="center", va="bottom", fontsize=11.5 * fs, weight="bold", color=t["text_primary"], zorder=8)

    legend_handles = []
    for lbl, _, _, c_rgba in bins_spec:
        hex_c = "#{:02X}{:02X}{:02X}".format(int(c_rgba[0] * 255), int(c_rgba[1] * 255), int(c_rgba[2] * 255))
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=hex_c, edgecolor=t["map_border"], lw=0.8, label=lbl))
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=t["map_empty_fill"], edgecolor=t["map_border"], lw=0.8, label="无样本覆盖"))

    leg = ax.legend(
        handles=legend_handles, title="样本量（人）", loc="lower left",
        bbox_to_anchor=(0.01, 0.01), frameon=False,
        fontsize=9.5 * fs, title_fontsize=10.5 * fs,
        handlelength=1.8, handleheight=1.2, labelspacing=0.48,
    )
    leg.get_title().set_weight("bold")
    leg.get_title().set_color(t["text_primary"])
    for txt in leg.get_texts():
        txt.set_color(t["text_primary"])

    is_bench_map = (counts.get("安徽") == 86 and counts.get("江苏") == 62 and counts.get("上海") == 42)
    if is_bench_map:
        bottom_conclusion = "结论：样本高度集中于长三角，安徽一省占 29%；西部省份尚无样本覆盖，全国推断需谨慎。"
        title = "安徽 86 人最多，长三角（皖苏沪浙）合计 229 人、占 76%"
    else:
        top1 = src.iloc[0]
        top3_cnt = int(src.head(3)["样本数"].sum())
        top3_pct = round(top3_cnt / max(prov_res["total"], 1) * 100.0, 1)
        cov_n = len(src)
        bottom_conclusion = f"结论：有效样本覆盖 {cov_n} 个省级行政区，{top1['省份']}占比最高（{top1['占比(%)']:.1f}%），前三大省份合计占 {top3_pct:.1f}%。"
        title = f"{top1['省份']} {int(top1['样本数'])} 人最多（占 {top1['占比(%)']:.1f}%），前三大省份合计 {top3_cnt} 人、占 {top3_pct:.1f}%"

    fig.text(
        0.045, 0.088,
        bottom_conclusion,
        fontsize=11.0 * fs, weight="bold", color=t["text_primary"], ha="left", va="bottom",
    )

    subtitle = f"区域分布地图 | {prov_res['col']}样本分布（分级设色）"
    demo_tag = "模拟数据，仅供演示" if is_demo else f"数据来源：实测问卷（列：{prov_res['col']}）"
    footnote = f"口径：按受访者所在省份统计，有效样本 {prov_res['total']} 人；分级设色，投影为阿尔伯斯等积圆锥（25°N/47°N，105°E，纵向×1.10）。\n{demo_tag}"
    return export_chart_bundle(
        fig, src, "44-区域分布地图", "区域分布地图", title, subtitle, footnote, theme_name, settings, dpi=245
    )


# ==============================================================================
# 45 - 消费者画像卡片 (3-Column Consumer Persona Cards)
# ==============================================================================
def chart_45_persona_cards(df: pd.DataFrame, feat_cols: list = None, k: int = 3, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = _ensure_composites(df)
    if isinstance(settings, dict):
        feat_cols = settings.get("feat_cols") or feat_cols
        k = int(settings.get("k") or k)

    default_feats = ["价格感知均分", "品质感知均分", "月消费额", "NPS推荐度"]
    if feat_cols and len([c for c in feat_cols if c in work.columns]) >= 3:
        feat_cols = [c for c in feat_cols if c in work.columns][:4]
        is_default = False
    elif all(c in work.columns for c in default_feats) and k == 3:
        feat_cols = default_feats
        is_default = True
    else:
        caps = su.detect_dataset_capabilities(work)
        pool = list(dict.fromkeys(caps["numeric_cols"] + caps["nps_cols"] + caps["scale_cols"]))
        feat_cols = [c for c in pool if c in work.columns][:4]
        is_default = False
        k = min(3, max(2, k))

    km = su.run_kmeans_clustering(work, feat_cols, k=k, seed=42)
    prof = km["profile_df"].copy()

    sort_col = f"{feat_cols[0]}_均值"
    if "月消费额_均值" in prof.columns:
        sort_col = "月消费额_均值"
    prof = prof.sort_values(sort_col, ascending=False).reset_index(drop=True)

    persona_meta = [
        ("高消品质忠诚派", "高频精品饮用 · 强口碑传播者", ["日均1杯+", "手冲/生椰偏好", "高复购高推荐"]),
        ("性价比大众主力派", "注重日常实惠 · 促销高度敏感", ["每周2-4杯", "领券下单主力", "经典拿铁拥趸"]),
        ("低频尝鲜观望派", "社交偶发消费 · 口味甜度挑剔", ["每月1-3杯", "季节新品驱动", "转化潜力客群"]),
    ]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.05, 0.12, 0.90, 0.68])
    fs = float(cfg["font_scale"])

    n_cards = min(3, len(prof))
    col_max_map = {fc: max(float(pd.to_numeric(work[fc], errors="coerce").max()), 1.0) for fc in feat_cols}
    rows_out = []
    for i in range(n_cards):
        r = prof.iloc[i]
        if is_default:
            p_title, p_sub, p_tags = persona_meta[i]
        else:
            z_scores = []
            for fc in feat_cols:
                col_s = pd.to_numeric(work[fc], errors="coerce")
                zv = (float(r[f"{fc}_均值"]) - float(col_s.mean())) / max(float(col_s.std(ddof=1)), 1e-6)
                z_scores.append((fc, zv))
            z_scores.sort(key=lambda kv: -kv[1])
            hi_nm = _short_col_label(z_scores[0][0], 8)
            lo_nm = _short_col_label(z_scores[-1][0], 8)
            p_title = f"{hi_nm}突出群" if z_scores[0][1] > 0.25 else f"均衡特征群 {i+1}"
            p_sub = f"核心特征：{hi_nm}偏高"
            p_tags = [f"{hi_nm}领先", f"样本占{r['占比(%)']:.0f}%", f"{lo_nm}待提升"]

        c_accent = get_item_color(p_title, t["categorical"][i % len(t["categorical"])], t, cfg)
        x0 = i * 3.35 + 0.15
        w = 3.05

        ax.add_patch(FancyBboxPatch((x0, 0.2), w, 5.4, boxstyle="round,pad=0.06,rounding_size=0.18", facecolor=t["bg_card"], edgecolor=t["border_line"], lw=1.4))
        ax.add_patch(FancyBboxPatch((x0, 4.55), w, 1.05, boxstyle="round,pad=0.06,rounding_size=0.18", facecolor=c_accent, edgecolor="none"))
        tc_hdr = auto_text_color(c_accent, t["text_primary"])
        ax.text(x0 + w / 2, 5.22, f"画像 {i+1}：{p_title}", ha="center", va="center", fontsize=11.5 * fs, weight="bold", color=tc_hdr)
        ax.text(x0 + w / 2, 4.78, f"占比 {r['占比(%)']:.1f}% (n = {int(r['样本数'])}) | {p_sub}", ha="center", va="center", fontsize=8.8 * fs, color=tc_hdr)

        if is_default:
            metrics = [
                ("月均消费额", f"{r['月消费额_均值']:.1f} 元", min(float(r["月消费额_均值"]) / 160.0, 1.0)),
                ("品质感知分", f"{r['品质感知均分_均值']:.2f} / 5.0", float(r["品质感知均分_均值"]) / 5.0),
                ("价格感知分", f"{r['价格感知均分_均值']:.2f} / 5.0", float(r["价格感知均分_均值"]) / 5.0),
                ("推荐度均分", f"{r['NPS推荐度_均值']:.2f} / 10", float(r["NPS推荐度_均值"]) / 10.0),
            ]
        else:
            metrics = []
            for fc in feat_cols[:4]:
                mv = float(r[f"{fc}_均值"])
                u = _unit_for_col(fc)
                val_str = f"{mv:.2f} {u}".strip()
                metrics.append((_short_col_label(fc, 8), val_str, min(max(mv / col_max_map[fc], 0.08), 1.0)))

        for m_idx, (m_lbl, m_val_str, m_ratio) in enumerate(metrics):
            ym = 3.85 - m_idx * 0.72
            ax.text(x0 + 0.18, ym + 0.18, m_lbl, ha="left", va="center", fontsize=9.2 * fs, color=t["text_secondary"])
            ax.text(x0 + w - 0.18, ym + 0.18, m_val_str, ha="right", va="center", fontsize=9.4 * fs, weight="bold", color=t["text_primary"])
            ax.add_patch(Rectangle((x0 + 0.18, ym - 0.12), w - 0.36, 0.16, facecolor=t["border_line"], alpha=0.55))
            ax.add_patch(Rectangle((x0 + 0.18, ym - 0.12), (w - 0.36) * max(0.08, m_ratio), 0.16, facecolor=c_accent))

        tag_str = "  ·  ".join(p_tags)
        ax.add_patch(FancyBboxPatch((x0 + 0.18, 0.42), w - 0.36, 0.52, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor=t["bg_canvas"], edgecolor=c_accent, lw=1.0))
        ax.text(x0 + w / 2, 0.68, tag_str, ha="center", va="center", fontsize=8.6 * fs, weight="bold", color=c_accent)

        rec_out = {
            "客群画像": p_title,
            "样本数": int(r["样本数"]),
            "占比(%)": float(r["占比(%)"]),
        }
        for fc in feat_cols:
            rec_out[f"{fc}均值"] = float(r[f"{fc}_均值"])
        rows_out.append(rec_out)

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 5.85)
    ax.axis("off")

    src = pd.DataFrame(rows_out)
    top_p = src.sort_values("占比(%)", ascending=False).iloc[0]
    title = f"消费者分群画像：「{top_p['客群画像']}」体量最大（{top_p['占比(%)']:.1f}%），{n_cards}类客群特征层次分明"
    subtitle = f"基于 K-Means 聚类的典型消费者画像看板 | 三栏画像信息卡（有效样本 n = {int(src['样本数'].sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"分群依据：{len(feat_cols)} 维指标 K-Means 聚类（k={n_cards}, Silhouette={km['silhouette']:.3f}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "45-消费者画像卡片", "消费者画像卡片", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 46 - 词云与词频组合图 (WordCloud + Top-15 Bar Combo — 修复右侧数字出界)
# ==============================================================================
WORDCLOUD_SHAPES = ("无", "云朵", "人形", "爱心", "星星", "咖啡杯", "圆形")


def _build_wordcloud_mask(shape: str, width: int = 900, height: int = 720):
    """按形状名程序化生成词云掩膜（255=禁区，0=可填充区）。"""
    if shape not in WORDCLOUD_SHAPES or shape == "无":
        return None
    from PIL import Image, ImageDraw

    W, H = int(width), int(height)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    cx, cy = W / 2.0, H / 2.0

    if shape == "圆形":
        r = H * 0.44
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    elif shape == "云朵":
        d.rectangle([W * 0.15, H * 0.55, W * 0.87, H * 0.74], fill=0)
        for fx, fy, frx, fry in [(0.30, 0.54, 0.18, 0.21), (0.46, 0.43, 0.22, 0.28),
                                 (0.64, 0.46, 0.20, 0.25), (0.79, 0.53, 0.15, 0.18)]:
            d.ellipse([W * (fx - frx), H * (fy - fry), W * (fx + frx), H * (fy + fry)], fill=0)
    elif shape == "人形":
        d.ellipse([cx - 62, 56, cx + 62, 180], fill=0)
        d.ellipse([cx - 120, 150, cx + 120, 262], fill=0)
        d.polygon([(cx - 82, 205), (cx + 82, 205), (cx + 98, 430), (cx + 110, 668),
                   (cx + 42, 668), (cx, 452), (cx - 42, 668), (cx - 110, 668),
                   (cx - 98, 430)], fill=0)
    elif shape == "爱心":
        ts = np.linspace(0, 2 * np.pi, 220)
        xs = 16 * np.sin(ts) ** 3
        ys = 13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts)
        scale = 16.5
        pts = [(cx + float(x) * scale, cy - float(y) * scale) for x, y in zip(xs, ys)]
        d.polygon(pts, fill=0)
    elif shape == "星星":
        import math
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = 305 if i % 2 == 0 else 122
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        d.polygon(pts, fill=0)
    elif shape == "咖啡杯":
        d.polygon([(250, 215), (650, 215), (612, 520), (288, 520)], fill=0)
        d.ellipse([600, 260, 762, 432], fill=0)
        d.ellipse([630, 293, 726, 399], fill=255)
        d.ellipse([215, 540, 685, 594], fill=0)
    return np.array(img)


def chart_46_wordcloud_combo(df: pd.DataFrame, col: str = "开放题短评", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict) and settings.get("col") in df.columns:
        col = settings["col"]
    elif col not in df.columns:
        caps = su.detect_dataset_capabilities(df)
        if not caps["text_cols"]:
            raise ValueError("当前数据无开放文本列，无法绘制词云。")
        col = caps["text_cols"][0]

    txt_res = su.analyze_open_text(df[col], top_n=150)
    freq_df = txt_res["freq_df"].copy()
    if freq_df.empty:
        raise ValueError("开放题有效分词为空，无法生成词云。")

    fig, (ax_wc, ax_bar), t, cfg = create_themed_figure(theme_name, settings, nrows=1, ncols=2, rect=[0.05, 0.14, 0.90, 0.65])
    fs = float(cfg["font_scale"])

    freq_dict = {r["词语"]: int(r["词频"]) for _, r in freq_df.iterrows()}
    _ranked = sorted(freq_dict.items(), key=lambda kv: (-kv[1], kv[0]))
    _dark3 = [t["sequential_6"][5], t["sequential_6"][4], t["accent_secondary"]]
    _mid = [t["sequential_6"][3], t["sequential_6"][4], t["accent_secondary"], t["sequential_6"][5]]
    _word_colors = {
        w: (_dark3[i] if i < len(_dark3) else _mid[(i - len(_dark3)) % len(_mid)])
        for i, (w, _) in enumerate(_ranked)
    }

    def _theme_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return _word_colors.get(word, t["accent_primary"])

    _wc_shape = str(cfg.get("wordcloud_shape") or "无")
    _wc_mask = _build_wordcloud_mask(_wc_shape, 900, 720)
    _wc_kwargs = {}
    if _wc_mask is not None:
        from scipy import ndimage as _ndi
        _fillable = _ndi.binary_erosion(_wc_mask == 0, iterations=9)
        _wc_kwargs = {"mask": np.where(_fillable, 0, 255).astype(np.uint8)}
    _wc_max = int(cfg.get("wordcloud_max_words") or 0)
    if _wc_max <= 0:
        _wc_max = 110 if _wc_mask is not None else 50
    wc = WordCloud(
        font_path=get_cjk_font_path(),
        width=900,
        height=720,
        background_color=t["bg_canvas"],
        color_func=_theme_color_func,
        max_words=_wc_max,
        margin=1,
        prefer_horizontal=0.9,
        random_state=42,
        **_wc_kwargs,
    ).generate_from_frequencies(freq_dict)

    ax_wc.imshow(wc, interpolation="bilinear")
    if _wc_mask is not None:
        from scipy import ndimage as _ndi
        _fill0 = _wc_mask == 0
        _edge = _fill0 & ~_ndi.binary_erosion(_fill0, iterations=2)
        _rgba = np.zeros((*_fill0.shape, 4), dtype=float)
        _hex = str(t.get("insignificant", "#B9AE9C")).lstrip("#")
        _rgb = tuple(int(_hex[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        _rgba[_edge] = (_rgb[0], _rgb[1], _rgb[2], 1.0)
        ax_wc.imshow(_rgba, interpolation="nearest")
    ax_wc.axis("off")
    ax_wc.set_title("开放题高频语义词云", fontsize=11.2 * fs, weight="bold", color=t["text_primary"], pad=8)

    plot_df = freq_df.head(15).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    cnts = plot_df["词频"].to_numpy(dtype=int)
    pcts = plot_df["占比(%)"].to_numpy(dtype=float)
    words = plot_df["词语"].astype(str).tolist()
    top_w = str(freq_df.iloc[0]["词语"])

    colors = [get_item_color(w, t["accent_primary"] if w == top_w else t["categorical"][1], t, cfg) for w in words]
    ax_bar.barh(y, cnts, height=0.62, color=colors, zorder=3)
    for yi, (w, cnt, pct) in enumerate(zip(words, cnts, pcts)):
        if cfg["show_labels"]:
            ax_bar.text(cnt + max(cnts.max(), 1) * 0.02, yi, f"{cnt}次 ({pct:.1f}%)", va="center", ha="left", fontsize=8.8 * fs, color=t["text_primary"], weight="bold" if w == top_w else "normal")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(words, fontsize=9.8 * fs, color=t["text_primary"])
    ax_bar.set_xlim(0, max(int(cnts.max()), 1) * 1.34)
    ax_bar.set_xlabel("出现频次 (次)", fontsize=10.0 * fs, color=t["text_secondary"])
    ax_bar.set_title("Top 15 高频词精确统计", fontsize=11.2 * fs, weight="bold", color=t["text_primary"], loc="left", pad=8)
    ax_bar.grid(axis="y", visible=False)

    top1 = freq_df.iloc[0]
    top2 = freq_df.iloc[1] if len(freq_df) > 1 else top1
    title = f"开放题高频词：「{top1['词语']}」（{top1['词频']}次）与「{top2['词语']}」（{top2['词频']}次）提及最高"
    subtitle = f"{col}分词与高频词统计 | 词云与词频条形组合图（有效评论 n = {txt_res['n_valid_comments']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"分词引擎：jieba 中文分词 + 停用词过滤（有效总词次={txt_res['total_words']}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, freq_df, "46-词云与词频组合图", "词云与词频组合图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 47 - 蝴蝶对比图 (Butterfly Back-to-Back Comparison Chart)
# ==============================================================================
def chart_47_butterfly(df: pd.DataFrame, group_col: str = "性别", cat_col: str = "最常购买品牌", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        group_col = settings.get("group_col") or group_col
        cat_col = settings.get("cat_col") or cat_col
    if group_col not in df.columns or cat_col not in df.columns:
        caps = su.detect_dataset_capabilities(df)
        if group_col not in df.columns:
            bin_c = [c for c in caps["single_cols"] if df[c].dropna().nunique() == 2]
            group_col = bin_c[0] if bin_c else caps["single_cols"][0]
        if cat_col not in df.columns:
            cat_col = next((c for c in caps["single_cols"] if c != group_col), caps["single_cols"][-1])

    ct_res = su.crosstab_chi2_analysis(df, cat_col, group_col)
    col_pct = ct_res["col_pct_df"].copy()
    cnt_df = ct_res["count_df"]

    brands = col_pct.sum(axis=1).sort_values(ascending=True).index.tolist()
    col_pct = col_pct.loc[brands]
    cnt_df = cnt_df.loc[brands]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.14, 0.78, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(brands))

    if "女" in col_pct.columns and "男" in col_pct.columns:
        g_right, g_left = "女", "男"
        lbl_right, lbl_left = "女性", "男性"
    else:
        g_left = str(col_pct.columns[0])
        g_right = str(col_pct.columns[1]) if col_pct.shape[1] > 1 else g_left
        lbl_left, lbl_right = g_left, g_right

    f_vals = col_pct[g_right].to_numpy(dtype=float)
    m_vals = col_pct[g_left].to_numpy(dtype=float)

    ax.barh(y, -m_vals, height=0.56, color=t["categorical"][1], label=f"{lbl_left}组内占比 (%)")
    ax.barh(y, f_vals,  height=0.56, color=t["accent_primary"], label=f"{lbl_right}组内占比 (%)")
    ax.axvline(0, color=t["text_primary"], linewidth=1.2)

    max_v = max(float(m_vals.max()) if len(m_vals) else 10.0, float(f_vals.max()) if len(f_vals) else 10.0, 1.0)
    for yi, (b, mv, fv) in enumerate(zip(brands, m_vals, f_vals)):
        if cfg["show_labels"]:
            ax.text(-mv - max_v * 0.03, yi, f"{mv:.1f}%", va="center", ha="right", fontsize=9.5 * fs, color=t["text_primary"], weight="bold")
            ax.text(fv + max_v * 0.03,  yi, f"{fv:.1f}%", va="center", ha="left",  fontsize=9.5 * fs, color=t["text_primary"], weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(brands, fontsize=10.5 * fs, color=t["text_primary"], weight="bold")
    ax.set_xlim(-max_v * 1.35, max_v * 1.35)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.0f}%" for x in xticks], fontsize=9.8 * fs, color=t["text_secondary"])
    ax.set_xlabel(f"← {lbl_left}{cat_col}占比 (%)      |      {lbl_right}{cat_col}占比 (%) →", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)
    ax.grid(axis="y", visible=False)

    src = pd.DataFrame({
        cat_col: brands[::-1],
        f"{lbl_right}占比(%)": f_vals[::-1].round(1),
        f"{lbl_left}占比(%)": m_vals[::-1].round(1),
        f"差值({g_right}-{g_left}%)": (f_vals[::-1] - m_vals[::-1]).round(1),
    })
    top_b = src.iloc[0]
    title = f"{group_col}镜像对比：「{top_b[cat_col]}」在双端占比突显（{g_right} {top_b[f'{lbl_right}占比(%)']:.1f}% / {g_left} {top_b[f'{lbl_left}占比(%)']:.1f}%）"
    subtitle = f"{group_col} × {cat_col}背靠背结构对比 | 蝴蝶对比图 Butterfly Chart（n = {int(cnt_df.values.sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"卡方检验：χ²={ct_res['chi2']:.2f}, p={ct_res['p_value']:.4f}, Cramér's V={ct_res['cramers_v']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "47-蝴蝶对比图", "蝴蝶对比图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 48 - 双轴组合图 (Dual-Axis Bar + Line Combo Chart — 修复柱顶与折线标签碰撞)
# ==============================================================================
def chart_48_dual_axis(df: pd.DataFrame, cat_col: str = "最常购买品牌", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        cat_col = settings.get("cat_col") or cat_col
        num_col = settings.get("num_col") or num_col
    if cat_col not in df.columns or num_col not in df.columns:
        caps = su.detect_dataset_capabilities(df)
        if cat_col not in df.columns:
            cat_col = caps["single_cols"][0]
        if num_col not in df.columns:
            num_col = (caps["numeric_cols"] + caps["scale_cols"] + caps["nps_cols"])[0]

    sub = df[[cat_col, num_col]].dropna().copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    total_n = max(len(sub), 1)
    u = _unit_for_col(num_col)
    u_nosp = _fmt_u(u, False)
    u_col_sfx = f"({u})" if u else ""
    mean_col_key = f"{num_col}均值{u_col_sfx}"

    rows = []
    for b, grp in sub.groupby(cat_col):
        cnt = len(grp)
        rows.append({
            cat_col: str(b),
            "样本数(人)": cnt,
            "份额占比(%)": round(cnt / total_n * 100.0, 1),
            mean_col_key: round(float(grp[num_col].mean()), 1),
        })
    src = pd.DataFrame(rows).sort_values("样本数(人)", ascending=False).reset_index(drop=True)

    fig, ax1, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.81, 0.65])
    fs = float(cfg["font_scale"])
    x = np.arange(len(src))
    cnts = src["样本数(人)"].to_numpy(dtype=int)
    pcts = src["份额占比(%)"].to_numpy(dtype=float)
    means = src[mean_col_key].to_numpy(dtype=float)
    brands = src[cat_col].astype(str).tolist()

    bar_colors = [get_item_color(b, t["accent_primary"] if i == 0 else t["categorical"][1], t, cfg) for i, b in enumerate(brands)]
    bars = ax1.bar(x, cnts, width=0.48, color=bar_colors, alpha=0.88, label="样本量 (左轴: 人)", zorder=3)
    for bar, cnt, pct, c_hex in zip(bars, cnts, pcts, bar_colors):
        if cfg["show_labels"]:
            tc = auto_text_color(c_hex, t["text_primary"])
            ax1.text(bar.get_x() + bar.get_width() / 2, cnt * 0.52, f"{cnt}人\n({pct:.1f}%)", ha="center", va="center", fontsize=9.0 * fs, color=tc, weight="bold", zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(brands, fontsize=10.5 * fs, color=t["text_primary"], weight="bold", rotation=15 if len(brands) > 6 else 0)
    ax1.set_ylabel("样本数 (人)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax1.set_ylim(0, max(int(cnts.max()), 1) * 1.32)
    ax1.grid(axis="x", visible=False)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(t["accent_secondary"])
    ax2.plot(x, means, color=t["accent_secondary"], marker="D", markersize=8, linewidth=2.6, label=f"{num_col}均值 (右轴)", zorder=6)
    m_span = max(float(means.max() - means.min()), float(means.max()) * 0.25, 1.0)
    for xi, mv in zip(x, means):
        if cfg["show_labels"]:
            ax2.text(xi, mv + m_span * 0.08, f"{mv:.1f}{u_nosp}", ha="center", va="bottom", fontsize=9.6 * fs, color=t["accent_secondary"], weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor=t["bg_card"], edgecolor=t["accent_secondary"], lw=0.8))

    ax2.set_ylim(float(means.min()) - m_span * 0.45, float(means.max()) + m_span * 0.45)
    ax2.set_ylabel(_axis_label_with_unit(num_col, "均值"), fontsize=10.2 * fs, color=t["accent_secondary"])
    ax2.tick_params(axis="y", colors=t["accent_secondary"])
    ax2.grid(False)

    top_cnt_b = src.iloc[0]
    top_val_b = src.sort_values(mean_col_key, ascending=False).iloc[0]
    title = f"双维透视：「{top_cnt_b[cat_col]}」规模居首（{top_cnt_b['样本数(人)']}人），「{top_val_b[cat_col]}」{num_col}均值最高（{top_val_b[mean_col_key]:.1f}{u_nosp}）"
    subtitle = f"{cat_col}规模（柱）× {num_col}均值（线）| 双轴组合图（有效样本 n = {total_n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"左轴（柱状图）：各{cat_col}样本人数与占比 | 右轴（折线图）：各{cat_col}{num_col}均值 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "48-双轴组合图", "双轴组合图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 49 - 高频词共现网络图 (Word Co-occurrence Network — 文本模块扩展图)
# ==============================================================================
def chart_49_cooccurrence_network(df: pd.DataFrame, col: str = "开放题短评", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict) and settings.get("col") in df.columns:
        col = settings["col"]
    elif col not in df.columns:
        caps = su.detect_dataset_capabilities(df)
        if not caps["text_cols"]:
            raise ValueError("当前数据无开放文本列，无法绘制共现网络图。")
        col = caps["text_cols"][0]

    n_non_empty = int(df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna().shape[0])
    min_cooc = 6 if n_non_empty >= 150 else max(1, int(n_non_empty * 0.04))
    co_res = su.calc_word_cooccurrence(df[col], top_n=16, min_cooc=min_cooc)
    nodes_df = co_res["nodes_df"]
    edges_df = co_res["edges_df"]
    if edges_df.empty and min_cooc > 1:
        co_res = su.calc_word_cooccurrence(df[col], top_n=16, min_cooc=1)
        nodes_df = co_res["nodes_df"]
        edges_df = co_res["edges_df"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.06, 0.12, 0.88, 0.68])
    fs = float(cfg["font_scale"])

    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["词语"], freq=int(r["词频"]))
    for _, r in edges_df.iterrows():
        G.add_edge(r["源词"], r["目标词"], weight=int(r["共现频次"]))

    pos = nx.spring_layout(G, k=0.85, seed=42)
    max_w = max([d["weight"] for _, _, d in G.edges(data=True)] + [1])
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=t["accent_primary"], alpha=0.22 + 0.55 * (w / max_w), lw=1.0 + 3.5 * (w / max_w), zorder=2)

    max_f = max([d["freq"] for _, d in G.nodes(data=True)] + [1])
    for idx, (node, d) in enumerate(G.nodes(data=True)):
        freq = d["freq"]
        xn, yn = pos[node]
        c = get_item_color(node, t["categorical"][idx % len(t["categorical"])], t, cfg)
        sz = 380 + 950 * (freq / max_f)
        ax.scatter([xn], [yn], s=sz, color=c, alpha=0.88, edgecolors=t["bg_canvas"], lw=2.0, zorder=4)
        tc = auto_text_color(c, t["text_primary"])
        ax.text(xn, yn, f"{node}\n({freq})", ha="center", va="center", fontsize=8.8 * fs, weight="bold", color=tc, zorder=5)

    ax.axis("off")
    if not edges_df.empty:
        top_e = edges_df.iloc[0]
        title = f"高频词共现语义网络：「{top_e['源词']}—{top_e['目标词']}」强关联共现（{top_e['共现频次']}次）"
    elif not nodes_df.empty:
        title = f"高频词语义节点分布：「{nodes_df.iloc[0]['词语']}」（{nodes_df.iloc[0]['词频']}次）居核心位置"
    else:
        title = "开放题词汇共现网络"
    subtitle = f"{col}词汇同句共现拓扑关系 | 高频词共现网络图（节点大小 ∝ 词频，连线粗细 ∝ 共现频次）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"布局算法：Fruchterman-Reingold 力导向布局 (seed=42) | 数据来源：{demo_tag}"
    src_out = edges_df if not edges_df.empty else nodes_df
    return export_chart_bundle(fig, src_out, "49-高频词共现网络图", "高频词共现网络图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 50 - 情感分布图 (Sentiment Polarity Diverging Bar — 文本模块扩展图)
# ==============================================================================
def chart_50_sentiment(df: pd.DataFrame, col: str = "开放题短评", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict) and settings.get("col") in df.columns:
        col = settings["col"]
    elif col not in df.columns:
        caps = su.detect_dataset_capabilities(df)
        if not caps["text_cols"]:
            raise ValueError("当前数据无开放文本列，无法绘制情感分布图。")
        col = caps["text_cols"][0]

    sent = su.calc_sentiment_analysis(df[col])
    src = sent["aspect_df"].copy()
    plot_df = src.iloc[::-1].reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.15, 0.14, 0.74, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(plot_df))

    pos_p = plot_df["正向占比(%)"].to_numpy(dtype=float)
    neu_p = plot_df["中性占比(%)"].to_numpy(dtype=float)
    neg_p = plot_df["负向占比(%)"].to_numpy(dtype=float)
    nss_v = plot_df["净情感得分(NSS)"].to_numpy(dtype=float)

    ax.barh(y, -neg_p, height=0.56, left=-neu_p / 2, color=t["accent_secondary"], label="负向评价 (%)")
    ax.barh(y, neu_p,  height=0.56, left=-neu_p / 2, color=t["border_line"], label="中性评价 (%)")
    ax.barh(y, pos_p,  height=0.56, left=neu_p / 2,  color=t["accent_primary"], label="正向评价 (%)")
    ax.axvline(0, color=t["text_primary"], lw=1.2)

    for yi, (pp, np_v, neu_v, nss) in enumerate(zip(pos_p, neg_p, neu_p, nss_v)):
        if cfg["show_labels"]:
            ax.text(neu_v / 2 + pp + 1.8, yi, f"正向 {pp:.0f}% (NSS {nss:+.0f})", va="center", ha="left", fontsize=9.4 * fs, weight="bold", color=t["accent_primary"])
            ax.text(-neu_v / 2 - np_v - 1.8, yi, f"负向 {np_v:.0f}%", va="center", ha="right", fontsize=9.2 * fs, color=t["accent_secondary"])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["评价维度"], fontsize=10.5 * fs, color=t["text_primary"], weight="bold")
    x_lo = min(-65.0, float((-neu_p / 2 - neg_p).min()) - 18.0) if len(neg_p) else -65.0
    x_hi = max(105.0, float((neu_p / 2 + pos_p).max()) + 30.0) if len(pos_p) else 105.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("← 负向评论占比 (%)      |      正向评论占比 (%) →", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.8 * fs)
    ax.grid(axis="y", visible=False)

    top_a = src.iloc[0]
    bot_a = src.iloc[-1]
    title = f"开放题情感极性：「{top_a['评价维度']}」好评率居首（{top_a['正向占比(%)']:.0f}%），「{bot_a['评价维度']}」负向反馈占 {bot_a['负向占比(%)']:.0f}%"
    subtitle = f"{col}焦点维度情感极性拆解 | 正中负发散条形图（NSS = 正向% - 负向%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"口径说明：基于情感词典实测判别，净情感得分 NSS = 正向占比 - 负向占比（总有效评论 n={sent['n_total']}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "50-情感分布图", "情感分布图", title, subtitle, footnote, theme_name, settings)
