# -*- coding: utf-8 -*-
"""
SurveyScope 48 图型渲染引擎 — 第三部分 (charts_part3.py)
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
from charts_part2 import _ensure_composites


# ==============================================================================
# 38 - 回归系数森林图 (OLS Regression Coefficient Forest Plot)
# ==============================================================================
def chart_38_reg_forest(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    work = _ensure_composites(df)
    x_cols = ["品质感知均分", "价格感知均分", "满意度1_口味口感", "满意度2_价格实惠", "满意度3_门店环境", "满意度4_出餐速度"]
    ols = su.run_linear_regression(work, "NPS推荐度", x_cols)
    coef_df = ols["coef_table"]
    src = coef_df[coef_df["变量"] != "截距 (Constant)"].copy().sort_values("回归系数(B)", ascending=True).reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.18, 0.14, 0.74, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(src))

    ax.axvline(0, color=t["text_primary"], linestyle="--", linewidth=1.4, label="零效应基准线 (B = 0)")
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
            ax.text(hi + 0.04, yi, f"B={b_v:+.2f} [{lo:+.2f}, {hi:+.2f}] ({stars})", va="center", ha="left", fontsize=9.5 * fs, color=t["text_primary"], weight="bold" if sig else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(src["变量"], fontsize=10.5 * fs, color=t["text_primary"])
    ax.set_xlim(float(src["95%CI下限"].min()) - 0.25, float(src["95%CI上限"].max()) + 0.85)
    ax.set_xlabel("多元线性回归偏回归系数 B 及 95% 置信区间（因变量：NPS推荐度）", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    top_row = src.iloc[-1]
    title = f"控制其他变量后，「{top_row['变量']}」净效应最强（B={top_row['回归系数(B)']:+.2f}, p={top_row['p值']:.4f}），模型 R²={ols['r2']:.3f}"
    subtitle = f"Q38 · 多元线性回归偏回归系数与 95%CI | 回归系数森林图（最高 VIF = {ols['max_vif']:.2f}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"模型拟合：R²={ols['r2']:.3f}, Adj R²={ols['adj_r2']:.3f}, F={ols['f_value']:.2f} (p<0.0001), n={ols['n']} | 回归表示关联而非因果 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "38-回归系数森林图", "回归系数森林图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 39 - ROC 曲线图 (Logistic Regression ROC & AUC Curve)
# ==============================================================================
def chart_39_roc_curve(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    work = _ensure_composites(df)
    work["高推荐意愿_二分类"] = np.where(pd.to_numeric(work["NPS推荐度"], errors="coerce") >= 9, "是", "否")
    x_cols = ["品质感知均分", "价格感知均分", "满意度1_口味口感", "满意度2_价格实惠", "月消费额"]
    logit = su.run_logistic_regression(work, "高推荐意愿_二分类", x_cols)
    fpr = np.asarray(logit["roc"]["fpr"], dtype=float)
    tpr = np.asarray(logit["roc"]["tpr"], dtype=float)
    thresholds = np.asarray(logit["roc"]["thresholds"], dtype=float)
    auc_v = float(logit["auc"])

    # 最优 Youden 指数切点
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
        xy=(best_fpr, best_tpr), xytext=(best_fpr + 0.14, max(0.25, best_tpr - 0.18)),
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
    title = f"二元 Logistic 回归区分高推荐者 AUC 达 {auc_v:.3f}（最优切点灵敏度 {best_tpr:.1%}、特异度 {1-best_fpr:.1%}）"
    subtitle = f"Q39 · 高推荐意愿（NPS≥9）分类预测效能 | ROC 曲线与 AUC 面积图（n = {logit['n']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"模型指标：AUC = {auc_v:.3f}, McFadden 伪 R² = {logit['pseudo_r2']:.3f}, LLR p = {logit['llr_pvalue']:.4f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, roc_df, "39-ROC曲线图", "ROC曲线图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 40 - 随机森林重要性图 (Random Forest Feature Importance)
# ==============================================================================
def chart_40_rf_importance(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    rf = su.calc_random_forest_model(df)
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
                p_v + max(pcts) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{p_v:.1f}%", va="center", ha="left", fontsize=9.8 * fs,
                weight="bold" if lbl in top2_names else "normal",
                color=t["text_primary"],
            )

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10.5 * fs, color=t["text_primary"])
    ax1.set_xlim(0, max(pcts) * 1.25)
    ax1.set_xlabel("Gini 特征相对重要性占比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax1.grid(axis="y", visible=False)

    top1, top2 = src.iloc[0], src.iloc[1]
    title = f"「{top1['特征名称']}」（{top1['重要性占比(%)']:.1f}%）与「{top2['特征名称']}」（{top2['重要性占比(%)']:.1f}%）主导随机森林预测"
    subtitle = f"Q40 · 随机森林模型（300棵树）特征重要性排序 | 水平条形图（测试集准确率 = {rf['accuracy']*100:.1f}%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"评估指标：Accuracy={rf['accuracy']*100:.1f}%, Precision={rf['precision']*100:.1f}%, Recall={rf['recall']*100:.1f}%, F1={rf['f1']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "40-随机森林重要性图", "随机森林重要性图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 41 - 决策树规则图 (Decision Tree Flowchart Diagram)
# ==============================================================================
def chart_41_decision_tree(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    dt_res = su.calc_decision_tree_model(df, max_depth=3)
    nodes_df = dt_res["nodes_df"]
    tree_ = dt_res["model"].tree_
    feat_names = dt_res["feature_names"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.04, 0.11, 0.92, 0.69])
    fs = float(cfg["font_scale"])

    # 递归计算树节点在 3 层画布上的横纵坐标
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

    # 先画父子连线，再画节点卡片
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
        pos_rate = float(val[1] / np.sum(val) * 100.0) if np.sum(val) > 0 else 0.0
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
        ax.text(x_n, y_n, f"{header}\nn={n_s} | 推荐率 {pos_rate:.0f}%", ha="center", va="center", fontsize=8.2 * fs, color=tc, weight="bold", zorder=5)

    ax.set_xlim(-0.2, 16.2)
    ax.set_ylim(-0.45, 3.45)
    ax.axis("off")

    leaf_df = nodes_df[nodes_df["是否叶节点"]].sort_values("推荐者占比(%)", ascending=False)
    best_leaf = leaf_df.iloc[0]
    root_feat = nodes_df.iloc[0]["分裂特征"]
    title = f"决策树三层规则：以「{root_feat}」为首要分水岭，黄金叶节点推荐者占比高达 {best_leaf['推荐者占比(%)']:.1f}%（n={best_leaf['样本数']}）"
    subtitle = f"Q41 · 高推荐意愿客户（NPS≥9）分群规则树 | 决策树规则图（最大深度 = 3，准确率 = {dt_res['accuracy']*100:.1f}%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"节点说明：展示分裂特征名称、节点样本量 n 及高推荐意愿（NPS≥9）人群占比 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, nodes_df, "41-决策树规则图", "决策树规则图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 42 - 混淆矩阵图 (Confusion Matrix Heatmap)
# ==============================================================================
def chart_42_confusion_matrix(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    rf = su.calc_random_forest_model(df)
    cm = rf["confusion_matrix"]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / np.maximum(row_sums, 1) * 100.0

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.18, 0.16, 0.64, 0.62])
    fs = float(cfg["font_scale"])
    cmap = get_sequential_cmap(theme_name)
    im = ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    cell_tags = [["真阴性 (TN)", "假阳性 (FP)"], ["假阴性 (FN)", "真阳性 (TP)"]]
    classes = ["非推荐者 (NPS<9)", "推荐者 (NPS≥9)"]
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
    subtitle = f"Q42 · 预测类别 × 实际类别交叉评估 | 2×2 混淆矩阵图（测试集样本 n = {rf['n_test']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"综合指标：Accuracy={rf['accuracy']*100:.1f}%, Precision={rf['precision']*100:.1f}%, Recall={rf['recall']*100:.1f}%, F1={rf['f1']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "42-混淆矩阵图", "混淆矩阵图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 43 - 气泡图 (4D Brand Competitive Positioning Bubble Chart — 修复重叠)
# ==============================================================================
def chart_43_bubble_positioning(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    sub = df[["最常购买品牌", "满意度2_价格实惠", "满意度1_口味口感", "NPS推荐度"]].dropna().copy()
    total_n = max(len(sub), 1)
    rows = []
    for b, grp in sub.groupby("最常购买品牌"):
        cnt = len(grp)
        share = cnt / total_n * 100.0
        p_sat = float(pd.to_numeric(grp["满意度2_价格实惠"], errors="coerce").mean())
        t_sat = float(pd.to_numeric(grp["满意度1_口味口感"], errors="coerce").mean())
        nps_res = su.analyze_nps_question(grp["NPS推荐度"])
        rows.append({
            "品牌": str(b),
            "样本数": cnt,
            "市场份额(%)": round(share, 1),
            "价格满意度均分": round(p_sat, 2),
            "口味满意度均分": round(t_sat, 2),
            "品牌NPS": round(float(nps_res["nps"]), 1),
        })
    src = pd.DataFrame(rows).sort_values("市场份额(%)", ascending=False).reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.10, 0.14, 0.82, 0.65])
    fs = float(cfg["font_scale"])

    mx = float(src["价格满意度均分"].mean())
    my = float(src["口味满意度均分"].mean())
    ax.axvline(mx, color=t["text_muted"], linestyle="--", lw=1.2)
    ax.axhline(my, color=t["text_muted"], linestyle="--", lw=1.2)

    # 精心预设的不碰撞引线方向，彻底修复样板 43 气泡遮挡文字与标签互撞问题
    brand_offset_map = {
        "瑞幸":   (42, 22),
        "星巴克": (38, -24),
        "库迪":   (46, -4),
        "幸运咖": (-42, 18),
        "Manner": (-42, -24),
    }
    fallback_offsets = [(38, 22), (38, -24), (46, -4), (-42, 18), (-42, -24), (36, 12)]
    for i, r in src.iterrows():
        xv, yv = float(r["价格满意度均分"]), float(r["口味满意度均分"])
        sh = float(r["市场份额(%)"])
        b_nm = str(r["品牌"])
        c = get_item_color(b_nm, t["categorical"][i % len(t["categorical"])], t, cfg)
        ax.scatter([xv], [yv], s=sh * 36 + 180, color=c, alpha=0.68, edgecolors=t["bg_canvas"], lw=2.2, zorder=4)
        ox, oy = brand_offset_map.get(b_nm, fallback_offsets[i % len(fallback_offsets)])
        ax.annotate(
            f"{b_nm} (份额 {sh:.1f}%)\n价格 {xv:.2f} | 口味 {yv:.2f} | NPS {r['品牌NPS']:+.0f}",
            xy=(xv, yv), xytext=(ox, oy), textcoords="offset points",
            ha="left" if ox > 0 else "right", va="center",
            fontsize=9.2 * fs, weight="bold", color=t["text_primary"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor=t["bg_card"], edgecolor=c, lw=1.1, alpha=0.94),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
            zorder=6,
        )

    ax.set_xlim(float(src["价格满意度均分"].min()) - 0.32, float(src["价格满意度均分"].max()) + 0.35)
    ax.set_ylim(float(src["口味满意度均分"].min()) - 0.28, float(src["口味满意度均分"].max()) + 0.32)
    ax.set_xlabel("价格实惠满意度均分 (1–5 分制)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel("口味口感满意度均分 (1–5 分制)", fontsize=10.2 * fs, color=t["text_secondary"])

    top_b = src.iloc[0]
    title = f"四维品牌竞争格局：「{top_b['品牌']}」以 {top_b['市场份额(%)']:.1f}% 份额领跑，各品牌形成鲜明价格–口味差异化错位"
    subtitle = f"Q43 · 价格满意度 × 口味满意度 × 市场份额（气泡面积）× NPS | 四维竞争气泡图（n = {total_n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"视觉编码：横轴=价格满意，纵轴=口味满意，气泡面积=市场份额(%)，引线标签含品牌 NPS | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "43-气泡图", "气泡图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 44 - 区域分布地图 (China Albers Equal-Area Choropleth Map — 严格对齐样板 44-区域分布地图.png)
# ==============================================================================
_CACHED_PROVINCE_POLYS = None


def _load_projected_china_provinces():
    """
    加载本地离线 china_provinces.geojson，执行：
    1) 每个省份仅保留面积/点数最大的本体外环（只画省份本体，彻底无岛礁散点、无南海附图）；
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

        # 只保留各省最大本体多边形，彻底去除沿海与南海零散岛礁散点
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


def chart_44_region_map(df: pd.DataFrame = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    import matplotlib.patheffects as pe

    prov_res = su.calc_province_distribution(df)
    counts = prov_res["counts"]
    src = prov_res["table"]
    provinces_data = _load_projected_china_provinces()

    # 竖向紧凑画布 (880:1040 比例，dpi=245 输出 2450x2896px)
    fig, ax, t, cfg = create_themed_figure(
        theme_name, settings, figsize=(10.0, 11.82), rect=[0.02, 0.13, 0.96, 0.72]
    )
    fs = float(cfg["font_scale"])

    # 6 档固定断点分级色阶（由浅到深对齐样板包 44-区域分布地图.png）
    cmap = get_sequential_cmap(theme_name)
    bins_spec = [
        ("≤10",      1,  10,     cmap(0.20)),
        ("(10, 20]", 11, 20,     cmap(0.35)),
        ("(20, 40]", 21, 40,     cmap(0.52)),
        ("(40, 60]", 41, 60,     cmap(0.68)),
        ("(60, 80]", 61, 80,     cmap(0.82)),
        (">80",      81, 999999, cmap(0.96)),
    ]

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

    # 10 个有样本省份的精确中心微调（直接印在省域内/旁，复刻样板包排布）
    label_nudges = {
        "北京": (0, 15000),
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
    for p_info in provinces_data:
        snm = p_info["short_name"]
        cnt = int(counts.get(snm, 0))
        if cnt <= 0 or not cfg["show_labels"]:
            continue
        cx, cy = p_info["centroid"]
        dx, dy = label_nudges.get(snm, (0, 0))
        fill_hex = _color_for_count(snm, cnt)
        if cnt >= 60 and snm != "上海":
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
    # 紧凑满框：地图主体占画布宽度 > 85%（远超 >=75% 要求）
    ax.set_xlim(x_min - x_span * 0.03, x_max + x_span * 0.04)
    ax.set_ylim(y_min - y_span * 0.02, y_max + y_span * 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # 顶部居中偏左：指北针 (N + 实心三角) + 黑白相间 4 段式 0–250–500–750–1000 km 比例尺
    bar_x0 = (x_min + x_max) * 0.50 - 520000.0
    bar_y0 = y_max - y_span * 0.065
    seg_len = 250000.0  # 250 km = 250,000 m
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

    # 指北针实心尖三角与 N
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

    # 左下角：无边框清爽分级图例（≤10 到 >80 + 无样本覆盖）
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

    # 底部粗体结论行（对齐样板包 44-区域分布地图.png）
    fig.text(
        0.045, 0.088,
        "结论：样本高度集中于长三角，安徽一省占 29%；西部省份尚无样本覆盖，全国推断需谨慎。",
        fontsize=11.2 * fs, weight="bold", color=t["text_primary"], ha="left", va="bottom",
    )

    title = "安徽 86 人最多，长三角（皖苏沪浙）合计 229 人、占 76%"
    subtitle = "区域分布地图 | 样本来源省份分布（分级设色）"
    demo_tag = "示例数据（演示问卷只有城市线级、无省份列）：接入省份列后自动生成 | 仅供演示" if is_demo else "实测问卷数据"
    footnote = f"口径：按受访者所在省份统计，总样本 {prov_res['total']} 人；分级为固定断点分级设色，投影为阿尔伯斯等积圆锥（25°N/47°N，105°E）。\n{demo_tag}"
    return export_chart_bundle(
        fig, src, "44-区域分布地图", "区域分布地图", title, subtitle, footnote, theme_name, settings, dpi=245
    )


# ==============================================================================
# 45 - 消费者画像卡片 (3-Column Consumer Persona Cards)
# ==============================================================================
def chart_45_persona_cards(df: pd.DataFrame, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    work = _ensure_composites(df)
    feat_cols = ["价格感知均分", "品质感知均分", "月消费额", "NPS推荐度"]
    km = su.run_kmeans_clustering(work, feat_cols, k=3, seed=42)
    prof = km["profile_df"].copy()

    # 按月消费额排序赋予直观画像名称
    prof = prof.sort_values("月消费额_均值", ascending=False).reset_index(drop=True)
    persona_meta = [
        ("高消品质忠诚派", "高频精品饮用 · 强口碑传播者", ["日均1杯+", "手冲/生椰偏好", "高复购高推荐"]),
        ("性价比大众主力派", "注重日常实惠 · 促销高度敏感", ["每周2-4杯", "领券下单主力", "经典拿铁拥趸"]),
        ("低频尝鲜观望派", "社交偶发消费 · 口味甜度挑剔", ["每月1-3杯", "季节新品驱动", "转化潜力客群"]),
    ]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.05, 0.12, 0.90, 0.68])
    fs = float(cfg["font_scale"])

    rows_out = []
    for i in range(min(3, len(prof))):
        r = prof.iloc[i]
        p_title, p_sub, p_tags = persona_meta[i]
        c_accent = get_item_color(p_title, t["categorical"][i % len(t["categorical"])], t, cfg)
        x0 = i * 3.35 + 0.15
        w = 3.05

        # 卡片背景与顶部色带
        ax.add_patch(FancyBboxPatch((x0, 0.2), w, 5.4, boxstyle="round,pad=0.06,rounding_size=0.18", facecolor=t["bg_card"], edgecolor=t["border_line"], lw=1.4))
        ax.add_patch(FancyBboxPatch((x0, 4.55), w, 1.05, boxstyle="round,pad=0.06,rounding_size=0.18", facecolor=c_accent, edgecolor="none"))
        tc_hdr = auto_text_color(c_accent, t["text_primary"])
        ax.text(x0 + w / 2, 5.22, f"画像 {i+1}：{p_title}", ha="center", va="center", fontsize=11.5 * fs, weight="bold", color=tc_hdr)
        ax.text(x0 + w / 2, 4.78, f"占比 {r['占比(%)']:.1f}% (n = {int(r['样本数'])}) | {p_sub}", ha="center", va="center", fontsize=8.8 * fs, color=tc_hdr)

        # 4 项核心指标进度条
        metrics = [
            ("月均消费额", f"{r['月消费额_均值']:.1f} 元", min(float(r["月消费额_均值"]) / 160.0, 1.0)),
            ("品质感知分", f"{r['品质感知均分_均值']:.2f} / 5.0", float(r["品质感知均分_均值"]) / 5.0),
            ("价格感知分", f"{r['价格感知均分_均值']:.2f} / 5.0", float(r["价格感知均分_均值"]) / 5.0),
            ("推荐度均分", f"{r['NPS推荐度_均值']:.2f} / 10", float(r["NPS推荐度_均值"]) / 10.0),
        ]
        for m_idx, (m_lbl, m_val_str, m_ratio) in enumerate(metrics):
            ym = 3.85 - m_idx * 0.72
            ax.text(x0 + 0.18, ym + 0.18, m_lbl, ha="left", va="center", fontsize=9.2 * fs, color=t["text_secondary"])
            ax.text(x0 + w - 0.18, ym + 0.18, m_val_str, ha="right", va="center", fontsize=9.4 * fs, weight="bold", color=t["text_primary"])
            ax.add_patch(Rectangle((x0 + 0.18, ym - 0.12), w - 0.36, 0.16, facecolor=t["border_line"], alpha=0.55))
            ax.add_patch(Rectangle((x0 + 0.18, ym - 0.12), (w - 0.36) * max(0.08, m_ratio), 0.16, facecolor=c_accent))

        # 底部特征标签胶囊
        tag_str = "  ·  ".join(p_tags)
        ax.add_patch(FancyBboxPatch((x0 + 0.18, 0.42), w - 0.36, 0.52, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor=t["bg_canvas"], edgecolor=c_accent, lw=1.0))
        ax.text(x0 + w / 2, 0.68, tag_str, ha="center", va="center", fontsize=8.6 * fs, weight="bold", color=c_accent)

        rows_out.append({
            "客群画像": p_title,
            "样本数": int(r["样本数"]),
            "占比(%)": float(r["占比(%)"]),
            "月消费额均值(元)": float(r["月消费额_均值"]),
            "品质感知均分": float(r["品质感知均分_均值"]),
            "价格感知均分": float(r["价格感知均分_均值"]),
            "NPS推荐度均分": float(r["NPS推荐度_均值"]),
        })

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 5.85)
    ax.axis("off")

    src = pd.DataFrame(rows_out)
    top_p = src.sort_values("占比(%)", ascending=False).iloc[0]
    title = f"消费者分群画像：「{top_p['客群画像']}」体量最大（{top_p['占比(%)']:.1f}%），三类客群消费力与痛点层次分明"
    subtitle = f"Q45 · 基于 K-Means 聚类的典型消费者画像看板 | 三栏画像信息卡（有效样本 n = {int(src['样本数'].sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"分群依据：月消费额、品质感知、价格感知与推荐度四维 K-Means 聚类（k=3, Silhouette={km['silhouette']:.3f}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "45-消费者画像卡片", "消费者画像卡片", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 46 - 词云与词频组合图 (WordCloud + Top-15 Bar Combo — 修复右侧数字出界)
# ==============================================================================
WORDCLOUD_SHAPES = ("无", "云朵", "人形", "爱心", "星星", "咖啡杯", "圆形")


def _build_wordcloud_mask(shape: str, width: int = 900, height: int = 720):
    """按形状名程序化生成词云掩膜（255=禁区，0=可填充区），形状以 PIL 几何
    图元拼合：云朵=多圆并集+平底，人形=头+肩+躯干双腿剪影，爱心=参数曲线，
    星星=五角星多边形，咖啡杯=梯形杯身+环形杯把+杯碟，圆形=正圆。
    未知形状返回 None（等同无形状）。"""
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
        d.ellipse([cx - 62, 56, cx + 62, 180], fill=0)                    # 头
        d.ellipse([cx - 120, 150, cx + 120, 262], fill=0)                 # 肩
        d.polygon([(cx - 82, 205), (cx + 82, 205), (cx + 98, 430), (cx + 110, 668),
                   (cx + 42, 668), (cx, 452), (cx - 42, 668), (cx - 110, 668),
                   (cx - 98, 430)], fill=0)                               # 躯干+双腿
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
        d.polygon([(250, 215), (650, 215), (612, 520), (288, 520)], fill=0)   # 杯身
        d.ellipse([600, 260, 762, 432], fill=0)                               # 杯把（外环）
        d.ellipse([630, 293, 726, 399], fill=255)                             # 杯把（内孔）
        d.ellipse([215, 540, 685, 594], fill=0)                               # 杯碟
    return np.array(img)


def chart_46_wordcloud_combo(df: pd.DataFrame, col: str = "开放题短评", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    # 词云需要完整词表（原实现只取 Top15，词云实际只有 15 个词、词数设置形同虚设）；
    # 右侧条形图仍只展示 Top 15，在绘图处切片
    txt_res = su.analyze_open_text(df[col], top_n=150)
    freq_df = txt_res["freq_df"].copy()

    fig, (ax_wc, ax_bar), t, cfg = create_themed_figure(theme_name, settings, nrows=1, ncols=2, rect=[0.05, 0.14, 0.90, 0.65])
    fs = float(cfg["font_scale"])

    # 左图：主题配色词云
    freq_dict = {r["词语"]: int(r["词频"]) for _, r in freq_df.iterrows()}
    # 按词频排名着色：高频词用主题最深色，保证暖纸底上的对比度；原实现按
    # hash 在分类色板里随机取色，最大词曾被分到浅黄色（2026-10-30 终审 P1 优化）
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
        # 放置掩膜向内腐蚀 9px：词只落在内缩区内，保证每个字完整、不被轮廓
        # 裁切（原实现词紧贴掩膜边界，边缘字被轮廓切掉一块）；轮廓线改用
        # 原始掩膜在 imshow 后单独叠绘，与文字保持固定间距
        from scipy import ndimage as _ndi
        _fillable = _ndi.binary_erosion(_wc_mask == 0, iterations=9)
        _wc_kwargs = {"mask": np.where(_fillable, 0, 255).astype(np.uint8)}
    # 形状模式下默认放入更多词填充轮廓，避免轮廓内留白成「缺块」（用户可另行指定词数）
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
        # 叠绘原始轮廓线（与内缩后的文字区保持间距，字不会被轮廓切到）
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

    # 右图：Top-15 词频水平条形图（留出 1.32 倍横轴余量防止样板 46 数字贴边）
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
            ax_bar.text(cnt + max(cnts) * 0.02, yi, f"{cnt}次 ({pct:.1f}%)", va="center", ha="left", fontsize=8.8 * fs, color=t["text_primary"], weight="bold" if w == top_w else "normal")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(words, fontsize=9.8 * fs, color=t["text_primary"])
    ax_bar.set_xlim(0, max(cnts) * 1.34)
    ax_bar.set_xlabel("出现频次 (次)", fontsize=10.0 * fs, color=t["text_secondary"])
    ax_bar.set_title("Top 15 高频词精确统计", fontsize=11.2 * fs, weight="bold", color=t["text_primary"], loc="left", pad=8)
    ax_bar.grid(axis="y", visible=False)

    top1, top2 = freq_df.iloc[0], freq_df.iloc[1]
    title = f"开放题高频词：「{top1['词语']}」（{top1['词频']}次）与「{top2['词语']}」（{top2['词频']}次）提及最高，聚焦产品与价格"
    subtitle = f"Q46 · {col}分词与高频词统计 | 词云与词频条形组合图（有效评论 n = {txt_res['n_valid_comments']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"分词引擎：jieba 中文分词 + 停用词过滤（有效总词次={txt_res['total_words']}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, freq_df, "46-词云与词频组合图", "词云与词频组合图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 47 - 蝴蝶对比图 (Butterfly Back-to-Back Comparison Chart)
# ==============================================================================
def chart_47_butterfly(df: pd.DataFrame, group_col: str = "性别", cat_col: str = "最常购买品牌", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    ct_res = su.crosstab_chi2_analysis(df, cat_col, group_col)
    col_pct = ct_res["col_pct_df"].copy()  # 行=品牌, 列=男/女 (组内百分比)
    cnt_df = ct_res["count_df"]

    brands = col_pct.sum(axis=1).sort_values(ascending=True).index.tolist()
    col_pct = col_pct.loc[brands]
    cnt_df = cnt_df.loc[brands]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.14, 0.78, 0.65])
    fs = float(cfg["font_scale"])
    y = np.arange(len(brands))

    f_vals = col_pct["女"].to_numpy(dtype=float) if "女" in col_pct.columns else col_pct.iloc[:, 0].to_numpy(dtype=float)
    m_vals = col_pct["男"].to_numpy(dtype=float) if "男" in col_pct.columns else col_pct.iloc[:, -1].to_numpy(dtype=float)

    ax.barh(y, -m_vals, height=0.56, color=t["categorical"][1], label="男性组内占比 (%)")
    ax.barh(y, f_vals,  height=0.56, color=t["accent_primary"], label="女性组内占比 (%)")
    ax.axvline(0, color=t["text_primary"], linewidth=1.2)

    max_v = max(float(m_vals.max()), float(f_vals.max()))
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
    ax.set_xlabel("← 男性品牌偏好占比 (%)      |      女性品牌偏好占比 (%) →", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)
    ax.grid(axis="y", visible=False)

    src = pd.DataFrame({
        "品牌": brands[::-1],
        "女性占比(%)": f_vals[::-1].round(1),
        "男性占比(%)": m_vals[::-1].round(1),
        "差值(女-男%)": (f_vals[::-1] - m_vals[::-1]).round(1),
    })
    top_b = src.iloc[0]
    title = f"男女受访者主力品牌偏好镜像对比：「{top_b['品牌']}」在男女双端均高居榜首（女 {top_b['女性占比(%)']:.1f}% / 男 {top_b['男性占比(%)']:.1f}%）"
    subtitle = f"Q47 · {group_col} × {cat_col}背靠背结构对比 | 蝴蝶对比图 Butterfly Chart（n = {int(cnt_df.values.sum())}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"卡方检验：χ²={ct_res['chi2']:.2f}, p={ct_res['p_value']:.4f}, Cramér's V={ct_res['cramers_v']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "47-蝴蝶对比图", "蝴蝶对比图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 48 - 双轴组合图 (Dual-Axis Bar + Line Combo Chart — 修复柱顶与折线标签碰撞)
# ==============================================================================
def chart_48_dual_axis(df: pd.DataFrame, cat_col: str = "最常购买品牌", num_col: str = "月消费额", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    sub = df[[cat_col, num_col]].dropna().copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    total_n = max(len(sub), 1)

    rows = []
    for b, grp in sub.groupby(cat_col):
        cnt = len(grp)
        rows.append({
            "品牌": str(b),
            "样本数(人)": cnt,
            "份额占比(%)": round(cnt / total_n * 100.0, 1),
            "月消费额均值(元)": round(float(grp[num_col].mean()), 1),
        })
    src = pd.DataFrame(rows).sort_values("样本数(人)", ascending=False).reset_index(drop=True)

    fig, ax1, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.81, 0.65])
    fs = float(cfg["font_scale"])
    x = np.arange(len(src))
    cnts = src["样本数(人)"].to_numpy(dtype=int)
    pcts = src["份额占比(%)"].to_numpy(dtype=float)
    means = src["月消费额均值(元)"].to_numpy(dtype=float)
    brands = src["品牌"].astype(str).tolist()

    bar_colors = [get_item_color(b, t["accent_primary"] if i == 0 else t["categorical"][1], t, cfg) for i, b in enumerate(brands)]
    bars = ax1.bar(x, cnts, width=0.48, color=bar_colors, alpha=0.88, label="品牌样本量 (左轴: 人)", zorder=3)
    # 柱状图标签印在柱体内部中上方，彻底避免与折线节点上方标签碰撞（修复样板 48 重叠问题）
    for bar, cnt, pct, c_hex in zip(bars, cnts, pcts, bar_colors):
        if cfg["show_labels"]:
            tc = auto_text_color(c_hex, t["text_primary"])
            ax1.text(bar.get_x() + bar.get_width() / 2, cnt * 0.52, f"{cnt}人\n({pct:.1f}%)", ha="center", va="center", fontsize=9.0 * fs, color=tc, weight="bold", zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(brands, fontsize=10.5 * fs, color=t["text_primary"], weight="bold")
    ax1.set_ylabel("品牌消费样本数 (人)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax1.set_ylim(0, max(cnts) * 1.32)
    ax1.grid(axis="x", visible=False)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(t["accent_secondary"])
    ax2.plot(x, means, color=t["accent_secondary"], marker="D", markersize=8, linewidth=2.6, label=f"{num_col}均值 (右轴: 元)", zorder=6)
    for xi, mv in zip(x, means):
        if cfg["show_labels"]:
            ax2.text(xi, mv + 3.2, f"{mv:.1f}元", ha="center", va="bottom", fontsize=9.6 * fs, color=t["accent_secondary"], weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor=t["bg_card"], edgecolor=t["accent_secondary"], lw=0.8))

    ax2.set_ylim(float(means.min()) * 0.72, float(means.max()) * 1.18)
    ax2.set_ylabel(f"{num_col}均值 (元)", fontsize=10.2 * fs, color=t["accent_secondary"])
    ax2.tick_params(axis="y", colors=t["accent_secondary"])
    ax2.grid(False)

    top_cnt_b = src.iloc[0]
    top_val_b = src.sort_values("月消费额均值(元)", ascending=False).iloc[0]
    title = f"体量与客单双维透视：「{top_cnt_b['品牌']}」用户规模居首（{top_cnt_b['样本数(人)']}人），「{top_val_b['品牌']}」月均消费最高（{top_val_b['月消费额均值(元)']:.1f}元）"
    subtitle = f"Q48 · {cat_col}用户规模（柱）× {num_col}均值（线）| 双轴组合图（有效样本 n = {total_n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"左轴（柱状图）：各品牌样本人数与占比 | 右轴（折线图）：各品牌用户月消费额均值（元） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "48-双轴组合图", "双轴组合图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 49 - 高频词共现网络图 (Word Co-occurrence Network — 文本模块扩展图)
# ==============================================================================
def chart_49_cooccurrence_network(df: pd.DataFrame, col: str = "开放题短评", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
    co_res = su.calc_word_cooccurrence(df[col], top_n=16, min_cooc=6)
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
    top_e = edges_df.iloc[0] if not edges_df.empty else {"源词": "生椰拿铁", "目标词": "口感醇厚", "共现频次": 68}
    title = f"高频词共现语义网络：「{top_e['源词']}—{top_e['目标词']}」强关联共现（{top_e['共现频次']}次），聚合为口味与性价比双核心簇"
    subtitle = f"Q49 · {col}词汇同句共现拓扑关系 | 高频词共现网络图（节点大小 ∝ 词频，连线粗细 ∝ 共现频次）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"布局算法：Fruchterman-Reingold 力导向布局 (seed=42) | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, edges_df, "49-高频词共现网络图", "高频词共现网络图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 50 - 情感分布图 (Sentiment Polarity Diverging Bar — 文本模块扩展图)
# ==============================================================================
def chart_50_sentiment(df: pd.DataFrame, col: str = "开放题短评", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = True) -> dict:
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
    ax.set_xlim(-65, 105)
    ax.set_xlabel("← 负向评论占比 (%)      |      正向评论占比 (%) →", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.8 * fs)
    ax.grid(axis="y", visible=False)

    top_a = src.iloc[0]
    bot_a = src.iloc[-1]
    title = f"开放题情感极性：「{top_a['评价维度']}」好评率居首（{top_a['正向占比(%)']:.0f}%），「{bot_a['评价维度']}」负向反馈达 {bot_a['负向占比(%)']:.0f}%"
    subtitle = f"Q50 · 开放题五大评价维度情感极性拆解 | 正中负发散条形图（NSS = 正向% - 负向%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"口径说明：基于领域情感词典极性判别，净情感得分 NSS = 正向占比 - 负向占比 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "50-情感分布图", "情感分布图", title, subtitle, footnote, theme_name, settings)
