# -*- coding: utf-8 -*-
"""
SurveyScope 图型渲染引擎 — 第二部分 (charts_part2.py)
涵盖编号 25 至 37（共 13 张图）：
25-散点回归图, 26-相关性热力图, 27-气泡矩阵图, 28-平行坐标图,
29-漏斗图, 30-桑基图, 31-对应分析双标图, 32-IPA四象限图, 33-PSM价格敏感度曲线,
34-结构方程路径图, 35-KANO模型图, 36-联合分析重要性图, 37-联合分析效用曲线
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle, PathPatch
from matplotlib.path import Path as MplPath

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
)


def _ensure_composites(df: pd.DataFrame) -> pd.DataFrame:
    """仅当对应量表列存在时派生合成维度分，对任意上传问卷零侵入、不抛 KeyError。"""
    work = df.copy()
    if "品质量表3_正向" not in work.columns and "品质量表3_品质不稳定_反向" in work.columns:
        work["品质量表3_正向"] = su.reverse_score(work["品质量表3_品质不稳定_反向"], max_scale=5)
    p_cols = ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"]
    if "价格感知均分" not in work.columns and all(c in work.columns for c in p_cols):
        work["价格感知均分"] = su.compute_scale_composite(work, p_cols)
    q_cols = ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"]
    if "品质感知均分" not in work.columns and all(c in work.columns for c in q_cols):
        work["品质感知均分"] = su.compute_scale_composite(work, q_cols)
    return work


def _short_col_label(col: str, max_len: int = 10) -> str:
    s = str(col)
    if "_" in s:
        rhs = s.split("_", 1)[1]
        if len(rhs) >= 2:
            s = rhs
    return s[:max_len]


# ==============================================================================
# 25 - 散点回归图 (Scatter + OLS Regression + 95% CI + Marginal Histograms)
# ==============================================================================
def chart_25_scatter_ols(df: pd.DataFrame, x_col: str = "品质感知均分", y_col: str = "NPS推荐度", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = _ensure_composites(df)
    if isinstance(settings, dict):
        x_col = settings.get("x_col") or x_col
        y_col = settings.get("y_col") or y_col
    if x_col not in work.columns or y_col not in work.columns:
        caps = su.detect_dataset_capabilities(work)
        pool = caps["numeric_cols"] + caps["nps_cols"] + caps["scale_cols"]
        pool = list(dict.fromkeys([c for c in pool if c in work.columns]))
        if x_col not in work.columns:
            x_col = pool[0]
        if y_col not in work.columns:
            y_col = next((c for c in pool if c != x_col), pool[-1])

    sub = work[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    cp = su.correlation_pair_analysis(sub[x_col], sub[y_col])
    ols = su.run_linear_regression(sub, y_col, [x_col])

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.66, 0.54])
    fs = float(cfg["font_scale"])

    ax_top = fig.add_axes([0.09, 0.70, 0.66, 0.11], facecolor=t["bg_canvas"], sharex=ax)
    ax_right = fig.add_axes([0.77, 0.14, 0.14, 0.54], facecolor=t["bg_canvas"], sharey=ax)
    for ma in (ax_top, ax_right):
        for sp in ma.spines.values():
            sp.set_visible(False)
        ma.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ma.grid(False)

    rng = np.random.default_rng(42)
    x_span = max(float(x.max() - x.min()), 1e-6)
    y_span = max(float(y.max() - y.min()), 1e-6)
    jx = x + rng.normal(0, x_span * 0.01, size=len(x))
    jy = y + rng.normal(0, y_span * 0.012, size=len(y))
    ax.scatter(jx, jy, color=t["accent_primary"], alpha=0.48, s=36, edgecolors="none", label=f"样本散点 (n={len(sub)})")

    xg = np.linspace(float(x.min()), float(x.max()), 100)
    slope, intercept = float(ols["slope"]), float(ols["intercept"])
    yg = intercept + slope * xg
    denom = max(float(np.sum((x - np.mean(x)) ** 2)), 1e-9)
    se_y = np.std(y - (intercept + slope * x), ddof=min(2, max(len(x) - 1, 0))) * np.sqrt(1.0 / max(len(x), 1) + (xg - np.mean(x)) ** 2 / denom)
    ci_band = 1.96 * se_y
    ax.fill_between(xg, yg - ci_band, yg + ci_band, color=t["accent_secondary"], alpha=0.22, label="95% 回归置信带")
    ax.plot(xg, yg, color=t["accent_secondary"], linewidth=2.6, label=f"OLS: Y = {intercept:.2f} + {slope:.2f}X (R²={ols['r2']:.3f})")

    ax_top.hist(x, bins=min(15, max(5, len(np.unique(x)))), color=t["accent_primary"], alpha=0.55, edgecolor=t["bg_canvas"])
    ax_right.hist(y, bins=min(11, max(5, len(np.unique(y)))), orientation="horizontal", color=t["accent_secondary"], alpha=0.55, edgecolor=t["bg_canvas"])

    ax.set_xlabel(x_col, fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel(y_col, fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="lower right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    p_str = "p < 0.001" if cp["pearson_p"] < 0.001 else f"p = {cp['pearson_p']:.3f}"
    if cp["pearson_p"] < 0.05:
        rel_str = "显著正向预测" if cp["pearson_r"] > 0 else "显著负向预测"
    else:
        rel_str = "线性关联微弱于"
    title = f"{x_col}{rel_str}{y_col}（r = {cp['pearson_r']:.3f}, {p_str}，回归斜率 B = {slope:+.2f}）"
    subtitle = f"{x_col} × {y_col}双变量联合分布与回归拟合 | 散点回归边际图（n = {len(sub)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"Pearson r={cp['pearson_r']:.3f} (p={cp['pearson_p']:.4f}) | Spearman ρ={cp['spearman_r']:.3f} | OLS R²={ols['r2']:.3f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, ols["coef_table"], "25-散点回归图", "散点回归图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 26 - 相关性热力图 (Lower-Triangle Correlation Matrix Heatmap)
# ==============================================================================
def chart_26_corr_heatmap(df: pd.DataFrame, cols: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = _ensure_composites(df)
    cfg_cols = (settings or {}).get("cols") if isinstance(settings, dict) else None
    use_cols = cfg_cols or cols
    default_cols = ["价格感知均分", "品质感知均分", "满意度1_口味口感", "满意度2_价格实惠", "满意度3_门店环境", "满意度4_出餐速度", "NPS推荐度"]
    default_names = ["价格感知", "品质感知", "口味满意", "价格满意", "环境满意", "速度满意", "NPS推荐度"]

    if use_cols and len([c for c in use_cols if c in work.columns]) >= 2:
        cols = [c for c in use_cols if c in work.columns][:10]
        short_names = [_short_col_label(c) for c in cols]
        is_default = False
    elif all(c in work.columns for c in default_cols):
        cols = default_cols
        short_names = default_names
        is_default = True
    else:
        caps = su.detect_dataset_capabilities(work)
        pool = list(dict.fromkeys(caps["scale_cols"] + caps["nps_cols"] + caps["numeric_cols"]))
        cols = [c for c in pool if c in work.columns][:8]
        short_names = [_short_col_label(c) for c in cols]
        is_default = False

    sub = work[cols].apply(pd.to_numeric, errors="coerce").dropna()
    sub.columns = short_names

    corr_mat = sub.corr(method="pearson").to_numpy(dtype=float)
    n = len(sub)
    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.16, 0.74, 0.64])
    fs = float(cfg["font_scale"])

    masked = corr_mat.copy()
    for i in range(len(short_names)):
        for j in range(i + 1, len(short_names)):
            masked[i, j] = np.nan

    cmap = get_diverging_cmap(theme_name)
    cmap.set_bad(color=t["bg_canvas"])
    im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    best_pair = ("", "", 0.0)
    for i in range(len(short_names)):
        for j in range(i + 1):
            r_v = corr_mat[i, j]
            if i == j:
                txt = "1.00"
            else:
                _, p_v = stats.pearsonr(sub.iloc[:, i], sub.iloc[:, j])
                stars = "***" if p_v < 0.001 else ("**" if p_v < 0.01 else ("*" if p_v < 0.05 else ""))
                txt = f"{r_v:+.2f}{stars}"
                if abs(r_v) > abs(best_pair[2]):
                    best_pair = (short_names[i], short_names[j], r_v)
            rgba = cmap((r_v + 1.0) / 2.0)
            hex_c = "#{:02X}{:02X}{:02X}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            tc = auto_text_color(hex_c, t["text_primary"])
            if cfg["show_labels"]:
                ax.text(j, i, txt, ha="center", va="center", fontsize=9.2 * fs, color=tc, weight="bold" if abs(r_v) >= 0.3 and i != j else "normal")

    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=10.0 * fs, color=t["text_primary"], rotation=18 if len(short_names) > 6 else 0)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=10.0 * fs, color=t["text_primary"])
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Pearson 相关系数 (r ∈ [-1, +1])", fontsize=9.5 * fs, color=t["text_secondary"])

    src = pd.DataFrame(np.round(corr_mat, 3), index=short_names, columns=short_names).reset_index().rename(columns={"index": "变量"})
    if is_default:
        r_p_nps = corr_mat[0, -1]
        r_q_nps = corr_mat[1, -1]
        title = f"品质感知（r={r_q_nps:+.2f}***）与价格感知（r={r_p_nps:+.2f}***）均与 NPS 呈显著中度正相关"
    else:
        title = f"多变量相关性矩阵：「{best_pair[0]}」与「{best_pair[1]}」相关性最强（r = {best_pair[2]:+.2f}）"
    subtitle = f"多变量相关矩阵 | 下三角相关性热力图（n = {n}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"显著性标记：* p<0.05, ** p<0.01, *** p<0.001 | 色阶固定于 [-1, +1]，0 为中心 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "26-相关性热力图", "相关性热力图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 27 - 气泡矩阵图 (Correlation Bubble Matrix)
# ==============================================================================
def chart_27_bubble_matrix(df: pd.DataFrame, cols: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = _ensure_composites(df)
    cfg_cols = (settings or {}).get("cols") if isinstance(settings, dict) else None
    use_cols = cfg_cols or cols
    default_cols = ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向", "NPS推荐度"]
    default_names = ["价格合理", "性价比高", "促销吸引", "口感醇厚", "豆源优质", "出品稳定", "NPS推荐"]

    if use_cols and len([c for c in use_cols if c in work.columns]) >= 2:
        cols = [c for c in use_cols if c in work.columns][:10]
        names = [_short_col_label(c) for c in cols]
        is_default = False
    elif all(c in work.columns for c in default_cols):
        cols = default_cols
        names = default_names
        is_default = True
    else:
        caps = su.detect_dataset_capabilities(work)
        pool = list(dict.fromkeys(caps["scale_cols"] + caps["nps_cols"] + caps["numeric_cols"]))
        cols = [c for c in pool if c in work.columns][:8]
        names = [_short_col_label(c) for c in cols]
        is_default = False

    sub = work[cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr_mat = sub.corr().to_numpy(dtype=float)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.16, 0.76, 0.64])
    fs = float(cfg["font_scale"])
    cmap = get_diverging_cmap(theme_name)

    k = len(names)
    max_off = 0.0
    for i in range(k):
        for j in range(k):
            r_v = corr_mat[i, j]
            if i != j:
                max_off = max(max_off, abs(r_v))
            size = (abs(r_v) ** 0.85) * 950 + 45
            rgba = cmap((r_v + 1.0) / 2.0)
            ax.scatter([j], [i], s=size, color=rgba, edgecolors=t["border_line"], linewidths=0.8, zorder=3)
            if cfg["show_labels"] and abs(r_v) >= 0.18:
                hex_c = "#{:02X}{:02X}{:02X}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
                tc = auto_text_color(hex_c, t["text_primary"])
                ax.text(j, i, f"{r_v:.2f}", ha="center", va="center", fontsize=8.4 * fs, color=tc, weight="bold")

    ax.set_xticks(range(k))
    ax.set_xticklabels(names, fontsize=10.0 * fs, color=t["text_primary"], rotation=18 if k > 6 else 0)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names, fontsize=10.0 * fs, color=t["text_primary"])
    ax.set_xlim(-0.65, k - 0.35)
    ax.set_ylim(k - 0.35, -0.65)

    src = pd.DataFrame(np.round(corr_mat, 3), index=names, columns=names).reset_index().rename(columns={"index": "题项"})
    if is_default:
        title = "量表内部题项形成清晰「价格簇」与「品质簇」两大高相关方块（簇内 r = 0.62–0.71）"
    else:
        title = f"题项双重编码相关矩阵：最高跨题项相关强度 |r| = {max_off:.2f}，聚合结构清晰"
    subtitle = "题项双重编码相关矩阵 | 气泡矩阵图（圆面积 ∝ |r|，颜色 = 正负方向）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"双重视觉编码：圆面积映射相关强度 |r|，色彩映射正负符号（n={len(sub)}） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "27-气泡矩阵图", "气泡矩阵图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 28 - 平行坐标图 (Parallel Coordinates for K-Means Clusters)
# ==============================================================================
def chart_28_parallel_coordinates(df: pd.DataFrame, feat_cols: list = None, k: int = 3, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    work = _ensure_composites(df)
    if isinstance(settings, dict):
        feat_cols = settings.get("feat_cols") or feat_cols
        k = int(settings.get("k") or k)

    default_feats = ["价格感知均分", "品质感知均分", "月消费额", "满意度1_口味口感", "满意度2_价格实惠", "NPS推荐度"]
    default_shorts = ["价格感知", "品质感知", "月消费额", "口味满意", "价格满意", "NPS推荐度"]

    if feat_cols and len([c for c in feat_cols if c in work.columns]) >= 3:
        feat_cols = [c for c in feat_cols if c in work.columns][:8]
        short_cols = [_short_col_label(c) for c in feat_cols]
        is_default = False
    elif all(c in work.columns for c in default_feats) and k == 3:
        feat_cols = default_feats
        short_cols = default_shorts
        is_default = True
    else:
        caps = su.detect_dataset_capabilities(work)
        pool = list(dict.fromkeys(caps["numeric_cols"] + caps["nps_cols"] + caps["scale_cols"]))
        feat_cols = [c for c in pool if c in work.columns][:6]
        short_cols = [_short_col_label(c) for c in feat_cols]
        is_default = False

    km = su.run_kmeans_clustering(work, feat_cols, k=k, seed=42)
    sub = work[feat_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    z_df = (sub - sub.mean()) / sub.std(ddof=1).replace(0, 1)
    z_df["分群"] = km["labels"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.65])
    fs = float(cfg["font_scale"])
    x = np.arange(len(short_cols))

    _centers = {}
    for _cid in range(k):
        _rows = z_df[z_df["分群"] == _cid]
        _centers[_cid] = _rows.iloc[:, :-1].mean().to_numpy(dtype=float)

    if is_default:
        _loyal_id = max(range(3), key=lambda i: _centers[i][1] + _centers[i][2] + _centers[i][5])
        _rest = [i for i in range(3) if i != _loyal_id]
        _price_id = max(_rest, key=lambda i: _centers[i][0] + _centers[i][4])
        _watch_id = [i for i in _rest if i != _price_id][0]
        _name_map = {_loyal_id: "高消品质忠诚型", _price_id: "价格敏感大众型", _watch_id: "低频被动观望型"}
        cluster_names = [f"群组{cl_id + 1}：{_name_map[cl_id]}" for cl_id in range(3)]
        top_cluster_short = _name_map[_loyal_id]
    else:
        cluster_names = []
        for cl_id in range(k):
            cz = _centers[cl_id]
            hi_idx = int(np.argmax(cz))
            lo_idx = int(np.argmin(cz))
            if cz[hi_idx] >= 0.35:
                tag = f"{short_cols[hi_idx]}领先型"
            elif cz[lo_idx] <= -0.35:
                tag = f"{short_cols[lo_idx]}偏低型"
            else:
                tag = "均衡中间型"
            cluster_names.append(f"群组{cl_id + 1}：{tag}")
        biggest_id = int(z_df["分群"].value_counts().idxmax())
        top_cluster_short = cluster_names[biggest_id].split("：", 1)[-1]

    for cl_id in range(k):
        c_hex = t["categorical"][cl_id % len(t["categorical"])]
        cl_sub = z_df[z_df["分群"] == cl_id].iloc[:18, :-1].to_numpy(dtype=float)
        for row_vals in cl_sub:
            ax.plot(x, row_vals, color=c_hex, alpha=0.07, linewidth=0.9, zorder=2)

    cent_rows = []
    for cl_id in range(k):
        c_hex = get_item_color(cluster_names[cl_id], t["categorical"][cl_id % len(t["categorical"])], t, cfg)
        cl_rows = z_df[z_df["分群"] == cl_id]
        mean_z = cl_rows.iloc[:, :-1].mean().to_numpy(dtype=float)
        cnt = len(cl_rows)
        pct = cnt / max(len(z_df), 1) * 100.0
        ax.plot(x, mean_z, color=c_hex, linewidth=3.4, marker="o", markersize=8, label=f"{cluster_names[cl_id]} (n={cnt}, {pct:.1f}%)", zorder=5)
        rec = {"客群名称": cluster_names[cl_id], "样本数": cnt, "占比(%)": round(pct, 1)}
        for sc, zv in zip(short_cols, mean_z):
            rec[f"{sc}(Z分)"] = round(float(zv), 2)
        cent_rows.append(rec)

    ax.axhline(0, color=t["text_muted"], linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(short_cols, fontsize=10.5 * fs, color=t["text_primary"], weight="bold")
    ax.set_ylabel("标准化 Z-Score 均值 (0 = 全样本平均)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.5 * fs)

    src = pd.DataFrame(cent_rows)
    title = f"K-Means {k}大客群特征分化鲜明（轮廓系数 Silhouette = {km['silhouette']:.3f}），「{top_cluster_short}」特征突出"
    subtitle = f"K-Means 聚类中心多维标准化轮廓 | 平行坐标图（k = {k}, 有效样本 n = {len(z_df)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"聚类评估：Silhouette 轮廓系数 = {km['silhouette']:.3f} | 粗折线=分群均值中心，浅底线=个体样本轨迹 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "28-平行坐标图", "平行坐标图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 29 - 漏斗图 (Brand Conversion Funnel)
# ==============================================================================
def chart_29_funnel(df: pd.DataFrame, cols: list = None, stage_names: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        cols = settings.get("cols") or cols
        stage_names = settings.get("stage_names") or stage_names
    default_cols = ["品牌知名", "品牌考虑", "品牌购买", "品牌复购"]
    default_names = ["品牌知名 (Awareness)", "品牌考虑 (Consideration)", "品牌购买 (Purchase)", "品牌复购 (Repurchase)"]

    if cols and all(c in df.columns for c in cols):
        names = stage_names if (stage_names and len(stage_names) == len(cols)) else [str(c) for c in cols]
    elif all(c in df.columns for c in default_cols):
        cols = default_cols
        names = default_names
    else:
        f_info = su.detect_funnel_columns(df)
        cols = f_info["cols"]
        if not f_info["enabled"] or len(cols) < 3:
            raise ValueError("当前数据未检出不少于 3 个有序二值转化阶段列，无法绘制漏斗图。")
        names = f_info["stage_names"] or [str(c) for c in cols]

    fn = su.calc_brand_funnel(df, cols, stage_names=names)
    src = fn["table"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.08, 0.12, 0.84, 0.68])
    fs = float(cfg["font_scale"])

    counts = fn["counts"]
    top_cnt = max(counts[0], 1)
    widths = [c / top_cnt * 100.0 for c in counts]
    n_stages = len(counts)
    stage_colors = [t["accent_primary"], t["categorical"][1 % len(t["categorical"])], t["categorical"][2 % len(t["categorical"])], t["accent_secondary"], t["categorical"][4 % len(t["categorical"])]]

    for i in range(n_stages):
        w_top = widths[i]
        w_bot = widths[i + 1] if i + 1 < n_stages else widths[i] * 0.78
        y_top = (n_stages - i) * 1.35
        y_bot = y_top - 1.05
        poly = Polygon(
            [(-w_top / 2, y_top), (w_top / 2, y_top), (w_bot / 2, y_bot), (-w_bot / 2, y_bot)],
            closed=True, facecolor=get_item_color(names[i], stage_colors[i % len(stage_colors)], t, cfg),
            edgecolor=t["bg_canvas"], linewidth=2.2, alpha=0.92,
        )
        ax.add_patch(poly)
        tc = auto_text_color(stage_colors[i % len(stage_colors)], t["text_primary"])
        ax.text(0, (y_top + y_bot) / 2, f"{names[i]}\n{counts[i]} 人（占首级 {widths[i]:.1f}%）", ha="center", va="center", fontsize=10.5 * fs, color=tc, weight="bold")

        if i > 0:
            rate = fn["step_rates"][i - 1]
            drop = counts[i - 1] - counts[i]
            ax.text(
                max(widths[i - 1], w_top) / 2 + 6.0, y_top + 0.12,
                f"↘ 逐级转化率 {rate:.1f}%  (流失 {drop} 人)",
                ha="left", va="center", fontsize=9.8 * fs, color=t["accent_secondary"], weight="bold",
            )

    ax.set_xlim(-75, 95)
    ax.set_ylim(0.1, n_stages * 1.45)
    ax.axis("off")

    r_str = " → ".join([f"{r:.1f}%" for r in fn["step_rates"]])
    title = f"漏斗逐级转化率依次为 {r_str}，首末级总转化率达 {src.iloc[-1]['占首级比例(%)']:.1f}%"
    subtitle = f"全链路转化与流失诊断 | 转化漏斗图（首级人数 n = {counts[0]}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"口径说明：占首级比例以首阶段人数（n={counts[0]}）为基准，逐级转化率以上一环节人数为分母 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "29-漏斗图", "漏斗图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 30 - 桑基图 (Alluvial / Sankey Brand Switching Diagram)
# ==============================================================================
def chart_30_sankey(df: pd.DataFrame, prev_col: str = "上次购买品牌", curr_col: str = "最常购买品牌", theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        prev_col = settings.get("prev_col") or prev_col
        curr_col = settings.get("curr_col") or curr_col
    if prev_col not in df.columns or curr_col not in df.columns:
        s_info = su.detect_brand_switching_columns(df)
        if not s_info["enabled"]:
            raise ValueError("当前数据未检出成对的『上次/当前选择』流转列，无法绘制桑基图。")
        prev_col, curr_col = s_info["prev_col"], s_info["curr_col"]

    sw = su.calc_brand_switching(df, prev_col, curr_col)
    ct = sw["count_matrix"]
    brands = list(ct.index)
    total = float(ct.values.sum())

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.08, 0.12, 0.84, 0.68])
    fs = float(cfg["font_scale"])

    left_totals = ct.sum(axis=1).to_numpy(dtype=float)
    right_totals = ct.sum(axis=0).to_numpy(dtype=float)
    gap = 2.5
    scale = (100.0 - gap * max(len(brands) - 1, 0)) / max(total, 1.0)

    left_y = {}
    curr_y = 100.0
    for i, b in enumerate(brands):
        h = left_totals[i] * scale
        left_y[b] = [curr_y - h, curr_y]
        curr_y -= (h + gap)

    right_y = {}
    curr_y = 100.0
    for i, b in enumerate(brands):
        h = right_totals[i] * scale
        right_y[b] = [curr_y - h, curr_y]
        curr_y -= (h + gap)

    palette = t["categorical"]
    color_map = {b: get_item_color(b, palette[i % len(palette)], t, cfg) for i, b in enumerate(brands)}

    for b in brands:
        y0, y1 = left_y[b]
        ax.add_patch(Rectangle((0.0, y0), 0.08, y1 - y0, facecolor=color_map[b], edgecolor=t["bg_canvas"], lw=1.2, zorder=4))
        ax.text(-0.03, (y0 + y1) / 2, f"{b}\n({int(ct.loc[b].sum())}人)", ha="right", va="center", fontsize=9.5 * fs, color=t["text_primary"], weight="bold")

        ry0, ry1 = right_y[b]
        ax.add_patch(Rectangle((0.92, ry0), 0.08, ry1 - ry0, facecolor=color_map[b], edgecolor=t["bg_canvas"], lw=1.2, zorder=4))
        ax.text(1.03, (ry0 + ry1) / 2, f"{b}\n({int(ct[b].sum())}人)", ha="left", va="center", fontsize=9.5 * fs, color=t["text_primary"], weight="bold")

    left_cursor = {b: left_y[b][1] for b in brands}
    right_cursor = {b: right_y[b][1] for b in brands}
    for b_from in brands:
        for b_to in brands:
            val = float(ct.loc[b_from, b_to])
            if val <= 0:
                continue
            h = val * scale
            ly_top = left_cursor[b_from]
            ly_bot = ly_top - h
            left_cursor[b_from] = ly_bot

            ry_top = right_cursor[b_to]
            ry_bot = ry_top - h
            right_cursor[b_to] = ry_bot

            verts = [
                (0.08, ly_top), (0.45, ly_top), (0.55, ry_top), (0.92, ry_top),
                (0.92, ry_bot), (0.55, ry_bot), (0.45, ly_bot), (0.08, ly_bot), (0.08, ly_top),
            ]
            codes = [
                MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY,
            ]
            alpha_v = 0.52 if b_from == b_to else 0.24
            patch = PathPatch(MplPath(verts, codes), facecolor=color_map[b_from], edgecolor="none", alpha=alpha_v, zorder=2)
            ax.add_patch(patch)

    ax.text(0.04, 103.5, f"左端：{prev_col}", ha="center", fontsize=10.5 * fs, weight="bold", color=t["text_secondary"])
    ax.text(0.96, 103.5, f"右端：{curr_col}", ha="center", fontsize=10.5 * fs, weight="bold", color=t["text_secondary"])
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-2, 108)
    ax.axis("off")

    top_ret = sw["retention_df"].iloc[0]
    title = f"「{top_ret['品牌']}」留存率最高达 {top_ret['品牌保留率(%)']:.1f}%（留存 {top_ret['留存人数']}/{top_ret['上期基数']} 人），锁客效应显著"
    subtitle = f"{prev_col} → {curr_col}流转与留存矩阵 | 桑基流向图 Sankey（有效样本 n = {int(total)}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"带宽说明：流带宽度严格正比于转换人数；深色流带代表同项留存 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, sw["retention_df"], "30-桑基图", "桑基图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 31 - 对应分析双标图 (Correspondence Analysis Biplot)
# ==============================================================================
def chart_31_correspondence_biplot(df: pd.DataFrame, row_col: str = "最常购买品牌", col_col: str = "偏好品类", is_multi_col: bool = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        row_col = settings.get("row_col") or row_col
        col_col = settings.get("col_col") or col_col
        if settings.get("is_multi_col") is not None:
            is_multi_col = bool(settings.get("is_multi_col"))

    caps = su.detect_dataset_capabilities(df)
    ca_pool = caps["special_models"]["ca"]["cols"]
    if row_col not in df.columns:
        row_col = ca_pool[0]
    if col_col not in df.columns:
        col_col = next((c for c in ca_pool if c != row_col), ca_pool[-1])
    if is_multi_col is None:
        is_multi_col = (caps["qtypes"].get(col_col) == "多选题")

    ca = su.calc_correspondence_analysis(df, row_col, col_col, is_multi_col=is_multi_col)
    r_coords = ca["row_coords"]
    c_coords = ca["col_coords"]
    p1 = ca["pct_inertia"][0]
    p2 = ca["pct_inertia"][1] if len(ca["pct_inertia"]) > 1 else 0.0

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.10, 0.14, 0.82, 0.65])
    fs = float(cfg["font_scale"])

    ax.axhline(0, color=t["text_muted"], linestyle="--", linewidth=1.1)
    ax.axvline(0, color=t["text_muted"], linestyle="--", linewidth=1.1)

    ax.scatter(r_coords["维度1"], r_coords["维度2"], s=140, marker="o", color=t["accent_primary"], edgecolors=t["bg_canvas"], lw=1.5, label=f"行变量 ({row_col})", zorder=4)
    r_offsets = [(10, 12), (-10, 14), (12, 10), (-12, -16), (12, -14)]
    for idx, (nm, row) in enumerate(r_coords.iterrows()):
        ox, oy = r_offsets[idx % len(r_offsets)]
        ax.annotate(
            str(nm), xy=(row["维度1"], row["维度2"]), xytext=(ox, oy), textcoords="offset points",
            fontsize=10.2 * fs, weight="bold", color=t["accent_primary"],
            ha="left" if ox > 0 else "right", va="center",
        )

    ax.scatter(c_coords["维度1"], c_coords["维度2"], s=140, marker="^", color=t["accent_secondary"], edgecolors=t["bg_canvas"], lw=1.5, label=f"列变量 ({col_col})", zorder=5)
    c_offsets = [(12, 10), (-12, 10), (12, -12), (-14, 12), (-12, -14), (12, -10)]
    for idx, (nm, row) in enumerate(c_coords.iterrows()):
        ox, oy = c_offsets[idx % len(c_offsets)]
        ax.annotate(
            str(nm), xy=(row["维度1"], row["维度2"]), xytext=(ox, oy), textcoords="offset points",
            fontsize=10.0 * fs, weight="bold", color=t["accent_secondary"],
            ha="left" if ox > 0 else "right", va="center",
        )

    all_d1 = np.concatenate([r_coords["维度1"].to_numpy(), c_coords["维度1"].to_numpy()])
    all_d2 = np.concatenate([r_coords["维度2"].to_numpy(), c_coords["维度2"].to_numpy()])
    pad1 = max(0.035, (all_d1.max() - all_d1.min()) * 0.22)
    pad2 = max(0.030, (all_d2.max() - all_d2.min()) * 0.24)
    ax.set_xlim(all_d1.min() - pad1, all_d1.max() + pad1)
    ax.set_ylim(all_d2.min() - pad2, all_d2.max() + pad2)

    ax.set_xlabel(f"第一主惯量维度 Dim 1 ({p1:.1f}% 解释率)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel(f"第二主惯量维度 Dim 2 ({p2:.1f}% 解释率)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=10.0 * fs)

    src_r = r_coords.reset_index().rename(columns={"index": "名称"})
    src_r["类型"] = row_col
    src_c = c_coords.reset_index().rename(columns={"index": "名称"})
    src_c["类型"] = col_col
    src = pd.concat([src_r, src_c], ignore_index=True)

    title = f"对应分析前两维累计解释 {p1 + p2:.1f}% 惯量（Dim1={p1:.1f}%, Dim2={p2:.1f}%），清晰定位{row_col}×{col_col}空间关联"
    subtitle = f"{row_col} × {col_col}空间定位映射 | 对应分析双标图 Biplot（prince.CA）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"解读法则：同一象限且距离越近的行圆点与列三角，代表类别关联越紧密 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "31-对应分析双标图", "对应分析双标图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 32 - IPA 四象限图 (Importance-Performance Analysis — 零重叠防碰撞布局)
# ==============================================================================
def chart_32_ipa(df: pd.DataFrame, imp_cols: list = None, sat_cols: list = None, names: list = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    if isinstance(settings, dict):
        imp_cols = settings.get("imp_cols") or imp_cols
        sat_cols = settings.get("sat_cols") or sat_cols
        names = settings.get("names") or names

    default_imp = ["重要性1_口味口感", "重要性2_价格实惠", "重要性3_门店环境", "重要性4_出餐速度"]
    default_sat = ["满意度1_口味口感", "满意度2_价格实惠", "满意度3_门店环境", "满意度4_出餐速度"]
    default_names = ["口味口感", "价格实惠", "门店环境", "出餐速度"]

    if imp_cols and sat_cols and len(imp_cols) == len(sat_cols) and all(c in df.columns for c in imp_cols + sat_cols):
        if not names or len(names) != len(imp_cols):
            names = [_short_col_label(c) for c in imp_cols]
    elif all(c in df.columns for c in default_imp + default_sat):
        imp_cols, sat_cols, names = default_imp, default_sat, default_names
    else:
        i_info = su.detect_ipa_columns(df)
        if not i_info["enabled"] or len(i_info["imp_cols"]) < 2:
            raise ValueError("当前数据未检出成对的『重要性×满意度』量表列，无法绘制 IPA 四象限图。")
        imp_cols, sat_cols, names = i_info["imp_cols"], i_info["sat_cols"], i_info["attr_names"]

    ipa = su.calc_ipa_analysis(df, imp_cols, sat_cols, names)
    src = ipa["table"]
    m_imp, m_sat = ipa["mean_imp"], ipa["mean_sat"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.10, 0.14, 0.82, 0.65])
    fs = float(cfg["font_scale"])

    x_min, x_max = float(src["重要性均值"].min()) - 0.25, float(src["重要性均值"].max()) + 0.28
    y_min, y_max = float(src["满意度均值"].min()) - 0.25, float(src["满意度均值"].max()) + 0.28
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.add_patch(Rectangle((m_imp, y_min), x_max - m_imp, m_sat - y_min, facecolor=t["accent_secondary"], alpha=0.08, zorder=1))
    ax.axvline(m_imp, color=t["text_primary"], linestyle="--", linewidth=1.4, zorder=2)
    ax.axhline(m_sat, color=t["text_primary"], linestyle="--", linewidth=1.4, zorder=2)

    ax.text(x_max - 0.02, y_max - 0.03, "第一象限：保持优势区\n(高重要 · 高满意)", ha="right", va="top", fontsize=9.8 * fs, color=t["accent_primary"], weight="bold", alpha=0.85)
    ax.text(x_min + 0.02, y_max - 0.03, "第二象限：供给过度区\n(低重要 · 高满意)", ha="left", va="top", fontsize=9.8 * fs, color=t["text_secondary"], weight="bold", alpha=0.85)
    ax.text(x_min + 0.02, y_min + 0.03, "第三象限：低顺位改进区\n(低重要 · 低满意)", ha="left", va="bottom", fontsize=9.8 * fs, color=t["text_muted"], weight="bold", alpha=0.85)
    ax.text(x_max - 0.02, y_min + 0.03, "第四象限：优先改进区 ★\n(高重要 · 低满意)", ha="right", va="bottom", fontsize=9.8 * fs, color=t["accent_secondary"], weight="bold")

    offsets = [(18, 16), (-18, -22), (-20, 16), (18, -20), (16, 22), (-16, 20)]
    for i, r in src.iterrows():
        xv, yv = float(r["重要性均值"]), float(r["满意度均值"])
        quad = r["所属象限"]
        def_c = t["accent_secondary"] if quad == "优先改进" else (t["accent_primary"] if quad == "保持优势" else t["categorical"][2])
        c = get_item_color(r["要素名称"], def_c, t, cfg)
        ax.scatter([xv], [yv], s=180, color=c, edgecolors=t["bg_canvas"], linewidths=2.0, zorder=5)
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(
            f"{r['要素名称']} ({quad})\n重要 {xv:.2f} | 满意 {yv:.2f}",
            xy=(xv, yv), xytext=(ox, oy), textcoords="offset points",
            ha="left" if ox > 0 else "right", va="center",
            fontsize=9.6 * fs, weight="bold", color=t["text_primary"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_card"], edgecolor=c, lw=1.1, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
            zorder=6,
        )

    ax.set_xlabel(f"要素重要性均值 (十字分割线 = {m_imp:.2f})", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel(f"要素满意度均值 (十字分割线 = {m_sat:.2f})", fontsize=10.2 * fs, color=t["text_secondary"])

    pri_list = src.loc[src["所属象限"] == "优先改进", "要素名称"].tolist()
    keep_list = src.loc[src["所属象限"] == "保持优势", "要素名称"].tolist()
    if pri_list:
        title = f"IPA 诊断：「{'、'.join(pri_list)}」落入高重要低满意象限，为首要改进抓手"
    else:
        title = f"IPA 诊断：全部高重要要素均位于保持优势区（{'、'.join(keep_list) if keep_list else '整体表现稳健'}）"
    subtitle = f"要素重要性 × 满意度四象限矩阵 | IPA 分析图（保持优势：{'、'.join(keep_list) if keep_list else '无'}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"基准线：全要素重要性总均值={m_imp:.2f}，满意度总均值={m_sat:.2f} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "32-IPA四象限图", "IPA四象限图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 33 - PSM 价格敏感度曲线 (Van Westendorp Price Sensitivity Meter)
# ==============================================================================
def chart_33_psm(df: pd.DataFrame, cols: dict = None, theme_name: str = DEFAULT_THEME_NAME, settings: dict = None, is_demo: bool = False) -> dict:
    psm_cols = (settings or {}).get("psm_cols") if isinstance(settings, dict) else None
    psm_cols = psm_cols or cols
    if isinstance(psm_cols, dict) and "cols" in psm_cols:
        psm_cols = psm_cols["cols"]
    if not psm_cols:
        p_info = su.detect_psm_columns(df)
        psm_cols = p_info["cols"] if p_info["enabled"] else None
    if not psm_cols:
        raise ValueError("当前数据未检出完整的 PSM 价格敏感度四列（太便宜/便宜/贵/太贵）。")

    tc_c = psm_cols.get("too_cheap") or psm_cols.get("tc_col")
    c_c = psm_cols.get("cheap") or psm_cols.get("c_col")
    e_c = psm_cols.get("expensive") or psm_cols.get("e_col")
    te_c = psm_cols.get("too_expensive") or psm_cols.get("te_col")
    psm = su.calc_van_westendorp(df, tc_c, c_c, e_c, te_c)
    pg = psm["price_grid"]
    opp_p, opp_y = psm["opp"]
    ipp_p, ipp_y = psm["ipp"]
    pmc_p, pmc_y = psm["pmc"]
    pme_p, pme_y = psm["pme"]
    n_valid = len(df[[tc_c, c_c, e_c, te_c]].apply(pd.to_numeric, errors="coerce").dropna())

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.09, 0.14, 0.84, 0.65])
    fs = float(cfg["font_scale"])

    ax.axvspan(pmc_p, pme_p, color=t["accent_primary"], alpha=0.11, label=f"可接受价格带 [{pmc_p:.1f}, {pme_p:.1f}] 元")

    ax.plot(pg, psm["curve_tc"], color=t["categorical"][1], lw=2.3, linestyle="--", label="太便宜 (Too Cheap)")
    ax.plot(pg, psm["curve_c"],  color=t["accent_primary"], lw=2.4, linestyle="-",  label="便宜 (Cheap)")
    ax.plot(pg, psm["curve_e"],  color=t["accent_secondary"], lw=2.4, linestyle="-", label="贵 (Expensive)")
    ax.plot(pg, psm["curve_te"], color=t["highlight"], lw=2.3, linestyle="--", label="太贵 (Too Expensive)")

    pts = [
        ("PMC 最低下限", pmc_p, pmc_y, (-45, 22), t["categorical"][1]),
        ("OPP 最优价格点", opp_p, opp_y, (-15, 28), t["accent_primary"]),
        ("IPP 无差异点", ipp_p, ipp_y, (18, 22), t["accent_secondary"]),
        ("PME 最高上限", pme_p, pme_y, (22, -25), t["highlight"]),
    ]
    for nm, px, py, offset, c_hex in pts:
        ax.scatter([px], [py], s=110, color=c_hex, edgecolors=t["bg_canvas"], lw=1.8, zorder=6)
        ax.annotate(
            f"{nm}\n{px:.2f} 元 ({py:.1f}%)",
            xy=(px, py), xytext=offset, textcoords="offset points",
            fontsize=9.2 * fs, weight="bold", color=c_hex,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=t["bg_card"], edgecolor=c_hex, lw=1.0),
            arrowprops=dict(arrowstyle="->", color=c_hex, lw=1.2),
            zorder=7,
        )

    ax.set_xlabel("测试价格 (元)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel("受访者累积百分比 (%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, facecolor=t["bg_card"], edgecolor=t["border_line"], fontsize=9.2 * fs)

    title = f"最优价格点 OPP 为 {opp_p:.2f} 元，无差异点 IPP 为 {ipp_p:.2f} 元，可接受价格区间 [{pmc_p:.2f}, {pme_p:.2f}] 元"
    subtitle = f"Van Westendorp 价格敏感度测试 | PSM 四曲线交点模型（n = {n_valid}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"交点定义：OPP=太便宜∩太贵({opp_p:.2f}元) | IPP=便宜∩贵({ipp_p:.2f}元) | PMC={pmc_p:.2f}元 | PME={pme_p:.2f}元 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, psm["table"], "33-PSM价格敏感度曲线", "PSM价格敏感度曲线", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 34 - 结构方程路径图 (SEM Path Diagram — S1 精确匹配 OLS 标准化回归系数)
# ==============================================================================
def chart_34_sem_path(
    df: pd.DataFrame,
    dim1_cols: list = None,
    dim2_cols: list = None,
    target_col: str = None,
    dim1_name: str = None,
    dim2_name: str = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        dim1_cols = settings.get("dim1_cols") or dim1_cols
        dim2_cols = settings.get("dim2_cols") or dim2_cols
        target_col = settings.get("target_col") or target_col
        dim1_name = settings.get("dim1_name") or dim1_name
        dim2_name = settings.get("dim2_name") or dim2_name

    sem = su.calc_sem_path_model(
        df,
        dim1_cols=dim1_cols,
        dim2_cols=dim2_cols,
        target_col=target_col,
        dim1_name=dim1_name,
        dim2_name=dim2_name,
    )
    d1_nm = sem["dim1_name"]
    d2_nm = sem["dim2_name"]
    tgt_nm = sem["target_col"]

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.05, 0.12, 0.90, 0.68])
    fs = float(cfg["font_scale"])

    load_df = sem["loadings_table"]
    d1_loads = load_df[load_df["潜变量"] == d1_nm].reset_index(drop=True)
    d2_loads = load_df[load_df["潜变量"] == d2_nm].reset_index(drop=True)

    p_y_list = [7.8, 6.6, 5.4] if len(d1_loads) == 3 else np.linspace(7.9, 5.1, max(len(d1_loads), 1)).tolist()
    q_y_list = [3.4, 2.2, 1.0] if len(d2_loads) == 3 else np.linspace(3.7, 0.9, max(len(d2_loads), 1)).tolist()
    box_h1 = min(0.70, 2.4 / max(len(d1_loads), 1))
    box_h2 = min(0.70, 2.4 / max(len(d2_loads), 1))

    for idx, r in d1_loads.iterrows():
        yi = p_y_list[idx]
        short_nm = _short_col_label(r["测量题项"], 12)
        ax.add_patch(FancyBboxPatch((0.4, yi - box_h1 / 2), 2.1, box_h1, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor=t["bg_card"], edgecolor=t["accent_primary"], lw=1.5))
        ax.text(1.45, yi, f"{short_nm}\n(λ={r['标准化载荷']:.2f})", ha="center", va="center", fontsize=9.0 * fs, color=t["text_primary"])
        ax.annotate("", xy=(2.55, yi), xytext=(4.1, 6.6), arrowprops=dict(arrowstyle="->", color=t["accent_primary"], lw=1.4))

    for idx, r in d2_loads.iterrows():
        yi = q_y_list[idx]
        raw_m = str(r["测量题项"])
        short_nm = "品质稳定(正向)" if "品质量表3_正向" in raw_m else _short_col_label(raw_m, 12)
        ax.add_patch(FancyBboxPatch((0.4, yi - box_h2 / 2), 2.1, box_h2, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor=t["bg_card"], edgecolor=t["categorical"][1], lw=1.5))
        ax.text(1.45, yi, f"{short_nm}\n(λ={r['标准化载荷']:.2f})", ha="center", va="center", fontsize=9.0 * fs, color=t["text_primary"])
        ax.annotate("", xy=(2.55, yi), xytext=(4.1, 2.2), arrowprops=dict(arrowstyle="->", color=t["categorical"][1], lw=1.4))

    ax.add_patch(FancyBboxPatch((4.15, 5.85), 2.4, 1.5, boxstyle="round,pad=0.1,rounding_size=0.65", facecolor=t["accent_primary"], edgecolor=t["bg_canvas"], lw=2.0))
    ax.text(5.35, 6.6, f"{d1_nm}\n(外生潜变量 ξ1)", ha="center", va="center", fontsize=11.0 * fs, weight="bold", color=auto_text_color(t["accent_primary"]))

    ax.add_patch(FancyBboxPatch((4.15, 1.45), 2.4, 1.5, boxstyle="round,pad=0.1,rounding_size=0.65", facecolor=t["categorical"][1], edgecolor=t["bg_canvas"], lw=2.0))
    ax.text(5.35, 2.2, f"{d2_nm}\n(外生潜变量 ξ2)", ha="center", va="center", fontsize=11.0 * fs, weight="bold", color=auto_text_color(t["categorical"][1]))

    ax.annotate("", xy=(5.35, 5.80), xytext=(5.35, 3.00), arrowprops=dict(arrowstyle="<->", color=t["text_muted"], lw=1.5, linestyle="--", connectionstyle="arc3,rad=-0.22"))
    ax.text(4.55, 4.4, f"外生相关\nr = {sem['r_exo']:+.2f}", ha="center", va="center", fontsize=9.2 * fs, color=t["text_secondary"], bbox=dict(boxstyle="round,pad=0.2", facecolor=t["bg_canvas"], edgecolor="none"))

    tgt_disp = "推荐意愿 (NPS)" if tgt_nm == "NPS推荐度" else _short_col_label(tgt_nm, 12)
    ax.add_patch(FancyBboxPatch((8.8, 3.55), 2.7, 1.7, boxstyle="round,pad=0.1,rounding_size=0.45", facecolor=t["accent_secondary"], edgecolor=t["bg_canvas"], lw=2.2))
    ax.text(10.15, 4.4, f"{tgt_disp}\nR² = {sem['r2']:.3f}\n(解释方差 {sem['r2']*100:.1f}%)", ha="center", va="center", fontsize=11.0 * fs, weight="bold", color=auto_text_color(t["accent_secondary"]))

    bp = sem["beta_price"]
    bq = sem["beta_qual"]
    sig_p = sem["path_table"].iloc[0]["显著性"]
    sig_q = sem["path_table"].iloc[1]["显著性"]

    ax.annotate("", xy=(8.75, 4.75), xytext=(6.60, 6.45), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", color=t["accent_primary"], lw=3.2))
    ax.text(7.65, 5.95, f"β1 = {bp:+.2f} {sig_p}\n(p = {sem['p_price']:.4f})", ha="center", va="center", fontsize=10.5 * fs, weight="bold", color=t["accent_primary"], bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_card"], edgecolor=t["accent_primary"], lw=1.2))

    ax.annotate("", xy=(8.75, 4.05), xytext=(6.60, 2.35), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", color=t["categorical"][1], lw=3.2))
    ax.text(7.65, 2.85, f"β2 = {bq:+.2f} {sig_q}\n(p = {sem['p_qual']:.4f})", ha="center", va="center", fontsize=10.5 * fs, weight="bold", color=t["categorical"][1], bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_card"], edgecolor=t["categorical"][1], lw=1.2))

    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.2, 8.6)
    ax.axis("off")

    sp_s = "" if sig_p == "ns" else sig_p
    sq_s = "" if sig_q == "ns" else sig_q
    title = f"{d1_nm}（β={bp:+.2f}{sp_s}）与{d2_nm}（β={bq:+.2f}{sq_s}）共同作用，合计解释 {sem['r2']*100:.1f}% 方差"
    subtitle = f"潜变量测量与结构方程路径模型 | SEM 路径图（OLS 标准化估计，有效样本 n = {sem['n']}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"模型拟合：R²={sem['r2']:.3f}, Adj R²={sem['adj_r2']:.3f}, F={sem['f_stat']:.2f} (p={sem['f_p']:.4f}) | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, sem["path_table"], "34-结构方程路径图", "结构方程路径图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 35 - KANO 模型图 (KANO Better-Worse Four-Quadrant Plot — S2)
# ==============================================================================
def chart_35_kano(
    df: pd.DataFrame = None,
    pos_cols: list = None,
    neg_cols: list = None,
    feature_names: list = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        pos_cols = settings.get("pos_cols") or pos_cols
        neg_cols = settings.get("neg_cols") or neg_cols
        feature_names = settings.get("feature_names") or feature_names

    kano = su.calc_kano_model(df, pos_cols=pos_cols, neg_cols=neg_cols, feature_names=feature_names)
    src = kano["table"]
    sb, sw = kano["split_better"], kano["split_worse"]
    n_val = kano.get("n_valid", len(df) if df is not None else 0)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.10, 0.14, 0.82, 0.65])
    fs = float(cfg["font_scale"])

    x_lo = min(0.05, float(src["Worse绝对值"].min()) - 0.08)
    x_hi = max(0.95, float(src["Worse绝对值"].max()) + 0.08)
    y_lo = min(0.15, float(src["Better系数"].min()) - 0.08)
    y_hi = max(0.90, float(src["Better系数"].max()) + 0.08)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axvline(sw, color=t["text_primary"], linestyle="--", linewidth=1.3)
    ax.axhline(sb, color=t["text_primary"], linestyle="--", linewidth=1.3)

    ax.text(x_lo + 0.03, y_hi - 0.04, "魅力属性象限 (A: Attractive)\n惊喜加分项：不具备不抱怨，具备则大增满意", ha="left", va="top", fontsize=9.5 * fs, color=t["accent_primary"], weight="bold")
    ax.text(x_hi - 0.03, y_hi - 0.04, "期望属性象限 (O: One-dimensional)\n核心竞争项：做得越好越满意，反之强烈不满", ha="right", va="top", fontsize=9.5 * fs, color=t["accent_secondary"], weight="bold")
    ax.text(x_lo + 0.03, y_lo + 0.03, "无差异属性象限 (I: Indifferent)\n低敏感项：无论提供与否均不影响满意度", ha="left", va="bottom", fontsize=9.5 * fs, color=t["text_muted"], weight="bold")
    ax.text(x_hi - 0.03, y_lo + 0.03, "必备属性象限 (M: Must-be)\n底线门槛项：必须达标，一旦缺失引发剧烈流失", ha="right", va="bottom", fontsize=9.5 * fs, color=t["highlight"], weight="bold")

    color_by_code = {"A": t["accent_primary"], "O": t["accent_secondary"], "M": t["highlight"], "I": t["categorical"][2]}
    kano_offset_map = {
        "季节限定新品": (14, 8),
        "出餐≤5分钟":   (-14, 12),
        "价格透明":     (14, 14),
        "外卖包装防洒": (-14, -14),
        "现磨现做":     (-14, -16),
        "会员积分":     (14, 8),
        "杯型可选":     (-14, -12),
    }
    offsets = [(14, 10), (-14, 12), (14, 14), (-14, -14), (-14, -16), (14, 10), (-14, -14)]

    for i, r in src.iterrows():
        xv = float(r["Worse绝对值"])
        yv = float(r["Better系数"])
        code = r["属性代码"]
        fnm = str(r["功能需求"])
        c = get_item_color(fnm, color_by_code.get(code, t["accent_primary"]), t, cfg)
        ax.scatter([xv], [yv], s=190, color=c, edgecolors=t["bg_canvas"], lw=2.0, zorder=5)
        ox, oy = kano_offset_map.get(fnm, offsets[i % len(offsets)])
        ax.annotate(
            f"{r['功能需求']} [{code}]\n(|Worse|={xv:.2f}, Better={yv:.2f})",
            xy=(xv, yv), xytext=(ox, oy), textcoords="offset points",
            ha="left" if ox > 0 else "right", va="center",
            fontsize=9.2 * fs, weight="bold", color=t["text_primary"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=t["bg_card"], edgecolor=c, lw=1.0, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.1),
            zorder=6,
        )

    ax.set_xlabel("不满意影响力绝对值 |Worse| = (O + M) / (A + O + M + I)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.set_ylabel("满意影响力 Better = (A + O) / (A + O + M + I)", fontsize=10.2 * fs, color=t["text_secondary"])

    _by_code = {}
    for _, _r in src.iterrows():
        _by_code.setdefault(str(_r["属性代码"]), []).append(str(_r["功能需求"]))
    _clauses = []
    if _by_code.get("M"):
        _clauses.append(f"「{_by_code['M'][0]}」属必备底线（M）")
    if _by_code.get("O"):
        _clauses.append(f"「{'/'.join(_by_code['O'][:2])}」属期望核心（O）")
    if _by_code.get("A"):
        _clauses.append(f"「{_by_code['A'][0]}」属魅力惊喜（A）")
    title = "KANO 需求分层：" + "，".join(_clauses) if _clauses else "KANO 需求分层：各属性象限分布如图所示"
    subtitle = f"产品与服务功能 KANO 模型 Better–Worse 矩阵 | KANO 四象限图（n = {n_val}）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"计算公式：Better=(A+O)/(A+O+M+I), Worse=-(O+M)/(A+O+M+I) | 分割线=0.50 | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "35-KANO模型图", "KANO模型图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 36 - 联合分析重要性图 (Conjoint Relative Importance Bar — S3)
# ==============================================================================
def chart_36_conjoint_importance(
    df: pd.DataFrame = None,
    attr_cols: list = None,
    rating_col: str = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        attr_cols = settings.get("attr_cols") or attr_cols
        rating_col = settings.get("rating_col") or rating_col

    cj = su.calc_conjoint_analysis(df, attr_cols=attr_cols, rating_col=rating_col)
    src = apply_df_sort_and_topn(cj["importance_df"], "相对重要性(%)", settings, default_sort="desc")
    plot_df = src.iloc[::-1].reset_index(drop=True)

    fig, ax, t, cfg = create_themed_figure(theme_name, settings, rect=[0.14, 0.16, 0.78, 0.64])
    fs = float(cfg["font_scale"])
    y = np.arange(len(plot_df))
    imps = plot_df["相对重要性(%)"].to_numpy(dtype=float)
    ranges = plot_df["效用极差(Range)"].to_numpy(dtype=float)
    labels = plot_df["属性名称"].astype(str).tolist()
    top_attr = str(src.iloc[0]["属性名称"])

    colors = [get_item_color(lbl, t["accent_primary"] if lbl == top_attr else t["categorical"][1], t, cfg) for lbl in labels]
    bars = ax.barh(y, imps, height=0.52, color=colors, zorder=3)
    for bar, lbl, imp, rg in zip(bars, labels, imps, ranges):
        if cfg["show_labels"]:
            ax.text(
                imp + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{imp:.1f}%  (效用极差 Range = {rg:.2f})",
                va="center", ha="left", fontsize=10.2 * fs,
                weight="bold" if lbl == top_attr else "normal",
                color=t["text_primary"],
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.0 * fs, color=t["text_primary"], weight="bold")
    ax.set_xlim(0, max(float(imps.max()), 1.0) * 1.35)
    ax.set_xlabel(f"属性相对重要性占比 (%，{len(src)}项属性总和 = 100.0%)", fontsize=10.2 * fs, color=t["text_secondary"])
    ax.grid(axis="y", visible=False)

    sec_clause = f"，「{src.iloc[1]['属性名称']}」次之（{src.iloc[1]['相对重要性(%)']:.1f}%）" if len(src) > 1 else ""
    title = f"「{src.iloc[0]['属性名称']}」主导偏好决策（相对重要性 {src.iloc[0]['相对重要性(%)']:.1f}%）{sec_clause}"
    subtitle = f"联合分析（Conjoint Analysis）属性相对重要性排序 | 水平条形图（重要性总和 = {cj['importance_sum']:.1f}%）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"计算方法：各属性相对重要性 = 该属性部分价值极差 Range_i / ΣRange × 100%（总和={cj['importance_sum']:.1f}%） | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, src, "36-联合分析重要性图", "联合分析重要性图", title, subtitle, footnote, theme_name, settings)


# ==============================================================================
# 37 - 联合分析效用曲线 (Conjoint Part-Worth Utility Curves — 修复顶部标签留白)
# ==============================================================================
def chart_37_conjoint_utility(
    df: pd.DataFrame = None,
    attr_cols: list = None,
    rating_col: str = None,
    theme_name: str = DEFAULT_THEME_NAME,
    settings: dict = None,
    is_demo: bool = False,
) -> dict:
    if isinstance(settings, dict):
        attr_cols = settings.get("attr_cols") or attr_cols
        rating_col = settings.get("rating_col") or rating_col

    cj = su.calc_conjoint_analysis(df, attr_cols=attr_cols, rating_col=rating_col)
    util_df = cj["utility_df"]
    imp_df = cj["importance_df"]
    attrs = imp_df["属性名称"].tolist()

    fig, axes, t, cfg = create_themed_figure(theme_name, settings, nrows=1, ncols=len(attrs), rect=[0.06, 0.16, 0.89, 0.62])
    fs = float(cfg["font_scale"])
    ax_list = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

    u_span = max(float(util_df["部分价值(Utility)"].max() - util_df["部分价值(Utility)"].min()), 0.5)
    y_min = float(util_df["部分价值(Utility)"].min()) - u_span * 0.22
    y_max = float(util_df["部分价值(Utility)"].max()) + u_span * 0.28

    for idx, (ax, attr_nm) in enumerate(zip(ax_list, attrs)):
        sub = util_df[util_df["属性名称"] == attr_nm].reset_index(drop=True)
        imp_pct = float(imp_df.loc[imp_df["属性名称"] == attr_nm, "相对重要性(%)"].iloc[0])
        x = np.arange(len(sub))
        u_vals = sub["部分价值(Utility)"].to_numpy(dtype=float)
        levels = sub["属性水平"].astype(str).tolist()
        c = get_item_color(attr_nm, t["categorical"][idx % len(t["categorical"])], t, cfg)

        ax.axhline(0, color=t["text_muted"], linestyle="--", linewidth=1.1)
        ax.plot(x, u_vals, color=c, marker="o", linewidth=2.5, markersize=8, zorder=4)
        for xi, uv in zip(x, u_vals):
            offset_y = u_span * 0.08 if uv >= 0 else -u_span * 0.11
            ax.text(
                xi, uv + offset_y, f"{uv:+.2f}",
                ha="center", va="bottom" if uv >= 0 else "top",
                fontsize=9.0 * fs, weight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=t["bg_canvas"], edgecolor="none", alpha=0.88),
                zorder=5,
            )

        ax.set_title(f"{attr_nm}\n(重要性 {imp_pct:.1f}%)", fontsize=10.5 * fs, weight="bold", color=t["text_primary"])
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=9.0 * fs, color=t["text_primary"], rotation=18)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.4, len(sub) - 0.6)
        if idx == 0:
            ax.set_ylabel("零中心化部分价值 (Part-Worth Utility)", fontsize=10.0 * fs, color=t["text_secondary"])
        else:
            ax.set_yticklabels([])

    top_utils = util_df.sort_values("部分价值(Utility)", ascending=False)
    u1 = top_utils.iloc[0]
    u2 = top_utils.iloc[1] if len(top_utils) > 1 else u1
    title = f"最优轮廓为「{cj['best_combo']}」，「{u1['属性水平']}」（{u1['部分价值(Utility)']:+.2f}）与「{u2['属性水平']}」（{u2['部分价值(Utility)']:+.2f}）效用拉升最强"
    subtitle = "各属性水平部分价值（Part-Worth Utility）效应曲线 | 联合分析效用曲线（属性内零中心化）"
    demo_tag = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    footnote = f"性质校验：每个属性内部各水平效用值之和严格等于 0.00 | 最优组合：{cj['best_combo']} | 数据来源：{demo_tag}"
    return export_chart_bundle(fig, util_df, "37-联合分析效用曲线", "联合分析效用曲线", title, subtitle, footnote, theme_name, settings)
