# -*- coding: utf-8 -*-
"""
SurveyScope 统计与数据处理核心模块 (stats_utils.py)
全部统计计算、数据质控、变量加工、演示数据生成及市调特色算法均集中在本文件。
"""

import io
import re
import json
import inspect
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_curve, auc, roc_auc_score
import jieba
import prince

# 兼容 factor_analyzer 与新版 scikit-learn (check_array 参数 force_all_finite -> ensure_all_finite)
import factor_analyzer.factor_analyzer as _fa_mod
import factor_analyzer.rotator as _rot_mod
from sklearn.utils.validation import check_array as _sk_check_array

def _compat_check_array(*args, **kwargs):
    sig_params = inspect.signature(_sk_check_array).parameters
    if "force_all_finite" in kwargs and "force_all_finite" not in sig_params:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _sk_check_array(*args, **kwargs)

_fa_mod.check_array = _compat_check_array
_rot_mod.check_array = _compat_check_array

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


# ==============================================================================
# 模块一：数据上传与题型识别
# ==============================================================================

def load_survey_file(uploaded_file, filename=None) -> pd.DataFrame:
    """读取 CSV 或 Excel 问卷数据文件，自动处理中文编码，兼容问卷星导出格式。

    uploaded_file 可为：本地路径 str / Streamlit 上传文件对象 / 原始 bytes
    （bytes 时用 filename 判断扩展名，app.py 的上传入口即此用法）。
    """
    if isinstance(uploaded_file, bytes):
        _bio = io.BytesIO(uploaded_file)
        _bio.name = filename or "data.csv"
        uploaded_file = _bio
    if isinstance(uploaded_file, str):
            if uploaded_file.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(uploaded_file)
            _last_err = None
            for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
                try:
                    return pd.read_csv(uploaded_file, encoding=enc)
                except Exception as _e:
                    _last_err = _e
                    continue
            raise ValueError(f"无法解析 CSV 文件：可能是编码（请用 UTF-8 或 GBK 保存）或列数不一致等格式问题。原始错误：{_last_err}")
    name = getattr(uploaded_file, "name", "data.csv").lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes))
    _last_err = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except Exception as _e:
            _last_err = _e
            continue
    raise ValueError(f"无法解析上传的 CSV 文件：可能是编码（请用 UTF-8 或 GBK 保存）或列数不一致等格式问题。原始错误：{_last_err}")


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """返回样本量、题目数以及每道题的缺失数和缺失率表。"""
    n_samples = int(len(df))
    n_questions = int(df.shape[1])
    missing_rows = []
    for col in df.columns:
        s = df[col]
        # 空字符串视同缺失
        is_na = s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None", "NULL"]))
        n_miss = int(is_na.sum())
        miss_rate = round(n_miss / n_samples * 100, 2) if n_samples > 0 else 0.0
        missing_rows.append({
            "题目名称": col,
            "非空样本数": n_samples - n_miss,
            "缺失样本数": n_miss,
            "缺失率(%)": miss_rate,
        })
    missing_df = pd.DataFrame(missing_rows)
    return {
        "n_samples": n_samples,
        "n_questions": n_questions,
        "missing_df": missing_df,
    }


def auto_detect_question_types(df: pd.DataFrame) -> dict:
    """
    自动识别每道题的题型：
    数值题 / 单选题 / 量表题（1-7整数）/ 多选题（含「；」或「、」分隔）/ 开放文本题 / NPS题（0-10整数）
    问卷星导出文件自带的元信息列（序号/提交答卷时间/所用时间/来源/来源详情/来自IP 等，
    精确列名匹配）先行识别为「元信息」，不计入题目，避免被误判为单选/文本题参与分析。
    """
    _WJX_META_COLS = {"序号", "提交答卷时间", "所用时间", "来源", "来源详情", "来自IP", "IP", "答卷编号", "提交时间"}
    q_types = {}
    for col in df.columns:
        col_str = str(col)
        if col_str.strip() in _WJX_META_COLS:
            q_types[col] = "元信息"
            continue
        s = df[col].dropna()
        if len(s) == 0:
            q_types[col] = "单选题"
            continue

        # 优先检查字符串分隔的多选题与开放文本题
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            str_vals = s.astype(str).str.strip()
            str_vals = str_vals[~str_vals.isin(["", "nan", "None"])]
            if len(str_vals) == 0:
                q_types[col] = "单选题"
                continue
            avg_len = str_vals.str.len().mean()
            nunique_ratio = str_vals.nunique() / max(len(str_vals), 1)
            has_sent_punct = str_vals.str.contains(r"[，。！？!?]", regex=True).mean()
            is_open_kw = any(k in col_str for k in ["短评", "建议", "开放", "留言", "原因", "看法", "评论", "主观", "补充"])
            if ("多选" not in col_str) and (
                is_open_kw
                or (has_sent_punct >= 0.25 and avg_len >= 8)
                or (avg_len >= 18 and nunique_ratio > 0.5)
            ):
                q_types[col] = "开放文本题"
                continue
            has_sep = str_vals.str.contains(r"[；;、]", regex=True).mean()
            if has_sep >= 0.15 or ("多选" in col_str and has_sep > 0):
                q_types[col] = "多选题"
                continue
            # 检查开放文本题（唯一值比例高或平均字数较长）
            if avg_len >= 10 or (avg_len >= 6 and nunique_ratio > 0.4):
                q_types[col] = "开放文本题"
                continue
            q_types[col] = "单选题"
            continue

        # 数值型列判断
        num_vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(num_vals) == 0:
            q_types[col] = "单选题"
            continue

        is_int = np.all(np.isclose(num_vals, np.round(num_vals)))
        vmin, vmax = float(num_vals.min()), float(num_vals.max())
        nunique = num_vals.nunique()

        # NPS 题识别：列名含 NPS/推荐度，或取值为 0-10 的整数且跨度覆盖 0-10
        if is_int and vmin >= 0 and vmax <= 10:
            if any(k in col_str.upper() for k in ["NPS", "推荐", "净推荐"]):
                q_types[col] = "NPS题"
                continue
            if vmin == 0 and vmax >= 8 and nunique >= 6:
                q_types[col] = "NPS题"
                continue

        # 二值题 (0/1) 识别为单选题
        if is_int and set(num_vals.astype(int).unique()).issubset({0, 1}):
            q_types[col] = "单选题"
            continue

        # 量表题识别：取值为 1-7 的整数
        if is_int and vmin >= 1 and vmax <= 7 and nunique >= 2:
            if any(k in col_str for k in ["时长", "年龄", "消费", "金额", "价格敏感"]):
                q_types[col] = "数值题"
            else:
                q_types[col] = "量表题"
            continue

        q_types[col] = "数值题"

    return q_types


def split_multiple_choice(series: pd.Series, sep_pattern: str = r"[；;、,]") -> dict:
    """
    A5 多选拆分：把一格内以「；」或「、」分隔的多选题自动拆分为选项 0/1 矩阵，并按选项统计频数与占比。
    """
    clean_s = series.dropna().astype(str).str.strip()
    clean_s = clean_s[~clean_s.isin(["", "nan", "None"])]
    n_valid = len(clean_s)

    option_counter = Counter()
    option_order = []
    parsed_rows = []

    for val in clean_s:
        parts = [p.strip() for p in re.split(sep_pattern, val) if p.strip()]
        # 同一受访者同一选项去重
        unique_parts = list(dict.fromkeys(parts))
        parsed_rows.append(unique_parts)
        for p in unique_parts:
            option_counter[p] += 1
            if p not in option_order:
                option_order.append(p)

    # 构建 0/1 矩阵
    dummy_dict = {opt: [] for opt in option_order}
    for parts in parsed_rows:
        part_set = set(parts)
        for opt in option_order:
            dummy_dict[opt].append(1 if opt in part_set else 0)
    dummy_df = pd.DataFrame(dummy_dict, index=clean_s.index) if option_order else pd.DataFrame()

    total_mentions = sum(option_counter.values())
    rows = []
    for opt, cnt in sorted(option_counter.items(), key=lambda x: (-x[1], option_order.index(x[0]))):
        resp_pct = (cnt / n_valid * 100.0) if n_valid > 0 else 0.0
        case_pct = (cnt / total_mentions * 100.0) if total_mentions > 0 else 0.0
        p_hat = cnt / n_valid if n_valid > 0 else 0.0
        se = np.sqrt(p_hat * (1.0 - p_hat) / n_valid) if n_valid > 0 else 0.0
        ci_low = max(0.0, (p_hat - 1.96 * se) * 100.0)
        ci_high = min(100.0, (p_hat + 1.96 * se) * 100.0)
        rows.append({
            "选项": opt,
            "频数": int(cnt),
            "选中次数": int(cnt),
            "受访者占比(%)": round(resp_pct, 2),
            "普及率(% of Cases)": round(resp_pct, 2),
            "选项提及占比(%)": round(case_pct, 2),
            "响应占比(% of Responses)": round(case_pct, 2),
            "95%CI下限(%)": round(ci_low, 2),
            "95%CI上限(%)": round(ci_high, 2),
        })

    summary_df = pd.DataFrame(rows)
    top_opt_str = f"「{summary_df.iloc[0]['选项']}」普及率最高（{summary_df.iloc[0]['受访者占比(%)']:.1f}%）" if not summary_df.empty else "无有效数据"
    return {
        "counts": dict(option_counter),
        "n_valid": n_valid,
        "n_cases": n_valid,
        "total_mentions": total_mentions,
        "total_responses": total_mentions,
        "dummy_df": dummy_df,
        "summary_df": summary_df,
        "freq_df": summary_df,
        "conclusion": f"多选题有效作答人数 n={n_valid}（总响应 {total_mentions} 人次），{top_opt_str}。",
    }


# ==============================================================================
# 模块二：数据质控中心
# ==============================================================================

def detect_duplicate_responses(df: pd.DataFrame, cols: list = None) -> pd.Series:
    """重复作答识别：全部题目答案完全相同的受访者标记为 True。"""
    _meta = {"序号", "提交答卷时间", "所用时间", "来源", "来源详情", "来自IP", "IP", "答卷编号", "提交时间"}
    if cols:
        target_cols = [c for c in cols if c in df.columns]
    else:
        non_meta = [c for c in df.columns if str(c).strip() not in _meta]
        target_cols = non_meta if non_meta else list(df.columns)
    if not target_cols or len(df) == 0:
        return pd.Series(False, index=df.index)
    return df[target_cols].duplicated(keep=False)


def detect_straightlining(df: pd.DataFrame, scale_cols: list) -> pd.Series:
    """
    A12 直线作答识别：选定一组量表题，受访者在这组题上的回答标准差为 0（全部选同一选项）即标记为 True。
    """
    if not scale_cols or len(scale_cols) < 2:
        return pd.Series(False, index=df.index)
    sub = df[scale_cols].apply(pd.to_numeric, errors="coerce")
    # 至少回答了2道题且标准差为0
    valid_cnt = sub.notna().sum(axis=1)
    row_std = sub.std(axis=1, ddof=0)
    return (valid_cnt >= 2) & np.isclose(row_std.fillna(1.0), 0.0)


def detect_speeders(df: pd.DataFrame, duration_col: str, min_seconds: float = 60.0) -> pd.Series:
    """答题时长筛查：答题时间低于设定秒数者标记为 True（兼容纯数字与问卷星「125秒」格式）。"""
    if not duration_col or duration_col not in df.columns:
        return pd.Series(False, index=df.index)
    raw_s = df[duration_col]
    dur = pd.to_numeric(raw_s, errors="coerce")
    if dur.notna().sum() == 0 and raw_s.notna().sum() > 0:
        extracted = raw_s.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        dur = pd.to_numeric(extracted, errors="coerce")
    return dur < float(min_seconds)


def run_quality_control(
    df: pd.DataFrame,
    check_dup: bool = True,
    dup_cols: list = None,
    check_straight: bool = True,
    straight_cols: list = None,
    check_speed: bool = True,
    duration_col: str = None,
    min_seconds: float = 60.0,
    remove_flagged: bool = True,
) -> dict:
    """综合执行数据质控，输出每条规则标记、汇总指标、自动跳过原因与质控后有效样本集。"""
    total_n = len(df)
    skipped_rules = []
    valid_straight_cols = [c for c in (straight_cols or []) if c in df.columns]
    has_straight = bool(check_straight and len(valid_straight_cols) >= 2)
    if check_straight and len(valid_straight_cols) < 2:
        skipped_rules.append("直线作答检测：当前数据集量表题不足 2 道，已自动跳过")

    has_speed = bool(check_speed and duration_col and duration_col in df.columns)
    if check_speed and (not duration_col or duration_col not in df.columns):
        skipped_rules.append("作答过快检测：当前数据集不包含答题时长列，已自动跳过")

    dup_flag = detect_duplicate_responses(df, dup_cols) if check_dup else pd.Series(False, index=df.index)
    straight_flag = detect_straightlining(df, valid_straight_cols) if has_straight else pd.Series(False, index=df.index)
    speed_flag = detect_speeders(df, duration_col, min_seconds) if has_speed else pd.Series(False, index=df.index)

    any_flag = dup_flag | straight_flag | speed_flag
    valid_df = df.loc[~any_flag].copy() if remove_flagged else df.copy()

    dup_n = int(dup_flag.sum())
    straight_n = int(straight_flag.sum())
    speed_n = int(speed_flag.sum())
    flagged_n = int(any_flag.sum())
    valid_n = int(len(valid_df))
    valid_rate = round(valid_n / total_n * 100.0, 2) if total_n > 0 else 0.0

    flags_df = pd.DataFrame({
        "重复作答": dup_flag,
        "直线作答": straight_flag,
        "作答过快": speed_flag,
        "异常汇总标记": any_flag,
        "是否异常样本": any_flag,
    }, index=df.index)

    skip_note = f"（注：{'；'.join(skipped_rules)}）" if skipped_rules else ""
    conclusion = (
        f"原始总样本量为 {total_n} 份，其中识别出重复作答 {dup_n} 份、直线作答 {straight_n} 份、"
        f"作答过快 {speed_n} 份（合计异常样本 {flagged_n} 份）；"
        f"{'剔除后' if remove_flagged else '当前保留全部样本，'}有效样本量为 {valid_n} 份，问卷有效率为 {valid_rate:.1f}%。{skip_note}"
    )

    return {
        "total_n": total_n,
        "dup_n": dup_n,
        "straight_n": straight_n,
        "speed_n": speed_n,
        "flagged_n": flagged_n,
        "valid_n": valid_n,
        "valid_rate": valid_rate,
        "skipped_rules": skipped_rules,
        "flags_df": flags_df,
        "marked_df": flags_df,
        "valid_df": valid_df,
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块三：变量加工
# ==============================================================================

def recode_categories(series: pd.Series, mapping_dict: dict) -> pd.Series:
    """选项合并/重编码：将原选项映射为新类别，未在映射字典中的保持原值。"""
    return series.map(lambda x: mapping_dict.get(x, mapping_dict.get(str(x), x)) if pd.notna(x) else np.nan)


def bin_numeric(series: pd.Series, cut_points: list, labels: list = None, right: bool = True) -> pd.Series:
    """数值分箱：按用户给定的分界点将数值变量切分为分组变量。"""
    num_s = pd.to_numeric(series, errors="coerce")
    sorted_pts = sorted([float(x) for x in cut_points])
    bins = [-np.inf] + sorted_pts + [np.inf]
    if not labels or len(labels) != len(bins) - 1:
        auto_labels = []
        for i in range(len(bins) - 1):
            if i == 0:
                auto_labels.append(f"≤{sorted_pts[0]:g}")
            elif i == len(bins) - 2:
                auto_labels.append(f">{sorted_pts[-1]:g}")
            else:
                auto_labels.append(f"{sorted_pts[i-1]:g}-{sorted_pts[i]:g}")
        labels = auto_labels
    return pd.cut(num_s, bins=bins, labels=labels, right=right).astype(object)


def reverse_score(series_or_list, max_scale: int = 5, min_scale: int = 1):
    """
    A13 反向计分：新值 = (量表级数 + 1) - 原值（当最小值为 1 时，即 max_scale + min_scale - 原值）。
    """
    if isinstance(series_or_list, list):
        return [(max_scale + min_scale) - x for x in series_or_list]
    num_s = pd.to_numeric(series_or_list, errors="coerce")
    return (max_scale + min_scale) - num_s


def compute_scale_composite(df: pd.DataFrame, cols: list, agg: str = "mean") -> pd.Series:
    """量表合成：勾选多道题，计算维度均分作为新变量。"""
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    if agg == "sum":
        return sub.sum(axis=1, min_count=1)
    return sub.mean(axis=1)


# ==============================================================================
# 模块四：单题分析
# ==============================================================================

def calc_max_sampling_error(n: int, p: float = 0.5, confidence: float = 0.95) -> float:
    """
    A10 抽样误差计算：
    公式：E = z_{α/2} * sqrt(p * (1 - p) / n)
    在 95% 置信水平下 z = 1.96（精确正态分位数 stats.norm.ppf(0.975) ≈ 1.95996），返回百分比数值（如 4.9 表示 4.9%）。
    """
    if n <= 0:
        return 0.0
    z = 1.96 if np.isclose(confidence, 0.95) else float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    err = z * np.sqrt(p * (1.0 - p) / float(n))
    return float(err * 100.0)


def analyze_categorical_question(
    series: pd.Series,
    weights: pd.Series = None,
    is_multi: bool = False,
) -> dict:
    """
    单选/多选题分析：输出频数、占比、95%置信区间，以及 p=0.5 时的最大抽样误差。
    支持未加权与加权模式。
    """
    if is_multi:
        res = split_multiple_choice(series)
        n_valid = res["n_valid"]
        max_err = calc_max_sampling_error(n_valid, p=0.5)
        df_out = res["summary_df"].copy()
        if not df_out.empty:
            top_row = df_out.iloc[0]
            conclusion = (
                f"有效受访者共 {n_valid} 人，占比最高的选项为「{top_row['选项']}」"
                f"（频数 {top_row['频数']}，受访者占比 {top_row['受访者占比(%)']:.1f}%，"
                f"95%CI [{top_row['95%CI下限(%)']:.1f}%, {top_row['95%CI上限(%)']:.1f}%]）；"
                f"当前样本量下最大抽样误差为 ±{max_err:.1f}%。"
            )
        else:
            conclusion = "当前题目无有效作答数据。"
        return {
            "table": df_out,
            "n_valid": n_valid,
            "max_error_pct": round(max_err, 2),
            "conclusion": conclusion,
            "is_multi": True,
        }

    valid_mask = series.notna() & (~series.astype(str).str.strip().isin(["", "nan", "None"]))
    s_valid = series[valid_mask].astype(str)
    n_valid = int(len(s_valid))
    if n_valid == 0:
        return {
            "table": pd.DataFrame(),
            "n_valid": 0,
            "max_error_pct": 0.0,
            "conclusion": "当前题目无有效作答数据。",
            "is_multi": False,
        }

    if weights is not None:
        w_valid = pd.to_numeric(weights[valid_mask], errors="coerce").fillna(1.0)
        grouped = w_valid.groupby(s_valid).sum()
        total_w = float(grouped.sum())
    else:
        grouped = s_valid.value_counts()
        total_w = float(n_valid)

    grouped = grouped.sort_values(ascending=False)
    rows = []
    for opt, cnt in grouped.items():
        p_hat = float(cnt) / total_w if total_w > 0 else 0.0
        pct = p_hat * 100.0
        se = np.sqrt(p_hat * (1.0 - p_hat) / n_valid) if n_valid > 0 else 0.0
        ci_low = max(0.0, (p_hat - 1.96 * se) * 100.0)
        ci_high = min(100.0, (p_hat + 1.96 * se) * 100.0)
        rows.append({
            "选项": str(opt),
            "频数": round(float(cnt), 2) if weights is not None else int(cnt),
            "占比(%)": round(pct, 2),
            "95%CI下限(%)": round(ci_low, 2),
            "95%CI上限(%)": round(ci_high, 2),
        })

    df_out = pd.DataFrame(rows)
    max_err = calc_max_sampling_error(n_valid, p=0.5)
    top_row = df_out.iloc[0]
    weight_note = "（基于加权数据，仅供参考）" if weights is not None else ""
    conclusion = (
        f"有效样本量 n={n_valid}，占比最高的选项为「{top_row['选项']}」"
        f"（频数 {top_row['频数']}，占比 {top_row['占比(%)']:.1f}%，"
        f"95%CI [{top_row['95%CI下限(%)']:.1f}%, {top_row['95%CI上限(%)']:.1f}%]）；"
        f"在 p=0.5、95% 置信水平下最大抽样误差为 ±{max_err:.1f}%{weight_note}。"
    )
    return {
        "table": df_out,
        "n_valid": n_valid,
        "max_error_pct": round(max_err, 2),
        "conclusion": conclusion,
        "is_multi": False,
    }


def analyze_numeric_question(series_or_list, weights: pd.Series = None) -> dict:
    """
    A1 描述统计：计算均值、样本标准差 (ddof=1)、中位数、四分位数 (Q1, Q3)、最小值、最大值。
    """
    s = pd.Series(series_or_list)
    num_s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(num_s))
    if n == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
            "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan,
            "conclusion": "无有效数值数据。",
        }

    vals = num_s.to_numpy(dtype=float)
    if weights is not None and isinstance(series_or_list, pd.Series):
        w = pd.to_numeric(weights.loc[num_s.index], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        mean_val = float(np.average(vals, weights=w))
        # 加权样本标准差
        var_val = float(np.average((vals - mean_val) ** 2, weights=w) * (n / max(n - 1, 1)))
        std_val = float(np.sqrt(var_val))
    else:
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    median_val = float(np.median(vals))
    q25_val = float(np.percentile(vals, 25))
    q75_val = float(np.percentile(vals, 75))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))

    summary_table = pd.DataFrame([{
        "样本量(n)": n,
        "均值": round(mean_val, 3),
        "标准差(SD)": round(std_val, 3),
        "中位数": round(median_val, 3),
        "下四分位数(Q1)": round(q25_val, 3),
        "上四分位数(Q3)": round(q75_val, 3),
        "最小值": round(min_val, 3),
        "最大值": round(max_val, 3),
    }])

    conclusion = (
        f"有效样本量 n={n}，均值为 {mean_val:.2f}，样本标准差为 {std_val:.3f}，"
        f"中位数为 {median_val:.2f}（四分位距 Q1={q25_val:.2f}，Q3={q75_val:.2f}）。"
    )
    norm_p = float(stats.shapiro(vals[:5000])[1]) if n >= 3 else 1.0
    se_val = std_val / np.sqrt(n) if n > 1 else 0.0
    skew_val = float(stats.skew(vals, bias=False)) if n >= 3 and std_val > 0 else 0.0
    kurt_val = float(stats.kurtosis(vals, bias=False)) if n >= 4 and std_val > 0 else 0.0
    return {
        "n": n,
        "n_valid": n,
        "mean": mean_val,
        "std": std_val,
        "sd": std_val,
        "se": se_val,
        "ci_low": mean_val - 1.96 * se_val,
        "ci_high": mean_val + 1.96 * se_val,
        "median": median_val,
        "q25": q25_val,
        "q75": q75_val,
        "iqr": round(q75_val - q25_val, 3),
        "skew": skew_val,
        "kurt": kurt_val,
        "normality_p": norm_p,
        "min": min_val,
        "max": max_val,
        "table": summary_table,
        "values": vals,
        "conclusion": conclusion,
    }


def analyze_scale_question(series: pd.Series, max_scale: int = None, weights: pd.Series = None) -> dict:
    """量表题分析：除数值描述统计外，额外计算负向/中立/正向分歧占比，供分歧堆积条形图使用。"""
    num_res = analyze_numeric_question(series, weights=weights)
    num_s = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if len(num_s) == 0:
        return num_res

    if max_scale is None:
        max_scale = 7 if num_s.max() > 5 else 5
    mid_pt = (max_scale + 1) / 2.0

    levels = list(range(1, max_scale + 1))
    if weights is not None:
        w = pd.to_numeric(weights.loc[num_s.index], errors="coerce").fillna(1.0)
        total_w = float(w.sum())
        level_pcts = {lv: float(w[num_s == lv].sum()) / total_w * 100.0 for lv in levels}
    else:
        total_n = float(len(num_s))
        level_pcts = {lv: float((num_s == lv).sum()) / total_n * 100.0 for lv in levels}

    neg_pct = sum(v for k, v in level_pcts.items() if k < mid_pt)
    neu_pct = sum(v for k, v in level_pcts.items() if np.isclose(k, mid_pt))
    pos_pct = sum(v for k, v in level_pcts.items() if k > mid_pt)

    num_res["max_scale"] = max_scale
    num_res["level_pcts"] = level_pcts
    num_res["neg_pct"] = round(neg_pct, 2)
    num_res["neu_pct"] = round(neu_pct, 2)
    num_res["pos_pct"] = round(pos_pct, 2)
    num_res["conclusion"] = (
        f"量表均值为 {num_res['mean']:.2f}（标准差 {num_res['std']:.2f}），"
        f"正向评价占比 {pos_pct:.1f}%，中立评价占比 {neu_pct:.1f}%，负向评价占比 {neg_pct:.1f}%。"
    )
    return num_res


def analyze_nps_question(
    series_or_counts=None,
    promoters: int = None,
    passives: int = None,
    detractors: int = None,
    weights: pd.Series = None,
) -> dict:
    """
    A9 NPS 净推荐值计算：
    公式：NPS = (推荐者占比 - 贬损者占比) * 100
    其中 9–10 分为推荐者 (Promoters)，7–8 分为中立者 (Passives)，0–6 分为贬损者 (Detractors)。
    """
    if promoters is not None and passives is not None and detractors is not None:
        p_cnt = float(promoters)
        neu_cnt = float(passives)
        d_cnt = float(detractors)
        total = p_cnt + neu_cnt + d_cnt
    else:
        s = pd.Series(series_or_counts)
        num_s = pd.to_numeric(s, errors="coerce").dropna()
        if weights is not None and isinstance(series_or_counts, pd.Series):
            w = pd.to_numeric(weights.loc[num_s.index], errors="coerce").fillna(1.0)
            p_cnt = float(w[num_s >= 9].sum())
            neu_cnt = float(w[(num_s >= 7) & (num_s <= 8)].sum())
            d_cnt = float(w[num_s <= 6].sum())
            total = float(w.sum())
        else:
            p_cnt = float((num_s >= 9).sum())
            neu_cnt = float(((num_s >= 7) & (num_s <= 8)).sum())
            d_cnt = float((num_s <= 6).sum())
            total = float(len(num_s))

    if total <= 0:
        return {"nps": 0.0, "promoter_pct": 0.0, "passive_pct": 0.0, "detractor_pct": 0.0, "total": 0}

    p_pct = p_cnt / total * 100.0
    neu_pct = neu_cnt / total * 100.0
    d_pct = d_cnt / total * 100.0
    # NPS 公式：推荐者占比(%) - 贬损者占比(%)
    nps_val = p_pct - d_pct

    table = pd.DataFrame([
        {"人群分类": "推荐者 (9-10分)", "人数": round(p_cnt, 1), "占比(%)": round(p_pct, 2)},
        {"人群分类": "中立者 (7-8分)", "人数": round(neu_cnt, 1), "占比(%)": round(neu_pct, 2)},
        {"人群分类": "贬损者 (0-6分)", "人数": round(d_cnt, 1), "占比(%)": round(d_pct, 2)},
    ])

    conclusion = (
        f"净推荐值 NPS = {nps_val:.1f}，其中推荐者占比 {p_pct:.1f}%（{p_cnt:g} 人），"
        f"中立者占比 {neu_pct:.1f}%（{neu_cnt:g} 人），贬损者占比 {d_pct:.1f}%（{d_cnt:g} 人）。"
    )
    return {
        "nps": float(round(nps_val, 4)),
        "promoter_pct": round(p_pct, 2),
        "passive_pct": round(neu_pct, 2),
        "detractor_pct": round(d_pct, 2),
        "promoter_cnt": p_cnt,
        "passive_cnt": neu_cnt,
        "detractor_cnt": d_cnt,
        "promoters": int(round(p_cnt)),
        "passives": int(round(neu_cnt)),
        "detractors": int(round(d_cnt)),
        "total": total,
        "table": table,
        "conclusion": conclusion,
    }


def analyze_wave_trend(df: pd.DataFrame, target_col: str, wave_col: str, q_type: str) -> pd.DataFrame:
    """按波次变量汇总同一道题的趋势数据。"""
    sub = df[[wave_col, target_col]].dropna()
    if sub.empty:
        return pd.DataFrame()
    waves = sorted(sub[wave_col].astype(str).unique())
    rows = []
    for wv in waves:
        w_df = sub[sub[wave_col].astype(str) == wv]
        if q_type in ["数值题", "量表题"]:
            vals = pd.to_numeric(w_df[target_col], errors="coerce").dropna()
            rows.append({"波次": wv, "指标名称": "均值", "数值": round(float(vals.mean()), 3) if len(vals) else np.nan, "样本量": len(vals)})
        elif q_type == "NPS题":
            res = analyze_nps_question(w_df[target_col])
            rows.append({"波次": wv, "指标名称": "NPS净推荐值", "数值": round(res["nps"], 2), "样本量": int(res["total"])})
        else:
            vc = w_df[target_col].astype(str).value_counts(normalize=True) * 100.0
            for opt, pct in vc.items():
                rows.append({"波次": wv, "指标名称": str(opt), "数值": round(float(pct), 2), "样本量": len(w_df)})
    return pd.DataFrame(rows)


# ==============================================================================
# 模块五：交叉分析
# ==============================================================================

def crosstab_chi2_analysis(
    df_or_table,
    row_col: str = None,
    col_col: str = None,
    weights: pd.Series = None,
    bonferroni: bool = True,
    alpha: float = 0.05,
) -> dict:
    """
    A2 交叉表卡方检验与列比例 z 检验：
    - 使用 scipy.stats.chi2_contingency(obs, correction=False) 计算卡方值与 p 值；
    - 报告效应量 Cramér's V；
    - 检查期望频数 < 5 的格子占比，2×2 表额外提供 Fisher 精确检验；
    - 对各列做列比例两两 z 检验，并用字母 (A, B, C...) 标注显著差异（α=0.05，可选 Bonferroni 校正）。
    """
    if isinstance(df_or_table, (list, np.ndarray)):
        obs_df = pd.DataFrame(df_or_table)
        obs_df.index = [f"行{i+1}" for i in range(obs_df.shape[0])]
        obs_df.columns = [f"列{j+1}" for j in range(obs_df.shape[1])]
    elif isinstance(df_or_table, pd.DataFrame) and row_col is None and col_col is None:
        obs_df = df_or_table.copy()
    else:
        sub = df_or_table[[row_col, col_col]].dropna()
        if weights is not None:
            w = pd.to_numeric(weights.loc[sub.index], errors="coerce").fillna(1.0)
            obs_df = pd.crosstab(sub[row_col].astype(str), sub[col_col].astype(str), values=w, aggfunc="sum").fillna(0.0)
        else:
            obs_df = pd.crosstab(sub[row_col].astype(str), sub[col_col].astype(str))

    obs = obs_df.to_numpy(dtype=float)
    n_total = float(obs.sum())
    r_cnt, c_cnt = obs.shape

    if n_total <= 0 or r_cnt < 2 or c_cnt < 2:
        return {"chi2": 0.0, "p_value": 1.0, "cramers_v": 0.0, "conclusion": "样本量或类别数不足以执行卡方检验。"}

    chi2_val, p_val, dof, expected = stats.chi2_contingency(obs, correction=False)
    min_dim = min(r_cnt - 1, c_cnt - 1)
    cramers_v = float(np.sqrt(chi2_val / (n_total * min_dim))) if (n_total > 0 and min_dim > 0) else 0.0

    low_exp_cells = int((expected < 5.0).sum())
    total_cells = int(expected.size)
    low_exp_pct = low_exp_cells / total_cells * 100.0
    exp_warning = low_exp_pct > 20.0 or bool((expected < 1.0).any())

    fisher_p = None
    fisher_or = None
    if r_cnt == 2 and c_cnt == 2:
        fisher_or, fisher_p = stats.fisher_exact(np.round(obs).astype(int))
        fisher_or = float(fisher_or)
        fisher_p = float(fisher_p)

    # 行百分比与列百分比
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    row_pct_mat = np.divide(obs, row_sums, out=np.zeros_like(obs), where=row_sums > 0) * 100.0
    col_pct_mat = np.divide(obs, col_sums, out=np.zeros_like(obs), where=col_sums > 0) * 100.0

    row_pct_df = pd.DataFrame(np.round(row_pct_mat, 2), index=obs_df.index, columns=obs_df.columns)
    col_pct_df = pd.DataFrame(np.round(col_pct_mat, 2), index=obs_df.index, columns=obs_df.columns)

    # 列比例两两 z 检验与字母标注 (A, B, C...)
    col_letters = [chr(ord("A") + j) for j in range(c_cnt)]
    col_header_map = {obs_df.columns[j]: f"{obs_df.columns[j]} [{col_letters[j]}]" for j in range(c_cnt)}
    n_pairs = c_cnt * (c_cnt - 1) / 2.0
    alpha_adj = (alpha / n_pairs) if (bonferroni and n_pairs > 1) else alpha

    letter_matrix = pd.DataFrame("", index=obs_df.index, columns=obs_df.columns)
    annotated_col_pct_df = pd.DataFrame(index=obs_df.index, columns=[col_header_map[c] for c in obs_df.columns])
    sig_pairs_text = []

    for i in range(r_cnt):
        for j1 in range(c_cnt):
            higher_than = []
            for j2 in range(c_cnt):
                if j1 == j2:
                    continue
                n1, n2 = col_sums[0, j1], col_sums[0, j2]
                x1, x2 = obs[i, j1], obs[i, j2]
                if n1 <= 0 or n2 <= 0:
                    continue
                p1, p2 = x1 / n1, x2 / n2
                if p1 <= p2:
                    continue
                p_pool = (x1 + x2) / (n1 + n2)
                if p_pool <= 0.0 or p_pool >= 1.0:
                    continue
                se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
                if se <= 0:
                    continue
                z_stat = (p1 - p2) / se
                p_two = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
                if p_two < alpha_adj:
                    higher_than.append(col_letters[j2])
            letter_str = "".join(higher_than)
            letter_matrix.iloc[i, j1] = letter_str
            pct_val = col_pct_mat[i, j1]
            col_label = col_header_map[obs_df.columns[j1]]
            annotated_col_pct_df.loc[obs_df.index[i], col_label] = (
                f"{pct_val:.1f}% ({letter_str})" if letter_str else f"{pct_val:.1f}%"
            )
            if letter_str:
                sig_pairs_text.append(f"在「{obs_df.index[i]}」上，{obs_df.columns[j1]}({pct_val:.1f}%)显著高于{','.join(higher_than)}组")

    sig_word = "存在显著关联" if p_val < alpha else "无统计学显著关联"
    weight_note = "（基于加权数据，仅供参考）" if weights is not None else ""
    warn_note = f"；注意：期望频数小于5的单元格占 {low_exp_pct:.1f}%，建议参考精确检验或合并类别" if exp_warning else ""
    fisher_note = f"；2×2 Fisher 精确检验 p={fisher_p:.4f}" if fisher_p is not None else ""

    conclusion = (
        f"卡方独立性检验结果显示：χ²={chi2_val:.3f}，p={p_val:.4f}，效应量 Cramér's V={cramers_v:.3f}，"
        f"两变量之间{sig_word}{fisher_note}{warn_note}{weight_note}。"
    )

    # 调整后标准化残差 (Adjusted Standardized Residuals)
    row_prop = row_sums / max(n_total, 1.0)
    col_prop = col_sums / max(n_total, 1.0)
    resid_denom = np.sqrt(np.maximum(expected * (1.0 - row_prop) * (1.0 - col_prop), 1e-12))
    std_resid_mat = (obs - expected) / resid_denom
    std_resid_df = pd.DataFrame(np.round(std_resid_mat, 2), index=obs_df.index, columns=obs_df.columns)

    return {
        "obs_df": obs_df,
        "count_df": obs_df,
        "row_pct_df": row_pct_df,
        "col_pct_df": col_pct_df,
        "std_resid_df": std_resid_df,
        "annotated_col_pct_df": annotated_col_pct_df,
        "letter_matrix": letter_matrix,
        "col_letters": col_letters,
        "chi2": float(chi2_val),
        "p_value": float(p_val),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "expected": expected,
        "low_exp_pct": round(low_exp_pct, 2),
        "exp_warning": exp_warning,
        "fisher_p": fisher_p,
        "fisher_or": fisher_or,
        "conclusion": conclusion,
    }


def group_mean_comparison(df: pd.DataFrame, group_col: str, num_col: str, weights: pd.Series = None) -> dict:
    """
    单选题 × 数值题：
    - 两组：独立样本 t 检验，报告 Cohen's d 与 η²；
    - 三组及以上：单因素方差分析 (ANOVA)，报告 η² 效应量，显著时执行 Tukey HSD 事后两两比较并生成显著性字母。
    """
    sub = df[[group_col, num_col]].copy()
    sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return {"conclusion": "无有效样本数据进行均值比较。"}

    groups_order = list(sub[group_col].astype(str).unique())
    group_arrays = []
    rows = []

    for g in groups_order:
        vals = sub.loc[sub[group_col].astype(str) == g, num_col].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        group_arrays.append(vals)
        n_g = len(vals)
        mean_g = float(np.mean(vals))
        sd_g = float(np.std(vals, ddof=1)) if n_g > 1 else 0.0
        se_g = sd_g / np.sqrt(n_g) if n_g > 1 else 0.0
        ci_low = mean_g - 1.96 * se_g
        ci_high = mean_g + 1.96 * se_g
        rows.append({
            "分组": g,
            "组别": g,
            "样本量(n)": n_g,
            "样本量": n_g,
            "均值": round(mean_g, 3),
            "标准差(SD)": round(sd_g, 3),
            "标准差": round(sd_g, 3),
            "95%CI下限": round(ci_low, 3),
            "95%CI上限": round(ci_high, 3),
        })

    summary_df = pd.DataFrame(rows).sort_values("均值", ascending=False).reset_index(drop=True)
    k = len(group_arrays)
    if k < 2:
        return {"table": summary_df, "conclusion": "分组数量不足 2 组，无法执行假设检验。"}

    # 计算总体 η²
    all_vals = sub[num_col].to_numpy(dtype=float)
    grand_mean = float(np.mean(all_vals))
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))
    ss_between = float(sum(len(arr) * (float(np.mean(arr)) - grand_mean) ** 2 for arr in group_arrays))
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    tukey_df = None
    letters_map = {r["分组"]: "" for _, r in summary_df.iterrows()}
    col_letters = {summary_df.loc[i, "分组"]: chr(ord("A") + i) for i in range(len(summary_df))}

    if k == 2:
        g1_name, g2_name = summary_df.loc[0, "分组"], summary_df.loc[1, "分组"]
        v1 = sub.loc[sub[group_col].astype(str) == g1_name, num_col].to_numpy(dtype=float)
        v2 = sub.loc[sub[group_col].astype(str) == g2_name, num_col].to_numpy(dtype=float)
        t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
        n1, n2 = len(v1), len(v2)
        s_pool = np.sqrt(((n1 - 1) * np.var(v1, ddof=1) + (n2 - 1) * np.var(v2, ddof=1)) / max(n1 + n2 - 2, 1))
        cohens_d = abs(float(np.mean(v1) - np.mean(v2))) / s_pool if s_pool > 0 else 0.0
        if p_val < 0.05:
            letters_map[g1_name] = col_letters[g2_name]
        sig_desc = (
            f"「{g1_name}」的均值（{np.mean(v1):.2f}）显著高于「{g2_name}」（{np.mean(v2):.2f}）"
            if p_val < 0.05 else f"「{g1_name}」（{np.mean(v1):.2f}）与「{g2_name}」（{np.mean(v2):.2f}）均值差异不显著"
        )
        conclusion = (
            f"独立样本 t 检验结果显示：{sig_desc}（t={t_stat:.3f}，p={p_val:.4f}，"
            f"效应量 Cohen's d={cohens_d:.3f}，η²={eta_sq:.3f}）。"
        )
        test_name = "独立样本 t 检验"
        stat_val = float(t_stat)
    else:
        f_stat, p_val = stats.f_oneway(*group_arrays)
        test_name = "单因素方差分析 (ANOVA)"
        stat_val = float(f_stat)
        cohens_d = None
        if p_val < 0.05:
            tukey = pairwise_tukeyhsd(endog=sub[num_col].to_numpy(dtype=float), groups=sub[group_col].astype(str).to_numpy(), alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            # 生成显著性字母标注
            means_dict = dict(zip(summary_df["分组"], summary_df["均值"]))
            for _, trow in tukey_df.iterrows():
                if bool(trow["reject"]):
                    ga, gb = str(trow["group1"]), str(trow["group2"])
                    if means_dict.get(ga, 0) > means_dict.get(gb, 0):
                        letters_map[ga] += col_letters.get(gb, "")
                    else:
                        letters_map[gb] += col_letters.get(ga, "")
        top_g = summary_df.iloc[0]
        bot_g = summary_df.iloc[-1]
        sig_desc = "各组均值存在显著差异" if p_val < 0.05 else "各组均值差异不显著"
        conclusion = (
            f"单因素方差分析结果显示：{sig_desc}（F={f_stat:.3f}，p={p_val:.4f}，效应量 η²={eta_sq:.3f}）；"
            f"其中「{top_g['分组']}」均值最高（{top_g['均值']:.2f}），「{bot_g['分组']}」均值最低（{bot_g['均值']:.2f}）。"
        )

    summary_df["组别代号"] = summary_df["分组"].map(col_letters)
    summary_df["显著高于组(α=0.05)"] = summary_df["分组"].map(letters_map)

    return {
        "table": summary_df,
        "group_df": summary_df,
        "test_name": test_name,
        "stat_val": stat_val,
        "stat": stat_val,
        "p_value": float(p_val),
        "eta_sq": float(eta_sq),
        "cohens_d": cohens_d,
        "effect_name": "Cohen's d" if k == 2 else "η²",
        "effect_size": float(cohens_d) if (k == 2 and cohens_d is not None) else float(eta_sq),
        "tukey_df": tukey_df,
        "conclusion": conclusion,
        "raw_df": sub,
    }


def correlation_pair_analysis(x, y, method: str = "pearson") -> dict:
    """
    A3 双数值变量相关分析：支持 Pearson 或 Spearman 相关，报告 r、p 值、效应量 R²=r²，并计算线性拟合线与 95% 置信带。
    """
    df_xy = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"), "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    n = len(df_xy)
    if n < 3:
        return {"r": np.nan, "p_value": np.nan, "r_sq": np.nan, "n": n, "conclusion": "有效配对样本不足 3 个。"}

    xv = df_xy["x"].to_numpy(dtype=float)
    yv = df_xy["y"].to_numpy(dtype=float)

    pr_v, pp_v = stats.pearsonr(xv, yv)
    sr_v, sp_v = stats.spearmanr(xv, yv)

    if method.lower() == "spearman":
        r_val, p_val = sr_v, sp_v
        method_label = "Spearman 秩相关"
    else:
        r_val, p_val = pr_v, pp_v
        method_label = "Pearson 相关"

    r_val = float(r_val)
    p_val = float(p_val)
    r_sq = r_val ** 2

    # 拟合线与 95% 置信带
    X_design = sm.add_constant(xv)
    ols_model = sm.OLS(yv, X_design).fit()
    x_grid = np.linspace(xv.min(), xv.max(), 100)
    pred = ols_model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)

    direction = "正相关" if r_val > 0 else ("负相关" if r_val < 0 else "无相关")
    sig_str = f"显著{direction}" if p_val < 0.05 else "相关性不显著"
    conclusion = (
        f"{method_label}分析结果显示：r={r_val:.3f}，p={p_val:.4f}，决定系数 R²={r_sq:.3f}（n={n}），"
        f"两变量呈{sig_str}。"
    )

    return {
        "r": r_val,
        "p_value": p_val,
        "r_sq": r_sq,
        "pearson_r": float(pr_v),
        "pearson_p": float(pp_v),
        "spearman_r": float(sr_v),
        "spearman_p": float(sp_v),
        "n": n,
        "method": method_label,
        "x": xv,
        "y": yv,
        "x_grid": x_grid,
        "y_hat": pred["mean"].to_numpy(),
        "ci_low": pred["mean_ci_lower"].to_numpy(),
        "ci_high": pred["mean_ci_upper"].to_numpy(),
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块六：信度与效度
# ==============================================================================

def calc_reliability(data_or_df, cols: list = None) -> dict:
    """
    A4 信度分析（Cronbach's α、CITC、删除该题后的 α）：
    Cronbach's α 公式：
        α = (k / (k - 1)) * (1 - sum(s_i^2) / s_T^2)
    其中 k 为题目数，s_i^2 为第 i 题的样本方差 (ddof=1)，s_T^2 为各受访者总分的样本方差 (ddof=1)。
    """
    if isinstance(data_or_df, pd.DataFrame):
        sub = data_or_df[cols].copy() if cols else data_or_df.copy()
    else:
        arr = np.asarray(data_or_df, dtype=float)
        sub = pd.DataFrame(arr, columns=[f"题项{i+1}" for i in range(arr.shape[1])])

    sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
    n_samples, k = sub.shape
    if k < 2 or n_samples < 2:
        return {"alpha": np.nan, "n": n_samples, "k": k, "item_table": pd.DataFrame(), "conclusion": "题项数或样本量不足。"}

    item_vars = sub.var(axis=0, ddof=1)
    total_scores = sub.sum(axis=1)
    total_var = float(total_scores.var(ddof=1))

    # Cronbach's α 计算公式：α = (k / (k - 1)) * (1 - Σs_i² / s_T²)
    alpha_val = float((k / (k - 1.0)) * (1.0 - float(item_vars.sum()) / total_var)) if total_var > 0 else 0.0

    item_rows = []
    for col in sub.columns:
        rest_df = sub.drop(columns=[col])
        rest_sum = rest_df.sum(axis=1)
        citc_val, _ = stats.pearsonr(sub[col], rest_sum)
        if k > 2:
            rest_vars_sum = float(rest_df.var(axis=0, ddof=1).sum())
            rest_total_var = float(rest_sum.var(ddof=1))
            alpha_del = float(((k - 1.0) / (k - 2.0)) * (1.0 - rest_vars_sum / rest_total_var)) if rest_total_var > 0 else 0.0
        else:
            alpha_del = np.nan

        item_rows.append({
            "题项名称": str(col),
            "题项均值": round(float(sub[col].mean()), 3),
            "题项标准差": round(float(sub[col].std(ddof=1)), 3),
            "校正题总相关(CITC)": round(float(citc_val), 3),
            "删除该题后的α": round(float(alpha_del), 3) if pd.notna(alpha_del) else None,
        })

    item_table = pd.DataFrame(item_rows)
    conclusion = (
        f"量表共包含 {k} 个题项（有效样本 n={n_samples}），Cronbach's α 系数为 {alpha_val:.3f}，"
        f"各题项校正题总相关（CITC）介于 {item_table['校正题总相关(CITC)'].min():.3f} 至 {item_table['校正题总相关(CITC)'].max():.3f} 之间。"
    )
    return {
        "alpha": alpha_val,
        "level": "极佳" if alpha_val >= 0.8 else ("良好" if alpha_val >= 0.7 else "可接受"),
        "n": n_samples,
        "k": k,
        "item_table": item_table,
        "conclusion": conclusion,
    }


def calc_eigenvalues_from_corr(corr_matrix) -> np.ndarray:
    """A11 计算相关矩阵的特征值并按从大到小排序返回。"""
    mat = np.asarray(corr_matrix, dtype=float)
    eigvals = np.linalg.eigvalsh(mat)
    return np.sort(eigvals)[::-1]


def calc_validity_efa(df: pd.DataFrame, cols: list, n_factors: int = None, loading_threshold: float = 0.4) -> dict:
    """
    模块六效度与探索性因子分析 (EFA)：
    - 计算 KMO 与 Bartlett 球形检验；
    - 采用主成分法 (method='principal') 提取公因子 + 最大方差法 (rotation='varimax') 旋转；
    - 输出特征值表、旋转后因子载荷矩阵（绝对值 < 0.4 不显示数值）、累计方差解释率及因子得分。
    """
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_samples, k = sub.shape
    if k < 2 or n_samples < k + 2:
        raise ValueError("有效样本量或题项数不足以执行探索性因子分析。")

    _, kmo_model = calculate_kmo(sub)
    bart_chi2, bart_p = calculate_bartlett_sphericity(sub)

    corr_mat = sub.corr().to_numpy()
    eigvals = calc_eigenvalues_from_corr(corr_mat)

    if n_factors is None or n_factors <= 0:
        n_factors = int(np.sum(eigvals > 1.0))
        n_factors = max(1, min(n_factors, k - 1))

    rotation_method = "varimax" if n_factors > 1 else None
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation_method, method="principal")
    fa.fit(sub)

    ss_loadings, prop_var, cum_var = fa.get_factor_variance()
    eig_rows = []
    for i in range(k):
        row_info = {
            "成分序号": f"成分{i+1}",
            "初始特征值": round(float(eigvals[i]), 3),
            "初始方差解释率(%)": round(float(eigvals[i] / k * 100.0), 2),
            "初始累计解释率(%)": round(float(np.sum(eigvals[:i+1]) / k * 100.0), 2),
        }
        if i < n_factors:
            row_info["旋转后特征值"] = round(float(ss_loadings[i]), 3)
            row_info["旋转后方差解释率(%)"] = round(float(prop_var[i] * 100.0), 2)
            row_info["旋转后累计解释率(%)"] = round(float(cum_var[i] * 100.0), 2)
        else:
            row_info["旋转后特征值"] = None
            row_info["旋转后方差解释率(%)"] = None
            row_info["旋转后累计解释率(%)"] = None
        eig_rows.append(row_info)

    eigen_table = pd.DataFrame(eig_rows)

    factor_names = [f"因子{j+1}" for j in range(n_factors)]
    raw_loadings_df = pd.DataFrame(fa.loadings_, index=cols, columns=factor_names)
    communalities = fa.get_communalities()

    # 旋转后载荷表（绝对值 < 0.4 的载荷显示为空字符串）
    masked_rows = []
    for idx, col_name in enumerate(cols):
        r_dict = {"题项": col_name}
        for fname in factor_names:
            val = float(raw_loadings_df.loc[col_name, fname])
            r_dict[fname] = f"{val:.3f}" if abs(val) >= loading_threshold else ""
        r_dict["公因子方差(共同度)"] = round(float(communalities[idx]), 3)
        masked_rows.append(r_dict)
    masked_loadings_df = pd.DataFrame(masked_rows)

    scores_arr = fa.transform(sub)
    scores_df = pd.DataFrame(scores_arr, index=sub.index, columns=[f"因子得分_{j+1}" for j in range(n_factors)])

    cum_pct = float(cum_var[-1] * 100.0)
    conclusion = (
        f"KMO 检验值为 {float(kmo_model):.3f}，Bartlett 球形检验 χ²={float(bart_chi2):.2f}（p={float(bart_p):.4f}）；"
        f"共提取 {n_factors} 个公因子，累计方差解释率为 {cum_pct:.2f}%。"
    )

    return {
        "kmo": float(kmo_model),
        "bartlett_chi2": float(bart_chi2),
        "bartlett_p": float(bart_p),
        "eigenvalues": eigvals,
        "eigen_table": eigen_table,
        "raw_loadings_df": raw_loadings_df,
        "masked_loadings_df": masked_loadings_df,
        "loadings_df": masked_loadings_df,
        "cum_var_pct": round(cum_pct, 2),
        "n_factors": n_factors,
        "scores_df": scores_df,
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块七：相关与回归建模
# ==============================================================================

def calc_correlation_matrix(df: pd.DataFrame, cols: list, method: str = "pearson") -> dict:
    """多变量相关系数矩阵，附显著性星号标记 (* p<0.05, ** p<0.01, *** p<0.001)。"""
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    k = len(cols)
    r_mat = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((k, k)), index=cols, columns=cols)
    annot_mat = pd.DataFrame("", index=cols, columns=cols)

    for i in range(k):
        for j in range(k):
            if i == j:
                r_mat.iloc[i, j] = 1.0
                p_mat.iloc[i, j] = 0.0
                annot_mat.iloc[i, j] = "1.000"
            else:
                pair = sub[[cols[i], cols[j]]].dropna()
                if len(pair) < 3:
                    r_val, p_val = np.nan, np.nan
                    annot_mat.iloc[i, j] = "N/A"
                else:
                    if method.lower() == "spearman":
                        r_val, p_val = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                    else:
                        r_val, p_val = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                    stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                    r_mat.iloc[i, j] = round(float(r_val), 3)
                    p_mat.iloc[i, j] = round(float(p_val), 4)
                    annot_mat.iloc[i, j] = f"{r_val:.3f}{stars}"

    return {
        "r_df": r_mat,
        "p_df": p_mat,
        "annot_df": annot_mat,
        "conclusion": f"共完成 {k} 个变量间的 {method.capitalize()} 相关矩阵分析（* p<0.05，** p<0.01，*** p<0.001）。",
    }


def run_linear_regression(df_or_x, y_col_or_y=None, x_cols: list = None) -> dict:
    """
    A6 多元线性回归：
    输出回归系数、标准误、t 值、p 值、95%CI、R²、调整 R²、F 检验、各变量 VIF（>5 警告），以及残差与拟合值。
    """
    if isinstance(df_or_x, pd.DataFrame) and isinstance(y_col_or_y, str):
        sub = df_or_x[[y_col_or_y] + list(x_cols)].apply(pd.to_numeric, errors="coerce").dropna()
        y_vec = sub[y_col_or_y]
        X_raw = sub[list(x_cols)]
    else:
        x_arr = np.asarray(df_or_x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        y_arr = np.asarray(y_col_or_y, dtype=float)
        X_raw = pd.DataFrame(x_arr, columns=[f"X{i+1}" for i in range(x_arr.shape[1])])
        y_vec = pd.Series(y_arr, name="Y")

    X_const = sm.add_constant(X_raw, has_constant="add")
    model = sm.OLS(y_vec, X_const).fit()

    # 计算 VIF
    vif_map = {"const": np.nan}
    if X_raw.shape[1] == 1:
        vif_map[X_raw.columns[0]] = 1.0
    else:
        X_mat = X_const.to_numpy(dtype=float)
        for idx, col_name in enumerate(X_const.columns):
            if col_name == "const":
                continue
            vif_map[col_name] = float(variance_inflation_factor(X_mat, idx))

    ci_df = model.conf_int(alpha=0.05)
    coef_rows = []
    for term in model.params.index:
        label = "截距 (Constant)" if term == "const" else str(term)
        vif_v = vif_map.get(term, np.nan)
        coef_rows.append({
            "变量": label,
            "回归系数(B)": round(float(model.params[term]), 4),
            "标准误(SE)": round(float(model.bse[term]), 4),
            "t值": round(float(model.tvalues[term]), 4) if np.isfinite(model.tvalues[term]) else np.nan,
            "p值": round(float(model.pvalues[term]), 4) if np.isfinite(model.pvalues[term]) else 0.0,
            "95%CI下限": round(float(ci_df.loc[term, 0]), 4),
            "95%CI上限": round(float(ci_df.loc[term, 1]), 4),
            "VIF": round(float(vif_v), 3) if pd.notna(vif_v) else None,
        })

    coef_table = pd.DataFrame(coef_rows)
    max_vif = float(coef_table["VIF"].dropna().max()) if coef_table["VIF"].notna().any() else 1.0
    vif_warning = max_vif > 5.0

    fitted = model.fittedvalues.to_numpy(dtype=float)
    residuals = model.resid.to_numpy(dtype=float)
    std_resid = (residuals - np.mean(residuals)) / (np.std(residuals, ddof=1) if np.std(residuals, ddof=1) > 0 else 1.0)

    slope_val = float(model.params.iloc[1]) if len(model.params) > 1 else np.nan
    intercept_val = float(model.params["const"])
    r2_val = float(model.rsquared)
    adj_r2_val = float(model.rsquared_adj)
    f_val = float(model.fvalue) if np.isfinite(model.fvalue) else np.nan
    f_pval = float(model.f_pvalue) if np.isfinite(model.f_pvalue) else 0.0

    vif_msg = f"；注意：存在 VIF > 5 的自变量（最高 VIF={max_vif:.2f}），提示潜在多重共线性" if vif_warning else ""
    conclusion = (
        f"多元线性回归模型 R²={r2_val:.3f}，调整后 R²={adj_r2_val:.3f}，"
        f"F={f_val:.3f}（p={f_pval:.4f}）{vif_msg}。回归结果表示关联关系，不直接证明因果。"
    )

    return {
        "slope": slope_val,
        "intercept": intercept_val,
        "r2": r2_val,
        "adj_r2": adj_r2_val,
        "f_value": f_val,
        "f_pvalue": f_pval,
        "coef_table": coef_table,
        "vif_warning": vif_warning,
        "max_vif": max_vif,
        "fitted": fitted,
        "residuals": residuals,
        "std_residuals": std_resid,
        "n": int(model.nobs),
        "conclusion": conclusion,
    }


def calc_roc_auc(y_true, y_score) -> dict:
    """A17 计算 ROC 曲线坐标 (FPR, TPR) 与 AUC 面积。"""
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(yt, ys)
    auc_val = float(auc(fpr, tpr))
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc_val,
    }


def run_logistic_regression(df: pd.DataFrame, y_col: str, x_cols: list) -> dict:
    """
    二元 Logistic 回归：
    输出回归系数 B、OR 值 exp(B) 及 95%CI、McFadden 伪 R²，以及 ROC 曲线与 AUC。
    """
    sub = df[[y_col] + list(x_cols)].dropna().copy()
    # 将二分类因变量编码为 0/1
    y_raw = sub[y_col]
    uniq_vals = sorted(y_raw.unique(), key=lambda x: str(x))
    if len(uniq_vals) != 2:
        raise ValueError(f"二元 Logistic 回归要求因变量恰有 2 个取值，当前共有 {len(uniq_vals)} 个取值。")

    pos_label = uniq_vals[1]
    if set(uniq_vals) == {"是", "否"}:
        pos_label = "是"
    y_bin = (y_raw == pos_label).astype(int)

    X_raw = sub[list(x_cols)].apply(pd.to_numeric, errors="coerce")
    valid_idx = X_raw.dropna().index
    y_bin = y_bin.loc[valid_idx]
    X_raw = X_raw.loc[valid_idx]

    X_const = sm.add_constant(X_raw, has_constant="add")
    logit_mod = sm.Logit(y_bin, X_const).fit(disp=False)

    ci_df = logit_mod.conf_int(alpha=0.05)
    rows = []
    for term in logit_mod.params.index:
        b_val = float(logit_mod.params[term])
        ci_l = float(ci_df.loc[term, 0])
        ci_u = float(ci_df.loc[term, 1])
        rows.append({
            "变量": "截距 (Constant)" if term == "const" else str(term),
            "回归系数(B)": round(b_val, 4),
            "标准误(SE)": round(float(logit_mod.bse[term]), 4),
            "Wald z值": round(float(logit_mod.tvalues[term]), 4),
            "p值": round(float(logit_mod.pvalues[term]), 4),
            "OR值 exp(B)": round(float(np.exp(b_val)), 4),
            "OR 95%CI下限": round(float(np.exp(ci_l)), 4),
            "OR 95%CI上限": round(float(np.exp(ci_u)), 4),
        })

    coef_table = pd.DataFrame(rows)
    y_prob = logit_mod.predict(X_const).to_numpy(dtype=float)
    roc_res = calc_roc_auc(y_bin.to_numpy(), y_prob)
    pseudo_r2 = float(logit_mod.prsquared)

    conclusion = (
        f"二元 Logistic 回归（目标类别：{pos_label}）伪 R²={pseudo_r2:.3f}，"
        f"ROC 曲线下面积 AUC={roc_res['auc']:.3f}（样本量 n={len(y_bin)}）。回归结果表示关联关系，不直接证明因果。"
    )

    return {
        "coef_table": coef_table,
        "pseudo_r2": pseudo_r2,
        "llr_pvalue": float(logit_mod.llr_pvalue),
        "auc": roc_res["auc"],
        "fpr": roc_res["fpr"],
        "tpr": roc_res["tpr"],
        "roc": roc_res,
        "pos_label": str(pos_label),
        "n": int(len(y_bin)),
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块八：聚类分群
# ==============================================================================

def evaluate_kmeans_range(df: pd.DataFrame, cols: list, k_min: int = 2, k_max: int = 8) -> dict:
    """对 k=2..8 评估 K-Means 惯量（肘部图）与平均轮廓系数，推荐轮廓系数最高的 k。"""
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_samples = len(sub)
    max_k = min(k_max, n_samples - 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    k_list, inertias, silhouettes = [], [], []
    for k in range(k_min, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        k_list.append(k)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_scaled, labels))
        silhouettes.append(sil)

    best_idx = int(np.argmax(silhouettes))
    best_k = k_list[best_idx]
    eval_df = pd.DataFrame({
        "类别数(k)": k_list,
        "簇内误差平方和(SSE)": [round(x, 2) for x in inertias],
        "平均轮廓系数": [round(x, 4) for x in silhouettes],
    })
    return {
        "k_list": k_list,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "best_silhouette": silhouettes[best_idx],
        "eval_df": eval_df,
        "conclusion": f"k 在 {k_min}–{max_k} 范围内，k={best_k} 时平均轮廓系数最高（{silhouettes[best_idx]:.3f}），建议优先考虑分为 {best_k} 群。",
    }


def run_kmeans_clustering(
    data_or_df,
    cols: list = None,
    k: int = 2,
    standardize: bool = True,
    cluster_names: dict = None,
    seed: int = 42,
) -> dict:
    """
    A7 K-Means 聚类分群：
    - 先对变量做标准化 (StandardScaler)；
    - 聚类后在原始量纲下计算每类均值中心（画像表）与相对指数（以总体均值为 100）。
    """
    if isinstance(data_or_df, pd.DataFrame):
        sub = data_or_df[cols].copy() if cols else data_or_df.copy()
    else:
        arr = np.asarray(data_or_df, dtype=float)
        sub = pd.DataFrame(arr, columns=[f"V{i+1}" for i in range(arr.shape[1])])
        cols = list(sub.columns)

    sub_clean = sub.apply(pd.to_numeric, errors="coerce").dropna()
    X_orig = sub_clean.to_numpy(dtype=float)

    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_orig)
    else:
        X_fit = X_orig

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    raw_labels = km.fit_predict(X_fit)
    sil_val = float(silhouette_score(X_fit, raw_labels)) if k >= 2 and len(X_fit) > k else 0.0

    # 按第一维均值从小到大固定簇编号顺序，保证结果确定性
    cluster_means_first = [X_orig[raw_labels == c, 0].mean() for c in range(k)]
    order = np.argsort(cluster_means_first)
    remap = {old_c: new_c for new_c, old_c in enumerate(order)}
    labels = np.array([remap[c] for c in raw_labels], dtype=int)

    overall_means = sub_clean.mean(axis=0)
    centers_orig = []
    size_rows = []
    profile_rows = []
    index_rows = []

    n_total = len(sub_clean)
    for c_idx in range(k):
        c_name = (cluster_names or {}).get(c_idx, f"群体{c_idx+1}")
        mask = labels == c_idx
        c_df = sub_clean.loc[mask]
        c_n = int(mask.sum())
        c_pct = round(c_n / n_total * 100.0, 2) if n_total > 0 else 0.0
        c_mean = c_df.mean(axis=0)
        centers_orig.append(c_mean.to_numpy(dtype=float))

        size_rows.append({"群体编号": c_idx + 1, "群体名称": c_name, "样本量": c_n, "占比(%)": c_pct})

        prof_row = {"群体名称": c_name, "样本量": c_n, "样本数": c_n, "占比(%)": c_pct}
        idx_row = {"群体名称": c_name}
        for col_name in sub_clean.columns:
            prof_row[col_name] = round(float(c_mean[col_name]), 3)
            prof_row[f"{col_name}_均值"] = round(float(c_mean[col_name]), 3)
            ov = float(overall_means[col_name])
            idx_row[col_name] = round(float(c_mean[col_name]) / ov * 100.0, 1) if not np.isclose(ov, 0.0) else 100.0
        profile_rows.append(prof_row)
        index_rows.append(idx_row)

    # PCA 降维用于二维聚类散点图
    if X_fit.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X_fit)
        pca_var = pca.explained_variance_ratio_ * 100.0
    else:
        pca_coords = np.column_stack([X_fit[:, 0], np.zeros(len(X_fit))])
        pca_var = np.array([100.0, 0.0])

    label_series = pd.Series(
        [(cluster_names or {}).get(int(lb), f"群体{int(lb)+1}") for lb in labels],
        index=sub_clean.index,
        name="所属群体",
    )

    size_df = pd.DataFrame(size_rows)
    profile_df = pd.DataFrame(profile_rows)
    index_df = pd.DataFrame(index_rows)

    top_cluster = size_df.sort_values("样本量", ascending=False).iloc[0]
    conclusion = (
        f"K-Means 聚类将 {n_total} 份有效样本划分为 {k} 个群体，"
        f"其中占比最大的为「{top_cluster['群体名称']}」（{top_cluster['样本量']} 人，占 {top_cluster['占比(%)']:.1f}%）。"
    )

    return {
        "k": k,
        "labels": labels,
        "label_series": label_series,
        "centers": np.array(centers_orig),
        "counts": [int((labels == c).sum()) for c in range(k)],
        "size_df": size_df,
        "profile_df": profile_df,
        "index_df": index_df,
        "silhouette": sil_val,
        "pca_coords": pca_coords,
        "pca_var": pca_var,
        "clean_index": sub_clean.index,
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块九：开放题文本分析
# ==============================================================================

DEFAULT_STOPWORDS = {
    "的", "了", "是", "我", "也", "都", "很", "在", "有", "和", "就", "不", "人", "这", "那",
    "还", "比较", "一个", "可以", "觉得", "非常", "但是", "还是", "而且", "因为", "所以",
    "不过", "有些", "一点", "如果", "就是", "没有", "什么", "他们", "我们", "自己", "知道",
}


def analyze_open_text(series: pd.Series, stopwords: set = None, top_n: int = 30, extra_stopwords: set = None) -> dict:
    """用 jieba 分词，过滤停用词与标点符号，输出前 30 词频表与词频字典。"""
    stop_set = set(DEFAULT_STOPWORDS) if stopwords is None else set(stopwords)
    if extra_stopwords:
        stop_set = stop_set | set(extra_stopwords)
    clean_s = series.dropna().astype(str).str.strip()
    clean_s = clean_s[~clean_s.isin(["", "nan", "None"])]

    word_counter = Counter()
    for text in clean_s:
        words = jieba.lcut(text)
        for w in words:
            w_clean = w.strip()
            if len(w_clean) < 2:
                continue
            if w_clean in stop_set:
                continue
            if re.match(r"^[\d\W_]+$", w_clean):
                continue
            word_counter[w_clean] += 1

    top_items = word_counter.most_common(top_n)
    total_words = sum(word_counter.values())
    rows = []
    for rank, (word, freq) in enumerate(top_items, start=1):
        pct_w = round(freq / total_words * 100.0, 2) if total_words > 0 else 0.0
        rows.append({
            "排名": rank,
            "词语": word,
            "出现频次": int(freq),
            "词频": int(freq),
            "占总词频比例(%)": pct_w,
            "占比(%)": pct_w,
        })

    freq_df = pd.DataFrame(rows)
    if not freq_df.empty:
        top1 = freq_df.iloc[0]
        conclusion = f"有效文本回答共 {len(clean_s)} 条，高频词首位为「{top1['词语']}」（共出现 {top1['出现频次']} 次）。"
    else:
        conclusion = "未提取到有效分词结果。"

    return {
        "freq_df": freq_df,
        "word_freq_dict": dict(word_counter),
        "n_valid": len(clean_s),
        "n_valid_comments": len(clean_s),
        "total_words": total_words,
        "conclusion": conclusion,
    }


def categorize_by_keywords(series: pd.Series, rules_dict: dict) -> dict:
    """
    关键词规则归类：
    给定 {类别名: [关键词1, 关键词2...]}，统计每类被提及人数与占比；一条回答可命中多类；未命中的归入「其他」。
    """
    clean_s = series.dropna().astype(str).str.strip()
    clean_s = clean_s[~clean_s.isin(["", "nan", "None"])]
    n_valid = len(clean_s)

    cat_counts = {cat: 0 for cat in rules_dict}
    other_count = 0

    for text in clean_s:
        matched_any = False
        for cat, kw_list in rules_dict.items():
            if any(kw and (kw in text) for kw in kw_list):
                cat_counts[cat] += 1
                matched_any = True
        if not matched_any:
            other_count += 1

    rows = []
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        rows.append({
            "归类类别": cat,
            "提及人数": int(cnt),
            "受访者占比(%)": round(cnt / n_valid * 100.0, 2) if n_valid > 0 else 0.0,
        })
    rows.append({
        "归类类别": "其他",
        "提及人数": int(other_count),
        "受访者占比(%)": round(other_count / n_valid * 100.0, 2) if n_valid > 0 else 0.0,
    })

    cat_df = pd.DataFrame(rows)
    top_row = cat_df.iloc[0] if not cat_df.empty else None
    conclusion = (
        f"基于关键词规则对 {n_valid} 条有效回答进行归类，提及率最高的类别为「{top_row['归类类别']}」"
        f"（{top_row['提及人数']} 人，占 {top_row['受访者占比(%)']:.1f}%）。"
        if top_row is not None else "无有效文本可归类。"
    )
    return {
        "table": cat_df,
        "n_valid": n_valid,
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块十：加权处理
# ==============================================================================

def compute_post_stratification_weights(
    series_or_counts,
    target_proportions: dict,
) -> dict:
    """
    A8 事后分层加权计算：
    公式：权重 w_i = 目标占比(target_p) ÷ 样本占比(sample_p)
    加权后有效样本量（Kish's Effective Sample Size）：n_eff = (Σw_i)² / Σ(w_i²)
    """
    # Normalize target proportions so sum = 1.0 (supports 0-1 or 0-100 input)
    raw_targets = {str(k): float(v) for k, v in target_proportions.items()}
    t_sum = sum(raw_targets.values())
    norm_targets = {k: (v / t_sum if t_sum > 0 else 0.0) for k, v in raw_targets.items()}

    if isinstance(series_or_counts, dict):
        counts_map = {str(k): float(v) for k, v in series_or_counts.items()}
        total_n = sum(counts_map.values())
        s_valid = None
    else:
        s_valid = pd.Series(series_or_counts).astype(str)
        vc = s_valid.value_counts()
        counts_map = {str(k): float(v) for k, v in vc.items()}
        total_n = float(len(s_valid))

    weights_by_cat = {}
    rows = []
    sum_w = 0.0
    sum_w2 = 0.0

    for cat, cnt in counts_map.items():
        sample_p = cnt / total_n if total_n > 0 else 0.0
        target_p = norm_targets.get(cat, sample_p)
        # 事后分层权重公式：权重 = 目标占比 ÷ 样本占比
        w_val = (target_p / sample_p) if sample_p > 0 else 1.0
        weights_by_cat[cat] = float(w_val)
        weighted_cnt = cnt * w_val
        sum_w += weighted_cnt
        sum_w2 += cnt * (w_val ** 2)

        rows.append({
            "类别": cat,
            "未加权样本数": int(cnt),
            "未加权占比(%)": round(sample_p * 100.0, 2),
            "目标占比(%)": round(target_p * 100.0, 2),
            "样本权重": round(w_val, 4),
            "加权后频数": round(weighted_cnt, 2),
            "加权后占比(%)": round(target_p * 100.0, 2),
        })

    n_eff = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0.0
    weight_series = s_valid.map(weights_by_cat).fillna(1.0) if s_valid is not None else None
    table_df = pd.DataFrame(rows)

    conclusion = (
        f"事后分层加权已完成：原始样本量 n={int(total_n)}，加权后频数总和为 {sum_w:.1f}，"
        f"Kish 有效样本量 n_eff={n_eff:.1f}。"
    )

    return {
        "weights_by_cat": weights_by_cat,
        "weight_series": weight_series,
        "table": table_df,
        "total_n": int(total_n),
        "weighted_total": sum_w,
        "effective_n": float(round(n_eff, 2)),
        "conclusion": conclusion,
    }


def compare_weighted_unweighted(df: pd.DataFrame, target_col: str, weight_col: str, is_numeric: bool = False) -> pd.DataFrame:
    """
    加权前后对比表（模块 09 使用）：
    - 分类题：各类别未加权占比 vs 加权后占比（%）与变化（百分点），按未加权占比降序。
    - 数值题：未加权 vs 加权后的均值与标准差（加权均值 Σwx/Σw，加权方差 Σw(x-x̄w)²/Σw）。
    权重缺失按 1.0 处理；目标列缺失的行不纳入计算。
    """
    if target_col not in df.columns:
        raise ValueError(f"数据中不存在题目列：{target_col}")
    if weight_col in df.columns:
        w_all = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    else:
        w_all = pd.Series(1.0, index=df.index)

    if is_numeric:
        x = pd.to_numeric(df[target_col], errors="coerce")
        mask = x.notna()
        x, w = x[mask], w_all[mask]
        if len(x) == 0 or float(w.sum()) == 0:
            return pd.DataFrame([{"统计量": "均值", "未加权": np.nan, "加权后": np.nan, "变化": np.nan}])
        mean_u = float(x.mean())
        mean_w = float((x * w).sum() / w.sum())
        sd_u = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        sd_w = float(((w * (x - mean_w) ** 2).sum() / w.sum()) ** 0.5)
        return pd.DataFrame([
            {"统计量": "均值", "未加权": round(mean_u, 2), "加权后": round(mean_w, 2), "变化": round(mean_w - mean_u, 2)},
            {"统计量": "标准差", "未加权": round(sd_u, 2), "加权后": round(sd_w, 2), "变化": round(sd_w - sd_u, 2)},
        ])

    mask = df[target_col].notna()
    s = df.loc[mask, target_col].astype(str)
    w = w_all[mask]
    total_w = float(w.sum())
    rows = []
    for cat, cnt in s.value_counts().items():
        share_u = float(cnt) / len(s) * 100.0 if len(s) > 0 else 0.0
        share_w = float(w[s == cat].sum()) / total_w * 100.0 if total_w > 0 else 0.0
        rows.append({
            "类别": cat,
            "未加权占比(%)": round(share_u, 2),
            "加权后占比(%)": round(share_w, 2),
            "变化(百分点)": round(share_w - share_u, 2),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 模块十二：演示模式数据生成 (seed=42, 300份模拟咖啡消费问卷)
# ==============================================================================

COMMENT_TEMPLATES_20 = [
    "拿铁口感醇厚，奶香和浓缩咖啡融合得很好",
    "美式咖啡清爽提神，早上上班前必喝一杯",
    "价格合理，经常发放优惠券，性价比非常高",
    "门店环境安静整洁，座位宽敞适合办公学习",
    "小程序点单出餐速度很快，不用排队等待",
    "生椰拿铁风味独特，同事们每天都在拼单",
    "咖啡豆烘焙香气浓郁，回甘明显品质稳定",
    "希望配送费能再降低一些，外卖包装很严实",
    "新品推出频率高，每次季节限定都想尝试",
    "冷萃咖啡酸度适中，夏天冰饮非常解暑",
    "部分门店高峰期出餐稍慢，需要提前下单",
    "会员积分兑换福利丰富，复购意愿很强",
    "价格偏贵，没有折扣券的时候购买频次会降低",
    "燕麦拿铁对乳糖不耐受人群很友好，味道好",
    "门店位置便利，写字楼楼下就能买到热咖啡",
    "甜度选择灵活，不另外加糖的选项很健康",
    "咖啡风味层次丰富，油脂细腻，拉花也很漂亮",
    "早餐搭配烘焙面包套餐很划算，省时省心",
    "偶尔遇到出品冷热不均，希望品控更加稳定",
    "包装设计时尚好看，品牌联名活动很有吸引力",
]


def generate_demo_coffee_data(seed: int = 42, n: int = 300, include_extended: bool = False) -> pd.DataFrame:
    """
    内部基准测试数据生成器（仅供 self_check.py 自检使用，禁止在 app.py 生产前端调用）：
    以三个潜在消费分群（品质型 / 性价比型 / 便利型）为生成骨架，驱动品牌选择、
    月消费额、量表、NPS、漏斗与开放题情感，固定随机种子 seed=42 生成 n 份问卷。
    当 include_extended=True 时，额外附加「省份」、成对 KANO 正反向列及联合分析正交轮廓列供 S3–S8 测试。
    """
    rng = np.random.default_rng(seed)

    # ---------------- 人口属性与潜在分群 ----------------
    genders = rng.choice(["男", "女"], size=n, p=[0.46, 0.54])
    age_groups = rng.choice(["18-24岁", "25-30岁", "31-40岁", "40岁以上"], size=n, p=[0.24, 0.38, 0.26, 0.12])
    cities = rng.choice(["一线城市", "新一线城市", "二线城市", "三线及以下"], size=n, p=[0.32, 0.35, 0.21, 0.12])
    waves = rng.choice(["2026Q1", "2026Q2", "2026Q3"], size=n, p=[0.32, 0.34, 0.34])
    seg = rng.choice(["品质型", "性价比型", "便利型"], size=n, p=[0.30, 0.45, 0.25])
    is_q, is_v, is_c = seg == "品质型", seg == "性价比型", seg == "便利型"

    duration = rng.integers(85, 420, size=n)
    drink_p = np.select(
        [age_groups == "18-24岁", age_groups == "25-30岁", age_groups == "31-40岁"],
        [0.93, 0.92, 0.86], default=0.74,
    )
    drink_coffee = np.where(rng.random(n) < drink_p, "是", "否")

    freq_levels = ["每天1杯及以上", "每周3-5杯", "每周1-2杯", "偶尔饮用"]
    freq_list = np.empty(n, dtype=object)
    for mask, p in [(is_q, [0.22, 0.38, 0.28, 0.12]), (is_v, [0.20, 0.34, 0.30, 0.16]), (is_c, [0.48, 0.32, 0.14, 0.06])]:
        freq_list[mask] = rng.choice(freq_levels, size=int(mask.sum()), p=p)

    # 潜变量提前生成（消费额等行为变量也要用到满意度潜变量）
    halo = rng.normal(0, 0.35, size=n)
    qual_lat = np.select([is_q, is_v], [4.08, 3.66], default=3.78) + halo + rng.normal(0, 0.52, size=n)
    price_lat = np.select([is_q, is_v], [3.30, 3.92], default=3.58) + halo * 0.8 + rng.normal(0, 0.55, size=n)

    freq_base = np.select(
        [freq_list == "每天1杯及以上", freq_list == "每周3-5杯", freq_list == "每周1-2杯"],
        [370.0, 235.0, 125.0], default=55.0,
    )
    seg_mult = np.select([is_q, is_v], [1.28, 0.85], default=1.02)
    city_mult = np.select([cities == "一线城市", cities == "新一线城市", cities == "二线城市"], [1.18, 1.05, 0.95], default=0.85)
    monthly_spend = np.round(np.clip(freq_base * seg_mult * city_mult * (1 + 0.15 * (qual_lat - 3.8) + 0.05 * (price_lat - 3.6)) + rng.normal(0, 38, size=n), 15, 950), 0)

    # ---------------- 偏好品类（多选，分群修正基础概率） ----------------
    cat_pool = ["拿铁", "美式", "生椰拿铁", "冷萃", "摩卡", "手冲单品"]
    cat_base = np.array([0.68, 0.52, 0.48, 0.30, 0.22, 0.18])
    multi_choices = []
    for i in range(n):
        p = cat_base.copy()
        if is_q[i]:
            p[3] += 0.22; p[5] += 0.25; p[2] -= 0.10
        if is_v[i]:
            p[2] += 0.15
        if age_groups[i] == "18-24岁":
            p[2] += 0.10; p[4] += 0.08
        picked = [c for c, pi in zip(cat_pool, np.clip(p, 0.02, 0.95)) if rng.random() < pi]
        if not picked:
            picked = [str(rng.choice(["拿铁", "美式"]))]
        multi_choices.append("；".join(picked))

    # ---------------- 量表（双因子结构，潜变量已在上方生成） ----------------
    def _to_scale(latent, offset=0.0, noise=0.5):
        return np.clip(np.round(latent + offset + rng.normal(0, noise, size=n)), 1, 5).astype(int)

    price_1 = _to_scale(price_lat)
    price_2 = _to_scale(price_lat, 0.08)
    price_3 = _to_scale(price_lat, -0.08)
    qual_1 = _to_scale(qual_lat, noise=0.42)
    qual_2 = _to_scale(qual_lat, 0.05, noise=0.42)
    qual_3_rev = 6 - _to_scale(qual_lat, -0.05, noise=0.42)  # 第 3 题为反向题

    # ---------------- NPS（校准至均分≈7.8、NPS≈+25） ----------------
    wave_drift = np.select([waves == "2026Q2", waves == "2026Q3"], [0.12, 0.25], default=0.0)
    nps_scores = np.clip(np.round(1.15 + 1.25 * qual_lat + 0.6 * price_lat + wave_drift + rng.normal(0, 1.5, size=n)), 0, 10).astype(int)

    # ---------------- 品牌（分群份额 + 忠诚度转移） ----------------
    brands = ["瑞幸", "星巴克", "库迪", "Manner", "幸运咖"]
    brand_p = np.where(is_q[:, None], [0.20, 0.42, 0.08, 0.24, 0.06],
                       np.where(is_v[:, None], [0.48, 0.09, 0.22, 0.07, 0.14],
                                [0.40, 0.22, 0.18, 0.12, 0.08]))
    last_brand = np.array([rng.choice(brands, p=brand_p[i]) for i in range(n)], dtype=object)
    stay_p = np.select([is_q, is_v], [0.78, 0.62], default=0.70)
    most_brand = np.array([
        lb if rng.random() < sp else rng.choice(brands, p=brand_p[i])
        for i, (lb, sp) in enumerate(zip(last_brand, stay_p))
    ], dtype=object)

    # ---------------- 品牌漏斗（复购与满意度挂钩） ----------------
    aware = (rng.random(n) < 0.93).astype(int)
    consider = np.where(aware == 1, (rng.random(n) < np.select([is_q, is_v], [0.80, 0.74], default=0.77)).astype(int), 0)
    purchase = np.where(consider == 1, (rng.random(n) < 0.72).astype(int), 0)
    rep_p = 1.0 / (1.0 + np.exp(-(-2.1 + 0.5 * qual_lat + 0.25 * price_lat)))
    repurchase = np.where(purchase == 1, (rng.random(n) < rep_p).astype(int), 0)

    # ---------------- IPA（速度为典型短板：重要性高、满意度低） ----------------
    imp_taste = _to_scale(rng.normal(4.5, 0.5, size=n), noise=0.3)
    imp_price = _to_scale(rng.normal(4.3, 0.55, size=n) + np.where(is_v, 0.15, 0.0), noise=0.3)
    imp_env = _to_scale(rng.normal(3.6, 0.7, size=n), noise=0.3)
    imp_speed = _to_scale(rng.normal(4.0, 0.6, size=n) + np.where(is_c, 0.35, 0.0), noise=0.3)
    sat_taste = qual_1.copy()
    sat_price = price_1.copy()
    sat_env = _to_scale(rng.normal(3.82, 0.62, size=n), noise=0.4)
    sat_speed = _to_scale(rng.normal(3.32, 0.72, size=n), noise=0.4)

    # ---------------- Van Westendorp PSM（可接受价心由分群决定） ----------------
    psm_center = np.select([is_q, is_v], [24.0, 16.5], default=19.0) + rng.normal(0, 4.0, size=n)
    _hw = rng.uniform(4.5, 8.5, size=n)  # 个人可接受价格带半宽（个体差异大，四条曲线才能在中段真实相交）
    psm_draws = np.clip(np.sort(np.column_stack([
        psm_center - _hw * 1.15,
        psm_center - _hw * 0.55,
        psm_center + _hw * 0.55,
        psm_center + _hw * 1.15,
    ]), axis=1), 2.0, None)
    psm_too_cheap, psm_cheap, psm_expensive, psm_too_expensive = [np.round(psm_draws[:, k], 1) for k in range(4)]

    # ---------------- 开放题（情感与满意度一致，词库大幅扩充） ----------------
    pos_pool = [
        "咖啡口感醇厚，香气浓郁，入口很顺滑", "生椰拿铁味道清爽，椰香和咖啡融合得很好",
        "咖啡拉花漂亮，油脂细腻，看着就很有食欲", "咖啡豆新鲜，能喝出明显的回甘和层次",
        "门店环境安静整洁，适合办公和学习", "店员服务热情，出餐速度也很快",
        "性价比很高，优惠券叠加下来非常划算", "包装精致好看，联名设计很有吸引力",
        "甜度选择灵活，少糖选项对控糖人群友好", "冷萃咖啡清爽干净，夏天的下午来一杯很舒服",
        "早餐套餐搭配烘焙面包，省时又省心", "手冲咖啡风味突出，能喝出产区特点",
        "下单取餐动线合理，基本不用排队等候", "会员积分兑换很实在，老顾客有专属福利",
        "杯量充足，冰块比例恰到好处不冲淡", "工作日提神效果好，已经成为日常习惯",
    ]
    neg_pool = [
        "高峰期排队太久，出餐速度明显变慢", "最近感觉价格涨了，优惠券门槛也变高",
        "个别门店品控不稳定，冷热和浓度不一致", "冰块给得太多，喝到后面味道变淡了",
        "部分门店座位太少，想坐下办公很难", "外卖送到时包装有漏洒，体验不太好",
        "新品咖啡甜度偏高，对不爱甜的人不友好", "附近门店太少，想喝的时候不方便",
        "小程序经常卡顿，下单流程不够顺畅", "咖啡香气比以前弱了，风味层次变单薄",
        "排队取餐区域拥挤，高峰期体验差", "希望价格能更亲民，学生党负担有点重",
    ]
    sug_pool = [
        "希望增加低因和无咖啡因的选择", "要是周边能再多开两家门店就好了",
        "期待推出更多季节限定新品", "早餐种类要是再丰富一些更好",
        "希望小程序能提前预约取餐时间", "建议大杯型选择更多，杯型划分更细",
        "希望环保杯自带优惠能长期保留", "要是能增加燕麦奶等植物奶选项就好了",
        "门店要是能提供充电插座和热水就更方便", "希望会员生日福利再大方一点",
    ]
    comments = []
    for i in range(n):
        lat = qual_lat[i]
        r = rng.random()
        if lat >= 3.9:
            pair = list(rng.choice(pos_pool, size=2, replace=False)) if r < 0.7 else [rng.choice(pos_pool), rng.choice(sug_pool)]
        elif lat < 3.2:
            pair = list(rng.choice(neg_pool, size=2, replace=False)) if r < 0.45 else [rng.choice(neg_pool), rng.choice(sug_pool)]
        else:
            pool_pick = rng.choice(["ps", "pn", "ns"], p=[0.4, 0.35, 0.25])
            pair = {"ps": [rng.choice(pos_pool), rng.choice(sug_pool)],
                    "pn": [rng.choice(pos_pool), rng.choice(neg_pool)],
                    "ns": [rng.choice(neg_pool), rng.choice(sug_pool)]}[pool_pick]
        comments.append("；".join(pair) + "。")

    # ---------------- 混入约 5% 异常样本（质控演示） ----------------
    straight_indices = rng.choice(n, size=9, replace=False)
    speed_indices = rng.choice(n, size=8, replace=False)
    for j, idx in enumerate(straight_indices):
        v = 3 if j < 5 else 4
        price_1[idx] = price_2[idx] = price_3[idx] = v
        qual_1[idx] = qual_2[idx] = qual_3_rev[idx] = v
    for idx in speed_indices:
        duration[idx] = int(rng.integers(12, 28))

    df = pd.DataFrame({
        "性别": genders,
        "年龄段": age_groups,
        "城市": cities,
        "调查波次": waves,
        "答题时长（秒）": duration,
        "是否喝咖啡": drink_coffee,
        "消费频次": freq_list,
        "月消费额": monthly_spend,
        "偏好品类": multi_choices,
        "NPS推荐度": nps_scores,
        "价格量表1_价格合理": price_1,
        "价格量表2_性价比高": price_2,
        "价格量表3_促销吸引": price_3,
        "品质量表1_口感醇厚": qual_1,
        "品质量表2_豆源优质": qual_2,
        "品质量表3_品质不稳定_反向": qual_3_rev,
        "上次购买品牌": last_brand,
        "最常购买品牌": most_brand,
        "品牌知名": aware,
        "品牌考虑": consider,
        "品牌购买": purchase,
        "品牌复购": repurchase,
        "重要性1_口味口感": imp_taste,
        "重要性2_价格实惠": imp_price,
        "重要性3_门店环境": imp_env,
        "重要性4_出餐速度": imp_speed,
        "满意度1_口味口感": sat_taste,
        "满意度2_价格实惠": sat_price,
        "满意度3_门店环境": sat_env,
        "满意度4_出餐速度": sat_speed,
        "价格敏感_太便宜": psm_too_cheap,
        "价格敏感_便宜": psm_cheap,
        "价格敏感_贵": psm_expensive,
        "价格敏感_太贵": psm_too_expensive,
        "开放题短评": comments,
    })

    if include_extended:
        rng_ext = np.random.default_rng(seed + 1000)
        # 1) 省份列（供 self_check S4/S5/S7/S8 测试真实列区域地图渲染）
        prov_names = ["安徽", "江苏", "上海", "浙江", "湖北", "广东", "北京", "四川", "山东", "河南"]
        prov_probs = np.array([86, 64, 41, 38, 22, 18, 12, 9, 6, 4], dtype=float)
        prov_probs = prov_probs / prov_probs.sum()
        df["省份"] = rng_ext.choice(prov_names, size=n, p=prov_probs)

        # 2) 成对 KANO 正向/反向评价列（1=喜欢, 2=理应如此, 3=无所谓, 4=勉强接受, 5=不喜欢）
        kano_specs = [
            ("季节限定新品", [0.54, 0.18, 0.00, 0.28]),  # A, O, M, I
            ("出餐≤5分钟",   [0.18, 0.50, 0.10, 0.22]),
            ("价格透明",     [0.09, 0.46, 0.26, 0.19]),
            ("现磨现做",     [0.05, 0.37, 0.41, 0.17]),
            ("外卖包装防洒", [0.15, 0.33, 0.22, 0.30]),
            ("会员积分",     [0.20, 0.15, 0.10, 0.55]),
            ("杯型可选",     [0.16, 0.14, 0.08, 0.62]),
        ]
        pair_for_code = {"A": (1, 2), "O": (1, 5), "M": (2, 5), "I": (3, 3)}
        for feat_nm, probs in kano_specs:
            chosen_codes = rng_ext.choice(["A", "O", "M", "I"], size=n, p=probs)
            df[f"KANO正向_{feat_nm}"] = [pair_for_code[c][0] for c in chosen_codes]
            df[f"KANO反向_{feat_nm}"] = [pair_for_code[c][1] for c in chosen_codes]

        # 3) 联合分析属性水平列与总体意愿评分列（供 S3/S4/S5 OLS 效用回归测试）
        u_price = {"9元": 0.82, "13元": 0.31, "17元": -0.28, "21元": -0.85}
        u_flavor = {"生椰拿铁": 0.60, "经典拿铁": 0.20, "美式": -0.20, "手冲单品": -0.60}
        u_brand = {"瑞幸": 0.35, "库迪": 0.12, "星巴克": -0.01, "Manner": -0.46}
        u_cup = {"大杯473ml": 0.34, "中杯355ml": 0.00, "小杯237ml": -0.34}
        c_p = rng_ext.choice(list(u_price.keys()), size=n)
        c_f = rng_ext.choice(list(u_flavor.keys()), size=n)
        c_b = rng_ext.choice(list(u_brand.keys()), size=n)
        c_c = rng_ext.choice(list(u_cup.keys()), size=n)
        c_score = [
            round(5.0 + u_price[p] + u_flavor[f] + u_brand[b] + u_cup[c] + float(rng_ext.normal(0, 0.08)), 3)
            for p, f, b, c in zip(c_p, c_f, c_b, c_c)
        ]
        df["联合属性_价格"] = c_p
        df["联合属性_口味类型"] = c_f
        df["联合属性_品牌"] = c_b
        df["联合属性_杯型容量"] = c_c
        df["联合评分_购买意愿"] = c_score

    return df


# ==============================================================================
# 模块十三：市调特色分析
# ==============================================================================

def calc_brand_funnel(df_or_counts, cols: list = None, stage_names: list = None) -> dict:
    """
    A18 品牌漏斗与逐级转化率计算：
    支持传入 (df, cols) 或直接传入各级人数列表（如 [300, 210, 120, 60]）。
    逐级转化率保留一位小数。
    """
    if isinstance(df_or_counts, pd.DataFrame) and cols:
        counts = []
        names = stage_names if stage_names else list(cols)
        for c in cols:
            s = df_or_counts[c]
            cnt = int(s.isin([1, "1", True, "是", "Yes", "yes"]).sum())
            counts.append(cnt)
    else:
        counts = [int(x) for x in df_or_counts]
        names = stage_names if stage_names else [f"阶段{i+1}" for i in range(len(counts))]

    top_cnt = counts[0] if counts and counts[0] > 0 else 1
    step_rates = []
    rows = []

    for i, (nm, cnt) in enumerate(zip(names, counts)):
        overall_pct = round(cnt / top_cnt * 100.0, 1)
        if i == 0:
            step_pct = 100.0
            drop_cnt = 0
        else:
            prev_cnt = counts[i - 1]
            step_pct = round(cnt / prev_cnt * 100.0, 1) if prev_cnt > 0 else 0.0
            drop_cnt = prev_cnt - cnt
            step_rates.append(step_pct)
        rows.append({
            "漏斗阶段": nm,
            "人数": cnt,
            "占首级比例(%)": overall_pct,
            "逐级转化率(%)": step_pct,
            "本级流失人数": drop_cnt,
        })

    funnel_df = pd.DataFrame(rows)
    rate_str = " → ".join([f"{r:.1f}%" for r in step_rates])
    conclusion = (
        f"品牌漏斗各级人数依次为 {' → '.join(map(str, counts))}，"
        f"相邻环节逐级转化率依次为 {rate_str}，整体总转化率为 {rows[-1]['占首级比例(%)']:.1f}%。"
    )
    return {
        "stage_names": names,
        "counts": counts,
        "step_rates": step_rates,
        "table": funnel_df,
        "conclusion": conclusion,
    }


def calc_brand_switching(df: pd.DataFrame, prev_col: str, curr_col: str) -> dict:
    """品牌转换分析：输出转换交叉矩阵、各品牌保留率及桑基图流向数据。"""
    sub = df[[prev_col, curr_col]].dropna().astype(str)
    brands = sorted(set(sub[prev_col].unique()) | set(sub[curr_col].unique()))
    ct = pd.crosstab(sub[prev_col], sub[curr_col]).reindex(index=brands, columns=brands, fill_value=0)
    row_sums = ct.sum(axis=1)
    pct_ct = ct.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    retention_rows = []
    for b in brands:
        prev_n = int(ct.loc[b].sum())
        curr_n = int(ct[b].sum())
        stay_n = int(ct.loc[b, b])
        ret_rate = round(stay_n / prev_n * 100.0, 2) if prev_n > 0 else 0.0
        inflow_n = curr_n - stay_n
        outflow_n = prev_n - stay_n
        retention_rows.append({
            "品牌": b,
            "上期基数": prev_n,
            "本期人数": curr_n,
            "留存人数": stay_n,
            "品牌保留率(%)": ret_rate,
            "流入人数": inflow_n,
            "流出人数": outflow_n,
            "净增减": curr_n - prev_n,
        })

    retention_df = pd.DataFrame(retention_rows).sort_values("品牌保留率(%)", ascending=False).reset_index(drop=True)
    top_ret = retention_df.iloc[0]
    conclusion = (
        f"品牌转换分析显示：保留率最高的品牌为「{top_ret['品牌']}」（保留率 {top_ret['品牌保留率(%)']:.1f}%，"
        f"上期 {top_ret['上期基数']} 人中留存 {top_ret['留存人数']} 人）。"
    )
    return {
        "count_matrix": ct,
        "pct_matrix": pct_ct.round(2),
        "retention_df": retention_df,
        "brands": brands,
        "conclusion": conclusion,
    }


def calc_correspondence_analysis(df: pd.DataFrame, row_col: str, col_col: str, is_multi_col: bool = False) -> dict:
    """用 prince.CA 执行对应分析，返回行坐标、列坐标和前两维惯量解释率。"""
    if is_multi_col:
        sub = df[[row_col, col_col]].dropna()
        split_res = split_multiple_choice(sub[col_col])
        dummy = split_res["dummy_df"]
        ct = dummy.groupby(sub.loc[dummy.index, row_col].astype(str)).sum()
    else:
        sub = df[[row_col, col_col]].dropna().astype(str)
        ct = pd.crosstab(sub[row_col], sub[col_col])

    # 过滤全 0 行列
    ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("对应分析要求行与列的有效类别数均不少于 2 个。")

    ca = prince.CA(n_components=2, n_iter=10, random_state=42)
    ca = ca.fit(ct)
    row_coords = ca.row_coordinates(ct)
    col_coords = ca.column_coordinates(ct)
    row_coords.columns = ["维度1", "维度2"]
    col_coords.columns = ["维度1", "维度2"]

    pct_var = [float(x) for x in ca.percentage_of_variance_]
    if len(pct_var) < 2:
        pct_var = pct_var + [0.0] * (2 - len(pct_var))

    conclusion = (
        f"对应分析前两个维度分别解释了 {pct_var[0]:.1f}% 和 {pct_var[1]:.1f}% 的惯量，"
        f"累计惯量解释率为 {pct_var[0] + pct_var[1]:.1f}%。"
    )
    return {
        "contingency_table": ct,
        "row_coords": row_coords,
        "col_coords": col_coords,
        "pct_inertia": pct_var,
        "eigenvalues": [float(x) for x in ca.eigenvalues_],
        "conclusion": conclusion,
    }


def calc_ipa_analysis(df: pd.DataFrame, imp_cols: list, sat_cols: list, attr_names: list = None) -> dict:
    """
    IPA 重要性-满意度四象限分析：
    以总体重要性均值和总体满意度均值为十字分割线，划分为：
    保持优势（高重要-高满意）、优先改进（高重要-低满意）、低优先（低重要-低满意）、供给过度（低重要-高满意）。
    """
    if len(imp_cols) != len(sat_cols):
        raise ValueError("重要性题项与满意度题项的数量必须一一对应。")
    names = attr_names if (attr_names and len(attr_names) == len(imp_cols)) else [f"要素{i+1}" for i in range(len(imp_cols))]

    rows = []
    for nm, ic, sc in zip(names, imp_cols, sat_cols):
        imp_v = float(pd.to_numeric(df[ic], errors="coerce").mean())
        sat_v = float(pd.to_numeric(df[sc], errors="coerce").mean())
        rows.append({
            "要素名称": nm,
            "重要性均值": round(imp_v, 3),
            "满意度均值": round(sat_v, 3),
            "差值(满意度-重要性)": round(sat_v - imp_v, 3),
        })

    ipa_df = pd.DataFrame(rows)
    mean_imp = float(ipa_df["重要性均值"].mean())
    mean_sat = float(ipa_df["满意度均值"].mean())

    quadrants = []
    for _, r in ipa_df.iterrows():
        hi_imp = r["重要性均值"] >= mean_imp
        hi_sat = r["满意度均值"] >= mean_sat
        if hi_imp and not hi_sat:
            quadrants.append("优先改进")
        elif hi_imp and hi_sat:
            quadrants.append("保持优势")
        elif not hi_imp and not hi_sat:
            quadrants.append("低优先")
        else:
            quadrants.append("供给过度")
    ipa_df["所属象限"] = quadrants

    priority_items = ipa_df.loc[ipa_df["所属象限"] == "优先改进", "要素名称"].tolist()
    keep_items = ipa_df.loc[ipa_df["所属象限"] == "保持优势", "要素名称"].tolist()
    conclusion = (
        f"IPA 分析以总体重要性均值（{mean_imp:.2f}）和总体满意度均值（{mean_sat:.2f}）为十字基准线："
        f"落入「优先改进」象限的要素为 {('、'.join(priority_items) if priority_items else '无')}，"
        f"落入「保持优势」象限的要素为 {('、'.join(keep_items) if keep_items else '无')}。"
    )
    return {
        "table": ipa_df,
        "mean_imp": mean_imp,
        "mean_sat": mean_sat,
        "conclusion": conclusion,
    }


def _find_curve_intersection(x_grid: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> tuple:
    """线性插值求解两条累积曲线的交点 (price, pct)。"""
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float(x_grid[idx]), float((y1[idx] + y2[idx]) / 2.0)
    idx = int(sign_changes[0])
    x0, x1 = x_grid[idx], x_grid[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]
    t = -d0 / (d1 - d0) if not np.isclose(d1, d0) else 0.5
    px = float(x0 + t * (x1 - x0))
    py = float(y1[idx] + t * (y1[idx + 1] - y1[idx]))
    return px, py


def calc_van_westendorp(df: pd.DataFrame, tc_col: str, c_col: str, e_col: str, te_col: str) -> dict:
    """
    Van Westendorp 价格敏感度模型 (PSM)：
    计算四条标准累积百分比曲线及四个关键价格交点：
    - OPP (最优价格点)：太便宜 ∩ 太贵
    - IPP (无差异价格点)：便宜 ∩ 贵
    - PMC (最低可接受价格/边际便宜点)：太便宜 ∩ 不便宜(100%-便宜)
    - PME (最高可接受价格/边际昂贵点)：太贵 ∩ 不贵(100%-贵)
    """
    sub = df[[tc_col, c_col, e_col, te_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError("有效价格样本不足 5 份，无法执行价格敏感度分析。")

    tc = sub[tc_col].to_numpy(dtype=float)
    ch = sub[c_col].to_numpy(dtype=float)
    ex = sub[e_col].to_numpy(dtype=float)
    te = sub[te_col].to_numpy(dtype=float)

    p_min = float(min(tc.min(), ch.min(), ex.min(), te.min()))
    p_max = float(max(tc.max(), ch.max(), ex.max(), te.max()))
    price_grid = np.linspace(p_min, p_max, 300)

    # 累积曲线定义：
    # 太便宜(Too Cheap)：认为价格 p 太便宜的比例（阈值 >= p）
    # 便宜(Cheap)：认为价格 p 便宜的比例（阈值 >= p）
    # 贵(Expensive)：认为价格 p 贵的比例（阈值 <= p）
    # 太贵(Too Expensive)：认为价格 p 太贵的比例（阈值 <= p）
    curve_tc = np.array([(tc >= p).mean() * 100.0 for p in price_grid])
    curve_c = np.array([(ch >= p).mean() * 100.0 for p in price_grid])
    curve_e = np.array([(ex <= p).mean() * 100.0 for p in price_grid])
    curve_te = np.array([(te <= p).mean() * 100.0 for p in price_grid])

    curve_not_c = 100.0 - curve_c
    curve_not_e = 100.0 - curve_e

    opp_p, opp_y = _find_curve_intersection(price_grid, curve_tc, curve_te)
    ipp_p, ipp_y = _find_curve_intersection(price_grid, curve_c, curve_e)
    pmc_p, pmc_y = _find_curve_intersection(price_grid, curve_tc, curve_not_c)
    pme_p, pme_y = _find_curve_intersection(price_grid, curve_te, curve_not_e)

    summary_df = pd.DataFrame([
        {"关键价格指标": "最低价格下限 (PMC: 太便宜 ∩ 不便宜)", "价格数值": round(pmc_p, 2), "对应累积占比(%)": round(pmc_y, 1)},
        {"关键价格指标": "最优价格点 (OPP: 太便宜 ∩ 太贵)", "价格数值": round(opp_p, 2), "对应累积占比(%)": round(opp_y, 1)},
        {"关键价格指标": "无差异价格点 (IPP: 便宜 ∩ 贵)", "价格数值": round(ipp_p, 2), "对应累积占比(%)": round(ipp_y, 1)},
        {"关键价格指标": "最高价格上限 (PME: 太贵 ∩ 不贵)", "价格数值": round(pme_p, 2), "对应累积占比(%)": round(pme_y, 1)},
    ])

    conclusion = (
        f"Van Westendorp 价格敏感度分析显示：最优价格点（OPP）为 {opp_p:.2f} 元，"
        f"无差异价格点（IPP）为 {ipp_p:.2f} 元，可接受价格区间（PMC–PME）为 [{pmc_p:.2f}, {pme_p:.2f}] 元。"
    )

    return {
        "price_grid": price_grid,
        "curve_tc": curve_tc,
        "curve_c": curve_c,
        "curve_e": curve_e,
        "curve_te": curve_te,
        "opp": (opp_p, opp_y),
        "ipp": (ipp_p, ipp_y),
        "pmc": (pmc_p, pmc_y),
        "pme": (pme_p, pme_y),
        "table": summary_df,
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块十四：自适应数据能力探测矩阵与扩展高阶市调模型（纯实测数据驱动，零合成回退）
# ==============================================================================

CHINA_PROVINCES_SHORT = (
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "台湾",
    "香港", "澳门",
)


def _normalize_province_name(val) -> str:
    """将省份全称或带地市前缀的字符串归一化为标准省份简称（如「安徽省」->「安徽」）。"""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s in ("nan", "None", "NULL"):
        return ""
    for p in CHINA_PROVINCES_SHORT:
        if s == p or s.startswith(p):
            return p
    for suf in ("维吾尔自治区", "壮族自治区", "回族自治区", "特别行政区", "自治区", "省", "市"):
        if s.endswith(suf) and len(s) > len(suf):
            cand = s[: -len(suf)]
            if cand in CHINA_PROVINCES_SHORT:
                return cand
    return ""


def detect_province_column(df: pd.DataFrame) -> str:
    """自动检测数据集中能匹配中国省份名称的地域列；若不存在则返回 None。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    priority_cols = [
        c for c in df.columns
        if any(k in str(c).lower() for k in ("省", "区域", "地区", "地域", "所在地", "籍贯", "province", "region"))
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    for col in priority_cols + other_cols:
        s = df[col].dropna().astype(str).str.strip()
        s = s[~s.isin(["", "nan", "None"])]
        if len(s) < 2:
            continue
        mapped = s.map(_normalize_province_name)
        valid_mapped = mapped[mapped != ""]
        if len(valid_mapped) / len(s) >= 0.50 and valid_mapped.nunique() >= 2:
            return col
    return None


def detect_duration_column(df: pd.DataFrame) -> str:
    """检测问卷作答时长列（兼容「答题时长（秒）」与问卷星「所用时间」等列名）。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for col in df.columns:
        cs = str(col).strip()
        if any(k in cs for k in ("答题时长", "所用时间", "作答时长", "用时", "duration")):
            raw_s = df[col].dropna()
            if len(raw_s) == 0:
                continue
            num_s = pd.to_numeric(raw_s, errors="coerce")
            if num_s.notna().sum() >= max(1, int(len(raw_s) * 0.5)):
                return col
            ext_s = pd.to_numeric(raw_s.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
            if ext_s.notna().sum() >= max(1, int(len(raw_s) * 0.5)):
                return col
    return None


def detect_funnel_columns(df: pd.DataFrame) -> dict:
    """检测可排序的品牌/业务转化漏斗二值列（至少 3 个递进环节）。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"enabled": False, "cols": [], "stage_names": []}

    stage_patterns = [
        ("知名", ("知名", "知晓", "听说", "认知", "触达", "曝光", "aware")),
        ("考虑", ("考虑", "兴趣", "意向", "加购", "收藏", "consider")),
        ("购买", ("购买", "尝试", "下单", "转化", "付费", "使用", "purchase", "buy")),
        ("复购", ("复购", "重购", "留存", "忠诚", "续费", "推荐", "repurchase", "loyal")),
    ]
    valid_vals = {0, 1, "0", "1", True, False, "是", "否", "Yes", "No", "yes", "no"}

    def _is_binary_col(c):
        s = df[c].dropna()
        if len(s) < 3:
            return False
        uniq = set(s.unique()) | set(s.astype(str).str.strip().unique())
        num_s = pd.to_numeric(s, errors="coerce").dropna()
        if len(num_s) == len(s) and set(num_s.unique()).issubset({0, 1, 0.0, 1.0}):
            return True
        return set(s.astype(str).str.strip().unique()).issubset({str(v) for v in valid_vals})

    matched_cols = []
    matched_names = []
    for stage_label, kws in stage_patterns:
        for col in df.columns:
            if col in matched_cols:
                continue
            cs = str(col).lower()
            if any(k in cs for k in kws) and _is_binary_col(col):
                matched_cols.append(col)
                matched_names.append(str(col))
                break

    if len(matched_cols) >= 3:
        return {"enabled": True, "cols": matched_cols, "stage_names": matched_names}
    return {"enabled": False, "cols": [], "stage_names": []}


def detect_psm_columns(df: pd.DataFrame) -> dict:
    """检测 Van Westendorp PSM 四条价格阈值列（太便宜 / 便宜 / 贵 / 太贵）。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"enabled": False, "cols": {}}

    tc_col = c_col = e_col = te_col = None
    for col in df.columns:
        cs = str(col).lower()
        num_ok = pd.to_numeric(df[col], errors="coerce").notna().sum() >= 5
        if not num_ok:
            continue
        if tc_col is None and any(k in cs for k in ("太便宜", "过低", "too_cheap", "toocheap")):
            tc_col = col
        elif te_col is None and any(k in cs for k in ("太贵", "过高", "too_expensive", "tooexpensive")):
            te_col = col

    for col in df.columns:
        if col in (tc_col, te_col):
            continue
        cs = str(col).lower()
        num_ok = pd.to_numeric(df[col], errors="coerce").notna().sum() >= 5
        if not num_ok:
            continue
        if c_col is None and any(k in cs for k in ("便宜", "较便宜", "划算", "cheap")) and "太" not in cs:
            c_col = col
        elif e_col is None and any(k in cs for k in ("贵", "偏贵", "较贵", "expensive")) and "太" not in cs:
            e_col = col

    if tc_col and c_col and e_col and te_col:
        return {
            "enabled": True,
            "cols": {
                "tc_col": tc_col,
                "c_col": c_col,
                "e_col": e_col,
                "te_col": te_col,
                "too_cheap": tc_col,
                "cheap": c_col,
                "expensive": e_col,
                "too_expensive": te_col,
            },
        }
    return {"enabled": False, "cols": {}}


def detect_kano_columns(df: pd.DataFrame) -> dict:
    """检测 KANO 模型成对的正向（具备）与反向（不具备）题项列（至少 2 对）。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"enabled": False, "pos_cols": [], "neg_cols": [], "feature_names": []}

    pos_map = {}
    neg_map = {}
    prefix_pairs = [
        ("KANO正向_", "KANO反向_"),
        ("kano正向_", "kano反向_"),
        ("正向_", "反向_"),
        ("具备_", "不具备_"),
    ]
    suffix_pairs = [
        ("_正向", "_反向"),
        ("_具备", "_不具备"),
    ]

    for col in df.columns:
        cs = str(col).strip()
        num_s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(num_s) < 5 or num_s.min() < 1 or num_s.max() > 7:
            continue
        matched = False
        for p_pre, n_pre in prefix_pairs:
            if cs.startswith(p_pre):
                pos_map[cs[len(p_pre):]] = col
                matched = True
                break
            if cs.startswith(n_pre):
                neg_map[cs[len(n_pre):]] = col
                matched = True
                break
        if not matched:
            for p_suf, n_suf in suffix_pairs:
                if cs.endswith(p_suf):
                    pos_map[cs[:-len(p_suf)]] = col
                    break
                if cs.endswith(n_suf):
                    neg_map[cs[:-len(n_suf)]] = col
                    break

    common_feats = [f for f in pos_map.keys() if f in neg_map]
    if len(common_feats) >= 2:
        return {
            "enabled": True,
            "pos_cols": [pos_map[f] for f in common_feats],
            "neg_cols": [neg_map[f] for f in common_feats],
            "feature_names": common_feats,
        }
    return {"enabled": False, "pos_cols": [], "neg_cols": [], "feature_names": []}


def detect_conjoint_columns(df: pd.DataFrame) -> dict:
    """检测联合分析（Conjoint）轮廓属性列与总体意愿评分列。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"enabled": False, "attr_cols": [], "rating_col": None}

    attr_cols = []
    rating_col = None
    for col in df.columns:
        cs = str(col).strip()
        cs_low = cs.lower()
        if any(cs.startswith(p) for p in ("联合属性_", "轮廓属性_", "conjoint_attr_")):
            if df[col].dropna().nunique() >= 2:
                attr_cols.append(col)
        elif rating_col is None and any(k in cs_low for k in ("联合评分", "轮廓评分", "conjoint_rating", "conjoint_score")):
            if pd.to_numeric(df[col], errors="coerce").notna().sum() >= 8:
                rating_col = col

    if len(attr_cols) >= 2 and rating_col is not None:
        return {"enabled": True, "attr_cols": attr_cols, "rating_col": rating_col}
    return {"enabled": False, "attr_cols": [], "rating_col": None}


def detect_ipa_columns(df: pd.DataFrame, scale_cols: list = None) -> dict:
    """检测成对的「重要性 × 满意度」列（如 重要性1_口味口感 <-> 满意度1_口味口感）。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"enabled": False, "auto_paired": False, "imp_cols": [], "sat_cols": [], "attr_names": []}

    imp_map = {}
    sat_map = {}
    for col in df.columns:
        cs = str(col).strip()
        num_s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(num_s) < 3:
            continue
        m_imp = re.match(r"^(?:重要性|重要度)\d*[_:：\-]?(.*)$", cs)
        m_sat = re.match(r"^(?:满意度|满意)\d*[_:：\-]?(.*)$", cs)
        if m_imp and m_imp.group(1):
            imp_map[m_imp.group(1).strip()] = col
        elif m_sat and m_sat.group(1):
            sat_map[m_sat.group(1).strip()] = col

    common = [k for k in imp_map.keys() if k in sat_map]
    if len(common) >= 2:
        return {
            "enabled": True,
            "auto_paired": True,
            "imp_cols": [imp_map[k] for k in common],
            "sat_cols": [sat_map[k] for k in common],
            "attr_names": common,
        }
    return {"enabled": False, "auto_paired": False, "imp_cols": [], "sat_cols": [], "attr_names": []}


def detect_brand_switching_columns(df: pd.DataFrame) -> dict:
    """检测成对的品牌/选项前期→当期流转列（如 上次购买品牌 -> 最常购买品牌）。"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"enabled": False, "prev_col": None, "curr_col": None}

    prev_cands = [c for c in df.columns if any(k in str(c) for k in ("上次", "上期", "过去", "首次", "之前", "原品牌"))]
    curr_cands = [c for c in df.columns if any(k in str(c) for k in ("最常", "本期", "现在", "目前", "当前", "现品牌"))]
    for pc in prev_cands:
        for cc in curr_cands:
            if pc != cc and df[pc].dropna().nunique() >= 2 and df[cc].dropna().nunique() >= 2:
                return {"enabled": True, "prev_col": pc, "curr_col": cc}
    return {"enabled": False, "prev_col": None, "curr_col": None}


def detect_scale_dimensions(df: pd.DataFrame, scale_cols: list = None) -> dict:
    """
    自动将量表题按题干前缀聚类为潜变量测量维度（排除 KANO 正反向列）：
    例如「价格量表1_价格合理」「价格量表2_性价比高」归入「价格量表」维度。
    返回 {维度名: [列名列表]}（仅保留题项数 >= 2 的维度）。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    if scale_cols is None:
        qt = auto_detect_question_types(df)
        scale_cols = [c for c, t in qt.items() if t == "量表题"]

    groups = {}
    for col in scale_cols:
        cs = str(col).strip()
        if cs.startswith(("KANO正向_", "KANO反向_", "kano正向_", "kano反向_")):
            continue
        # 提取下划线/冒号/数字编号前的维度词根
        m = re.match(r"^([^\d_:：\-]+?)\d*[_:：\-].+$", cs)
        if m:
            prefix = m.group(1).strip()
        else:
            m2 = re.match(r"^([^\d]+)\d+$", cs)
            prefix = m2.group(1).strip() if m2 else "综合量表"
        groups.setdefault(prefix, []).append(col)

    valid_dims = {k: v for k, v in groups.items() if len(v) >= 2}
    if not valid_dims and len(scale_cols) >= 2:
        non_kano = [c for c in scale_cols if not str(c).startswith(("KANO正向_", "KANO反向_"))]
        if len(non_kano) >= 2:
            valid_dims["综合量表"] = non_kano
    return valid_dims


def detect_dataset_capabilities(df: pd.DataFrame, qtypes: dict = None) -> dict:
    """
    核心自适应探测引擎：扫描当前上传数据集的真实列结构，输出题型分组与各高级模型可用性矩阵。
    严格对应《06-产品化改造要求》2.4 能力可用性矩阵，不满足条件的能力一律返回 enabled=False。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"has_data": False}

    qt = qtypes if isinstance(qtypes, dict) else auto_detect_question_types(df)
    duration_col = detect_duration_column(df)
    kano_info = detect_kano_columns(df)
    kano_col_set = set(kano_info["pos_cols"] + kano_info["neg_cols"])
    conjoint_info = detect_conjoint_columns(df)
    conjoint_col_set = set(conjoint_info["attr_cols"] + ([conjoint_info["rating_col"]] if conjoint_info["rating_col"] else []))

    single_cols = [c for c, t in qt.items() if t == "单选题" and c not in conjoint_col_set]
    multi_cols = [c for c, t in qt.items() if t == "多选题"]
    scale_cols = [c for c, t in qt.items() if t == "量表题" and c not in kano_col_set]
    numeric_cols = [c for c, t in qt.items() if t == "数值题" and c != duration_col and c not in conjoint_col_set]
    nps_cols = [c for c, t in qt.items() if t == "NPS题"]
    text_cols = [c for c, t in qt.items() if t == "开放文本题"]
    meta_cols = [c for c, t in qt.items() if t == "元信息"]

    scale_dims = detect_scale_dimensions(df, scale_cols)
    non_imp_dims = {k: v for k, v in scale_dims.items() if not k.startswith(("重要性", "重要度"))}
    prov_col = detect_province_column(df)
    funnel_info = detect_funnel_columns(df)
    psm_info = detect_psm_columns(df)
    ipa_info = detect_ipa_columns(df, scale_cols)
    switch_info = detect_brand_switching_columns(df)

    # 对应分析：>=2 道分类/多选题且有效类别数 >= 3
    ca_cands = []
    for c in single_cols:
        if df[c].dropna().nunique() >= 3:
            ca_cands.append(c)
    for c in multi_cols:
        mc_res = split_multiple_choice(df[c])
        if len(mc_res["counts"]) >= 3:
            ca_cands.append(c)
    ca_enabled = len(ca_cands) >= 2

    # 聚类画像：>=3 道数值/量表/NPS题 且有效样本 >= 10
    cluster_cands = scale_cols + numeric_cols + nps_cols
    cluster_enabled = (len(cluster_cands) >= 3) and (len(df) >= 10)

    # SEM 路径模型：>=2 个量表维度（每维度 >=2 题）且存在目标结果列
    sem_dims = non_imp_dims if len(non_imp_dims) >= 2 else scale_dims
    sem_targets = nps_cols + numeric_cols + [c for c in scale_cols if c not in sum(list(sem_dims.values())[:2], [])]
    sem_enabled = (len(sem_dims) >= 2) and (len(sem_targets) >= 1) and (len(df) >= 12)

    # 加权可用列：类别数在 2~10 之间的单选题（优先人口学列）
    demo_kws = ("性别", "年龄", "城市", "地区", "区域", "省份", "学历", "教育", "职业", "收入", "年级", "身份")
    weight_cols = [
        c for c in single_cols
        if 2 <= df[c].dropna().nunique() <= 10
    ]
    weight_cols_sorted = sorted(weight_cols, key=lambda c: (0 if any(k in str(c) for k in demo_kws) else 1))

    # 文本有效条数统计
    text_counts = {}
    for tc in text_cols:
        s_txt = df[tc].dropna().astype(str).str.strip()
        s_txt = s_txt[~s_txt.isin(["", "nan", "None", "无", "没有", "略"])]
        text_counts[tc] = int(len(s_txt))

    special_models = {
        "correspondence": {
            "enabled": ca_enabled,
            "name": "对应分析 (CA)",
            "requirement": "至少 2 道分类题/多选题（每题有效选项 ≥ 3 个）",
            "status_reason": f"已匹配 {len(ca_cands)} 道多类别题目" if ca_enabled else "缺少至少 2 道含 3 个以上选项的分类/多选题",
            "cols": ca_cands,
        },
        "ipa": {
            "enabled": ipa_info["enabled"],
            "name": "IPA 重要性-满意度四象限",
            "requirement": "成对「重要性 × 满意度」量表题项（≥ 2 对）",
            "status_reason": f"已识别 {len(ipa_info['attr_names'])} 对重要性×满意度要素" if ipa_info["enabled"] else "未检测到成对「重要性_X / 满意度_X」量表列",
            "info": ipa_info,
        },
        "funnel": {
            "enabled": funnel_info["enabled"],
            "name": "品牌转化漏斗",
            "requirement": "至少 3 个递进转化阶段的二值列（知名/考虑/购买/复购等）",
            "status_reason": f"已识别 {len(funnel_info['cols'])} 级转化漏斗列" if funnel_info["enabled"] else "未检测到可排序的漏斗阶段 0/1 二值列",
            "cols": funnel_info["cols"],
            "info": funnel_info,
        },
        "brand_switching": {
            "enabled": switch_info["enabled"],
            "name": "品牌转换与留存桑基图",
            "requirement": "成对前期与当期品牌/偏好单选列（如上次购买 vs 最常购买）",
            "status_reason": f"已匹配「{switch_info['prev_col']} → {switch_info['curr_col']}」" if switch_info["enabled"] else "未检测到前期/当期成对转换列",
            "cols": [switch_info["prev_col"], switch_info["curr_col"]] if switch_info["enabled"] else [],
            "info": switch_info,
        },
        "psm": {
            "enabled": psm_info["enabled"],
            "name": "PSM 价格敏感度模型",
            "requirement": "4 道 Van Westendorp 价格阈值数值列（太便宜/便宜/贵/太贵）",
            "status_reason": "已完整识别 4 道价格敏感度阈值列" if psm_info["enabled"] else "未检测到完整的太便宜/便宜/贵/太贵 4 道价格列",
            "cols": psm_info["cols"],
            "info": psm_info,
        },
        "kano": {
            "enabled": kano_info["enabled"],
            "name": "KANO 需求分层模型",
            "requirement": "成对正向（具备）与反向（不具备）5 级评价列（≥ 2 对）",
            "status_reason": f"已识别 {len(kano_info['feature_names'])} 对 KANO 正反向题项" if kano_info["enabled"] else "未检测到成对的正向/反向功能评价列",
            "cols": kano_info["pos_cols"] + kano_info["neg_cols"],
            "info": kano_info,
        },
        "conjoint": {
            "enabled": conjoint_info["enabled"],
            "name": "联合分析 (Conjoint)",
            "requirement": "产品轮廓属性列（≥ 2 个属性）+ 总体购买意愿/效用评分列",
            "status_reason": f"已识别 {len(conjoint_info['attr_cols'])} 个轮廓属性与评分列" if conjoint_info["enabled"] else "未检测到正交轮廓属性列与对应评分列",
            "cols": conjoint_info,
            "info": conjoint_info,
        },
        "sem": {
            "enabled": sem_enabled,
            "name": "结构方程路径模型 (SEM)",
            "requirement": "至少 2 个量表测量维度（每维度 ≥ 2 题）+ 1 个结果变量列",
            "status_reason": f"已识别 {len(sem_dims)} 个量表潜变量维度" if sem_enabled else "量表测量维度不足 2 个或缺少结果变量列",
            "dims": sem_dims,
            "targets": sem_targets,
        },
        "clustering": {
            "enabled": cluster_enabled,
            "name": "消费者聚类画像 (K-Means)",
            "requirement": "至少 3 道数值题/量表题且有效样本 n ≥ 10",
            "status_reason": f"可用连续/量表特征共 {len(cluster_cands)} 列" if cluster_enabled else "数值题与量表题合计不足 3 列或样本过少",
            "cols": cluster_cands,
        },
        "region_map": {
            "enabled": prov_col is not None,
            "name": "区域分布地图",
            "requirement": "包含中国省份/直辖市/自治区名称的地域列",
            "status_reason": f"已识别省份列「{prov_col}」" if prov_col else "数据中不含可匹配中国省份名称的列",
            "col": prov_col,
        },
    }

    # 补充简写键名别名，供报告导出与前端统一读取（展示列表枚举时使用 canonical_special_keys）
    canonical_special_keys = [
        "region_map", "funnel", "psm", "kano", "conjoint",
        "sem", "correspondence", "clustering", "ipa", "brand_switching",
    ]
    special_models["map"] = special_models["region_map"]
    special_models["sankey"] = special_models["brand_switching"]
    special_models["ca"] = special_models["correspondence"]

    quant_all = scale_cols + numeric_cols + nps_cols
    analysis_cols = single_cols + multi_cols + scale_cols + numeric_cols + nps_cols + text_cols
    cross_cat_cat = len(single_cols) >= 2
    cross_cat_num = (len(single_cols) >= 1) and (len(quant_all) >= 1)
    cross_num_num = len(quant_all) >= 2

    modules = {
        "overview": True,
        "qc": True,
        "single_analysis": len(analysis_cols) > 0,
        "single_question": len(analysis_cols) > 0,
        "cross_analysis": len(single_cols + multi_cols + quant_all) >= 2,
        "crosstab": len(single_cols + multi_cols + quant_all) >= 2,
        "scale_modeling": (len(scale_cols) >= 2) or (len(quant_all) >= 2),
        "special_models": any(special_models[k]["enabled"] for k in canonical_special_keys),
        "text_analysis": len(text_cols) >= 1,
        "weighting": len(weight_cols_sorted) >= 1,
        "export": True,
    }

    return {
        "has_data": True,
        "qtypes": qt,
        "analysis_cols": analysis_cols,
        "single_cols": single_cols,
        "multi_cols": multi_cols,
        "scale_cols": scale_cols,
        "numeric_cols": numeric_cols,
        "nps_cols": nps_cols,
        "text_cols": text_cols,
        "meta_cols": meta_cols,
        "duration_col": duration_col,
        "scale_dimensions": scale_dims,
        "scale_dims": scale_dims,
        "weight_cols": weight_cols_sorted,
        "text_valid_counts": text_counts,
        "cross_cat_cat": cross_cat_cat,
        "cross_cat_num": cross_cat_num,
        "cross_num_num": cross_num_num,
        "canonical_special_keys": canonical_special_keys,
        "modules": modules,
        "special_models": special_models,
    }


def calc_sem_path_model(
    df: pd.DataFrame,
    price_cols: list = None,
    qual_cols: list = None,
    target_col: str = None,
    dim1_name: str = None,
    dim2_name: str = None,
    dim1_cols: list = None,
    dim2_cols: list = None,
) -> dict:
    """
    S1 结构方程路径模型（基于标准化变量 OLS 路径估计）：
    支持任意两个外生潜变量测量维度指向结果变量的路径估计；
    路径系数 β 与对标准化后的合成变量执行多元线性回归 (run_linear_regression) 的回归系数完全一致。
    """
    work_df = df.copy()
    price_cols = dim1_cols or price_cols
    qual_cols = dim2_cols or qual_cols
    default_p = ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"]
    default_q_check = ["品质量表1_口感醇厚", "品质量表2_豆源优质"]

    if price_cols is None and qual_cols is None:
        if all(c in work_df.columns for c in default_p + default_q_check):
            price_cols = default_p
            if "品质量表3_正向" not in work_df.columns and "品质量表3_品质不稳定_反向" in work_df.columns:
                work_df["品质量表3_正向"] = reverse_score(work_df["品质量表3_品质不稳定_反向"], max_scale=5)
            qual_cols = ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"]
            qual_cols = [c for c in qual_cols if c in work_df.columns]
            dim1_name = dim1_name or "价格感知"
            dim2_name = dim2_name or "品质感知"
            target_col = target_col or ("NPS推荐度" if "NPS推荐度" in work_df.columns else None)
        else:
            dims = detect_scale_dimensions(work_df)
            non_imp = {k: v for k, v in dims.items() if not k.startswith(("重要性", "重要度"))}
            use_dims = non_imp if len(non_imp) >= 2 else dims
            if len(use_dims) < 2:
                raise ValueError("当前数据集量表维度不足 2 个（每维度需 ≥ 2 题），无法拟合结构方程路径模型。")
            dim_keys = list(use_dims.keys())[:2]
            dim1_name = dim1_name or dim_keys[0]
            dim2_name = dim2_name or dim_keys[1]
            price_cols = use_dims[dim_keys[0]]
            qual_cols = use_dims[dim_keys[1]]

    dim1_name = dim1_name or "价格感知"
    dim2_name = dim2_name or "品质感知"
    if target_col is None or target_col not in work_df.columns:
        qt = auto_detect_question_types(work_df)
        used_set = set(price_cols + qual_cols)
        cands = [c for c, t in qt.items() if t in ("NPS题", "数值题", "量表题") and c not in used_set]
        if not cands:
            raise ValueError("当前数据集缺少可用于结构方程路径模型的内生结果变量列。")
        target_col = cands[0]

    work_df["_dim1_raw"] = compute_scale_composite(work_df, price_cols)
    work_df["_dim2_raw"] = compute_scale_composite(work_df, qual_cols)
    work_df["_target_raw"] = pd.to_numeric(work_df[target_col], errors="coerce")

    sub = work_df[["_dim1_raw", "_dim2_raw", "_target_raw"] + price_cols + qual_cols].dropna().copy()
    n_valid = len(sub)
    if n_valid < 10:
        raise ValueError("有效样本量不足 10 份，无法估计结构方程路径模型。")

    # 标准化合成变量 (Z-score, ddof=1)
    std_df = pd.DataFrame(index=sub.index)
    for raw_c, std_c in [("_dim1_raw", dim1_name), ("_dim2_raw", dim2_name), ("_target_raw", target_col)]:
        s = sub[raw_c].astype(float)
        sd = float(s.std(ddof=1))
        std_df[std_c] = (s - float(s.mean())) / (sd if sd > 0 else 1.0)

    ols_res = run_linear_regression(std_df, target_col, [dim1_name, dim2_name])
    coef_df = ols_res["coef_table"]

    row_price = coef_df[coef_df["变量"] == dim1_name].iloc[0]
    row_qual = coef_df[coef_df["变量"] == dim2_name].iloc[0]

    beta_price = float(row_price["回归系数(B)"])
    se_price = float(row_price["标准误(SE)"])
    t_price = float(row_price["t值"])
    p_price = float(row_price["p值"])

    beta_qual = float(row_qual["回归系数(B)"])
    se_qual = float(row_qual["标准误(SE)"])
    t_qual = float(row_qual["t值"])
    p_qual = float(row_qual["p值"])

    r_exo, p_exo = stats.pearsonr(std_df[dim1_name], std_df[dim2_name])
    r_exo = float(r_exo)
    p_exo = float(p_exo)

    def _sig_stars(p_val: float) -> str:
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "ns"

    path_table = pd.DataFrame([
        {
            "路径关系": f"{dim1_name} → {target_col}",
            "外生变量": dim1_name,
            "内生变量": target_col,
            "标准化路径系数(β)": beta_price,
            "标准误(SE)": se_price,
            "t统计量": t_price,
            "p值": p_price,
            "显著性": _sig_stars(p_price),
        },
        {
            "路径关系": f"{dim2_name} → {target_col}",
            "外生变量": dim2_name,
            "内生变量": target_col,
            "标准化路径系数(β)": beta_qual,
            "标准误(SE)": se_qual,
            "t统计量": t_qual,
            "p值": p_qual,
            "显著性": _sig_stars(p_qual),
        },
    ])

    loading_rows = []
    for c in price_cols:
        r_load = float(np.corrcoef(sub[c].astype(float), sub["_dim1_raw"].astype(float))[0, 1])
        loading_rows.append({"潜变量": dim1_name, "测量题项": c, "标准化载荷": round(r_load, 3)})
    for c in qual_cols:
        r_load = float(np.corrcoef(sub[c].astype(float), sub["_dim2_raw"].astype(float))[0, 1])
        loading_rows.append({"潜变量": dim2_name, "测量题项": c, "标准化载荷": round(r_load, 3)})
    loadings_table = pd.DataFrame(loading_rows)

    r2 = float(ols_res["r2"])
    conclusion = (
        f"结构方程路径模型显示：{dim1_name}（β={beta_price:+.2f}, p={p_price:.4f}）与"
        f"{dim2_name}（β={beta_qual:+.2f}, p={p_qual:.4f}）对{target_col}的联合解释方差达 "
        f"{r2 * 100:.1f}%（R²={r2:.3f}, n={n_valid}）。"
    )
    return {
        "dim1_name": dim1_name,
        "dim2_name": dim2_name,
        "target_col": target_col,
        "path_table": path_table,
        "loadings_table": loadings_table,
        "beta_price": beta_price,
        "beta_qual": beta_qual,
        "p_price": p_price,
        "p_qual": p_qual,
        "r_exo": r_exo,
        "p_exo": p_exo,
        "r2": r2,
        "adj_r2": float(ols_res["adj_r2"]),
        "f_stat": float(ols_res["f_value"]),
        "f_p": float(ols_res["f_pvalue"]),
        "n": n_valid,
        "std_df": std_df,
        "ols_res": ols_res,
        "conclusion": conclusion,
    }


# KANO 5x5 评价对照表：行=正向题(1~5)，列=反向题(1~5)
_KANO_EVAL_MATRIX = {
    (1, 1): "Q", (1, 2): "A", (1, 3): "A", (1, 4): "A", (1, 5): "O",
    (2, 1): "R", (2, 2): "I", (2, 3): "I", (2, 4): "I", (2, 5): "M",
    (3, 1): "R", (3, 2): "I", (3, 3): "I", (3, 4): "I", (3, 5): "M",
    (4, 1): "R", (4, 2): "I", (4, 3): "I", (4, 4): "I", (4, 5): "M",
    (5, 1): "R", (5, 2): "R", (5, 3): "R", (5, 4): "R", (5, 5): "Q",
}


def calc_kano_model(
    df: pd.DataFrame = None,
    pos_cols: list = None,
    neg_cols: list = None,
    feature_names: list = None,
    counts_records: list = None,
) -> dict:
    """
    S2 KANO 模型 Better-Worse 系数与四象限分类计算（纯真实数据驱动，无合成回退）：
    - Better (满意影响力) = (A + O) / (A + O + M + I)
    - Worse (不满意影响力) = - (O + M) / (A + O + M + I)
    支持：
    1) 传入 counts_records 汇总计数记录（用于单元测试与汇总表分析）；
    2) 传入问卷正反向配对列 (df, pos_cols, neg_cols) 或由 df 自动识别成对 KANO 列。
    """
    rows = []
    n_respondents = 0
    if counts_records is not None:
        for rec in counts_records:
            nm = str(rec.get("功能", rec.get("feature", "功能项")))
            A = int(rec.get("A", 0))
            O = int(rec.get("O", 0))
            M = int(rec.get("M", 0))
            I = int(rec.get("I", 0))
            R = int(rec.get("R", 0))
            Q = int(rec.get("Q", 0))
            denom = A + O + M + I
            n_respondents = max(n_respondents, denom + R + Q)
            better = (A + O) / denom if denom > 0 else 0.0
            worse = - (O + M) / denom if denom > 0 else 0.0
            rows.append({
                "功能需求": nm, "A(魅力)": A, "O(期望)": O, "M(必备)": M,
                "I(无差异)": I, "R(反向)": R, "Q(可疑)": Q,
                "Better系数": round(better, 4),
                "Worse系数": round(worse, 4),
                "Worse绝对值": round(abs(worse), 4),
            })
    elif isinstance(df, pd.DataFrame) and not df.empty:
        if not (pos_cols and neg_cols):
            k_info = detect_kano_columns(df)
            if not k_info["enabled"]:
                raise ValueError("当前数据集未检测到成对的正向/反向 KANO 评价列，无法执行 KANO 分析。")
            pos_cols = k_info["pos_cols"]
            neg_cols = k_info["neg_cols"]
            if not feature_names:
                feature_names = k_info["feature_names"]
        if len(pos_cols) != len(neg_cols) or len(pos_cols) == 0:
            raise ValueError("KANO 模型正向题项列与反向题项列数量必须一致且不少于 1 对。")
        names = feature_names if (feature_names and len(feature_names) == len(pos_cols)) else [f"功能{i+1}" for i in range(len(pos_cols))]
        for nm, pc, nc in zip(names, pos_cols, neg_cols):
            sub = df[[pc, nc]].apply(pd.to_numeric, errors="coerce").dropna()
            n_respondents = max(n_respondents, len(sub))
            cats = [
                _KANO_EVAL_MATRIX.get((int(round(r[pc])), int(round(r[nc]))), "I")
                for _, r in sub.iterrows()
            ]
            s_cat = pd.Series(cats).value_counts()
            A = int(s_cat.get("A", 0))
            O = int(s_cat.get("O", 0))
            M = int(s_cat.get("M", 0))
            I = int(s_cat.get("I", 0))
            R = int(s_cat.get("R", 0))
            Q = int(s_cat.get("Q", 0))
            denom = A + O + M + I
            better = (A + O) / denom if denom > 0 else 0.0
            worse = - (O + M) / denom if denom > 0 else 0.0
            rows.append({
                "功能需求": nm, "A(魅力)": A, "O(期望)": O, "M(必备)": M,
                "I(无差异)": I, "R(反向)": R, "Q(可疑)": Q,
                "Better系数": round(better, 4),
                "Worse系数": round(worse, 4),
                "Worse绝对值": round(abs(worse), 4),
            })
    else:
        raise ValueError("KANO 模型缺少有效的成对正反向问卷数据列或统计输入。")

    kano_df = pd.DataFrame(rows)
    split_better = 0.50
    split_worse = 0.50
    quadrants = []
    short_types = []
    for _, r in kano_df.iterrows():
        b_hi = r["Better系数"] >= split_better
        w_hi = r["Worse绝对值"] >= split_worse
        if b_hi and not w_hi:
            quadrants.append("魅力属性 (A)")
            short_types.append("A")
        elif b_hi and w_hi:
            quadrants.append("期望属性 (O)")
            short_types.append("O")
        elif not b_hi and w_hi:
            quadrants.append("必备属性 (M)")
            short_types.append("M")
        else:
            quadrants.append("无差异属性 (I)")
            short_types.append("I")

    kano_df["KANO属性归类"] = quadrants
    kano_df["属性代码"] = short_types

    m_items = kano_df.loc[kano_df["属性代码"] == "M", "功能需求"].tolist()
    o_items = kano_df.loc[kano_df["属性代码"] == "O", "功能需求"].tolist()
    a_items = kano_df.loc[kano_df["属性代码"] == "A", "功能需求"].tolist()

    conclusion = (
        f"KANO 模型分析显示：必备属性（M）为{'、'.join(m_items) if m_items else '无'}，"
        f"期望属性（O）为{'、'.join(o_items) if o_items else '无'}，"
        f"魅力属性（A）为{'、'.join(a_items) if a_items else '无'}。"
    )
    return {
        "table": kano_df,
        "n_valid": n_respondents,
        "split_better": split_better,
        "split_worse": split_worse,
        "conclusion": conclusion,
    }


def calc_conjoint_analysis(
    profiles_df: pd.DataFrame = None,
    attr_cols: list = None,
    rating_col: str = None,
) -> dict:
    """
    S3 联合分析 (Conjoint Analysis，纯真实数据 OLS 估计，无合成回退)：
    基于轮廓实验虚拟变量效应编码 OLS 回归估计各属性水平的部分价值（Part-Worth Utility），
    对每个属性内部严格执行零中心化（Zero-Centered Utilities），并计算属性极差与相对重要性（总和 = 100.0%）。
    """
    if not isinstance(profiles_df, pd.DataFrame) or profiles_df.empty:
        raise ValueError("联合分析需要传入包含轮廓属性列与总体评分列的真实数据集。")

    if not attr_cols or not rating_col:
        cj_info = detect_conjoint_columns(profiles_df)
        if not cj_info["enabled"]:
            raise ValueError("当前数据集未检测到联合分析属性水平列与总体评分列。")
        attr_cols = attr_cols or cj_info["attr_cols"]
        rating_col = rating_col or cj_info["rating_col"]

    sub = profiles_df[attr_cols + [rating_col]].dropna().copy()
    sub[rating_col] = pd.to_numeric(sub[rating_col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 6:
        raise ValueError("联合分析有效评价记录不足 6 条，无法稳健估计部分价值效用。")

    y = sub[rating_col].to_numpy(dtype=float)
    X_parts = [np.ones((len(sub), 1))]
    attr_levels_map = {}
    col_slices = {}
    curr_idx = 1

    for ac in attr_cols:
        levels = list(pd.unique(sub[ac].astype(str)))
        attr_levels_map[ac] = levels
        if len(levels) > 1:
            dummies = pd.get_dummies(sub[ac].astype(str), dtype=float)[levels[:-1]].to_numpy(copy=True)
            last_mask = (sub[ac].astype(str) == levels[-1]).to_numpy()
            dummies[last_mask, :] = -1.0
            X_parts.append(dummies)
            col_slices[ac] = (curr_idx, curr_idx + len(levels) - 1)
            curr_idx += len(levels) - 1
        else:
            col_slices[ac] = (curr_idx, curr_idx)

    X_mat = np.hstack(X_parts)
    betas, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    y_hat = X_mat @ betas
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def _clean_attr_label(col_nm: str) -> str:
        s = str(col_nm).strip()
        for pre in ("联合属性_", "轮廓属性_", "conjoint_attr_"):
            if s.startswith(pre):
                return s[len(pre):]
        return s

    utility_rows = []
    importance_rows = []
    ranges = {}
    raw_utils_by_attr = {}
    clean_names = {}

    for ac in attr_cols:
        disp_nm = _clean_attr_label(ac)
        clean_names[ac] = disp_nm
        levels = attr_levels_map[ac]
        s_i, e_i = col_slices[ac]
        if e_i > s_i:
            b_sub = betas[s_i:e_i]
            u_vals = np.append(b_sub, -np.sum(b_sub))
        else:
            u_vals = np.zeros(len(levels))
        u_vals = u_vals - np.mean(u_vals)
        # 四舍五入后修正末项残差，严格保证每个属性内部效用值之和精确为 0.0000
        u_rounded = np.round(u_vals, 4)
        if len(u_rounded) > 1:
            u_rounded[-1] = round(-float(np.sum(u_rounded[:-1])), 4)
        raw_utils_by_attr[ac] = (levels, u_rounded)
        ranges[ac] = float(np.max(u_rounded) - np.min(u_rounded))

    total_range = sum(ranges.values())
    if total_range <= 0:
        total_range = 1.0

    for ac in attr_cols:
        disp_nm = clean_names[ac]
        levels, u_rounded = raw_utils_by_attr[ac]
        imp_pct = ranges[ac] / total_range * 100.0
        importance_rows.append({
            "属性名称": disp_nm,
            "效用极差(Range)": round(ranges[ac], 4),
            "相对重要性(%)": imp_pct,
        })
        for lv, uv in zip(levels, u_rounded):
            utility_rows.append({
                "属性名称": disp_nm,
                "属性水平": lv,
                "部分价值(Utility)": float(uv),
            })

    imp_df = pd.DataFrame(importance_rows).sort_values("相对重要性(%)", ascending=False).reset_index(drop=True)
    rounded_imps = imp_df["相对重要性(%)"].round(1).to_numpy(copy=True)
    diff = round(100.0 - float(np.sum(rounded_imps)), 1)
    if len(rounded_imps) > 0 and diff != 0:
        rounded_imps[0] = round(rounded_imps[0] + diff, 1)
    imp_df["相对重要性(%)"] = rounded_imps
    util_df = pd.DataFrame(utility_rows)

    best_levels = []
    for ac_nm in imp_df["属性名称"]:
        sub_u = util_df[util_df["属性名称"] == ac_nm].sort_values("部分价值(Utility)", ascending=False)
        if not sub_u.empty:
            best_levels.append(f"{sub_u.iloc[0]['属性水平']}")
    top_attr = imp_df.iloc[0]
    sec_attr = imp_df.iloc[1] if len(imp_df) > 1 else top_attr
    conclusion = (
        f"联合分析显示：{top_attr['属性名称']}对决策影响最大（相对重要性 {top_attr['相对重要性(%)']:.1f}%），"
        f"{sec_attr['属性名称']}次之（{sec_attr['相对重要性(%)']:.1f}%）；"
        f"最优轮廓组合为「{' + '.join(best_levels)}」。"
    )
    return {
        "importance_df": imp_df,
        "utility_df": util_df,
        "importance_sum": round(float(imp_df["相对重要性(%)"].sum()), 4),
        "best_combo": " + ".join(best_levels),
        "r2": r2,
        "n_valid": len(sub),
        "conclusion": conclusion,
    }


def calc_word_cooccurrence(series: pd.Series, stopwords: set = None, top_n: int = 16, min_cooc: int = 4) -> dict:
    """计算开放题高频词共现网络（节点词频与同句共现边权重，纯实测文本统计）。"""
    sw = set(DEFAULT_STOPWORDS)
    if stopwords:
        sw |= set(stopwords)

    docs_tokens = []
    word_counts = Counter()
    clean_texts = series.dropna().astype(str).str.strip()
    clean_texts = clean_texts[~clean_texts.isin(["", "nan", "None", "无", "没有"])]

    for text in clean_texts:
        words = [w.strip() for w in jieba.lcut(text) if len(w.strip()) >= 2 and w.strip() not in sw and not w.strip().isdigit()]
        uniq_w = sorted(set(words))
        docs_tokens.append(uniq_w)
        for w in words:
            word_counts[w] += 1

    top_words = [w for w, _ in word_counts.most_common(top_n)]
    top_set = set(top_words)

    pair_counts = Counter()
    for uniq_w in docs_tokens:
        filtered = [w for w in uniq_w if w in top_set]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                pair = (filtered[i], filtered[j])
                pair_counts[pair] += 1

    nodes_df = pd.DataFrame([
        {"词语": w, "词频": int(word_counts[w])}
        for w in top_words
    ])
    edges_rows = [
        {"源词": u, "目标词": v, "共现频次": int(cnt)}
        for (u, v), cnt in pair_counts.most_common()
        if cnt >= min_cooc
    ]
    if not edges_rows and pair_counts:
        edges_rows = [
            {"源词": u, "目标词": v, "共现频次": int(cnt)}
            for (u, v), cnt in pair_counts.most_common(15)
        ]
    edges_df = pd.DataFrame(edges_rows, columns=["源词", "目标词", "共现频次"])
    if not edges_df.empty:
        top_edge = edges_df.iloc[0]
        conclusion = (
            f"高频词共现网络显示：「{top_edge['源词']}—{top_edge['目标词']}」同句共现频次最高（{top_edge['共现频次']} 次），"
            f"共提取 {len(nodes_df)} 个核心语义节点与 {len(edges_df)} 条共现关联。"
        )
    else:
        conclusion = f"共提取 {len(nodes_df)} 个高频词语，但同句共现频次较低，未形成显著语义共现簇。"
    return {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "n_valid_texts": int(len(clean_texts)),
        "conclusion": conclusion,
    }


def calc_sentiment_analysis(series: pd.Series) -> dict:
    """开放题评论情感极性及细分主题维度情感分布分析（100% 基于实测文本词典命中计算，无写死数值）。"""
    pos_words = {
        "醇厚", "好喝", "性价比", "划算", "惊艳", "喜欢", "推荐", "快", "方便", "清爽",
        "顺滑", "稳定", "香气", "满意", "优秀", "丰富", "合理", "热情", "整洁", "安静",
        "漂亮", "细腻", "新鲜", "回甘", "精致", "友好", "舒服", "省心", "实在", "充足",
        "好", "棒", "赞", "不错", "贴心", "优质", "清晰", "流畅", "实惠",
    }
    neg_words = {
        "偏高", "慢", "排队", "淡", "贵", "少", "不稳定", "洒", "漏洒", "差",
        "失望", "卡顿", "拥挤", "涨", "单薄", "难", "负担", "抱怨", "糟糕", "不满",
        "繁琐", "混乱", "嘈杂", "敷衍", "延迟", "太久", "不便",
    }

    clean_s = series.dropna().astype(str).str.strip()
    clean_s = clean_s[~clean_s.isin(["", "nan", "None", "无", "没有"])]
    texts = clean_s.tolist()

    def _score_text(txt: str) -> str:
        p_hit = sum(1 for w in pos_words if w in txt)
        n_hit = sum(1 for w in neg_words if w in txt)
        if p_hit > n_hit:
            return "正向"
        if n_hit > p_hit:
            return "负向"
        return "中性"

    sentiments = [_score_text(t) for t in texts]
    s_cnt = pd.Series(sentiments, dtype=object).value_counts()
    total = max(len(sentiments), 1)
    overall_df = pd.DataFrame([
        {"情感极性": k, "评论数": int(s_cnt.get(k, 0)), "占比(%)": round(s_cnt.get(k, 0) / total * 100.0, 1)}
        for k in ["正向", "中性", "负向"]
    ])

    default_aspects = {
        "口味与品质": ["口感", "醇厚", "香气", "味道", "风味", "回甘", "顺滑", "咖啡豆", "冷萃", "生椰", "拿铁", "冰块", "品质"],
        "新品与品类": ["新品", "季节", "限定", "联名", "手冲", "低因", "燕麦奶", "植物奶", "早餐", "面包", "品类"],
        "门店与包装": ["门店", "环境", "安静", "整洁", "座位", "办公", "包装", "漏洒", "杯量", "杯型", "插座"],
        "价格性价比": ["价格", "性价比", "优惠", "划算", "便宜", "涨", "贵", "负担", "积分", "福利", "门槛"],
        "服务与效率": ["出餐", "速度", "排队", "服务", "店员", "动线", "外卖", "小程序", "卡顿", "预约", "品控"],
    }

    aspect_rows = []
    for asp_nm, kws in default_aspects.items():
        matched_txts = [t for t in texts if any(kw in t for kw in kws)]
        if len(matched_txts) >= 2:
            sub_sents = [_score_text(t) for t in matched_txts]
            n_m = len(sub_sents)
            pp = round(sub_sents.count("正向") / n_m * 100.0, 1)
            neup = round(sub_sents.count("中性") / n_m * 100.0, 1)
            negp = round(100.0 - pp - neup, 1)
            aspect_rows.append({
                "评价维度": asp_nm,
                "提及样本数": n_m,
                "正向占比(%)": pp,
                "中性占比(%)": neup,
                "负向占比(%)": max(0.0, negp),
                "净情感得分(NSS)": round(pp - max(0.0, negp), 1),
            })

    # 若通用非餐饮问卷未命中默认词簇，则自动提取前 5 高频主题词计算各主题情感极性
    if len(aspect_rows) < 2 and texts:
        txt_info = analyze_open_text(clean_s, top_n=5)
        for _, r_w in txt_info["freq_df"].head(5).iterrows():
            w = str(r_w["词语"])
            matched_txts = [t for t in texts if w in t]
            if not matched_txts:
                continue
            sub_sents = [_score_text(t) for t in matched_txts]
            n_m = len(sub_sents)
            pp = round(sub_sents.count("正向") / n_m * 100.0, 1)
            neup = round(sub_sents.count("中性") / n_m * 100.0, 1)
            negp = round(100.0 - pp - neup, 1)
            aspect_rows.append({
                "评价维度": f"「{w}」相关",
                "提及样本数": n_m,
                "正向占比(%)": pp,
                "中性占比(%)": neup,
                "负向占比(%)": max(0.0, negp),
                "净情感得分(NSS)": round(pp - max(0.0, negp), 1),
            })

    if not aspect_rows:
        pp = float(overall_df.loc[overall_df["情感极性"] == "正向", "占比(%)"].iloc[0])
        neup = float(overall_df.loc[overall_df["情感极性"] == "中性", "占比(%)"].iloc[0])
        negp = float(overall_df.loc[overall_df["情感极性"] == "负向", "占比(%)"].iloc[0])
        aspect_rows = [{
            "评价维度": "总体评价",
            "提及样本数": len(texts),
            "正向占比(%)": pp,
            "中性占比(%)": neup,
            "负向占比(%)": negp,
            "净情感得分(NSS)": round(pp - negp, 1),
        }]

    aspect_df = pd.DataFrame(aspect_rows).sort_values("净情感得分(NSS)", ascending=False).reset_index(drop=True)
    pos_pct = float(overall_df.loc[overall_df["情感极性"] == "正向", "占比(%)"].iloc[0])
    neg_pct = float(overall_df.loc[overall_df["情感极性"] == "负向", "占比(%)"].iloc[0])
    top_asp = aspect_df.iloc[0]
    bot_asp = aspect_df.sort_values("负向占比(%)", ascending=False).iloc[0]
    conclusion = (
        f"开放题情感分析显示（有效文本 n={len(texts)}）：整体正向评论占 {pos_pct:.1f}%、负向占 {neg_pct:.1f}%；"
        f"细分主题中「{top_asp['评价维度']}」净情感最高（正向 {top_asp['正向占比(%)']:.1f}%），"
        f"「{bot_asp['评价维度']}」负向反馈占比为 {bot_asp['负向占比(%)']:.1f}%。"
    )
    return {
        "overall_df": overall_df,
        "aspect_df": aspect_df,
        "n_valid_texts": len(texts),
        "conclusion": conclusion,
    }


def _prepare_ml_features(df: pd.DataFrame, target_col: str = None, feature_cols: list = None):
    """内部辅助：构建分类预测模型的特征矩阵 X 与二分类目标 y（自适应通用数据集与咖啡基准数据集）。"""
    work = df.copy()
    if "品质量表3_正向" not in work.columns and "品质量表3_品质不稳定_反向" in work.columns:
        work["品质量表3_正向"] = reverse_score(work["品质量表3_品质不稳定_反向"], max_scale=5)
    if all(c in work.columns for c in ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"]):
        work["品质感知均分"] = compute_scale_composite(work, ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"])
    if all(c in work.columns for c in ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"]):
        work["价格感知均分"] = compute_scale_composite(work, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"])

    if "消费频次" in work.columns:
        freq_map = {"每天1杯及以上": 4, "每周3-5杯": 3, "每周1-2杯": 2, "每月1-3杯": 1, "偶尔饮用": 1, "几乎不喝": 0}
        work["消费频次得分"] = work["消费频次"].map(freq_map).fillna(2)
    if "年龄段" in work.columns:
        age_map = {"18-24岁": 1, "25-30岁": 2, "31-35岁": 3, "31-40岁": 3, "36-45岁": 4, "40岁以上": 5, "46岁及以上": 5}
        work["年龄段编码"] = work["年龄段"].map(age_map).fillna(2)

    if target_col is None or target_col not in work.columns:
        if "NPS推荐度" in work.columns:
            target_col = "NPS推荐度"
        else:
            qt = auto_detect_question_types(work)
            cands = [c for c, t in qt.items() if t in ("NPS题", "数值题", "量表题")]
            if not cands:
                raise ValueError("当前数据集缺少可用于预测建模的数值/量表目标列。")
            target_col = cands[-1]

    feature_map = {
        "品质感知均分": "品质感知均分",
        "价格感知均分": "价格感知均分",
        "月消费额": "月消费额",
        "满意度1_口味口感": "口味满意度",
        "满意度2_价格实惠": "价格满意度",
        "消费频次得分": "消费频次",
        "满意度4_出餐速度": "速度满意度",
        "满意度3_门店环境": "环境满意度",
        "年龄段编码": "年龄段",
    }
    if feature_cols:
        raw_cols = [c for c in feature_cols if c in work.columns and c != target_col]
        rename_dict = {c: str(c) for c in raw_cols}
    else:
        raw_cols = [c for c in feature_map.keys() if c in work.columns and c != target_col]
        if len(raw_cols) >= 2:
            rename_dict = feature_map
        else:
            qt = auto_detect_question_types(work)
            dur_c = detect_duration_column(work)
            raw_cols = [
                c for c, t in qt.items()
                if t in ("量表题", "数值题", "NPS题") and c != target_col and c != dur_c
                and not str(c).startswith(("KANO正向_", "KANO反向_"))
            ]
            rename_dict = {c: str(c) for c in raw_cols}

    if len(raw_cols) < 2:
        raise ValueError("可用数值/量表特征不足 2 列，无法训练分类树/随机森林模型。")

    sub = work[raw_cols + [target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 20:
        raise ValueError("有效完整样本不足 20 份，无法执行训练/测试集划分与树模型评估。")

    X = sub[raw_cols].rename(columns=rename_dict)
    t_s = sub[target_col]
    if float(t_s.max()) >= 9 and float(t_s.min()) >= 0 and float(t_s.max()) <= 10:
        y = (t_s >= 9).astype(int)
        pos_label_desc = f"{target_col}≥9"
    else:
        med_v = float(t_s.median())
        y = (t_s > med_v).astype(int)
        if y.nunique() < 2:
            y = (t_s >= med_v).astype(int)
        pos_label_desc = f"{target_col}>{med_v:.1f}"
    return X, y, pos_label_desc


def calc_random_forest_model(df: pd.DataFrame, target_col: str = None, feature_cols: list = None, seed: int = 42) -> dict:
    """随机森林分类模型：输出特征重要性排序及测试集混淆矩阵。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    X, y, pos_desc = _prepare_ml_features(df, target_col=target_col, feature_cols=feature_cols)
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=strat
    )
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    imp_df = pd.DataFrame({
        "特征名称": list(X.columns),
        "重要性得分": np.round(rf.feature_importances_, 4),
        "重要性占比(%)": np.round(rf.feature_importances_ * 100.0, 1),
    }).sort_values("重要性得分", ascending=False).reset_index(drop=True)
    imp_df["累计重要性(%)"] = imp_df["重要性占比(%)"].cumsum().round(1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    top2 = imp_df.head(2)["特征名称"].tolist()
    conclusion = (
        f"随机森林模型显示：{'与'.join(top2)}是预测高水平组（{pos_desc}）的最核心特征"
        f"（合计贡献 {imp_df.head(2)['重要性占比(%)'].sum():.1f}%），测试集准确率达 {acc * 100:.1f}%。"
    )
    return {
        "importance_df": imp_df,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_label_desc": pos_desc,
        "conclusion": conclusion,
    }


def calc_decision_tree_model(df: pd.DataFrame, target_col: str = None, feature_cols: list = None, max_depth: int = 3, seed: int = 42) -> dict:
    """决策树规则提取模型：输出树结构对象与关键节点分群规则表。"""
    from sklearn.tree import DecisionTreeClassifier

    X, y, pos_desc = _prepare_ml_features(df, target_col=target_col, feature_cols=feature_cols)
    min_leaf = max(3, min(15, len(X) // 12))
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)
    dt.fit(X, y)
    acc = float(dt.score(X, y))

    tree_ = dt.tree_
    feature_names = list(X.columns)
    nodes_info = []
    for node_id in range(tree_.node_count):
        n_s = int(tree_.n_node_samples[node_id])
        val = tree_.value[node_id][0]
        pos_rate = float(val[1] / np.sum(val) * 100.0) if len(val) > 1 and np.sum(val) > 0 else 0.0
        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
        feat_nm = "叶节点" if is_leaf else feature_names[tree_.feature[node_id]]
        thresh = None if is_leaf else float(tree_.threshold[node_id])
        nodes_info.append({
            "节点ID": node_id,
            "是否叶节点": is_leaf,
            "分裂特征": feat_nm,
            "分裂阈值": round(thresh, 2) if thresh is not None else np.nan,
            "样本数": n_s,
            "推荐者占比(%)": round(pos_rate, 1),
            "左子节点": int(tree_.children_left[node_id]),
            "右子节点": int(tree_.children_right[node_id]),
        })
    nodes_df = pd.DataFrame(nodes_info)
    leaf_df = nodes_df[nodes_df["是否叶节点"]].sort_values("推荐者占比(%)", ascending=False)
    best_leaf = leaf_df.iloc[0]
    conclusion = (
        f"决策树规则（深度={max_depth}）显示：根节点按「{nodes_df.iloc[0]['分裂特征']}」分裂，"
        f"最高转化叶节点目标占比达 {best_leaf['推荐者占比(%)']:.1f}%（n={best_leaf['样本数']}，整体准确率 {acc * 100:.1f}%）。"
    )
    return {
        "model": dt,
        "feature_names": feature_names,
        "nodes_df": nodes_df,
        "accuracy": acc,
        "pos_label_desc": pos_desc,
        "conclusion": conclusion,
    }


def calc_province_distribution(df: pd.DataFrame, col: str = None) -> dict:
    """
    区域样本分布统计（用于 44-区域分布地图）。
    严格只接收真实数据集中的省份/地域列，若数据中不含有效省份列则抛出异常（无内置假数据回退）。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("区域分布统计需要传入非空数据集。")
    target_col = col if (col and col in df.columns) else detect_province_column(df)
    if not target_col or target_col not in df.columns:
        raise ValueError("当前数据集不包含可识别的省份/地域列，无法绘制区域分布地图。")

    raw_s = df[target_col].dropna().astype(str).str.strip()
    raw_s = raw_s[~raw_s.isin(["", "nan", "None"])]
    mapped_s = raw_s.map(_normalize_province_name)
    valid_s = mapped_s[mapped_s != ""]
    if valid_s.empty:
        raise ValueError(f"列「{target_col}」未匹配到有效的中国省份名称。")

    vc = valid_s.value_counts()
    counts = {str(k): int(v) for k, v in vc.items()}
    total = max(sum(counts.values()), 1)
    rows = [
        {"省份": k, "样本数": v, "占比(%)": round(v / total * 100.0, 1)}
        for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    ]
    prov_df = pd.DataFrame(rows)
    yrd_n = sum(counts.get(p, 0) for p in ["安徽", "江苏", "上海", "浙江"])
    yrd_pct = yrd_n / total * 100.0
    top1 = prov_df.iloc[0]
    top3_names = "、".join(prov_df.head(3)["省份"].tolist())
    top3_pct = float(prov_df.head(3)["占比(%)"].sum())

    if yrd_pct >= 50.0:
        conclusion = (
            f"区域分布显示：{top1['省份']}样本量居首（n={top1['样本数']}，占 {top1['占比(%)']:.1f}%），"
            f"长三角四省市（皖苏沪浙）合计占样本 {yrd_pct:.1f}%（{yrd_n}/{total}）。"
        )
    else:
        conclusion = (
            f"区域分布显示：{top1['省份']}样本量居首（n={top1['样本数']}，占 {top1['占比(%)']:.1f}%），"
            f"前三大省份（{top3_names}）合计占样本 {top3_pct:.1f}%（共覆盖 {len(counts)} 个省区市）。"
        )
    return {
        "col": target_col,
        "counts": counts,
        "table": prov_df,
        "total": total,
        "yrd_n": yrd_n,
        "yrd_pct": yrd_pct,
        "conclusion": conclusion,
    }


# ==============================================================================
# 模块十一：Word 报告、PPT 演示文稿与清洗后 Excel 导出
# ==============================================================================

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from pptx import Presentation
from pptx.util import Inches as PptInches, Pt as PptPt
from pptx.dml.color import RGBColor as PptRGBColor
from pptx.enum.text import PP_ALIGN


def _add_docx_dataframe(doc: Document, df: pd.DataFrame, max_rows: int = 30):
    """在 Word 文档中插入格式化的 DataFrame 表格。"""
    if df is None or df.empty:
        return
    sub = df.head(max_rows)
    table = doc.add_table(rows=1, cols=len(sub.columns))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for j, col_name in enumerate(sub.columns):
        hdr_cells[j].text = str(col_name)
        for p in hdr_cells[j].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)

    for _, row in sub.iterrows():
        row_cells = table.add_row().cells
        for j, val in enumerate(row):
            row_cells[j].text = "" if pd.isna(val) else str(val)
            for p in row_cells[j].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)


def build_word_report(
    overview_res: dict,
    qc_res: dict,
    report_sections: list,
    is_demo: bool = False,
    is_weighted: bool = False,
    theme_name: str = "暖纸杂志",
) -> bytes:
    """委托 ppt_report.build_word_report 生成含完整图表与源数据附录的 Word 报告 (.docx)。"""
    import ppt_report as pr
    return pr.build_word_report(
        overview_res=overview_res,
        qc_res=qc_res,
        report_sections=report_sections,
        is_demo=is_demo,
        is_weighted=is_weighted,
        theme_name=theme_name,
    )


def build_ppt_report(
    overview_res: dict,
    qc_res: dict,
    report_sections: list = None,
    is_demo: bool = False,
    is_weighted: bool = False,
    theme_name: str = "暖纸杂志",
    df: pd.DataFrame = None,
    export_options: dict = None,
) -> bytes:
    """委托 ppt_report.build_ppt_report 生成出版级 16:9 完整 PPT 演示报告 (.pptx)。"""
    import ppt_report as pr
    return pr.build_ppt_report(
        overview_res=overview_res,
        qc_res=qc_res,
        report_sections=report_sections,
        is_demo=is_demo,
        is_weighted=is_weighted,
        theme_name=theme_name,
        df=df,
        export_options=export_options,
    )


def export_cleaned_excel(df: pd.DataFrame) -> bytes:
    """导出清洗及加工后的有效样本数据为 Excel (.xlsx) 字节流。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="清洗后有效数据")
    return buf.getvalue()

