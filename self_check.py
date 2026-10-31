# -*- coding: utf-8 -*-
"""
SurveyScope 验收基准自检脚本 (self_check.py)
严格按照第六部分 A1–A18 给定测试数据逐条验证 stats_utils.py 与 chart_utils.py 的计算与导出结果。
支持按阶段运行：
  python self_check.py --stage 1
  python self_check.py --stage 2
  python self_check.py --stage 3
  python self_check.py --stage 4
  python self_check.py --stage all
"""

import io
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from docx import Document
from pptx import Presentation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import stats_utils as su
import chart_utils as cu


def check_A1():
    """A1 描述统计：对数据 [1,2,3,4,5]，均值 = 3.0，样本标准差 ≈ 1.581。"""
    res = su.analyze_numeric_question([1, 2, 3, 4, 5])
    mean_v = res["mean"]
    std_v = res["std"]
    ok = np.isclose(mean_v, 3.0, atol=1e-6) and np.isclose(std_v, 1.58113883, atol=1e-3)
    print(f"[A1 描述统计] 输入=[1,2,3,4,5] -> 实测均值={mean_v:.4f} (基准=3.0), 实测样本标准差={std_v:.4f} (基准≈1.581) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A1 自检未通过"


def check_A2():
    """A2 卡方检验：对列联表 [[50,30],[30,50]]（不做连续性校正），χ² = 10.0，p ≈ 0.0016。"""
    res = su.crosstab_chi2_analysis([[50, 30], [30, 50]])
    chi2_v = res["chi2"]
    p_v = res["p_value"]
    ok = np.isclose(chi2_v, 10.0, atol=1e-6) and np.isclose(p_v, 0.0015654, atol=1e-4)
    print(f"[A2 卡方检验] 输入=[[50,30],[30,50]] -> 实测 χ²={chi2_v:.4f} (基准=10.0), 实测 p={p_v:.4f} (基准≈0.0016), Cramér's V={res['cramers_v']:.4f} | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A2 自检未通过"


def check_A3():
    """A3 相关：x = [1,2,3,4,5]，y = [2,4,5,4,5]，Pearson r ≈ 0.775。"""
    res = su.correlation_pair_analysis([1, 2, 3, 4, 5], [2, 4, 5, 4, 5], method="pearson")
    r_v = res["r"]
    ok = np.isclose(r_v, 0.77459667, atol=1e-3)
    print(f"[A3 相关分析] x=[1,2,3,4,5], y=[2,4,5,4,5] -> 实测 Pearson r={r_v:.4f} (基准≈0.775), p={res['p_value']:.4f} | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A3 自检未通过"


def check_A4():
    """A4 信度：对数据 [[4,4,5],[3,3,4],[5,5,5],[2,2,3],[4,3,4]]，Cronbach's α ≈ 0.962。"""
    data = [[4, 4, 5], [3, 3, 4], [5, 5, 5], [2, 2, 3], [4, 3, 4]]
    res = su.calc_reliability(data)
    alpha_v = res["alpha"]
    ok = np.isclose(alpha_v, 0.9619565, atol=1e-3)
    print(f"[A4 信度分析] 5×3量表矩阵 -> 实测 Cronbach's α={alpha_v:.4f} (基准≈0.962) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A4 自检未通过"


def check_A5():
    """A5 多选拆分：对「拿铁；美式」和「美式」两行数据拆分统计，「美式」计数 = 2。"""
    s = pd.Series(["拿铁；美式", "美式"])
    res = su.split_multiple_choice(s)
    ame_cnt = res["counts"].get("美式", 0)
    lat_cnt = res["counts"].get("拿铁", 0)
    ok = (ame_cnt == 2) and (lat_cnt == 1)
    print(f"[A5 多选拆分] 输入=['拿铁；美式', '美式'] -> 实测「美式」计数={ame_cnt} (基准=2), 「拿铁」计数={lat_cnt} | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A5 自检未通过"


def check_A6():
    """A6 线性回归：x = [1,2,3,4,5]，y = [3,5,7,9,11]，斜率 = 2.0，截距 = 1.0，R² = 1.0。"""
    res = su.run_linear_regression([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    slope_v = res["slope"]
    inter_v = res["intercept"]
    r2_v = res["r2"]
    ok = np.isclose(slope_v, 2.0, atol=1e-6) and np.isclose(inter_v, 1.0, atol=1e-6) and np.isclose(r2_v, 1.0, atol=1e-6)
    print(f"[A6 线性回归] x=[1..5], y=[3,5,7,9,11] -> 实测斜率={slope_v:.4f} (基准=2.0), 截距={inter_v:.4f} (基准=1.0), R²={r2_v:.4f} (基准=1.0) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A6 自检未通过"


def check_A7():
    """A7 聚类：对点 (1,1)、(1,2)、(9,9)、(9,10) 做 k=2 的 K-Means，两类中心应 ≈ (1, 1.5) 与 (9, 9.5)，每类 2 个点。"""
    pts = [[1, 1], [1, 2], [9, 9], [9, 10]]
    res = su.run_kmeans_clustering(pts, k=2, standardize=True)
    centers = res["centers"]
    counts = res["counts"]
    ok = (
        np.allclose(centers[0], [1.0, 1.5], atol=1e-3)
        and np.allclose(centers[1], [9.0, 9.5], atol=1e-3)
        and counts == [2, 2]
    )
    print(f"[A7 K-Means聚类] 4个二维点(k=2) -> 实测中心1=({centers[0][0]:.2f}, {centers[0][1]:.2f}), 中心2=({centers[1][0]:.2f}, {centers[1][1]:.2f}), 各簇样本数={counts} (基准: (1, 1.5)与(9, 9.5), 每类2点) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A7 自检未通过"


def check_A8():
    """A8 加权：样本中男 80 人、女 20 人，目标占比各 50% 时，权重应为男 0.625、女 2.5，加权后男性占比 = 50%。"""
    s = pd.Series(["男"] * 80 + ["女"] * 20)
    res = su.compute_post_stratification_weights(s, {"男": 0.5, "女": 0.5})
    w_male = res["weights_by_cat"]["男"]
    w_female = res["weights_by_cat"]["女"]
    tbl = res["table"].set_index("类别")
    male_weighted_pct = tbl.loc["男", "加权后占比(%)"]
    ok = np.isclose(w_male, 0.625, atol=1e-6) and np.isclose(w_female, 2.5, atol=1e-6) and np.isclose(male_weighted_pct, 50.0, atol=1e-6)
    print(f"[A8 事后分层加权] 男80/女20, 目标各50% -> 实测男权重={w_male:.3f} (基准=0.625), 女权重={w_female:.3f} (基准=2.5), 加权后男性占比={male_weighted_pct:.1f}% (基准=50%) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A8 自检未通过"


def check_A9():
    """A9 NPS：推荐者 40 人、中立者 35 人、贬损者 25 人时，NPS = 15。"""
    res = su.analyze_nps_question(promoters=40, passives=35, detractors=25)
    nps_v = res["nps"]
    ok = np.isclose(nps_v, 15.0, atol=1e-6)
    print(f"[A9 NPS计算] 推荐者40/中立者35/贬损者25 -> 实测 NPS={nps_v:g} (基准=15) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A9 自检未通过"


def check_A10():
    """A10 抽样误差：n = 400、p = 0.5、95% 置信水平下，最大抽样误差 ≈ 4.9%。"""
    err_pct = su.calc_max_sampling_error(n=400, p=0.5, confidence=0.95)
    ok = np.isclose(err_pct, 4.9, atol=0.05)
    print(f"[A10 最大抽样误差] n=400, p=0.5, 95%置信水平 -> 实测最大抽样误差={err_pct:.2f}% (基准≈4.9%) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A10 自检未通过"


def check_A11():
    """A11 特征值：相关矩阵 [[1, 0.775],[0.775, 1]] 的两个特征值 ≈ 1.775 与 0.225。"""
    eigvals = su.calc_eigenvalues_from_corr([[1.0, 0.775], [0.775, 1.0]])
    ok = np.isclose(eigvals[0], 1.775, atol=1e-4) and np.isclose(eigvals[1], 0.225, atol=1e-4)
    print(f"[A11 特征值计算] 相关矩阵 [[1, 0.775],[0.775, 1]] -> 实测特征值=[{eigvals[0]:.4f}, {eigvals[1]:.4f}] (基准≈1.775 与 0.225) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A11 自检未通过"


def check_A12():
    """A12 直线作答：5 道量表题全部答 3 的受访者必须被标记；答案有任何变化的受访者不被标记。"""
    df_test = pd.DataFrame(
        [[3, 3, 3, 3, 3], [3, 3, 4, 3, 3], [1, 2, 3, 4, 5]],
        columns=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    flags = su.detect_straightlining(df_test, ["Q1", "Q2", "Q3", "Q4", "Q5"]).tolist()
    ok = flags == [True, False, False]
    print(f"[A12 直线作答识别] 样本1=[3,3,3,3,3], 样本2=[3,3,4,3,3], 样本3=[1,2,3,4,5] -> 实测标记={flags} (基准=[True, False, False]) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A12 自检未通过"


def check_A13():
    """A13 反向计分：5 点量表的 [1,2,5] 反向后 = [5,4,1]。"""
    rev = su.reverse_score([1, 2, 5], max_scale=5, min_scale=1)
    ok = rev == [5, 4, 1]
    print(f"[A13 反向计分] 5点量表输入=[1, 2, 5] -> 实测反向结果={rev} (基准=[5, 4, 1]) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A13 自检未通过"


def check_A14():
    """A14 导出文件：生成的 Word 和 PPT 文件能被 python-docx / python-pptx 重新打开读取，PPT 页数 ≥ 3。"""
    demo_df = su.generate_demo_coffee_data(seed=42, n=300)
    ov = su.get_dataset_overview(demo_df)
    qc = su.run_quality_control(demo_df, straight_cols=["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"], duration_col="答题时长（秒）", min_seconds=30)
    cat_res = su.analyze_categorical_question(qc["valid_df"]["性别"])
    fig_bar = cu.plot_categorical_bar(cat_res["table"], title="性别分布图")
    png_b = cu.fig_to_png_bytes(fig_bar)

    sections = [
        {"title": "单题分析：性别分布", "conclusion": cat_res["conclusion"], "table": cat_res["table"], "png_bytes": png_b},
        {"title": "单题分析：消费频次", "conclusion": "消费频次分布正常。", "table": cat_res["table"], "png_bytes": png_b},
    ]
    word_bytes = su.build_word_report(ov, qc, sections, is_demo=True)
    ppt_bytes = su.build_ppt_report(ov, qc, sections, is_demo=True)

    doc_reopen = Document(io.BytesIO(word_bytes))
    prs_reopen = Presentation(io.BytesIO(ppt_bytes))
    n_paras = len(doc_reopen.paragraphs)
    n_slides = len(prs_reopen.slides)
    ok = (n_paras > 5) and (n_slides >= 3)
    print(f"[A14 报告导出验证] Word 段落数={n_paras}, 表格数={len(doc_reopen.tables)}; PPT 幻灯片页数={n_slides} (基准: 可正常读取且 PPT页数>=3) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A14 自检未通过"


def check_A15():
    """A15 演示模式：示例数据能一键跑通全部模块，应用启动和运行无报错。"""
    df = su.generate_demo_coffee_data(seed=42, n=300)
    ov = su.get_dataset_overview(df)
    qtypes = su.auto_detect_question_types(df)
    mc = su.split_multiple_choice(df["偏好品类"])
    qc = su.run_quality_control(
        df,
        check_dup=True,
        check_straight=True,
        straight_cols=["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_品质不稳定_反向"],
        check_speed=True,
        duration_col="答题时长（秒）",
        min_seconds=30,
    )
    vdf = qc["valid_df"].copy()
    vdf["品质量表3_正向"] = su.reverse_score(vdf["品质量表3_品质不稳定_反向"], max_scale=5)
    vdf["价格维度均分"] = su.compute_scale_composite(vdf, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"])
    s_cat = su.analyze_categorical_question(vdf["性别"])
    s_num = su.analyze_numeric_question(vdf["月消费额"])
    s_nps = su.analyze_nps_question(vdf["NPS推荐度"])
    ct = su.crosstab_chi2_analysis(vdf, "消费频次", "性别")
    gm = su.group_mean_comparison(vdf, "性别", "月消费额")
    cp = su.correlation_pair_analysis(vdf["价格维度均分"], vdf["月消费额"])
    rel = su.calc_reliability(vdf, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"])
    efa = su.calc_validity_efa(vdf, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"], n_factors=2)
    ols = su.run_linear_regression(vdf, "月消费额", ["价格量表1_价格合理", "品质量表1_口感醇厚"])
    logit = su.run_logistic_regression(vdf, "是否喝咖啡", ["价格量表1_价格合理", "品质量表1_口感醇厚"])
    km = su.run_kmeans_clustering(vdf, ["价格量表1_价格合理", "品质量表1_口感醇厚", "月消费额"], k=3)
    txt = su.analyze_open_text(vdf["开放题短评"])
    kw = su.categorize_by_keywords(vdf["开放题短评"], {"口感品质": ["醇厚", "清爽"], "价格实惠": ["价格", "性价比"]})
    wt = su.compute_post_stratification_weights(vdf["性别"], {"男": 50, "女": 50})
    fn = su.calc_brand_funnel(vdf, ["品牌知名", "品牌考虑", "品牌购买", "品牌复购"])
    sw = su.calc_brand_switching(vdf, "上次购买品牌", "最常购买品牌")
    ca = su.calc_correspondence_analysis(vdf, "最常购买品牌", "偏好品类", is_multi_col=True)
    ipa = su.calc_ipa_analysis(
        vdf,
        ["重要性1_口味口感", "重要性2_价格实惠", "重要性3_门店环境", "重要性4_出餐速度"],
        ["满意度1_口味口感", "满意度2_价格实惠", "满意度3_门店环境", "满意度4_出餐速度"],
        ["口味口感", "价格实惠", "门店环境", "出餐速度"],
    )
    psm = su.calc_van_westendorp(vdf, "价格敏感_太便宜", "价格敏感_便宜", "价格敏感_贵", "价格敏感_太贵")

    ok = (
        len(df) == 300
        and qc["straight_n"] > 0
        and qc["speed_n"] > 0
        and qc["valid_n"] < 300
        and efa["kmo"] > 0.5
        and psm["opp"][0] > 0
    )
    print(
        f"[A15 演示模式全链路跑通] 生成样本={len(df)}份, 题型自动识别={len(qtypes)}列, "
        f"质控识别(直线={qc['straight_n']}, 过快={qc['speed_n']}, 有效={qc['valid_n']}), "
        f"NPS={s_nps['nps']:.1f}, α={rel['alpha']:.3f}, KMO={efa['kmo']:.3f}, "
        f"OLS R²={ols['r2']:.3f}, Logit AUC={logit['auc']:.3f}, 聚类k={km['k']}, "
        f"分词Top1='{txt['freq_df'].iloc[0]['词语']}', PSM最优价格点={psm['opp'][0]:.2f}元 | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "A15 自检未通过"


def check_A16():
    """A16 图表导出：任选一张 Plotly 图导出为 PNG，文件能正常打开、宽度 ≥ 1200px、中文不乱码不缺字。"""
    demo_df = su.generate_demo_coffee_data(seed=42, n=300)
    cat_res = su.analyze_categorical_question(demo_df["消费频次"])
    fig = cu.plot_categorical_bar(cat_res["table"], title="咖啡消费频次分布图（中文渲染测试）")
    png_bytes = cu.fig_to_png_bytes(fig, width=1360, height=760)
    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    ok = (w >= 1200) and (h >= 600) and (len(png_bytes) > 10000)
    print(f"[A16 图表导出PNG] 实测导出 PNG 尺寸={w}×{h}px (基准: 宽度>=1200px), 文件字节数={len(png_bytes)} bytes | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A16 自检未通过"


def check_A17():
    """A17 ROC：预测分数 [0.9, 0.8, 0.4, 0.2] 对应标签 [1, 1, 0, 0] 时，AUC = 1.0。"""
    res = su.calc_roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.4, 0.2])
    auc_v = res["auc"]
    ok = np.isclose(auc_v, 1.0, atol=1e-6)
    print(f"[A17 ROC与AUC] 预测分数=[0.9,0.8,0.4,0.2], 标签=[1,1,0,0] -> 实测 AUC={auc_v:.4f} (基准=1.0) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A17 自检未通过"


def check_A18():
    """A18 漏斗转化：各级人数依次为 300 → 210 → 120 → 60 时，逐级转化率依次为 70.0%、57.1%、50.0%（保留一位小数）。"""
    res = su.calc_brand_funnel([300, 210, 120, 60], stage_names=["知名", "考虑", "购买", "复购"])
    rates = res["step_rates"]
    formatted = [f"{r:.1f}%" for r in rates]
    ok = formatted == ["70.0%", "57.1%", "50.0%"]
    print(f"[A18 漏斗转化率] 各级人数=[300, 210, 120, 60] -> 实测逐级转化率={formatted} (基准=['70.0%', '57.1%', '50.0%']) | 结果: {'PASS' if ok else 'FAIL'}")
    assert ok, "A18 自检未通过"


# ==============================================================================
# S1–S8 新增专项自检（《04-图表与PPT全面优化要求》第 8 节）
# ==============================================================================
_CACHE_49_BUNDLES = None
_CACHE_PPTX_BYTES = None


def _get_valid_demo_df():
    df = su.generate_demo_coffee_data(seed=42, n=300, include_extended=True)
    qc = su.run_quality_control(
        df,
        straight_cols=["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_品质不稳定_反向"],
        duration_col="答题时长（秒）",
        min_seconds=30,
    )
    return df, qc, qc["valid_df"].copy()


def _get_cached_49_bundles():
    global _CACHE_49_BUNDLES
    if _CACHE_49_BUNDLES is None:
        _, _, vdf = _get_valid_demo_df()
        _CACHE_49_BUNDLES = cu.generate_all_49_charts(vdf, theme_name="暖纸杂志", is_demo=True)
    return _CACHE_49_BUNDLES


def _get_cached_pptx_bytes():
    global _CACHE_PPTX_BYTES
    if _CACHE_PPTX_BYTES is None:
        df, qc, vdf = _get_valid_demo_df()
        ov = su.get_dataset_overview(df)
        _CACHE_PPTX_BYTES = su.build_ppt_report(ov, qc, is_demo=True, theme_name="暖纸杂志", df=vdf)
    return _CACHE_PPTX_BYTES


def check_S1():
    """S1 结构方程路径图：基于 OLS 路径系数近似的标准化路径模型计算正确，路径系数与多元线性回归标准化系数完全一致。"""
    _, _, vdf = _get_valid_demo_df()
    sem = su.calc_sem_path_model(vdf)
    std_df = sem["std_df"]
    ols_direct = su.run_linear_regression(std_df, "NPS推荐度", ["价格感知", "品质感知"])
    ct = ols_direct["coef_table"]
    b_p_ols = float(ct.loc[ct["变量"] == "价格感知", "回归系数(B)"].iloc[0])
    b_q_ols = float(ct.loc[ct["变量"] == "品质感知", "回归系数(B)"].iloc[0])

    ok = (
        np.isclose(sem["beta_price"], b_p_ols, atol=1e-12)
        and np.isclose(sem["beta_qual"], b_q_ols, atol=1e-12)
        and np.isclose(sem["r2"], ols_direct["r2"], atol=1e-12)
    )
    print(
        f"[S1 结构方程路径模型] SEM路径系数(β_价格={sem['beta_price']:.4f}, β_品质={sem['beta_qual']:.4f}, R²={sem['r2']:.4f}) "
        f"vs 标准化多元线性回归(B_价格={b_p_ols:.4f}, B_品质={b_q_ols:.4f}, R²={ols_direct['r2']:.4f}) | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S1 自检未通过"


def check_S2():
    """S2 KANO 模型：Better = (A+O)/(A+O+M+I)、Worse = -(O+M)/(A+O+M+I) 计算正确，象限划分无误。"""
    test_records = [
        {"功能": "测试魅力项", "A": 60, "O": 10, "M": 5,  "I": 25},  # Better=0.70, Worse=-0.15 -> A
        {"功能": "测试期望项", "A": 20, "O": 50, "M": 15, "I": 15},  # Better=0.70, Worse=-0.65 -> O
        {"功能": "测试必备项", "A": 5,  "O": 20, "M": 55, "I": 20},  # Better=0.25, Worse=-0.75 -> M
        {"功能": "测试无差异", "A": 10, "O": 10, "M": 10, "I": 70},  # Better=0.20, Worse=-0.20 -> I
    ]
    res = su.calc_kano_model(counts_records=test_records)
    tdf = res["table"]
    b_list = tdf["Better系数"].tolist()
    w_list = tdf["Worse系数"].tolist()
    codes = tdf["属性代码"].tolist()
    ok = (
        np.allclose(b_list, [0.70, 0.70, 0.25, 0.20], atol=1e-6)
        and np.allclose(w_list, [-0.15, -0.65, -0.75, -0.20], atol=1e-6)
        and codes == ["A", "O", "M", "I"]
    )
    print(
        f"[S2 KANO模型系数与象限] 实测 Better={b_list}, Worse={w_list}, 象限归类={codes} "
        f"(基准 Better=[0.7,0.7,0.25,0.2], Worse=[-0.15,-0.65,-0.75,-0.2], 象限=['A','O','M','I']) | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S2 自检未通过"


def check_S3():
    """S3 联合分析：OLS 效用值（属性内零中心化）与属性相对重要性之和 = 100.0%。"""
    _, _, vdf = _get_valid_demo_df()
    cj = su.calc_conjoint_analysis(vdf)
    imp_sum = cj["importance_sum"]
    util_df = cj["utility_df"]
    attr_sums = util_df.groupby("属性名称")["部分价值(Utility)"].sum().round(6).to_dict()
    all_zero_centered = all(abs(v) < 1e-4 for v in attr_sums.values())
    ok = np.isclose(imp_sum, 100.0, atol=1e-4) and all_zero_centered
    print(
        f"[S3 联合分析效用与重要性] 属性相对重要性之和={imp_sum:.1f}% (基准=100.0%), "
        f"各属性内部效用值之和={attr_sums} (基准=全0.0) | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S3 自检未通过"


def check_S4():
    """S4 48种图型（49张样板图）逐张视觉自检并打包 zip 交付。"""
    import os
    import zipfile

    bundles = _get_cached_49_bundles()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deliverables", "S4_49张图表PNG")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deliverables", "S4_49张样板图高清PNG合集.zip")

    min_w = 99999
    bg_match_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cid, b in bundles.items():
            img = Image.open(io.BytesIO(b["png_bytes"])).convert("RGB")
            w, h = img.size
            min_w = min(min_w, w)
            px = img.getpixel((5, 5))
            if px == (247, 242, 234):  # #F7F2EA
                bg_match_count += 1
            png_file = os.path.join(out_dir, f"{cid}.png")
            with open(png_file, "wb") as f:
                f.write(b["png_bytes"])
            zf.writestr(f"{cid}.png", b["png_bytes"])

    ok = (len(bundles) == 49) and (min_w >= 2400) and (bg_match_count == 49) and os.path.exists(zip_path)
    print(
        f"[S4 49张样板图全量渲染与打包] 渲染图数={len(bundles)}/49, 最小宽度={min_w}px (基准>=2400px), "
        f"暖纸底色(#F7F2EA)匹配={bg_match_count}/49, 打包路径={zip_path} | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S4 自检未通过"


def check_S5():
    """S5 PPT 逐页渲染自检并打包示例 PPTX + 逐页高清截图 zip。"""
    import os
    import zipfile
    import ppt_report as pr

    pptx_bytes = _get_cached_pptx_bytes()
    prs = Presentation(io.BytesIO(pptx_bytes))
    n_slides = len(prs.slides)

    base_deliv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deliverables")
    slides_dir = os.path.join(base_deliv, "S5_PPT逐页截图")
    os.makedirs(slides_dir, exist_ok=True)
    pptx_path = os.path.join(base_deliv, "SurveyScope_示例分析报告.pptx")
    with open(pptx_path, "wb") as f:
        f.write(pptx_bytes)

    img_paths = pr.render_pptx_slides_to_images(pptx_bytes, slides_dir, theme_name="暖纸杂志")
    zip_path = os.path.join(base_deliv, "S5_PPT报告与逐页截图合集.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pptx_path, arcname="SurveyScope_示例分析报告.pptx")
        for p in img_paths:
            zf.write(p, arcname=f"slides_png/{os.path.basename(p)}")

    ok = (n_slides >= 18) and (len(img_paths) == n_slides) and os.path.exists(zip_path)
    print(
        f"[S5 PPT报告与逐页截图自检] PPT总页数={n_slides}页 (基准>=18页), "
        f"逐页PNG截图={len(img_paths)}张, 打包路径={zip_path} | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S5 自检未通过"


def check_S6():
    """S6 可编辑导出物自检：49张图SVG含<text>>=3且CSV行数>0；PPTX含chart*.xml>=8、*.svg>=10、备注Markdown表>=10。"""
    import zipfile

    bundles = _get_cached_49_bundles()
    svg_ok_cnt = 0
    csv_ok_cnt = 0
    for cid, b in bundles.items():
        svg_txt = b["svg_bytes"].decode("utf-8", errors="replace")
        if svg_txt.count("<text") >= 3:
            svg_ok_cnt += 1
        csv_df = pd.read_csv(io.BytesIO(b["csv_bytes"]))
        if len(csv_df) > 0:
            csv_ok_cnt += 1

    pptx_bytes = _get_cached_pptx_bytes()
    with zipfile.ZipFile(io.BytesIO(pptx_bytes), "r") as zf:
        names = zf.namelist()
        chart_xmls = [n for n in names if n.startswith("ppt/charts/chart") and n.endswith(".xml")]
        svg_medias = [n for n in names if n.startswith("ppt/media/") and n.endswith(".svg")]

    prs = Presentation(io.BytesIO(pptx_bytes))
    notes_md_cnt = 0
    for s in prs.slides:
        if s.has_notes_slide:
            nt = s.notes_slide.notes_text_frame.text
            if "| --- |" in nt:
                notes_md_cnt += 1

    ok = (
        svg_ok_cnt == 49
        and csv_ok_cnt == 49
        and len(chart_xmls) >= 8
        and len(svg_medias) >= 10
        and notes_md_cnt >= 10
    )
    print(
        f"[S6 可编辑导出物自检] SVG含<text>>=3: {svg_ok_cnt}/49, CSV非空: {csv_ok_cnt}/49; "
        f"PPTX原生图表(chart*.xml)={len(chart_xmls)} (基准>=8), 矢量SVG(ppt/media/*.svg)={len(svg_medias)} (基准>=10), "
        f"含Markdown源数据表备注页={notes_md_cnt} (基准>=10) | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S6 自检未通过"


def check_S7():
    """S7 高清度、格式面板与 AI 改图自检。"""
    from ai_edit import apply_ai_instruction

    _, _, vdf = _get_valid_demo_df()
    bundles = _get_cached_49_bundles()

    # 1) 任取 5 张 PNG 校验宽度 >= 2400px
    sample_ids = ["01-棒棒糖图", "11-发散堆叠条形图", "26-相关性热力图", "34-结构方程路径图", "44-区域分布地图"]
    widths = [Image.open(io.BytesIO(bundles[cid]["png_bytes"])).size[0] for cid in sample_ids]
    w_ok = all(w >= 2400 for w in widths)

    # 2) 程序化修改 settings['title'] 并验证导出 SVG <text> 随之改变，重置后恢复默认动态标题
    custom_t = "S7_自定义结论标题验证_2026"
    b_custom = cu.render_chart_by_id("01-棒棒糖图", vdf, settings={"title": custom_t})
    b_reset = cu.render_chart_by_id("01-棒棒糖图", vdf, settings={})
    title_mod_ok = (custom_t in b_custom["svg_bytes"].decode("utf-8")) and (custom_t not in b_reset["svg_bytes"].decode("utf-8"))

    # 3) Stub 模拟 AI 改图返回（合法高亮指令 / 非法 JSON / 超白名单篡改数据指令）
    stub_valid = '[{"action": "highlight", "target": "安徽", "color": "#A50026"}]'
    res_valid = apply_ai_instruction("把安徽标红", current_settings={}, stub_response=stub_valid)
    b_map_hl = cu.render_chart_by_id("44-区域分布地图", vdf, settings=res_valid["new_settings"])
    hl_ok = res_valid["ok"] and ("#A50026" in b_map_hl["svg_bytes"].decode("utf-8").upper())

    res_bad_json = apply_ai_instruction("乱码测试", current_settings={"sort": "desc"}, stub_response="NOT_A_JSON_STRING")
    bad_json_ok = (not res_bad_json["ok"]) and (res_bad_json["new_settings"].get("sort") == "desc")

    stub_out = '[{"action": "tamper_p_value", "p": 0.0001}]'
    res_out = apply_ai_instruction("把p值改成显著", current_settings={"sort": "desc"}, stub_response=stub_out)
    out_ok = (not res_out["ok"]) and (res_out["new_settings"].get("sort") == "desc")

    ok = w_ok and title_mod_ok and hl_ok and bad_json_ok and out_ok
    print(
        f"[S7 高清度、格式面板与AI改图自检] 5图宽度={widths}px (>=2400px), "
        f"标题修改与重置={title_mod_ok}, AI高亮安徽(#A50026)={hl_ok}, "
        f"非法JSON拦截={bad_json_ok}, 超白名单指令拦截={out_ok} | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S7 自检未通过"


def check_S8():
    """S8 多主题与无障碍对比度自检：5套主题×3张代表图表，验证背景像素RGB、无#F7F2EA硬编码残留及WCAG对比度>=4.5:1。"""
    import theme as th

    _, _, vdf = _get_valid_demo_df()
    rep_charts = ["01-棒棒糖图", "26-相关性热力图", "44-区域分布地图"]
    bg_pixel_ok = True
    no_hardcode_leak_ok = True
    contrast_ok = True
    min_contrast = 99.0

    for t_name, t_dict in th.THEMES.items():
        expected_rgb = th.hex_to_rgb255(t_dict["bg_canvas"])
        c_ink = th.calc_contrast_ratio(t_dict["ink"], t_dict["bg_canvas"])
        c_main = th.calc_contrast_ratio(t_dict["main"], t_dict["bg_canvas"])
        c_acc = th.calc_contrast_ratio(t_dict["accent"], t_dict["bg_canvas"])
        min_contrast = min(min_contrast, c_ink, c_main, c_acc)
        if min(c_ink, c_main, c_acc) < 4.5:
            contrast_ok = False

        for cid in rep_charts:
            b = cu.render_chart_by_id(cid, vdf, theme_name=t_name)
            img = Image.open(io.BytesIO(b["png_bytes"])).convert("RGB")
            if img.getpixel((5, 5)) != expected_rgb:
                bg_pixel_ok = False
            if t_name != "暖纸杂志":
                svg_str = b["svg_bytes"].decode("utf-8").upper()
                if "#F7F2EA" in svg_str:
                    no_hardcode_leak_ok = False

    ok = bg_pixel_ok and no_hardcode_leak_ok and contrast_ok
    print(
        f"[S8 多主题与WCAG无障碍自检] 5主题×3图背景像素RGB严格一致={bg_pixel_ok}, "
        f"非默认主题零#F7F2EA残留={no_hardcode_leak_ok}, "
        f"全主题最低WCAG对比度={min_contrast:.2f}:1 (基准>=4.5:1) | 结果: {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "S8 自检未通过"


STAGE_CHECKS = {
    "1": [check_A1, check_A2, check_A3, check_A4, check_A5, check_A10, check_A12, check_A13],
    "2": [check_A4, check_A6, check_A7, check_A8, check_A9, check_A11, check_A17],
    "3": [check_A14, check_A15, check_A16, check_S1, check_S2, check_S3, check_S4, check_S5, check_S6, check_S7, check_S8],
    "4": [check_A18, check_A15],
    "all": [
        check_A1, check_A2, check_A3, check_A4, check_A5, check_A6,
        check_A7, check_A8, check_A9, check_A10, check_A11, check_A12,
        check_A13, check_A14, check_A15, check_A16, check_A17, check_A18,
        check_S1, check_S2, check_S3, check_S4, check_S5, check_S6, check_S7, check_S8,
    ],
}


def main():
    parser = argparse.ArgumentParser(description="SurveyScope A1-A18 基准自检程序")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="指定自检阶段 (1/2/3/4/all)")
    args = parser.parse_args()

    print("=" * 78)
    print(f"SurveyScope 统计与系统功能基准自检 (执行阶段: {args.stage})")
    print("=" * 78)
    funcs = STAGE_CHECKS[args.stage]
    for fn in funcs:
        fn()
    print("=" * 78)
    print(f"阶段 [{args.stage}] 共 {len(funcs)} 项基准自检全部通过 (ALL PASSED)！")
    print("=" * 78)


if __name__ == "__main__":
    main()
