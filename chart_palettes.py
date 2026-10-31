# -*- coding: utf-8 -*-
"""
SurveyScope 逐图配色注册表 (chart_palettes.py)

设计原则（2026-10-31 定版，见项目文档《05-逐图配色方案-20261031.md》）：
- 不做全平台一刀切统一色：统一的是画布底色、墨色与字体（主题层），
  每张图的数据色使用一张「与该图气质对应」的命名色板，互不重复。
- 色板全部取自公开成熟来源，禁止临时自配：
  * MetBrewer（大都会艺术博物馆馆藏作品取色，GitHub: BlakeRMills/MetBrewer，
    Python: met_brewer）——分类/序渐色以包内推荐 order 顺序排列；
  * ggsci 期刊色（Nature NPG / NEJM / Lancet / JCO，Python: ggsci）；
  * Crameri Scientific Colour Maps（roma / vik / cork / lajolla，
    逐点采样 11 级，色值与官方版本一致）；
  * Paul Tol / ColorBrewer 仅保留在主题层作为未注册图表的回退。
- 语义约束优先于好看：发散图（Likert/残差/相关）必须用发散色带；
  有序分组（频次/漏斗）必须用单族序渐；名义分类才用彩虹式分类色。
- 注入方式：chart_utils 在渲染每张图前设定当前 chart_id，
  theme.resolve_effective_theme 读取本注册表覆盖主题的颜色令牌；
  用户在格式面板/AI 中显式选择的配色方案与自定义主色优先级更高。
- 44-区域分布地图为用户已验收的 v10 定制分级色，不注册覆盖。
"""

import matplotlib.colors as mcolors
import numpy as np


def _lum(hex_str: str) -> float:
    r, g, b = mcolors.to_rgb(hex_str)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _sat(hex_str: str) -> float:
    r, g, b = mcolors.to_rgb(hex_str)
    mx, mn = max(r, g, b), min(r, g, b)
    return 0.0 if mx == 0 else (mx - mn) / mx


def _interp(colors, n):
    cmap = mcolors.LinearSegmentedColormap.from_list("pal_interp", list(colors), N=256)
    return [mcolors.to_hex(cmap(x)).upper() for x in np.linspace(0, 1, n)]


def _seq6(colors):
    """由色板颜色按明度浅→深插值出 6 档序渐色。"""
    ordered = sorted(colors, key=_lum, reverse=True)
    if len(ordered) >= 6:
        idx = np.linspace(0, len(ordered) - 1, 6).round().astype(int)
        return [ordered[i] for i in idx]
    return _interp(ordered, 6)


def _mk(colors, div=None, seq=None, main=None, accent=None, source=""):
    colors = [c.upper() for c in colors]
    by_lum = sorted(colors, key=_lum)
    entry = {
        "categorical": colors,
        "sequential_6": [c.upper() for c in seq] if seq else _seq6(colors),
        "main_dark": by_lum[0],
        "main": main.upper() if main else by_lum[1] if len(by_lum) > 1 else by_lum[0],
        "main_light": by_lum[int(len(by_lum) * 0.62)],
        "main_pale": by_lum[int(len(by_lum) * 0.85)],
        "accent": accent.upper() if accent else max(colors, key=_sat),
        "source": source,
    }
    entry["highlight"] = entry["accent"]
    if div is not None:
        div = [c.upper() for c in div]
        entry["diverging"] = div if len(div) == 11 else _interp(div, 11)
    return entry


# Crameri 发散色带（官方版本逐点采样 11 级）
_CRAMERI_ROMA = ["#7E1700", "#974D13", "#AC7726", "#C1A343", "#D2D484", "#C0EAC3", "#89DAD7", "#4BB2CE", "#2D88BE", "#1E5FAC", "#033198"]
_CRAMERI_VIK = ["#001261", "#02397A", "#116496", "#5496B7", "#A7C9DA", "#ECE5E0", "#E1B8A0", "#CC875F", "#B75A26", "#852206", "#590008"]
_CRAMERI_CORK = ["#2C194C", "#284073", "#3D6B98", "#6F92B3", "#ADC1D4", "#E6EDEC", "#B7CFB7", "#78A578", "#438142", "#195615", "#0F2903"]
_CRAMERI_LAJOLLA_SEQ = ["#FFFECB", "#F9E384", "#F0BD57", "#EA9E53", "#E38050", "#D9604E", "#B34947", "#7D3B35", "#512D1E", "#30210D", "#191900"]

# MetBrewer Paquin 天然发散色带（红→奶油→绿，元组即渐变序）
_PAQUIN_DIV = ["#831818", "#C62320", "#F05B43", "#F78462", "#FEAC81", "#F7DEA3", "#CED1AF", "#98AB76", "#748F46", "#47632A", "#275024"]
# MetBrewer Hiroshige 天然发散色带（暖橙→奶油→青蓝，元组即渐变序）
_HIROSHIGE_DIV = ["#E76254", "#EF8A47", "#F7AA58", "#FFD06F", "#FFE6B7", "#AADCE0", "#72BCD5", "#528FAD", "#376795", "#1E466E"]


CHART_PALETTES = {
    # ---------------- 基础分布 ----------------
    "01-棒棒糖图": _mk(
        ["#134B73", "#72AEB6", "#2F70A1", "#ABC9C8", "#0A3351", "#4692B0"],
        main="#134B73", accent="#0A3351", source="MetBrewer · Hokusai2（葛饰北斋《神奈川冲浪里》海色）"),
    "02-水平条形图": _mk(
        ["#1E3D14", "#447243", "#E7E5CC", "#669D62", "#192813", "#C2D6A4", "#9CC184", "#1F5B25"],
        main="#447243", accent="#1E3D14", source="MetBrewer · VanGogh3（梵高绿野）"),
    "03-帕累托图": _mk(
        ["#8D1C06", "#E67424", "#F5C34D", "#3C0D03", "#ED9B49"],
        main="#8D1C06", accent="#E67424", source="MetBrewer · Greek（古希腊陶罐土金）"),
    "04-华夫饼图": _mk(
        ["#DD5129", "#0F7BA2", "#43B284", "#FAB255"],
        main="#0F7BA2", accent="#DD5129", source="MetBrewer · Egypt（古埃及壁画四色）"),
    "05-甜甜圈图": _mk(
        ["#E3ABA7", "#9D9DC7", "#B1615C", "#5A5A83", "#EDD7D9", "#C9C9DD", "#D88782", "#8282AA"],
        main="#5A5A83", accent="#B1615C", source="MetBrewer · Cassatt1（卡萨特《一杯茶》）"),
    "06-多选题条形图": _mk(
        ["#749E89", "#C399A2", "#7D87B2", "#ABCCBE", "#E3CACF", "#C2CAE3", "#4E6D58", "#9F6E71", "#41507B"],
        main="#4E6D58", accent="#9F6E71", source="MetBrewer · Monet（莫奈睡莲池）"),
    "07-直方图与核密度": _mk(
        ["#B36C06", "#65150B", "#C0431F", "#F19425", "#961F1F", "#533D14", "#C59349"],
        main="#C0431F", accent="#F19425", source="MetBrewer · Peru2（秘鲁纺织土色）"),
    "08a-箱线图": _mk(
        ["#418979", "#96410E", "#591D06", "#E5A335", "#556219", "#053C29", "#2B614E"],
        main="#96410E", accent="#E5A335", source="MetBrewer · Degas（德加芭蕾）"),
    "08b-小提琴图": _mk(
        ["#A56457", "#FFB178", "#FFC680", "#DB8872", "#DFBBC8", "#B08BA5", "#7C668C"],
        main="#A56457", accent="#7C668C", source="MetBrewer · Morgenstern（晨星霞色）"),
    "09-雨云图": _mk(
        ["#7FA074", "#90719F", "#2C4B27", "#2D223C", "#DEC5DA", "#C1D1AA", "#574571", "#0E2810", "#B695BC", "#466C4B"],
        main="#574571", accent="#7FA074", source="MetBrewer · Cassatt2（卡萨特《窗边的丁香》）"),
    "10-经验累积分布图": _mk(
        ["#2E77AB", "#D1B252", "#18527E", "#A97F2F", "#06314E", "#7E5522", "#041D2C", "#472C0B"],
        main="#18527E", accent="#A97F2F", source="MetBrewer · Ingres（安格尔蓝金）"),
    "11-发散堆叠条形图": _mk(
        ["#033198", "#2D88BE", "#C0EAC3", "#D2D484", "#AC7726", "#7E1700"],
        div=_CRAMERI_ROMA, main="#2D88BE", accent="#7E1700",
        source="Crameri · roma（红-白-蓝发散，Likert 专用）"),
    "12-均值置信区间图": _mk(
        ["#2C6B67", "#67322E", "#122C43", "#99610A", "#C38F16", "#175449", "#6E948C"],
        main="#2C6B67", accent="#C38F16", source="MetBrewer · Veronese（委罗内塞青金）"),
    "13-NPS仪表盘": _mk(
        ["#47632A", "#F7DEA3", "#831818", "#98AB76", "#F78462", "#F05B43", "#FEAC81", "#748F46", "#C62320", "#CED1AF", "#275024"],
        div=_PAQUIN_DIV, main="#47632A", accent="#C62320",
        source="MetBrewer · Paquin（红-奶油-绿天然发散，NPS 三段语义）"),
    "14-人口学金字塔": _mk(
        ["#2F357C", "#B0799A", "#E69B00", "#355828", "#6C5D9E", "#BF3729", "#E48171", "#F5BB50", "#9D9CD5", "#17154F", "#F6B3B0", "#ADA43B"],
        main="#2F357C", accent="#BF3729", source="MetBrewer · Renoir（雷诺阿舞会）"),
    # ---------------- 交叉分析 ----------------
    "15-100%堆叠条形图": _mk(
        ["#2B9B81", "#FE9B00", "#F4C40F", "#FBE183", "#633372", "#9B3441", "#E6A2A6", "#92C051", "#1F6E9C", "#9F5691", "#E87B89", "#D8443C", "#DE597C", "#AA7AA1"],
        main="#1F6E9C", accent="#D8443C", source="MetBrewer · Signac（西涅克点彩）"),
    "16-分组簇状条形图": _mk(
        ["#AAB5D5", "#97C684", "#808FE1", "#454A74", "#EFC86E", "#6F9969", "#5C66A8"],
        main="#454A74", accent="#EFC86E", source="MetBrewer · Derain（德兰野兽派）"),
    "17-马赛克图": _mk(
        ["#3C4B99", "#C93F55", "#EACC62", "#469D76", "#924099", "#DF9ED4"],
        main="#3C4B99", accent="#C93F55", source="MetBrewer · Klimt（克里姆特金饰）"),
    "18-卡方残差热力图": _mk(
        ["#2C194C", "#3D6B98", "#78A578", "#195615"],
        div=_CRAMERI_CORK, main="#3D6B98", accent="#195615",
        source="Crameri · cork（蓝-白-绿发散，残差专用）"),
    "19-哑铃图": _mk(
        ["#04A3BD", "#F0BE3D", "#931E18", "#DA7901", "#247D3F", "#20235B"],
        main="#04A3BD", accent="#DA7901", source="MetBrewer · Lakota（拉科塔织物）"),
    "20-坡度图": _mk(
        ["#A82203", "#208CC0", "#F1AF3A", "#CF5E4E", "#637B31", "#003967"],
        main="#208CC0", accent="#A82203", source="MetBrewer · Juarez（华雷斯壁画）"),
    "21-分组小提琴箱线图": _mk(
        ["#1A5270", "#4C825D", "#8DC7DC", "#134130", "#0E2A4D", "#508CA7", "#8CAE9E"],
        main="#1A5270", accent="#4C825D", source="MetBrewer · Pissaro（毕沙罗田园）"),
    "22-均值标准误条形图": _mk(
        ["#92351E", "#FBE3C2", "#B9563F", "#E69C6B", "#F2C88F", "#D37750", "#ECB27D"],
        main="#B9563F", accent="#92351E", source="MetBrewer · OKeeffe2（欧姬芙沙漠暖色）"),
    "23-山脊图": _mk(
        ["#5A97C1", "#95C36E", "#295384", "#74C8C3", "#D8D97A", "#0A2E57"],
        main="#295384", accent="#95C36E", source="MetBrewer · Hokusai3（北斋瀑布）"),
    "24-分面小多图": _mk(
        ["#F6C200", "#A00E00", "#0086A8", "#D04E00", "#132B69"],
        main="#0086A8", accent="#A00E00", source="MetBrewer · Johnson（约翰逊版画）"),
    # ---------------- 关系与模型 ----------------
    "25-散点回归图": _mk(
        ["#792504", "#527BAA", "#BC7524", "#8DADCA", "#082844", "#421600", "#104839"],
        main="#527BAA", accent="#BC7524", source="MetBrewer · Moreau（莫罗象征派）"),
    "26-相关性热力图": _mk(
        ["#001261", "#5496B7", "#ECE5E0", "#B75A26", "#590008"],
        div=_CRAMERI_VIK, main="#116496", accent="#B75A26",
        source="Crameri · vik（蓝-白-红发散，相关专用）"),
    "27-气泡矩阵图": _mk(
        ["#AADCE0", "#EF8A47", "#376795", "#F7AA58", "#72BCD5", "#FFE6B7", "#E76254", "#1E466E", "#FFD06F", "#528FAD"],
        div=_HIROSHIGE_DIV, main="#376795", accent="#E76254",
        source="MetBrewer · Hiroshige（广重浮世绘发散）"),
    "28-平行坐标图": _mk(
        ["#802417", "#C06636", "#17486F", "#E8B960", "#CE9344", "#646E3B", "#508EA2", "#2B5851"],
        main="#17486F", accent="#C06636", source="MetBrewer · Tiepolo（提埃波罗壁画）"),
    "29-漏斗图": _mk(
        ["#6B6CA3", "#87BCBD", "#6F9954", "#969BC7", "#89AB7C", "#434475", "#2C2D54"],
        main="#434475", accent="#6F9954", source="MetBrewer · VanGogh1（梵高星夜暮色）"),
    "30-桑基图": _mk(
        ["#5B859E", "#1E395F", "#75884B", "#1E5A46", "#DF8D71", "#AF4F2F", "#D48F90", "#732F30", "#AB84A5", "#59385C", "#D8B847", "#B38711"],
        main="#1E395F", accent="#AF4F2F", source="MetBrewer · Redon（勒东梦境）"),
    "31-对应分析双标图": _mk(
        ["#582310", "#AA361D", "#82C45F", "#318F49", "#0CB4BB", "#2673A3", "#473D7D"],
        main="#2673A3", accent="#AA361D", source="MetBrewer · Tsimshian（钦西安图腾）"),
    "32-IPA四象限图": _mk(
        ["#34B6C6", "#DDC000", "#79AD41", "#4063A3", "#D7ACA1"],
        main="#4063A3", accent="#DDC000", source="MetBrewer · Isfahan2（伊斯法罕瓷砖）"),
    "33-PSM价格敏感度曲线": _mk(
        ["#225E92", "#D29C44", "#43429B", "#EBC174", "#4585B7", "#183571", "#5E65BE", "#80521C", "#7EC5F4", "#3B2319", "#EDE2CC"],
        main="#225E92", accent="#D29C44", source="MetBrewer · Manet（马奈蓝金）"),
    "34-结构方程路径图": _mk(
        ["#3B7C70", "#CE9642", "#898E9F", "#3B3A3E"],
        main="#3B7C70", accent="#CE9642", source="MetBrewer · Kandinsky（康定斯基包豪斯）"),
    "35-KANO模型图": _mk(
        ["#EB7926", "#62929A", "#C969A1", "#004F63", "#CE4441", "#859B6C", "#EE8577", "#FFBB44", "#122451"],
        main="#004F63", accent="#EB7926", source="MetBrewer · Cross（克罗斯新印象派）"),
    "36-联合分析重要性图": _mk(
        ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B"],
        main="#0073C2", accent="#EFC000", source="ggsci · JCO（临床肿瘤学期刊）"),
    "37-联合分析效用曲线": _mk(
        ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
        main="#0072B5", accent="#BC3C29", source="ggsci · NEJM（新英格兰医学杂志）"),
    "38-回归系数森林图": _mk(
        ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6"],
        main="#00468B", accent="#ED0000", source="ggsci · Lancet（柳叶刀）"),
    "39-ROC曲线图": _mk(
        ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"],
        main="#E64B35", accent="#3C5488", source="ggsci · NPG（Nature 期刊组）"),
    "40-随机森林重要性图": _mk(
        ["#D9604E", "#B34947", "#7D3B35", "#F0BD57", "#512D1E", "#191900"],
        seq=["#F9E384", "#F0BD57", "#EA9E53", "#E38050", "#D9604E", "#B34947"],
        main="#B34947", accent="#7D3B35", source="Crameri · lajolla（金红序渐，重要性排序）"),
    "41-决策树规则图": _mk(
        ["#811E18", "#4C6216", "#C88A2C", "#9E4013", "#B04948", "#1A472A"],
        main="#1A472A", accent="#C88A2C", source="MetBrewer · Gauguin（高更塔希提土色）"),
    "42-混淆矩阵图": _mk(
        ["#B75347", "#224B5E", "#E09351", "#94B594", "#EDC775", "#6D2F20", "#DF7E66"],
        main="#B75347", accent="#224B5E", source="MetBrewer · Hokusai1（北斋富士）"),
    "43-气泡图": _mk(
        ["#1C9D7C", "#B5361C", "#369CC9", "#E35E28", "#31C7BA", "#3A507F"],
        main="#369CC9", accent="#B5361C", source="MetBrewer · Peru1（秘鲁彩织）"),
    # 44-区域分布地图：用户已验收 v10 定制分级色，不注册（见文件头说明）
    "45-消费者画像卡片": _mk(
        ["#381A61", "#F9D14A", "#AB3329", "#88A0DC", "#E78429", "#ED968C", "#7C4B73"],
        main="#381A61", accent="#E78429", source="MetBrewer · Archambault（拉科塔当代织物）"),
    "46-词云与词频组合图": _mk(
        ["#EF8737", "#341648", "#FFD353", "#9F2D55", "#FFB242", "#62205F", "#DE4F33", "#BB292C"],
        main="#DE4F33", accent="#341648", source="MetBrewer · Tam（谭氏落日渐变）"),
    "47-蝴蝶对比图": _mk(
        ["#660D20", "#E59A52", "#EDCE79", "#094568", "#E1C59A"],
        main="#094568", accent="#660D20", source="MetBrewer · Navajo（纳瓦霍织毯）"),
    "48-双轴组合图": _mk(
        ["#7C291E", "#BA7233", "#2D5380", "#3A4421", "#4B1D0D"],
        main="#2D5380", accent="#BA7233", source="MetBrewer · Wissing（维辛格静物）"),
    "49-高频词共现网络图": _mk(
        ["#178F92", "#845D29", "#4FB6CA", "#175F5D", "#4E3910", "#1D1F54", "#D8C29D"],
        main="#178F92", accent="#845D29", source="MetBrewer · Isfahan1（伊斯法罕蓝瓷）"),
    "50-情感分布图": _mk(
        ["#E9851D", "#165D43", "#BF3626", "#F9C53B", "#788F33", "#E9724C", "#AEAC4C"],
        main="#165D43", accent="#BF3626", source="MetBrewer · Homer2（霍默海景）"),
}


def get_chart_palette_override(chart_id: str, theme_name: str = None):
    """按 chart_id 返回主题令牌覆盖字典（无注册返回 None）。

    逐图色板按「暖纸杂志」的纸面与墨色校准，仅在该主题（或默认主题）下生效；
    其他主题（墨蓝学术 / 深空夜读等）保留其自身令牌，避免深色底上对比度失配。"""
    if theme_name not in (None, "暖纸杂志"):
        return None
    if not chart_id:
        return None
    entry = CHART_PALETTES.get(chart_id)
    if entry is None:
        return None
    out = {k: (list(v) if isinstance(v, list) else v)
           for k, v in entry.items() if k != "source"}
    out["accent_primary"] = out["main"]
    out["accent_secondary"] = out["accent"]
    return out


# =============================================================================
# 命名配色方案注册表（AI 改图 color_scheme 与格式面板「单图配色覆盖」共用）
# - 期刊/色盲友好色值逐一核对自公开来源（2026-10-31）：
#   ggsci（GitHub: nanxstats/ggsci, R/palettes.R）的 NPG / AAAS / Lancet；
#   Okabe-Ito（GitHub: clauswilke/colorblindr, R/palettes.R，黑版配色）；
#   Tol Bright（Paul Tol 官方配色页 Bright 定性方案，默认色序）。
# - 艺术系列直接引用上方 CHART_PALETTES 的逐图色板（MetBrewer），不重复维护色值。
# =============================================================================
NAMED_COLOR_SCHEMES = {
    "NPG 自然": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "AAAS 期刊": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B", "#A20056", "#808180", "#1B1919"],
    "柳叶刀 Lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"],
    "Okabe-Ito 色盲友好": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "Tol Bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    "北斋海色": list(CHART_PALETTES["01-棒棒糖图"]["categorical"]),
    "梵高绿野": list(CHART_PALETTES["02-水平条形图"]["categorical"]),
    "希腊陶土": list(CHART_PALETTES["03-帕累托图"]["categorical"]),
    "卡萨特茶色": list(CHART_PALETTES["05-甜甜圈图"]["categorical"]),
    "莫奈睡莲": list(CHART_PALETTES["06-多选题条形图"]["categorical"]),
    "雷诺阿舞会": list(CHART_PALETTES["14-人口学金字塔"]["categorical"]),
}

# color_scheme 全部可选项（主题默认 + 主题层既有 3 方案 + 命名注册表），AI 与手动面板共用
COLOR_SCHEME_OPTIONS = ["主题默认", "Tol Muted", "松绿序渐", "BrBG发散"] + list(NAMED_COLOR_SCHEMES.keys())
