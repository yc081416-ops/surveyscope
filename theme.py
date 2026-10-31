# -*- coding: utf-8 -*-
"""
SurveyScope 多主题设计系统令牌模块 (theme.py)
提供 5 套完整主题（暖纸杂志、墨蓝学术、纯白极简、深空夜读、薄荷清新），
涵盖画布底色、主墨色、次级文字、网格线、主色阶、点睛色、高亮色、分类色(9色)、发散色(11色)、序渐色(6档)与连续色标。
全站图表、页面与 PPT/Word 导出均从当前主题令牌取色，严禁硬编码十六进制颜色。
"""

import contextvars

import matplotlib.colors as mcolors
import numpy as np

# 逐图配色上下文：chart_utils 渲染单图前设定当前 chart_id，
# resolve_effective_theme 据此叠加 chart_palettes 注册表中的专属色板。
_ACTIVE_CHART_ID = contextvars.ContextVar("surveyscope_active_chart_id", default=None)


def set_active_chart_id(chart_id):
    return _ACTIVE_CHART_ID.set(chart_id)


def reset_active_chart_id(token):
    _ACTIVE_CHART_ID.reset(token)

try:
    import cmcrameri.cm as cmc
    BATLOW_CMAP = cmc.batlow
except Exception:
    import matplotlib.pyplot as plt
    BATLOW_CMAP = plt.get_cmap("viridis")


FONT_FAMILY_NAME = "Noto Sans CJK SC"
FONT_FALLBACK_LIST = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "sans-serif",
]


THEMES = {
    "暖纸杂志": {
        "key": "warm_paper",
        "name": "暖纸杂志",
        "description": "默认主题：暖纸画布 + 松绿/陶土，Paul Tol Muted 分类色，杂志出版级质感",
        "bg_canvas": "#F7F2EA",
        "bg_secondary": "#EFE9DC",
        "bg_card": "#FBF9F5",
        "ink": "#22312B",
        "muted_text": "#7A7368",
        "grid": "#DDD3C2",
        "insignificant": "#B9AE9C",
        "main": "#2E6B57",
        "main_dark": "#1E4D3D",
        "main_light": "#6FA08A",
        "main_pale": "#A8C3B2",
        "accent": "#A64B2A",
        "highlight": "#A50026",
        "categorical": [
            "#88CCEE", "#CC6677", "#44AA99", "#332288", "#DDCC77",
            "#882255", "#999933", "#AA4499", "#117733",
        ],
        "diverging": [
            "#543005", "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",
            "#F5F5F5", "#C7EAE5", "#80CDC1", "#35978F", "#01665E", "#003C30",
        ],
        "sequential_6": [
            "#EFE9DC", "#D8E2D5", "#A8C3B2", "#6FA08A", "#2E6B57", "#1E4D3D",
        ],
        "continuous_name": "batlow",
    },
    "墨蓝学术": {
        "key": "academic_blue",
        "name": "墨蓝学术",
        "description": "象牙白底 + 深墨蓝/黛青，期刊论文气质，分类色采用 Paul Tol Bright 调整版",
        "bg_canvas": "#FAF9F5",
        "bg_secondary": "#EEF1F6",
        "bg_card": "#FFFFFF",
        "ink": "#182638",
        "muted_text": "#5A6978",
        "grid": "#D5DCE6",
        "insignificant": "#A3B1C2",
        "main": "#1F4E79",
        "main_dark": "#123252",
        "main_light": "#4B7BA8",
        "main_pale": "#9BBAD9",
        "accent": "#A84A28",
        "highlight": "#9E1B32",
        "categorical": [
            "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
            "#AA3377", "#335588", "#BB7733", "#2B6E68",
        ],
        "diverging": [
            "#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
            "#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061",
        ],
        "sequential_6": [
            "#EEF1F6", "#D0DDF0", "#9BBAD9", "#5C8AB8", "#2B5C8F", "#123252",
        ],
        "continuous_name": "batlow",
    },
    "纯白极简": {
        "key": "minimal_white",
        "name": "纯白极简",
        "description": "纯白底 + 石板灰阶 + 单一强调蓝，清爽通透、黑白/彩色打印友好",
        "bg_canvas": "#FFFFFF",
        "bg_secondary": "#F3F5F8",
        "bg_card": "#F8FAFC",
        "ink": "#111827",
        "muted_text": "#525E70",
        "grid": "#E2E8F0",
        "insignificant": "#94A3B8",
        "main": "#1D4ED8",
        "main_dark": "#1E3A8A",
        "main_light": "#60A5FA",
        "main_pale": "#BFDBFE",
        "accent": "#B45309",
        "highlight": "#BE123C",
        "categorical": [
            "#2563EB", "#0D9488", "#D97706", "#7C3AED", "#DB2777",
            "#4B5563", "#059669", "#EA580C", "#4338CA",
        ],
        "diverging": [
            "#7C2D12", "#9A3412", "#C2410C", "#FB923C", "#FED7AA",
            "#F8FAFC", "#BAE6FD", "#38BDF8", "#0284C7", "#0369A1", "#0C4A6E",
        ],
        "sequential_6": [
            "#F1F5F9", "#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A8A",
        ],
        "continuous_name": "batlow",
    },
    "深空夜读": {
        "key": "dark_night",
        "name": "深空夜读",
        "description": "深灰蓝底 #1E242E + 柔光高对比文字，分类色采用 Okabe-Ito 无障碍色板",
        "bg_canvas": "#1E242E",
        "bg_secondary": "#28303D",
        "bg_card": "#242B37",
        "ink": "#ECF2F8",
        "muted_text": "#A6B4C6",
        "grid": "#384354",
        "insignificant": "#637287",
        "main": "#4CC38A",
        "main_dark": "#2E9968",
        "main_light": "#7CE0AE",
        "main_pale": "#33584B",
        "accent": "#F08C5B",
        "highlight": "#FF5C77",
        "categorical": [
            "#56B4E9", "#E69F00", "#009E73", "#F0E442", "#CC79A7",
            "#78A9FF", "#FF8389", "#42BE65", "#BE95FF",
        ],
        "diverging": [
            "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3", "#4A4E58",
            "#2E3542", "#528E87", "#66C2A5", "#35978F", "#28B5A8", "#80EDE3",
        ],
        "sequential_6": [
            "#28303D", "#2E4745", "#31685A", "#388E72", "#4CC38A", "#89F0BC",
        ],
        "continuous_name": "batlow",
    },
    "薄荷清新": {
        "key": "mint_fresh",
        "name": "薄荷清新",
        "description": "浅青瓷底 + 青绿/珊瑚，分类色采用 Paul Tol Vibrant 调整版",
        "bg_canvas": "#F0F7F4",
        "bg_secondary": "#E1EFEA",
        "bg_card": "#F8FCFA",
        "ink": "#16302B",
        "muted_text": "#58736C",
        "grid": "#CBE0D8",
        "insignificant": "#9AB5AC",
        "main": "#0E6E64",
        "main_dark": "#0B5750",
        "main_light": "#4EB0A6",
        "main_pale": "#9ED9D2",
        "accent": "#B5462E",
        "highlight": "#B81D33",
        "categorical": [
            "#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311",
            "#EE3377", "#6699CC", "#44AA77", "#AA6688",
        ],
        "diverging": [
            "#662506", "#993404", "#CC4C02", "#FE9929", "#FEE391",
            "#F5FAF8", "#C7EAE5", "#80CDC1", "#35978F", "#01665E", "#004438",
        ],
        "sequential_6": [
            "#E1EFEA", "#BFE3DC", "#8CCBC1", "#45A89C", "#148277", "#0B5750",
        ],
        "continuous_name": "batlow",
    },
}

DEFAULT_THEME_NAME = "暖纸杂志"


def _enrich_theme_aliases(t: dict) -> dict:
    """补充语义别名键，兼容各图表与 PPT 模块的统一访问。"""
    out = t.copy()
    out.setdefault("text_primary", out["ink"])
    out.setdefault("text_secondary", out["muted_text"])
    out.setdefault("text_muted", out["muted_text"])
    out.setdefault("grid_line", out["grid"])
    out.setdefault("border_line", out["grid"])
    out.setdefault("accent_primary", out["main"])
    out.setdefault("accent_secondary", out["accent"])
    out.setdefault("map_empty_fill", out["bg_secondary"])
    out.setdefault("map_border", out["grid"])
    return out


def get_theme(theme_name: str = None) -> dict:
    """获取指定名称的主题令牌字典，未指定或不存在时返回默认「暖纸杂志」主题。

    若当前处于单图渲染上下文（chart_utils 已设定 chart_id），叠加该图的
    专属色板覆盖——保证经 get_diverging_cmap / get_sequential_cmap 等
    二次取色路径的图表（热力图、混淆矩阵等）同样使用逐图色板。"""
    if isinstance(theme_name, dict):
        return _enrich_theme_aliases(theme_name)
    if not theme_name or theme_name not in THEMES:
        t = _enrich_theme_aliases(THEMES[DEFAULT_THEME_NAME])
    else:
        t = _enrich_theme_aliases(THEMES[theme_name])
    try:
        from chart_palettes import get_chart_palette_override
        _ov = get_chart_palette_override(_ACTIVE_CHART_ID.get(), theme_name if isinstance(theme_name, str) else None)
        if _ov:
            t.update(_ov)
    except Exception:
        pass
    return t


def hex_to_rgb01(hex_str: str) -> tuple:
    """将 #RRGGBB 转换为 (r, g, b) ∈ [0, 1]^3。"""
    return mcolors.to_rgb(hex_str)


def hex_to_rgb255(hex_str: str) -> tuple:
    """将 #RRGGBB 转换为 (R, G, B) ∈ [0, 255]^3 整数元组。"""
    r, g, b = mcolors.to_rgb(hex_str)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def relative_luminance(hex_str: str) -> float:
    """按 WCAG 2.1 标准计算颜色的相对明度 Y ∈ [0, 1]。"""
    rgb = mcolors.to_rgb(hex_str)
    lin = []
    for c in rgb:
        if c <= 0.03928:
            lin.append(c / 12.92)
        else:
            lin.append(((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def calc_contrast_ratio(hex1: str, hex2: str) -> float:
    """计算两个颜色之间的 WCAG 对比度 (>= 4.5:1 为 AA 级可读标准)。"""
    l1 = relative_luminance(hex1)
    l2 = relative_luminance(hex2)
    bright = max(l1, l2)
    dark = min(l1, l2)
    return round((bright + 0.05) / (dark + 0.05), 2)


def auto_text_color(bg_hex: str, theme: dict = None) -> str:
    """按底色明度自动切换深色字或浅色反白字，保证数据标签清晰可读。"""
    lum = relative_luminance(bg_hex)
    if isinstance(theme, str):
        # 若第二个参数传入了 fallback 颜色字符串（如 t['text_primary']）
        if lum < 0.36:
            return "#FFFFFF"
        return theme if relative_luminance(theme) < 0.35 else "#22312B"
    if lum < 0.36:
        return "#FFFFFF" if (theme is None) else (theme["bg_canvas"] if relative_luminance(theme["bg_canvas"]) > 0.7 else "#FFFFFF")
    return "#22312B" if (theme is None) else (theme["ink"] if relative_luminance(theme["ink"]) < 0.3 else "#182638")


def get_continuous_cmap(theme=None):
    """获取连续色标（默认 Crameri batlow）。"""
    return BATLOW_CMAP


def get_diverging_cmap(theme=None):
    """根据当前主题的 11 级发散色板构建线性插值 Colormap。"""
    t = get_theme(theme)
    return mcolors.LinearSegmentedColormap.from_list("theme_diverging", t["diverging"], N=256)


def get_sequential_cmap(theme=None):
    """根据当前主题的 6 档序渐色构建连续 Colormap。"""
    t = get_theme(theme)
    return mcolors.LinearSegmentedColormap.from_list("theme_sequential", t["sequential_6"], N=256)


def resolve_effective_theme(theme_name: str = None, settings: dict = None) -> dict:
    """
    合并主题默认令牌、格式面板配色切换与自定义主色（优先级：自定义主色 > 面板配色方案 > 主题默认）。
    """
    t = get_theme(theme_name)

    # 逐图专属色板覆盖（chart_palettes 注册表）；优先级低于下方用户显式配色设置
    try:
        from chart_palettes import get_chart_palette_override
        _ov = get_chart_palette_override(_ACTIVE_CHART_ID.get(), theme_name if isinstance(theme_name, str) else None)
        if _ov:
            t.update(_ov)
    except Exception:
        pass

    if not settings:
        return t

    scheme = settings.get("color_scheme") or settings.get("palette")
    if scheme in ("松绿序渐", "sequential"):
        t["categorical"] = list(reversed(t["sequential_6"][1:])) + t["categorical"][:4]
    elif scheme in ("BrBG发散", "diverging"):
        t["categorical"] = [t["diverging"][9], t["diverging"][1], t["diverging"][8], t["diverging"][2], t["diverging"][7], t["diverging"][3], t["main"], t["accent"], t["highlight"]]
    elif scheme in ("Tol Muted", "muted"):
        # Paul Tol Muted 原色（旧实现误写为主题默认分类色，选了等于没换，特此修正）
        t["categorical"] = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
                            "#CC6677", "#AA4499", "#882255", "#999999"]

    custom_main = settings.get("custom_main_color")
    if custom_main and isinstance(custom_main, str) and custom_main.startswith("#") and len(custom_main) == 7:
        t["main"] = custom_main
        t["accent_primary"] = custom_main

    return _enrich_theme_aliases(t)


def check_all_themes_contrast() -> list:
    """自检 S8：计算并返回全部 5 套主题的主文字/画布底色对比度（均须 >= 4.5:1）。"""
    results = []
    for name, t in THEMES.items():
        ratio = calc_contrast_ratio(t["ink"], t["bg_canvas"])
        sec_ratio = calc_contrast_ratio(t["muted_text"], t["bg_canvas"])
        results.append({
            "theme_name": name,
            "bg_canvas": t["bg_canvas"],
            "ink": t["ink"],
            "contrast_ratio": ratio,
            "muted_contrast": sec_ratio,
            "pass_aa": ratio >= 4.5,
        })
    return results
