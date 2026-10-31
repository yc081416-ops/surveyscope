# -*- coding: utf-8 -*-
"""
SurveyScope AI 改图（图表助手）与手动格式设置共享管线 (ai_edit.py)
支持：
1. 将「图型 + 类别列表 + 当前设置 JSON + 用户自然语言指令」组装为结构化 Prompt（绝不发送原始问卷数据）；
2. 调用 OpenAI 兼容接口（DeepSeek / Gemini / 通义等）或测试桩模型 (stub)；
3. 严格执行 JSON Schema 与 17 项白名单操作校验（每项均已核验在渲染层真实生效），拒绝修改原始数据等越界请求并友好报错；
4. 与手动「格式面板」共用同一套 settings 字典，并维护最近 5 步撤销历史栈。
"""

import json
import copy
import re
import urllib.request
import urllib.error


ALLOWED_ACTIONS = {
    "set_title",
    "set_subtitle",
    "set_footnote",
    "highlight",
    "set_item_color",
    "color_scheme",
    "custom_main_color",
    "toggle_labels",
    "toggle_legend",
    "sort",
    "top_n",
    "set_decimals",
    "toggle_mean_line",
    "toggle_ci",
    "set_font_scale",
    "set_wordcloud_shape",
    "set_wordcloud_words",
}

# 曾出现在旧版白名单、但渲染层并无对应实现（回执假成功）的操作：
# 识别后给出诚实报错，不再假装执行（2026-10-30 AI 功能补全审计）
UNSUPPORTED_ACTIONS = {
    "label_mode", "orientation", "canvas_ratio",
    "axis_range", "chart_type",
}

# set_item_color 的中文颜色名 → 主题体系十六进制（与平台色板一致）
ITEM_COLOR_NAMES = {
    "红色": "#A50026", "深红色": "#A50026", "砖红色": "#A50026",
    "绿色": "#2E6B57", "深绿色": "#1E4D3D", "浅绿色": "#6FA08A",
    "蓝色": "#88CCEE", "深蓝色": "#332288", "天蓝色": "#88CCEE",
    "紫色": "#AA4499", "橙色": "#C4664A", "陶土色": "#C4664A",
    "黄色": "#DDCC77", "灰色": "#999933", "粉色": "#CC6677",
    "青色": "#44AA99", "酒红色": "#882255",
}

# 词云可选填充形状（set_wordcloud_shape 的合法取值）及口语同义词归一表
WORDCLOUD_SHAPES = {"无", "云朵", "人形", "爱心", "星星", "咖啡杯", "圆形"}
_WORDCLOUD_SHAPE_SYNONYMS = {
    "云": "云朵", "云形": "云朵", "云状": "云朵",
    "人物": "人形", "人像": "人形", "小人": "人形",
    "心形": "爱心", "心": "爱心",
    "五角星": "星星", "星形": "星星",
    "杯子": "咖啡杯",
    "圆圈": "圆形", "圆": "圆形",
    "默认": "无", "取消": "无", "无形状": "无",
}

COMPATIBLE_CHART_GROUPS = [
    {"棒棒糖图", "条形图", "柱形图", "折线图", "词频棒棒糖图"},
    {"箱线图", "小提琴图"},
    {"交叉热力图", "簇状条形图", "100%堆积条形图"},
]

FORBIDDEN_DATA_KEYWORDS = [
    "篡改数据", "修改样本量", "伪造", "编造数据", "把p值改成", "把人数改成",
]


def get_default_chart_settings(chart_type: str = "棒棒糖图") -> dict:
    """生成单张图的默认格式面板与 AI 改图共享 settings 结构。"""
    return {
        "custom_title": "",
        "custom_subtitle": "",
        "footnote": "",                  # 自定义脚注（口径注）
        "highlight_targets": [],
        "item_colors": {},               # 单类别指定颜色 {类别名: #RRGGBB}
        "color_scheme": "主题默认",       # 主题默认 / Tol Muted / 松绿序渐 / BrBG发散
        "custom_main_color": "",         # 如 #2E6B57
        "font_scale": 1.0,               # 字号倍率（与渲染层同键同型：0.85 紧凑 / 1.0 标准 / 1.18 大字）
        "show_labels": True,
        "decimals": 1,                   # 数据标签小数位数 0/1/2
        "show_mean_line": True,          # 均值参考线
        "show_ci": True,                 # 置信区间/误差棒
        "sort_mode": "按值降序",         # 按值降序 / 按值升序 / 原始顺序
        "sort": "desc",                  # 渲染层同键：desc / asc / original
        "show_legend": True,
        "top_n": 0,                      # 0 表示显示全部
        "chart_type": chart_type,
        "wordcloud_shape": "无",         # 词云填充形状：无/云朵/人形/爱心/星星/咖啡杯/圆形（仅词云图）
        "wordcloud_max_words": 0,        # 词云最多显示词数（仅词云图，0=默认 50）
    }


def build_ai_edit_prompt(
    chart_type: str,
    categories: list,
    current_settings: dict,
    user_instruction: str,
) -> str:
    """
    构建发送给大模型的结构化提示词（只包含图型元信息、类别名列表与当前设置 JSON，绝不发送受访者原始问卷数据）。
    附完整 JSON Schema 与 3 个标准示例。
    """
    safe_cats = [str(c) for c in (categories or [])[:50]]
    prompt = f"""你是一个专业的统计图表样式编辑助手。请将用户的自然语言改图指令解析为结构化 JSON。
严禁输出除 JSON 以外的任何解释文字或 Markdown 代码块标记。

【当前图表元信息（不含任何原始问卷敏感数据）】
- 当前图型：{chart_type}
- 可选类别列表：{json.dumps(safe_cats, ensure_ascii=False)}
- 当前样式设置：{json.dumps(current_settings, ensure_ascii=False)}

【白名单操作集（只允许以下 17 种 action，白名单外或要求修改原始统计数值的请求必须返回 reject）】
1. set_title: {{"action": "set_title", "value": "新结论标题"}}
2. set_subtitle: {{"action": "set_subtitle", "value": "新口径副标题"}}
3. set_footnote: {{"action": "set_footnote", "value": "新脚注（口径注）文本"}}
4. highlight: {{"action": "highlight", "target": "某类别名称"}} (将该类别设为高亮深红 #A50026)
5. set_item_color: {{"action": "set_item_color", "target": "某类别名称", "value": "#RRGGBB"}} (把指定类别改成指定颜色；用户说颜色名时填对应十六进制：绿色 #2E6B57 / 红色 #A50026 / 蓝色 #88CCEE / 紫色 #AA4499 / 橙色 #C4664A)
6. color_scheme: {{"action": "color_scheme", "value": "Tol Muted" | "松绿序渐" | "BrBG发散" | "主题默认"}}
7. custom_main_color: {{"action": "custom_main_color", "value": "#RRGGBB"}}
8. toggle_labels: {{"action": "toggle_labels", "value": true | false}}
9. toggle_legend: {{"action": "toggle_legend", "value": true | false}}
10. sort: {{"action": "sort", "value": "按值降序" | "按值升序" | "原始顺序"}}
11. top_n: {{"action": "top_n", "value": 整数}}
12. set_decimals: {{"action": "set_decimals", "value": 0 | 1 | 2}} (数据标签保留的小数位数)
13. toggle_mean_line: {{"action": "toggle_mean_line", "value": true | false}} (均值参考线开关)
14. toggle_ci: {{"action": "toggle_ci", "value": true | false}} (置信区间/误差棒开关)
15. set_font_scale: {{"action": "set_font_scale", "value": "紧凑" | "标准" | "大字"}} (整体字号)
16. set_wordcloud_shape: {{"action": "set_wordcloud_shape", "value": "云朵" | "人形" | "爱心" | "星星" | "咖啡杯" | "圆形" | "无"}} (仅适用于词云图：将词云填充成指定形状，如用户说“把词云做成云朵/人的形状”；"无" 表示取消形状恢复默认)
17. set_wordcloud_words: {{"action": "set_wordcloud_words", "value": 整数}} (仅适用于词云图：词云最多显示的词数，10–150)

【返回 JSON Schema】
单个操作：{{"edits": [{{"action": "...", ...}}]}}
拒绝越界请求：{{"reject": true, "reason": "拒绝原因说明"}}

【示例 1】
用户指令："把安徽标成深红色，并且只看前5名"
返回：{{"edits": [{{"action": "highlight", "target": "安徽"}}, {{"action": "top_n", "value": 5}}]}}

【示例 2】
用户指令："标题改成「长三角样本占比超七成」，去掉图例，换成横向"
返回：{{"edits": [{{"action": "set_title", "value": "长三角样本占比超七成"}}, {{"action": "toggle_legend", "value": false}}, {{"action": "orientation", "value": "横向"}}]}}

【示例 3】
用户指令："把瑞幸的样本人数改成 500 人，把 p 值改成 0.001"
返回：{{"reject": true, "reason": "请求涉及修改原始统计数据，超出样式编辑白名单范围，已拒绝执行。"}}

【当前用户指令】
{user_instruction}
"""
    return prompt


def validate_and_apply_edits(
    parsed_json: dict,
    current_settings: dict,
    current_chart_type: str = None,
) -> dict:
    """
    校验解析后的 JSON 对象并将合法白名单操作应用到 settings 副本。
    返回: {"ok": bool, "settings": new_settings, "receipt": str, "error": str}
    """
    if isinstance(parsed_json, list):
        edits = parsed_json
    elif isinstance(parsed_json, dict):
        if parsed_json.get("reject") is True:
            reason = parsed_json.get("reason", "请求超出白名单允许的图表样式修改范围（如篡改原始数据），已拒绝。")
            return {
                "ok": False,
                "settings": copy.deepcopy(current_settings),
                "new_settings": copy.deepcopy(current_settings),
                "receipt": "",
                "error": f"已拒绝执行：{reason}",
                "message": f"已拒绝执行：{reason}",
            }
        if "edits" in parsed_json and isinstance(parsed_json["edits"], list):
            edits = parsed_json["edits"]
        elif "action" in parsed_json:
            edits = [parsed_json]
        else:
            return {
                "ok": False,
                "settings": copy.deepcopy(current_settings),
                "new_settings": copy.deepcopy(current_settings),
                "receipt": "",
                "error": "友好报错：JSON 缺少 edits 或 action 字段，请检查指令。",
                "message": "友好报错：JSON 缺少 edits 或 action 字段，请检查指令。",
            }
    else:
        return {
            "ok": False,
            "settings": copy.deepcopy(current_settings),
            "new_settings": copy.deepcopy(current_settings),
            "receipt": "",
            "error": "友好报错：模型返回结构不是 JSON 对象，未执行修改。",
            "message": "友好报错：模型返回结构不是 JSON 对象，未执行修改。",
        }

    if len(edits) == 0:
        return {
            "ok": False,
            "settings": copy.deepcopy(current_settings),
            "new_settings": copy.deepcopy(current_settings),
            "receipt": "",
            "error": "友好报错：未解析到任何有效改图操作。",
            "message": "友好报错：未解析到任何有效改图操作。",
        }

    new_settings = copy.deepcopy(current_settings)
    receipts = []

    for item in edits:
        if not isinstance(item, dict) or "action" not in item:
            return {
                "ok": False,
                "settings": copy.deepcopy(current_settings),
                "new_settings": copy.deepcopy(current_settings),
                "receipt": "",
                "error": "友好报错：编辑项格式非法，缺少 action 字段。",
                "message": "友好报错：编辑项格式非法，缺少 action 字段。",
            }
        act = str(item["action"]).strip()
        if act in UNSUPPORTED_ACTIONS:
            err_s = f"友好报错：操作「{act}」在当前版本暂未接入渲染层，执行了也不会改变图表，已未执行；请改用手动格式面板或换一种调整。"
            return {
                "ok": False,
                "settings": copy.deepcopy(current_settings),
                "new_settings": copy.deepcopy(current_settings),
                "receipt": "",
                "error": err_s,
                "message": err_s,
            }
        if act not in ALLOWED_ACTIONS:
            err_s = f"友好报错：操作「{act}」不在白名单操作集内（严禁修改原始数据或执行未知操作），已拒绝执行。"
            return {
                "ok": False,
                "settings": copy.deepcopy(current_settings),
                "new_settings": copy.deepcopy(current_settings),
                "receipt": "",
                "error": err_s,
                "message": err_s,
            }

        if act == "set_title":
            val = str(item.get("value", item.get("title", ""))).strip()
            new_settings["custom_title"] = val
            new_settings["title"] = val
            receipts.append(f"标题设为「{val}」")

        elif act == "set_subtitle":
            val = str(item.get("value", item.get("subtitle", ""))).strip()
            new_settings["custom_subtitle"] = val
            new_settings["subtitle"] = val
            receipts.append(f"副标题设为「{val}」")

        elif act == "highlight":
            target = item.get("target") or item.get("value")
            color_v = str(item.get("color") or "#A50026")
            if isinstance(target, list):
                targets = [str(x) for x in target]
            elif target:
                targets = [str(target)]
            else:
                targets = []
            new_settings["highlight_targets"] = targets
            if targets:
                new_settings["highlight"] = {"target": targets[0], "color": color_v}
            receipts.append(f"高亮类别：{', '.join(targets)} ({color_v})")

        elif act == "color_scheme":
            val = str(item.get("value", "主题默认"))
            if val not in {"主题默认", "Tol Muted", "松绿序渐", "BrBG发散"}:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：不支持的色板方案「{val}」。"}
            new_settings["color_scheme"] = val
            receipts.append(f"切换色板为「{val}」")

        elif act == "custom_main_color":
            val = str(item.get("value", "")).strip()
            if not (val.startswith("#") and len(val) == 7):
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：自定义颜色「{val}」必须为 #RRGGBB 十六进制格式。"}
            new_settings["custom_main_color"] = val
            receipts.append(f"自定义主色设为 {val}")

        elif act == "toggle_labels":
            val = bool(item.get("value", True))
            new_settings["show_labels"] = val
            receipts.append("开启数据标签" if val else "隐藏数据标签")

        elif act == "label_mode":
            val = str(item.get("value", "双显"))
            if val not in {"百分比", "计数", "双显"}:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：不支持的标签口径「{val}」。"}
            new_settings["label_mode"] = val
            receipts.append(f"数据标签口径切换为「{val}」")

        elif act == "sort":
            val = str(item.get("value", "按值降序"))
            if val not in {"按值降序", "按值升序", "原始顺序"}:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：不支持的排序方式「{val}」。"}
            new_settings["sort_mode"] = val
            # 同步写渲染层实际读取的 sort 键（旧版只写 sort_mode，图表并不读取，排序从未真正生效）
            new_settings["sort"] = {"按值降序": "desc", "按值升序": "asc", "原始顺序": "original"}[val]
            receipts.append(f"排序方式切换为「{val}」")

        elif act == "orientation":
            val = str(item.get("value", "横向"))
            if val not in {"默认", "横向", "纵向"}:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：不支持的方向「{val}」。"}
            new_settings["orientation"] = val
            receipts.append(f"方向切换为「{val}」")

        elif act == "canvas_ratio":
            val = str(item.get("value", "16:9"))
            if val not in {"16:9", "4:3", "1:1"}:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：不支持的画布比例「{val}」。"}
            new_settings["canvas_ratio"] = val
            receipts.append(f"画布比例设为 {val}")

        elif act == "toggle_legend":
            val = bool(item.get("value", False))
            new_settings["show_legend"] = val
            receipts.append("显示图例" if val else "隐藏图例")

        elif act == "top_n":
            val = int(item.get("value", 0))
            new_settings["top_n"] = max(0, val)
            receipts.append(f"仅保留前 {val} 项" if val > 0 else "显示全部项")

        elif act == "axis_range":
            rng = item.get("value")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                new_settings["axis_range"] = [float(rng[0]), float(rng[1])]
                receipts.append(f"坐标轴范围设为 [{rng[0]}, {rng[1]}]")
            else:
                new_settings["axis_range"] = None
                receipts.append("重置坐标轴范围")

        elif act == "chart_type":
            target_ct = str(item.get("value", ""))
            base_ct = current_chart_type or current_settings.get("chart_type", "")
            compatible = False
            for grp in COMPATIBLE_CHART_GROUPS:
                if base_ct in grp and target_ct in grp:
                    compatible = True
                    break
            if not compatible and base_ct and target_ct != base_ct:
                return {
                    "ok": False,
                    "settings": copy.deepcopy(current_settings),
                    "receipt": "",
                    "error": f"友好报错：图型「{base_ct}」与「{target_ct}」数据结构不兼容，禁止跨类转换。",
                }
            new_settings["chart_type"] = target_ct
            receipts.append(f"图型切换为「{target_ct}」")

        elif act == "set_wordcloud_shape":
            raw_v = str(item.get("value", "")).strip()
            val = _WORDCLOUD_SHAPE_SYNONYMS.get(raw_v, raw_v)
            if val not in WORDCLOUD_SHAPES:
                err_s = f"友好报错：不支持的词云形状「{raw_v}」，可选：云朵 / 人形 / 爱心 / 星星 / 咖啡杯 / 圆形 / 无。"
                return {"ok": False, "settings": copy.deepcopy(current_settings), "new_settings": copy.deepcopy(current_settings), "receipt": "", "error": err_s, "message": err_s}
            base_ct = str(current_chart_type or current_settings.get("chart_type", ""))
            if base_ct and "词云" not in base_ct:
                err_s = f"友好报错：填充形状仅适用于词云图，当前图型为「{base_ct}」，未执行。"
                return {"ok": False, "settings": copy.deepcopy(current_settings), "new_settings": copy.deepcopy(current_settings), "receipt": "", "error": err_s, "message": err_s}
            new_settings["wordcloud_shape"] = val
            receipts.append(f"词云形状设为「{val}」" if val != "无" else "取消词云形状，恢复默认排布")

        elif act == "set_footnote":
            val = str(item.get("value", "")).strip()
            if not val:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": "友好报错：脚注文本为空白，无效修改指令。"}
            new_settings["footnote"] = val
            new_settings["custom_footnote"] = val
            receipts.append("脚注（口径注）已更新")

        elif act == "set_item_color":
            tgt = str(item.get("target", "")).strip()
            raw_c = str(item.get("value", "")).strip()
            hex_c = ITEM_COLOR_NAMES.get(raw_c, raw_c)
            if not tgt:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": "友好报错：未指定要改色的类别名称。"}
            if not re.match(r"^#[0-9A-Fa-f]{6}$", hex_c):
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：颜色值「{raw_c}」不合法，请使用 #RRGGBB 格式或中文颜色名（如 绿色/红色/蓝色）。"}
            merged = dict(new_settings.get("item_colors") or {})
            merged[tgt] = hex_c.upper()
            new_settings["item_colors"] = merged
            receipts.append(f"类别「{tgt}」颜色设为 {hex_c.upper()}")

        elif act == "set_decimals":
            try:
                val = int(item.get("value", 1))
            except Exception:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": "友好报错：小数位数必须为 0 / 1 / 2 的整数。"}
            if val not in (0, 1, 2):
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：小数位数 {val} 超出范围，仅支持 0 / 1 / 2。"}
            new_settings["decimals"] = val
            receipts.append(f"数据标签小数位数设为 {val} 位")

        elif act == "toggle_mean_line":
            val = bool(item.get("value", True))
            new_settings["show_mean_line"] = val
            receipts.append("显示均值参考线" if val else "隐藏均值参考线")

        elif act == "toggle_ci":
            val = bool(item.get("value", True))
            new_settings["show_ci"] = val
            receipts.append("显示置信区间" if val else "隐藏置信区间")

        elif act == "set_font_scale":
            raw_f = item.get("value", "标准")
            fs_map = {"紧凑": 0.85, "标准": 1.0, "大字": 1.18, "大": 1.18, "小": 0.85}
            if isinstance(raw_f, (int, float)) and 0.7 <= float(raw_f) <= 1.4:
                new_settings["font_scale"] = float(raw_f)
                receipts.append(f"字号倍率设为 {float(raw_f):.2f}")
            elif str(raw_f).strip() in fs_map:
                new_settings["font_scale"] = fs_map[str(raw_f).strip()]
                receipts.append(f"字号设为「{str(raw_f).strip()}」")
            else:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": f"友好报错：字号档位「{raw_f}」不合法，仅支持 紧凑 / 标准 / 大字。"}

        elif act == "set_wordcloud_words":
            base_ct = str(current_chart_type or current_settings.get("chart_type", ""))
            if base_ct and "词云" not in base_ct:
                err_s = f"友好报错：词数设置仅适用于词云图，当前图型为「{base_ct}」，未执行。"
                return {"ok": False, "settings": copy.deepcopy(current_settings), "new_settings": copy.deepcopy(current_settings), "receipt": "", "error": err_s, "message": err_s}
            try:
                val = int(item.get("value", 50))
            except Exception:
                return {"ok": False, "settings": copy.deepcopy(current_settings), "receipt": "", "error": "友好报错：词数必须为整数。"}
            val = max(10, min(150, val))
            new_settings["wordcloud_max_words"] = val
            receipts.append(f"词云最多显示 {val} 个词")

    msg_ok = "已执行：" + "；".join(receipts)
    return {
        "ok": True,
        "settings": new_settings,
        "new_settings": new_settings,
        "receipt": msg_ok,
        "message": msg_ok,
        "error": "",
    }


def call_openai_compatible_llm(prompt: str, api_config: dict) -> str:
    """调用用户配置的 OpenAI 兼容大模型接口（只存 session_state，不落盘）。"""
    base_url = api_config.get("base_url", "https://api.deepseek.com").rstrip("/")
    api_key = api_config.get("api_key", "").strip()
    model = api_config.get("model", "deepseek-flash").strip()
    if not api_key:
        raise ValueError("尚未配置 API Key，请在左侧边栏「AI 改图接口设置」中填写。")

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出严格合法的 JSON 对象，不输出任何多余字符。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = [ln for ln in content.splitlines() if not ln.strip().startswith("```")]
            content = "\n".join(lines).strip()
        return content


def _local_rule_fallback(user_instruction: str) -> str:
    """当未配置 API Key 时，本地轻量级白名单规则解析器将常见中文指令转为合法白名单 JSON。"""
    import re
    txt = (user_instruction or "").strip()
    # 词云形状指令（如「把词云做成云朵形状」「换成人的形状」「取消形状」）
    if any(k in txt for k in ("形状", "做成", "变成", "组合成")) or "取消形状" in txt:
        for _canon, _keys in [("云朵", ("云朵", "云的", "云形")), ("人形", ("人形", "人物", "人像", "人的", "小人")),
                              ("爱心", ("爱心", "心形")), ("星星", ("星星", "五角星")),
                              ("咖啡杯", ("咖啡杯", "杯子")), ("圆形", ("圆形", "圆圈"))]:
            if any(k in txt for k in _keys):
                return json.dumps({"action": "set_wordcloud_shape", "value": _canon}, ensure_ascii=False)
        if any(k in txt for k in ("取消形状", "去掉形状", "不要形状", "恢复默认")):
            return json.dumps({"action": "set_wordcloud_shape", "value": "无"}, ensure_ascii=False)
    # 单类别改色（「把瑞幸改成绿色」/「把美式换成 #123456」）
    m_ic = re.search(r"把(.+?)(?:改成|换成|设为)(#[0-9A-Fa-f]{6})", txt)
    if m_ic:
        return json.dumps({"action": "set_item_color", "target": m_ic.group(1).strip(), "value": m_ic.group(2)}, ensure_ascii=False)
    m_icn = re.search(r"把(.+?)(?:改成|换成|设为)(深红|砖红|红|深绿|浅绿|绿|深蓝|天蓝|蓝|紫|陶土|橙|黄|灰|粉|青|酒红)色", txt)
    if m_icn:
        return json.dumps({"action": "set_item_color", "target": m_icn.group(1).strip(), "value": m_icn.group(2) + "色"}, ensure_ascii=False)
    # 小数位数
    m_dec = re.search(r"保留(\d)\s*位小数|小数点后(\d)\s*位", txt)
    if m_dec:
        return json.dumps({"action": "set_decimals", "value": int(m_dec.group(1) or m_dec.group(2))}, ensure_ascii=False)
    if "整数显示" in txt or "不要小数" in txt:
        return json.dumps({"action": "set_decimals", "value": 0}, ensure_ascii=False)
    # 均值线 / 置信区间 / 图例开关
    if any(k in txt for k in ("去掉均值线", "隐藏均值线", "不要均值线", "去掉平均线", "隐藏平均线")):
        return json.dumps({"action": "toggle_mean_line", "value": False}, ensure_ascii=False)
    if "显示均值线" in txt or "加上均值线" in txt:
        return json.dumps({"action": "toggle_mean_line", "value": True}, ensure_ascii=False)
    if any(k in txt for k in ("去掉置信区间", "隐藏置信区间", "去掉误差棒", "隐藏误差棒", "不要误差棒")):
        return json.dumps({"action": "toggle_ci", "value": False}, ensure_ascii=False)
    if "显示置信区间" in txt or "显示误差棒" in txt:
        return json.dumps({"action": "toggle_ci", "value": True}, ensure_ascii=False)
    if "隐藏图例" in txt or "去掉图例" in txt:
        return json.dumps({"action": "toggle_legend", "value": False}, ensure_ascii=False)
    if "显示图例" in txt:
        return json.dumps({"action": "toggle_legend", "value": True}, ensure_ascii=False)
    # 字号
    if any(k in txt for k in ("字大一点", "字体大一点", "放大字号", "字号调大")):
        return json.dumps({"action": "set_font_scale", "value": "大字"}, ensure_ascii=False)
    if any(k in txt for k in ("字小一点", "字体小一点", "缩小字号")):
        return json.dumps({"action": "set_font_scale", "value": "紧凑"}, ensure_ascii=False)
    # 排序
    if "升序" in txt or "从小到大" in txt:
        return json.dumps({"action": "sort", "value": "按值升序"}, ensure_ascii=False)
    if "降序" in txt or "从大到小" in txt:
        return json.dumps({"action": "sort", "value": "按值降序"}, ensure_ascii=False)
    if "原始顺序" in txt:
        return json.dumps({"action": "sort", "value": "原始顺序"}, ensure_ascii=False)
    # 脚注
    m_fn = re.search(r"脚注(?:改为|改成|设为|换成)[「“\"']?(.+?)[」”\"']?$", txt)
    if m_fn:
        return json.dumps({"action": "set_footnote", "value": m_fn.group(1).strip()}, ensure_ascii=False)
    # 词云词数
    m_ww = re.search(r"词云.*?(?:显示|放|用)\s*(\d+)\s*个词", txt)
    if m_ww:
        return json.dumps({"action": "set_wordcloud_words", "value": int(m_ww.group(1))}, ensure_ascii=False)
    if "词云多显示" in txt or "词云显示更多" in txt:
        return json.dumps({"action": "set_wordcloud_words", "value": 100}, ensure_ascii=False)
    m_hl = re.search(r"把(.+?)(?:标红|高亮|突出)", txt)
    if m_hl:
        return json.dumps({"action": "highlight", "target": m_hl.group(1).strip(), "color": "#A50026"}, ensure_ascii=False)
    m_title = re.search(r"标题(?:改为|改成|设为)[「“\"']?(.+?)[」”\"']?$", txt)
    if m_title:
        return json.dumps({"action": "set_title", "value": m_title.group(1).strip()}, ensure_ascii=False)
    if "隐藏" in txt and "标签" in txt:
        return json.dumps({"action": "toggle_labels", "value": False}, ensure_ascii=False)
    if "显示" in txt and "标签" in txt:
        return json.dumps({"action": "toggle_labels", "value": True}, ensure_ascii=False)
    m_top = re.search(r"前\s*(\d+)\s*项", txt)
    if m_top:
        return json.dumps({"action": "top_n", "value": int(m_top.group(1))}, ensure_ascii=False)
    return json.dumps({"reject": True, "reason": "未能识别为白名单内的样式指令，或尚未填写大模型 API Key。"}, ensure_ascii=False)


def run_ai_chart_edit(
    user_instruction: str,
    chart_type: str = "",
    categories: list = None,
    current_settings: dict = None,
    api_config: dict = None,
    stub_response: str = None,
) -> dict:
    """
    执行完整的 AI 改图流程：
    1. 组装不含原始数据的安全提示词；
    2. 调用 stub_response（自检/测试桩）或 OpenAI 兼容模型接口；
    3. 校验 JSON 格式与白名单操作；
    4. 更新 settings 并返回执行回执或友好错误。
    """
    current_settings = current_settings or {}
    categories = categories or []
    for kw in FORBIDDEN_DATA_KEYWORDS:
        if kw in (user_instruction or ""):
            err_s = "友好报错：请求涉及修改或编造原始统计数据，超出白名单操作范围，已拒绝执行。"
            return {
                "ok": False,
                "settings": copy.deepcopy(current_settings),
                "new_settings": copy.deepcopy(current_settings),
                "receipt": "",
                "error": err_s,
                "message": err_s,
            }

    prompt = build_ai_edit_prompt(chart_type, categories, current_settings, user_instruction)

    if stub_response is not None:
        raw_text = stub_response(prompt) if callable(stub_response) else str(stub_response)
    else:
        api_key = (api_config or {}).get("api_key", "").strip()
        if not api_key:
            raw_text = _local_rule_fallback(user_instruction)
        else:
            try:
                raw_text = call_openai_compatible_llm(prompt, api_config or {})
            except Exception as e:
                err_s = f"友好报错：调用大模型接口失败（{e}）。"
                return {
                    "ok": False,
                    "settings": copy.deepcopy(current_settings),
                    "new_settings": copy.deepcopy(current_settings),
                    "receipt": "",
                    "error": err_s,
                    "message": err_s,
                }

    try:
        parsed = json.loads(raw_text)
    except Exception:
        err_s = "友好报错：模型返回的内容不是合法的 JSON 格式，已拦截并未修改当前图表。"
        return {
            "ok": False,
            "settings": copy.deepcopy(current_settings),
            "new_settings": copy.deepcopy(current_settings),
            "receipt": "",
            "error": err_s,
            "message": err_s,
        }

    return validate_and_apply_edits(parsed, current_settings, current_chart_type=chart_type)


def apply_ai_instruction(
    user_instruction: str,
    current_settings: dict = None,
    api_config: dict = None,
    stub_response: str = None,
    chart_type: str = "",
    categories: list = None,
) -> dict:
    """统一 AI 改图调用入口（兼容 app.py 与 self_check.py S7）。"""
    return run_ai_chart_edit(
        user_instruction=user_instruction,
        chart_type=chart_type,
        categories=categories or [],
        current_settings=current_settings or {},
        api_config=api_config,
        stub_response=stub_response,
    )


def push_settings_history(history_store, chart_id_or_settings, maybe_settings: dict = None, max_steps: int = 5):
    """支持按 chart_id 字典存储或单列表存储 5 步撤销历史。"""
    if isinstance(history_store, dict) and isinstance(chart_id_or_settings, str):
        cid = chart_id_or_settings
        hist = list(history_store.get(cid, []))
        hist.append(copy.deepcopy(maybe_settings or {}))
        history_store[cid] = hist[-max_steps:]
        return history_store[cid]
    hist = list(history_store or [])
    hist.append(copy.deepcopy(chart_id_or_settings or {}))
    return hist[-max_steps:]


def undo_settings(history_store: dict, chart_id: str):
    """从 history_store[chart_id] 弹出最近一步设置。"""
    if not isinstance(history_store, dict):
        return None
    hist = history_store.get(chart_id, [])
    if not hist:
        return None
    return hist.pop()


def reset_settings_to_default() -> dict:
    """重置单图配置为默认空字典。"""
    return {}


def pop_settings_history(history_list: list, current_settings: dict) -> tuple:
    """撤销最近一步修改，返回 (restored_settings, updated_history_list)。"""
    hist = list(history_list or [])
    if not hist:
        return copy.deepcopy(current_settings), []
    prev = hist.pop()
    return prev, hist
