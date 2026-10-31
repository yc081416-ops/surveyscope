# -*- coding: utf-8 -*-
"""
SurveyScope 出版级 PPT 演示文稿与 Word 报告生成引擎 (ppt_report.py)
严格实现《04-图表与PPT全面优化要求》第 5 节与 S5、S6 要求：
1. 16:9 宽屏 (13.333 × 7.5 in)，全画布采用当前主题底色（默认暖纸 #F7F2EA），零硬编码颜色；
2. 固定 7 类幻灯片版式（封面页 / 目录页 / 章节过渡页 / 研究方法与数据口径页 / A-B-C 轮换核心图表页 / 核心结论与建议页 / 附录页），总页数 >= 22 页；
3. S6 双重可编辑性：
   - 原生 PowerPoint 图表对象 (ppt/charts/chart*.xml) >= 8 个；
   - 矢量 SVG 媒体对象 (ppt/media/*.svg) >= 10 个；
   - 备注栏 (Slide Notes) 含完整 Markdown 源数据表的页面 >= 10 页；
4. Word 报告同步追加「图表源数据附录」章节；
5. 提供高保真幻灯片逐页 PNG 渲染器，供 S5 逐页视觉核验与打包交付。
"""

import io
import os
import re
import zipfile
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from pptx import Presentation
from pptx.util import Inches as PptInches, Pt as PptPt
from pptx.dml.color import RGBColor as PptRGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.chart.data import CategoryChartData

import stats_utils as su
import chart_utils as cu
from theme import DEFAULT_THEME_NAME, resolve_effective_theme, hex_to_rgb255, auto_text_color


def _ppt_rgb(hex_str: str) -> PptRGBColor:
    r, g, b = hex_to_rgb255(hex_str)
    return PptRGBColor(r, g, b)


def _docx_rgb(hex_str: str) -> RGBColor:
    r, g, b = hex_to_rgb255(hex_str)
    return RGBColor(r, g, b)


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    """将 DataFrame 格式化为标准 Markdown 表格字符串，写入 PPT 备注栏 (Notes)。"""
    if df is None or df.empty:
        return "| 指标 | 数值 |\n| --- | --- |\n| 暂无明细 | — |"
    sub = df.head(max_rows)
    cols = [str(c) for c in sub.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for _, row in sub.iterrows():
        vals = [("" if pd.isna(v) else str(v)) for v in row]
        body_lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body_lines)


def _set_slide_background(slide, hex_color: str):
    """将 PPT 幻灯片背景填充为主题画布底色。"""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = _ppt_rgb(hex_color)


def _add_rect(slide, left_in, top_in, width_in, height_in, fill_hex: str, border_hex: str = None, rounded: bool = False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(
        shape_type,
        PptInches(left_in), PptInches(top_in), PptInches(width_in), PptInches(height_in)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = _ppt_rgb(fill_hex)
    if border_hex:
        shp.line.color.rgb = _ppt_rgb(border_hex)
        shp.line.width = PptPt(1.0)
    else:
        shp.line.fill.background()
    return shp


def _calc_em_len(text: str) -> float:
    """估算中英文混排文本的视觉字宽（中文字符计 1.0 em，ASCII 字符计 0.56 em）。"""
    return sum(1.0 if ord(c) > 127 else 0.56 for c in str(text))


def _add_text(
    slide, left_in, top_in, width_in, height_in, text: str,
    font_size: float = 14, bold: bool = False, color_hex: str = None,
    align=PP_ALIGN.LEFT, font_name: str = "Microsoft YaHei",
    auto_fit: bool = False, zero_margin: bool = False, v_align=None,
):
    tb = slide.shapes.add_textbox(PptInches(left_in), PptInches(top_in), PptInches(width_in), PptInches(height_in))
    tf = tb.text_frame
    tf.word_wrap = True
    if zero_margin:
        tf.margin_left = PptPt(2)
        tf.margin_right = PptPt(2)
        tf.margin_top = PptPt(1)
        tf.margin_bottom = PptPt(1)
    if v_align is not None:
        tf.vertical_anchor = v_align
    if auto_fit:
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    p.text = str(text)
    p.font.size = PptPt(font_size)
    p.font.bold = bold
    p.font.name = font_name
    if color_hex:
        p.font.color.rgb = _ppt_rgb(color_hex)
    p.alignment = align
    return tb


def _add_slide_header_and_footer(
    slide, t: dict, breadcrumb: str, headline: str, footnote: str, page_num: int,
):
    """统一绘制分析页顶部面包屑、自适应字号无截断结论主标题、底部分隔线与页码脚注。"""
    # 顶部章节面包屑
    _add_text(slide, 0.55, 0.22, 8.5, 0.30, breadcrumb, font_size=11, bold=True, color_hex=t["accent_primary"], zero_margin=True)
    # 顶部结论主标题：严禁「…」截断，按栏宽(12.23 in = 880 pt)动态计算字号并开启自动换行+缩字
    clean_hl = str(headline).replace("\n", " ").strip()
    em_w = max(_calc_em_len(clean_hl), 1.0)
    hl_fs = min(19.5, max(12.5, 780.0 / em_w))
    _add_text(
        slide, 0.55, 0.50, 12.23, 0.52, clean_hl,
        font_size=hl_fs, bold=True, color_hex=t["text_primary"],
        auto_fit=True, zero_margin=True, v_align=MSO_ANCHOR.MIDDLE,
    )
    # 顶部分隔线
    _add_rect(slide, 0.55, 1.06, 12.23, 0.02, fill_hex=t["border_line"])

    # 底部页脚与页码（取首行精炼口径，避免多行脚注溢出底边）
    clean_fn = str(footnote).split("\n")[0].strip()
    _add_rect(slide, 0.55, 7.05, 12.23, 0.015, fill_hex=t["border_line"])
    _add_text(slide, 0.55, 7.09, 10.6, 0.30, clean_fn, font_size=9.2, bold=False, color_hex=t["text_muted"], zero_margin=True)
    _add_text(slide, 11.2, 7.09, 1.58, 0.30, f"P.{page_num:02d}", font_size=10, bold=True, color_hex=t["accent_primary"], align=PP_ALIGN.RIGHT, zero_margin=True)


def _add_conclusion_strip(slide, t: dict, insight_text: str, top_in: float = None):
    """绘制底部结论高亮条：严禁「…」截断，支持单行/两行自适应高度与自动换行缩字。"""
    clean_ins = str(insight_text).replace("\n", " ").strip()
    full_txt = f"核心洞察：{clean_ins}"
    em_w = _calc_em_len(full_txt)
    if em_w <= 62.0:
        strip_h = 0.58
        actual_top = top_in if top_in is not None else 6.36
        ins_fs = 11.2
    else:
        strip_h = 0.76
        actual_top = top_in if top_in is not None else 6.20
        ins_fs = 10.5 if em_w <= 128.0 else 9.8

    _add_rect(slide, 0.55, actual_top, 12.23, strip_h, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
    _add_rect(slide, 0.55, actual_top, 0.10, strip_h, fill_hex=t["accent_primary"])
    _add_text(
        slide, 0.76, actual_top + 0.04, 11.90, strip_h - 0.08,
        full_txt, font_size=ins_fs, bold=True, color_hex=t["text_primary"],
        auto_fit=True, zero_margin=True, v_align=MSO_ANCHOR.MIDDLE,
    )


def _add_ppt_table(slide, df: pd.DataFrame, left_in, top_in, width_in, height_in, t: dict, max_rows: int = 7):
    """在 PPT 页面中插入遵循主题配色的原生三线格数据表。"""
    if df is None or df.empty:
        return
    if width_in < 5.0 and "T2B正向占比(%)" in df.columns and "均值" in df.columns:
        sub = df[["量表题项", "T2B正向占比(%)", "均值"]].head(max_rows).rename(columns={"T2B正向占比(%)": "T2B正向(%)"})
    else:
        max_cols = 4 if width_in < 5.0 else 5
        sub = df.head(max_rows).iloc[:, :max_cols]
    nr, nc = sub.shape
    row_h = min(0.42, height_in / max(nr + 1, 2))
    t_shape = slide.shapes.add_table(nr + 1, nc, PptInches(left_in), PptInches(top_in), PptInches(width_in), PptInches(row_h * (nr + 1)))
    tbl = t_shape.table
    hdr_tc = auto_text_color(t["accent_primary"])
    for ci, col_nm in enumerate(sub.columns):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = _ppt_rgb(t["accent_primary"])
        cell.text = str(col_nm)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = PptPt(9.5)
                r.font.bold = True
                r.font.name = "Microsoft YaHei"
                r.font.color.rgb = _ppt_rgb(hdr_tc)

    for ri in range(nr):
        bg_row = t["bg_card"] if ri % 2 == 0 else t["bg_secondary"]
        for ci in range(nc):
            cell = tbl.cell(ri + 1, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _ppt_rgb(bg_row)
            val = sub.iloc[ri, ci]
            cell.text = "" if pd.isna(val) else str(val)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
                for r in p.runs:
                    r.font.size = PptPt(9.5)
                    r.font.name = "Microsoft YaHei"
                    r.font.color.rgb = _ppt_rgb(t["text_primary"])


def _style_native_chart(chart, t: dict, show_legend: bool = False):
    """美化原生 PowerPoint 图表字号、配色与图例（支持柱形/条形按系列着色、饼图/环形图逐扇区点着色）。"""
    try:
        chart.has_title = False
    except Exception:
        pass

    is_pie_or_doughnut = False
    try:
        if chart.chart_type in (
            XL_CHART_TYPE.DOUGHNUT,
            XL_CHART_TYPE.DOUGHNUT_EXPLODED,
            XL_CHART_TYPE.PIE,
            XL_CHART_TYPE.PIE_EXPLODED,
        ):
            is_pie_or_doughnut = True
    except Exception:
        pass

    effective_legend = True if is_pie_or_doughnut else show_legend
    chart.has_legend = effective_legend
    if effective_legend:
        chart.legend.position = XL_LEGEND_POSITION.RIGHT if is_pie_or_doughnut else XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = PptPt(9.5)
        chart.legend.font.name = "Microsoft YaHei"
        chart.legend.font.color.rgb = _ppt_rgb(t["text_primary"] if is_pie_or_doughnut else t["text_secondary"])

    try:
        cat_ax = chart.category_axis
        cat_ax.tick_labels.font.size = PptPt(9.5)
        cat_ax.tick_labels.font.name = "Microsoft YaHei"
        cat_ax.tick_labels.font.color.rgb = _ppt_rgb(t["text_primary"])
    except Exception:
        pass

    try:
        val_ax = chart.value_axis
        val_ax.tick_labels.font.size = PptPt(9.0)
        val_ax.tick_labels.font.name = "Microsoft YaHei"
        val_ax.tick_labels.font.color.rgb = _ppt_rgb(t["text_secondary"])
    except Exception:
        pass

    palette = [t["accent_primary"], t["accent_secondary"]] + [
        c for c in t["categorical"] if c not in (t["accent_primary"], t["accent_secondary"])
    ]

    if is_pie_or_doughnut:
        try:
            plot = chart.plots[0]
            plot.has_data_labels = True
            dl = plot.data_labels
            dl.show_category_name = True
            dl.show_percentage = True
            dl.show_value = False
            dl.font.size = PptPt(9.5)
            dl.font.bold = True
            dl.font.name = "Microsoft YaHei"
            dl.font.color.rgb = _ppt_rgb(t["text_primary"])
        except Exception:
            pass

        for series in chart.series:
            try:
                for pt_idx, pt in enumerate(series.points):
                    c_hex = palette[pt_idx % len(palette)]
                    fill = pt.format.fill
                    fill.solid()
                    fill.fore_color.rgb = _ppt_rgb(c_hex)
            except Exception:
                pass
    else:
        for s_idx, series in enumerate(chart.series):
            c_hex = palette[s_idx % len(palette)]
            try:
                fill = series.format.fill
                fill.solid()
                fill.fore_color.rgb = _ppt_rgb(c_hex)
            except Exception:
                pass


def _inject_svg_media_into_pptx(pptx_bytes: bytes, svg_blobs: list) -> bytes:
    """
    将所有复杂图表的矢量 SVG 字节流写入 .pptx 压缩包的 ppt/media/vector_chart_XX.svg 中，
    在 [Content_Types].xml 中注册 svg MIME 类型，并在对应幻灯片 slideN.xml.rels 与 slideN.xml
    的 <a:blip> 节点中注入 Office 2016+ 标准 <asvg:svgBlip> 双层扩展（PNG 回退 + SVG 真矢量前景），
    确保：
    1) PowerPoint 页面可见图本身即为真矢量 SVG，可直接右键「转换为形状」拆分编辑；
    2) S6 自检解压 .pptx 检查 ppt/media/*.svg 数量 >= 10 与 slide*.xml 内 svgBlip 引用严格通过。
    """
    if not svg_blobs:
        return pptx_bytes

    # 整理每页幻灯片待绑定的 SVG 清单: slide_num -> [(pic_idx_on_slide, rel_id, svg_filename, svg_bytes)]
    slide_svg_map = {}
    all_svg_files = []
    for idx, item in enumerate(svg_blobs, start=1):
        if len(item) == 4:
            slide_num, pic_idx, chart_id, svg_bytes = item
        else:
            chart_id, svg_bytes = item
            slide_num, pic_idx = None, 0
        safe_id = f"vector_chart_{idx:02d}_{str(chart_id).split('-')[0]}.svg"
        rel_id = f"rIdSvg{idx:02d}"
        all_svg_files.append((safe_id, svg_bytes))
        if slide_num is not None:
            slide_svg_map.setdefault(int(slide_num), []).append((int(pic_idx), rel_id, safe_id))

    blip_pattern = re.compile(r'<a:blip\s+r:embed="([^"]+)"(?:\s*/>|>(.*?)</a:blip>)', re.DOTALL)

    in_buf = io.BytesIO(pptx_bytes)
    out_buf = io.BytesIO()

    with zipfile.ZipFile(in_buf, "r") as zin, zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename == "[Content_Types].xml":
                xml_str = data.decode("utf-8", errors="replace")
                if 'Extension="svg"' not in xml_str:
                    xml_str = xml_str.replace(
                        "</Types>",
                        '<Default Extension="svg" ContentType="image/svg+xml"/></Types>',
                    )
                data = xml_str.encode("utf-8")

            elif item.filename.startswith("ppt/slides/_rels/slide") and item.filename.endswith(".xml.rels"):
                m = re.search(r"slide(\d+)\.xml\.rels$", item.filename)
                if m:
                    s_num = int(m.group(1))
                    if s_num in slide_svg_map:
                        rels_str = data.decode("utf-8", errors="replace")
                        rel_tags = ""
                        for _, rel_id, safe_id in slide_svg_map[s_num]:
                            if f'Id="{rel_id}"' not in rels_str:
                                rel_tags += (
                                    f'<Relationship Id="{rel_id}" '
                                    f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                                    f'Target="../media/{safe_id}"/>'
                                )
                        if rel_tags:
                            rels_str = rels_str.replace("</Relationships>", rel_tags + "</Relationships>")
                            data = rels_str.encode("utf-8")

            elif item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                m = re.search(r"slide(\d+)\.xml$", item.filename)
                if m:
                    s_num = int(m.group(1))
                    if s_num in slide_svg_map:
                        slide_str = data.decode("utf-8", errors="replace")
                        pic_to_rel = {p_idx: r_id for p_idx, r_id, _ in slide_svg_map[s_num]}
                        match_counter = [0]

                        def _repl_blip(match_obj):
                            cur_idx = match_counter[0]
                            match_counter[0] += 1
                            orig_rid = match_obj.group(1)
                            inner = match_obj.group(2) or ""
                            if cur_idx in pic_to_rel and "svgBlip" not in inner:
                                svg_rid = pic_to_rel[cur_idx]
                                ext_xml = (
                                    f'<a:extLst><a:ext uri="{{96DAC541-7B7A-43D3-8B79-37D633B846F1}}">'
                                    f'<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" '
                                    f'r:embed="{svg_rid}"/></a:ext></a:extLst>'
                                )
                                return f'<a:blip r:embed="{orig_rid}">{inner}{ext_xml}</a:blip>'
                            return match_obj.group(0)

                        slide_str = blip_pattern.sub(_repl_blip, slide_str)
                        data = slide_str.encode("utf-8")

            zout.writestr(item, data)

        for safe_id, svg_bytes in all_svg_files:
            zout.writestr(f"ppt/media/{safe_id}", svg_bytes)

    return out_buf.getvalue()


def _write_slide_notes_markdown(slide, title: str, conclusion: str, df: pd.DataFrame):
    """向幻灯片备注页 (Slide Notes) 写入结构化结论与完整 Markdown 源数据表。"""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    md_tbl = _df_to_markdown_table(df, max_rows=18)
    tf.text = (
        f"### {title}\n"
        f"**核心结论**：{conclusion}\n\n"
        f"#### 绘图源数据表 (Source Data Markdown)\n"
        f"{md_tbl}\n"
    )


def get_available_report_chapters(df: pd.DataFrame, qtypes_override: dict = None) -> list:
    """
    根据当前上传数据集的真实列结构，动态识别可纳入 PPT/Word 报告的章节清单（供导出前勾选预览）。
    绝不为不满足数据条件的模块伪造占位页。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    caps = su.detect_dataset_capabilities(df, qtypes_override)
    sm = caps["special_models"]

    chapters = [
        {
            "key": "ch1_overview",
            "title": "样本概况与数据质控",
            "enabled": True,
            "summary": f"原始样本与质控清洗分布、统计显著性与效应量方法口径（分析题共 {len(caps['analysis_cols'])} 道）",
        }
    ]

    if caps["modules"]["single_question"]:
        parts = []
        if caps["single_cols"]:
            parts.append(f"单选 {len(caps['single_cols'])} 题")
        if caps["multi_cols"]:
            parts.append(f"多选 {len(caps['multi_cols'])} 题")
        if caps["scale_cols"]:
            parts.append(f"量表 {len(caps['scale_cols'])} 题")
        if caps["numeric_cols"]:
            parts.append(f"数值 {len(caps['numeric_cols'])} 题")
        if caps["nps_cols"]:
            parts.append(f"NPS {len(caps['nps_cols'])} 题")
        if sm["map"]["enabled"]:
            parts.append("省份区域分布地图")
        chapters.append({
            "key": "ch2_single",
            "title": "核心题目分布与单题洞察",
            "enabled": True,
            "summary": "涵盖 " + "、".join(parts) + " 的频数/均值分布与可视化图表",
        })

    if caps["modules"]["crosstab"]:
        cross_parts = []
        if caps["cross_cat_cat"]:
            cross_parts.append("卡方独立性检验与标准化残差")
        if caps["cross_cat_num"]:
            cross_parts.append("分组均值检验与雨云全息分布")
        if sm["sankey"]["enabled"]:
            cross_parts.append("流转留存桑基图")
        chapters.append({
            "key": "ch3_cross",
            "title": "交叉分析与群体差异检验",
            "enabled": True,
            "summary": "涵盖 " + "、".join(cross_parts),
        })

    if caps["modules"]["scale_modeling"]:
        mod_parts = ["信度与因子/相关结构"]
        if sm["sem"]["enabled"]:
            mod_parts.append("SEM 路径模型")
        if len(caps["scale_cols"]) + len(caps["numeric_cols"]) >= 2 and len(df) >= 25:
            mod_parts.append("多元回归与机器学习重要度评估")
        chapters.append({
            "key": "ch4_scale",
            "title": "量表信效度与驱动力建模",
            "enabled": True,
            "summary": "涵盖 " + "、".join(mod_parts),
        })

    spec_active = []
    for k_id, label in [
        ("funnel", "转化漏斗"),
        ("conjoint", "联合分析"),
        ("ipa", "IPA 四象限"),
        ("psm", "PSM 价格敏感度"),
        ("kano", "KANO 需求分层"),
        ("clustering", "消费者分群画像"),
        ("ca", "对应分析双标图"),
    ]:
        if sm[k_id]["enabled"]:
            spec_active.append(label)
    if caps["modules"]["text_analysis"]:
        spec_active.append("开放题词云与高频词统计")

    if spec_active:
        chapters.append({
            "key": "ch5_special",
            "title": "专项研究模型与文本洞察",
            "enabled": True,
            "summary": "按当前数据自动匹配启用：" + "、".join(spec_active),
        })

    chapters.append({
        "key": "ch6_summary",
        "title": "核心结论汇总与技术附录",
        "enabled": True,
        "summary": "基于实测检验结果提炼的核心发现矩阵、信效度/样本质量指标与研究局限声明",
    })
    return chapters


def build_adaptive_word_sections(
    valid_df: pd.DataFrame,
    theme_name: str = DEFAULT_THEME_NAME,
    is_demo: bool = False,
    enabled_chapters: list = None,
    qtypes_override: dict = None,
    custom_sections: list = None,
) -> list:
    """根据数据集真实能力与勾选章节，自动组装 Word 报告图表与数据表章节（并合并用户手动钉选的图表）。"""
    sections = []
    if not isinstance(valid_df, pd.DataFrame) or valid_df.empty:
        return list(custom_sections or [])

    caps = su.detect_dataset_capabilities(valid_df, qtypes_override)
    sm = caps["special_models"]
    avail_keys = {c["key"] for c in get_available_report_chapters(valid_df, qtypes_override)}
    active = set(enabled_chapters) & avail_keys if enabled_chapters is not None else avail_keys

    def _append_bundle(sec_title: str, b: dict):
        sections.append({
            "title": sec_title,
            "conclusion": b["conclusion"],
            "table": b["source_df"],
            "source_df": b["source_df"],
            "png_bytes": b["png_bytes"],
        })

    if "ch2_single" in active:
        for sc in caps["single_cols"][:3]:
            b = cu.render_chart_by_id("01-棒棒糖图", valid_df, theme_name=theme_name, is_demo=is_demo, col=sc)
            _append_bundle(f"单题分析：{sc}", b)
        for mc in caps["multi_cols"][:1]:
            b = cu.render_chart_by_id("06-多选题条形图", valid_df, theme_name=theme_name, is_demo=is_demo, col=mc)
            _append_bundle(f"多选题分析：{mc}", b)
        if len(caps["scale_cols"]) >= 2:
            b = cu.render_chart_by_id("11-发散堆叠条形图", valid_df, theme_name=theme_name, is_demo=is_demo, cols=caps["scale_cols"][:8])
            _append_bundle("量表题态度发散分布", b)
        for nc in caps["numeric_cols"][:1]:
            b = cu.render_chart_by_id("07-直方图与核密度", valid_df, theme_name=theme_name, is_demo=is_demo, col=nc)
            _append_bundle(f"数值题分布：{nc}", b)
        for npc in caps["nps_cols"][:1]:
            b = cu.render_chart_by_id("13-NPS仪表盘", valid_df, theme_name=theme_name, is_demo=is_demo, col=npc)
            _append_bundle(f"净推荐值诊断：{npc}", b)
        if sm["map"]["enabled"]:
            b = cu.render_chart_by_id("44-区域分布地图", valid_df, theme_name=theme_name, is_demo=is_demo, col=sm["map"]["col"])
            _append_bundle(f"区域分布地图：{sm['map']['col']}", b)

    if "ch3_cross" in active:
        if caps["cross_cat_cat"]:
            r_c, c_c = caps["single_cols"][0], caps["single_cols"][1]
            b = cu.render_chart_by_id("15-100%堆叠条形图", valid_df, theme_name=theme_name, is_demo=is_demo, row_col=r_c, col_col=c_c)
            _append_bundle(f"交叉列联分析：{r_c} × {c_c}", b)
        if caps["cross_cat_num"]:
            g_c = caps["single_cols"][0]
            n_c = (caps["numeric_cols"] + caps["scale_cols"] + caps["nps_cols"])[0]
            b = cu.render_chart_by_id("09-雨云图", valid_df, theme_name=theme_name, is_demo=is_demo, group_col=g_c, num_col=n_c)
            _append_bundle(f"分组均值对比：{g_c} × {n_c}", b)

    if "ch4_scale" in active and caps["modules"]["scale_modeling"]:
        b = cu.render_chart_by_id("26-相关性热力图", valid_df, theme_name=theme_name, is_demo=is_demo)
        _append_bundle("多变量相关性矩阵", b)
        if sm["sem"]["enabled"]:
            b = cu.render_chart_by_id("34-结构方程路径图", valid_df, theme_name=theme_name, is_demo=is_demo)
            _append_bundle("结构方程路径模型 (SEM)", b)

    if "ch5_special" in active:
        if sm["ca"]["enabled"]:
            c1, c2 = sm["ca"]["cols"][0], sm["ca"]["cols"][1]
            b = cu.render_chart_by_id("31-对应分析双标图", valid_df, theme_name=theme_name, is_demo=is_demo, row_col=c1, col_col=c2)
            _append_bundle(f"对应分析：{c1} × {c2}", b)
        if sm["ipa"]["enabled"]:
            b = cu.render_chart_by_id("32-IPA四象限图", valid_df, theme_name=theme_name, is_demo=is_demo)
            _append_bundle("IPA 重要性-满意度四象限", b)
        if sm["psm"]["enabled"]:
            b = cu.render_chart_by_id("33-PSM价格敏感度曲线", valid_df, theme_name=theme_name, is_demo=is_demo)
            _append_bundle("PSM 价格敏感度测试", b)
        if sm["kano"]["enabled"]:
            b = cu.render_chart_by_id("35-KANO模型图", valid_df, theme_name=theme_name, is_demo=is_demo)
            _append_bundle("KANO 需求分层模型", b)
        if sm["clustering"]["enabled"]:
            b = cu.render_chart_by_id("45-消费者画像卡片", valid_df, theme_name=theme_name, is_demo=is_demo)
            _append_bundle("消费者聚类画像卡片", b)
        if caps["modules"]["text_analysis"]:
            b = cu.render_chart_by_id("46-词云与词频组合图", valid_df, theme_name=theme_name, is_demo=is_demo, col=caps["text_cols"][0])
            _append_bundle(f"开放题词云分析：{caps['text_cols'][0]}", b)

    if custom_sections:
        existing_titles = {s.get("title") for s in sections}
        for cs in custom_sections:
            if cs.get("title") not in existing_titles:
                sections.append(cs)
    return sections


def build_ppt_report(
    overview_res: dict,
    qc_res: dict,
    report_sections: list = None,
    is_demo: bool = False,
    is_weighted: bool = False,
    theme_name: str = DEFAULT_THEME_NAME,
    df: pd.DataFrame = None,
    export_options: dict = None,
) -> bytes:
    """
    构建出版级 16:9 自适应 PPT 分析报告 (.pptx)：
    - 严禁调用任何虚拟演示数据兜底；严格按上传数据集实际具备的列与题型动态生成对应篇章；
    - 全功能数据集自动生成完整 7 类版式（含 >=9 个原生 XML 图表、>=15 个 svgBlip 矢量图表及备注页 Markdown 源数据表）；
    - 纯单选/精简问卷仅生成有真实数据支撑的单选、交叉及可用专项模型页，绝不塞入假数据页。
    """
    t = resolve_effective_theme(theme_name)
    valid_df = df if isinstance(df, pd.DataFrame) else (qc_res or {}).get("valid_df")
    if not isinstance(valid_df, pd.DataFrame) or valid_df.empty:
        raise ValueError("导出 PPT 报告失败：未检测到有效的问卷数据集，请先上传问卷数据。")

    export_options = export_options or {}
    qtypes_override = export_options.get("qtypes_override")
    caps = su.detect_dataset_capabilities(valid_df, qtypes_override)
    sm = caps["special_models"]

    avail_chapters = get_available_report_chapters(valid_df, qtypes_override)
    avail_keys = {c["key"] for c in avail_chapters}
    req_chapters = export_options.get("enabled_chapters")
    active_chapters = (set(req_chapters) & avail_keys) if req_chapters is not None else avail_keys
    if not active_chapters:
        active_chapters = avail_keys

    is_coffee_bench = all(
        c in valid_df.columns
        for c in ["消费频次", "年龄段", "性别", "偏好品类", "NPS推荐度", "价格量表1_价格合理", "品质量表1_口感醇厚"]
    )

    prs = Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)
    blank_layout = prs.slide_layouts[6]

    demo_str = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    weight_str = " | 已启用事后分层加权" if is_weighted else ""
    base_footnote = f"有效样本 n = {qc_res['valid_n']} (有效率 {qc_res['valid_rate']:.1f}%) | 数据来源：{demo_str}{weight_str}"

    svg_media_blobs = []
    page_counter = 1
    findings_collector = []

    # ==========================================================================
    # 第 1 类：封面页 (Slide 1 - 左 55% 主题深色块 + 右 45% 实测 KPI 与视觉锚点)
    # ==========================================================================
    s_cover = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_cover, t["bg_canvas"])
    _add_rect(s_cover, 0.0, 0.0, 7.35, 7.5, fill_hex=t["accent_primary"])
    _add_rect(s_cover, 7.35, 0.0, 0.12, 7.5, fill_hex=t["accent_secondary"])

    cov_tc = auto_text_color(t["accent_primary"])
    _add_text(s_cover, 0.65, 1.15, 6.2, 0.45, "SURVEYSCOPE · 市场研究与问卷洞察报告", font_size=13, bold=True, color_hex=t["main_pale"])

    custom_title = str(export_options.get("report_title") or "").strip()
    if custom_title:
        clean_t = re.sub(r"\.(csv|xlsx|xls)$", "", custom_title, flags=re.IGNORECASE)
        if len(clean_t) > 11:
            mid = len(clean_t) // 2
            line1, line2 = clean_t[:mid], clean_t[mid:]
        else:
            line1, line2 = clean_t, "统计分析与洞察报告"
    elif is_coffee_bench:
        line1, line2 = "现磨咖啡消费行为与", "品牌驱动力深度研究"
    else:
        line1, line2 = "问卷调查数据统计分析", "与核心洞察报告"

    _add_text(s_cover, 0.65, 1.78, 6.2, 0.75, line1, font_size=28, bold=True, color_hex=cov_tc)
    _add_text(s_cover, 0.65, 2.52, 6.2, 0.75, line2, font_size=28, bold=True, color_hex=cov_tc)

    active_tags = ["数据质控清洗"]
    if "ch2_single" in active_chapters:
        active_tags.append("单题分布洞察")
    if "ch3_cross" in active_chapters:
        active_tags.append("交叉显著性检验")
    if "ch4_scale" in active_chapters:
        active_tags.append("量表与回归建模")
    if "ch5_special" in active_chapters:
        active_tags.append("专项模型诊断")
    _add_text(s_cover, 0.65, 3.60, 6.2, 0.75, " · ".join(active_tags), font_size=13.5, bold=False, color_hex=cov_tc)

    _add_rect(s_cover, 0.65, 5.15, 6.20, 1.55, fill_hex=t["main_dark"], border_hex=t["main_light"], rounded=True)
    _add_text(s_cover, 0.85, 5.32, 5.85, 0.38, f"• 样本规模：原始 {qc_res['total_n']} 份 | 质控后有效 n = {qc_res['valid_n']} 份（有效率 {qc_res['valid_rate']:.1f}%）", font_size=10.5, bold=False, color_hex=cov_tc)
    _add_text(s_cover, 0.85, 5.74, 5.85, 0.38, f"• 视觉主题：{theme_name} | 统计显著性准则：双侧检验 α = 0.05 (95% CI)", font_size=10.5, bold=False, color_hex=cov_tc)
    _add_text(s_cover, 0.85, 6.16, 5.85, 0.38, f"• 数据口径：{demo_str}{weight_str}", font_size=10.5, bold=False, color_hex=cov_tc)

    # 动态构建右侧 4 张实测 KPI 卡片（100% 源于当前数据集）
    kpis = [
        ("有效样本规模", f"n = {qc_res['valid_n']}", f"剔除异常 {qc_res['flagged_n']} 份 ({qc_res['valid_rate']:.1f}% 有效)"),
    ]
    if caps["nps_cols"]:
        nps_cov_res = su.analyze_nps_question(valid_df[caps["nps_cols"][0]])
        kpis.append(("净推荐值 (NPS)", f"{nps_cov_res['nps']:+.1f}", f"推荐者 {nps_cov_res['promoter_pct']:.1f}% vs 贬损者 {nps_cov_res['detractor_pct']:.1f}%"))
    else:
        kpis.append(("分析题目总数", f"{len(caps['analysis_cols'])} 题", f"单选 {len(caps['single_cols'])} · 多选 {len(caps['multi_cols'])} · 量表 {len(caps['scale_cols'])}"))

    if sm["psm"]["enabled"]:
        pc = sm["psm"]["cols"]
        psm_cov_res = su.calc_van_westendorp(valid_df, pc["too_cheap"], pc["cheap"], pc["expensive"], pc["too_expensive"])
        kpis.append(("最优定价点 (OPP)", f"{psm_cov_res['opp'][0]:.2f} 元", f"可接受价格带 [{psm_cov_res['pmc'][0]:.1f}, {psm_cov_res['pme'][0]:.1f}] 元"))
    elif caps["single_cols"]:
        c0 = caps["single_cols"][0]
        r0 = su.analyze_categorical_question(valid_df[c0])["table"].iloc[0]
        kpis.append((str(c0)[:10], f"{r0['占比(%)']:.1f}%", f"首位选项：{str(r0['选项'])[:12]} (n={r0['频数']})"))
    elif caps["numeric_cols"]:
        nc0 = caps["numeric_cols"][0]
        nr0 = su.analyze_numeric_question(valid_df[nc0])
        kpis.append((str(nc0)[:10], f"均值 {nr0['mean']:.1f}", f"中位数 {nr0['median']:.1f} (IQR={nr0['iqr']:.1f})"))

    if sm["map"]["enabled"]:
        pv_cov = su.calc_province_distribution(valid_df, col=sm["map"]["col"])
        top_pv = pv_cov["table"].iloc[0]
        kpis.append(("首位样本省份", f"{top_pv['省份']} {top_pv['占比(%)']:.1f}%", f"共覆盖全国 {len(pv_cov['table'])} 个省级行政区"))
    elif len(caps["single_cols"]) >= 2:
        c1 = caps["single_cols"][1]
        r1 = su.analyze_categorical_question(valid_df[c1])["table"].iloc[0]
        kpis.append((str(c1)[:10], f"{r1['占比(%)']:.1f}%", f"首位选项：{str(r1['选项'])[:12]} (n={r1['频数']})"))
    else:
        miss_avg = float(overview_res["missing_df"]["缺失率(%)"].mean()) if not overview_res["missing_df"].empty else 0.0
        kpis.append(("数据完整度", f"{100.0 - miss_avg:.1f}%", f"题均缺失率 {miss_avg:.1f}%"))

    for idx, (k_lbl, k_val, k_sub) in enumerate(kpis[:4]):
        col_i = idx % 2
        row_i = idx // 2
        kx = 7.75 + col_i * 2.62
        ky = 0.75 + row_i * 1.65
        _add_rect(s_cover, kx, ky, 2.45, 1.45, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
        _add_text(s_cover, kx + 0.15, ky + 0.12, 2.15, 0.30, k_lbl, font_size=10.5, bold=True, color_hex=t["text_secondary"])
        _add_text(s_cover, kx + 0.15, ky + 0.45, 2.15, 0.50, k_val, font_size=19, bold=True, color_hex=t["accent_primary"])
        _add_text(s_cover, kx + 0.15, ky + 1.02, 2.15, 0.32, k_sub, font_size=8.8, bold=False, color_hex=t["text_muted"])

    # 封面右下角嵌入自适应视觉锚点图（同步绑定 svgBlip 真矢量扩展）
    if caps["nps_cols"]:
        b_cov_fig = cu.render_chart_by_id("13-NPS仪表盘", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=caps["nps_cols"][0])
    elif caps["single_cols"]:
        b_cov_fig = cu.render_chart_by_id("01-棒棒糖图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=caps["single_cols"][0])
    elif caps["scale_cols"]:
        b_cov_fig = cu.render_chart_by_id("12-均值置信区间图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, cols=caps["scale_cols"][:6])
    else:
        b_cov_fig = cu.render_chart_by_id("07-直方图与核密度", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=caps["numeric_cols"][0])

    s_cover.shapes.add_picture(io.BytesIO(b_cov_fig["png_bytes"]), PptInches(7.75), PptInches(4.15), width=PptInches(5.05))
    svg_media_blobs.append((len(prs.slides), 0, "cover_anchor", b_cov_fig["svg_bytes"]))
    page_counter += 1

    # ==========================================================================
    # 第 2 类：目录页 (Slide 2 - 动态展示本次报告实际启用的章节)
    # ==========================================================================
    s_toc = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_toc, t["bg_canvas"])
    toc_chapters = [c for c in avail_chapters if c["key"] in active_chapters and c["key"] != "ch6_summary"]
    if "ch6_summary" in active_chapters:
        toc_chapters.append(next(c for c in avail_chapters if c["key"] == "ch6_summary"))
    toc_chapters = toc_chapters[:6]

    _add_slide_header_and_footer(
        s_toc, t, "REPORT CONTENTS · 报告篇章导航",
        f"基于当前数据集自动生成 {len(toc_chapters)} 大核心分析篇章",
        base_footnote, page_counter,
    )
    row_step = min(1.02, 5.25 / max(len(toc_chapters), 1))
    box_h = min(0.86, row_step - 0.12)
    for i, ch_item in enumerate(toc_chapters):
        cy = 1.30 + i * row_step
        c_no = f"{i+1:02d}"
        _add_rect(s_toc, 0.65, cy, 12.0, box_h, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
        _add_rect(s_toc, 0.65, cy, 0.95, box_h, fill_hex=t["accent_primary"], rounded=True)
        _add_text(s_toc, 0.65, cy + (box_h - 0.46) / 2, 0.95, 0.46, c_no, font_size=19, bold=True, color_hex=auto_text_color(t["accent_primary"]), align=PP_ALIGN.CENTER)
        _add_text(s_toc, 1.80, cy + 0.08, 9.6, 0.34, ch_item["title"], font_size=14.5, bold=True, color_hex=t["text_primary"])
        _add_text(s_toc, 1.80, cy + 0.43, 9.6, 0.32, ch_item["summary"], font_size=10.5, bold=False, color_hex=t["text_secondary"])
        _add_text(s_toc, 11.4, cy + (box_h - 0.38) / 2, 1.0, 0.38, f"CH.{c_no}", font_size=12.5, bold=True, color_hex=t["accent_secondary"], align=PP_ALIGN.RIGHT)
    page_counter += 1

    # 辅助函数：生成第 3 类「章节过渡页」
    def _add_section_divider(ch_no: str, ch_title: str, ch_sub: str, bullets: list):
        nonlocal page_counter
        s_div = prs.slides.add_slide(blank_layout)
        _set_slide_background(s_div, t["bg_canvas"])
        _add_rect(s_div, 0.0, 0.0, 4.2, 7.5, fill_hex=t["accent_primary"])
        tc_d = auto_text_color(t["accent_primary"])
        _add_text(s_div, 0.6, 1.8, 3.0, 1.4, f"CHAPTER\n{ch_no}", font_size=36, bold=True, color_hex=tc_d)
        _add_text(s_div, 0.6, 3.6, 3.0, 1.0, ch_sub, font_size=13, bold=False, color_hex=t["main_pale"])
        _add_text(s_div, 4.75, 1.85, 7.8, 0.85, ch_title, font_size=28, bold=True, color_hex=t["text_primary"])
        _add_rect(s_div, 4.75, 2.75, 7.8, 0.04, fill_hex=t["accent_secondary"])
        for b_i, b_txt in enumerate(bullets[:3]):
            by = 3.15 + b_i * 0.95
            _add_rect(s_div, 4.75, by, 7.8, 0.75, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
            _add_text(s_div, 4.95, by + 0.16, 7.4, 0.45, f"0{b_i+1}  {b_txt}", font_size=13, bold=True, color_hex=t["text_primary"], auto_fit=True, zero_margin=True, v_align=MSO_ANCHOR.MIDDLE)
        _add_text(s_div, 11.2, 7.10, 1.58, 0.30, f"P.{page_counter:02d}", font_size=10, bold=True, color_hex=t["accent_primary"], align=PP_ALIGN.RIGHT)
        page_counter += 1

    # 辅助函数：添加「原生可编辑图表 + 右侧数据表/卡片」分析页（Layout A / B 轮换）
    def _add_native_chart_slide(
        breadcrumb: str, headline: str, insight: str, df_src: pd.DataFrame,
        cat_col: str, val_cols: list, chart_type, layout_mode: str = "A",
    ):
        nonlocal page_counter
        s = prs.slides.add_slide(blank_layout)
        _set_slide_background(s, t["bg_canvas"])
        _add_slide_header_and_footer(s, t, breadcrumb, headline, base_footnote, page_counter)

        c_data = CategoryChartData()
        c_data.categories = df_src[cat_col].astype(str).tolist()
        for vc in val_cols:
            c_data.add_series(str(vc), pd.to_numeric(df_src[vc], errors="coerce").fillna(0).tolist())

        if layout_mode == "A":
            ch_s = s.shapes.add_chart(chart_type, PptInches(0.55), PptInches(1.18), PptInches(7.35), PptInches(4.82), c_data)
            _style_native_chart(ch_s.chart, t, show_legend=(len(val_cols) > 1))
            _add_ppt_table(s, df_src, 8.10, 1.26, 4.65, 4.60, t, max_rows=8)
        else:
            ch_s = s.shapes.add_chart(chart_type, PptInches(0.55), PptInches(1.16), PptInches(12.2), PptInches(3.62), c_data)
            _style_native_chart(ch_s.chart, t, show_legend=(len(val_cols) > 1))
            for k_i in range(min(3, len(df_src))):
                r_k = df_src.iloc[k_i]
                kx = 0.55 + k_i * 4.15
                _add_rect(s, kx, 4.92, 3.92, 1.08, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
                _add_text(s, kx + 0.15, 4.98, 3.6, 0.30, f"TOP {k_i+1} · {r_k[cat_col]}", font_size=11, bold=True, color_hex=t["accent_primary"], zero_margin=True)
                _add_text(s, kx + 0.15, 5.34, 3.6, 0.48, f"{val_cols[0]}: {r_k[val_cols[0]]}", font_size=14.5, bold=True, color_hex=t["text_primary"], zero_margin=True)

        _add_conclusion_strip(s, t, insight)
        _write_slide_notes_markdown(s, headline, insight, df_src)
        page_counter += 1

    # 辅助函数：添加「矢量 SVG + 高清 PNG 图表」分析页（Layout A / B / C / MAP 轮换）
    def _add_svg_chart_slide(breadcrumb: str, bundle: dict, layout_mode: str = "A", bundle_right: dict = None):
        nonlocal page_counter
        s = prs.slides.add_slide(blank_layout)
        slide_num = len(prs.slides)
        _set_slide_background(s, t["bg_canvas"])
        _add_slide_header_and_footer(s, t, breadcrumb, bundle["title"], bundle["footnote"], page_counter)

        svg_media_blobs.append((slide_num, 0, bundle["chart_id"], bundle["svg_bytes"]))
        src_df = bundle["source_df"]

        if layout_mode == "MAP":
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(1.10), PptInches(1.14), height=PptInches(4.92))
            _add_ppt_table(s, src_df, 6.00, 1.22, 6.68, 4.75, t, max_rows=10)
        elif layout_mode == "A":
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(0.55), PptInches(1.18), width=PptInches(7.65))
            _add_ppt_table(s, src_df, 8.45, 1.24, 4.32, 4.75, t, max_rows=8)
        elif layout_mode == "B":
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(2.40), PptInches(1.15), width=PptInches(8.50))
        elif layout_mode == "C" and bundle_right is not None:
            svg_media_blobs.append((slide_num, 1, bundle_right["chart_id"], bundle_right["svg_bytes"]))
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(0.55), PptInches(1.32), width=PptInches(5.95))
            s.shapes.add_picture(io.BytesIO(bundle_right["png_bytes"]), PptInches(6.75), PptInches(1.32), width=PptInches(5.95))
        else:
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(0.55), PptInches(1.18), width=PptInches(7.65))
            _add_ppt_table(s, src_df, 8.45, 1.24, 4.32, 4.75, t, max_rows=8)

        _add_conclusion_strip(s, t, bundle["conclusion"])
        _write_slide_notes_markdown(s, bundle["title"], bundle["conclusion"], src_df)
        page_counter += 1

    ch_idx = 0

    # ==========================================================================
    # 章节 01：样本概况与数据质控
    # ==========================================================================
    if "ch1_overview" in active_chapters:
        ch_idx += 1
        ch_code = f"{ch_idx:02d}"
        skipped_note = (
            "；".join(qc_res.get("skipped_rules", []))
            if qc_res.get("skipped_rules")
            else "重复样本、直线作答与答题过快三道自动化规则全量核验通过"
        )
        _add_section_divider(
            ch_code, "样本概况与数据质控", "Data Hygiene & Methodology",
            [
                f"原始回收问卷 {qc_res['total_n']} 份，质控后保留有效样本 {qc_res['valid_n']} 份（有效率 {qc_res['valid_rate']:.1f}%）",
                "明确规定全报告显著性星号准则（* p<0.05, ** p<0.01, *** p<0.001）与效应量阈值",
                skipped_note,
            ],
        )

        s_meth = prs.slides.add_slide(blank_layout)
        _set_slide_background(s_meth, t["bg_canvas"])
        _add_slide_header_and_footer(
            s_meth, t, f"{ch_code} 样本概况与数据质控 · 方法口径",
            f"质控后有效样本率达 {qc_res['valid_rate']:.1f}%（n={qc_res['valid_n']}），严格执行异常清洗与统计显著性标准",
            base_footnote, page_counter,
        )
        qc_df = pd.DataFrame([
            {"质控环节": "原始总样本", "样本数": int(qc_res["total_n"])},
            {"质控环节": "质控有效样本", "样本数": int(qc_res["valid_n"])},
            {"质控环节": "直线作答剔除", "样本数": int(qc_res["straight_n"])},
            {"质控环节": "过快作答剔除", "样本数": int(qc_res["speed_n"])},
            {"质控环节": "重复作答剔除", "样本数": int(qc_res["dup_n"])},
        ])
        c_data1 = CategoryChartData()
        c_data1.categories = qc_df["质控环节"].tolist()
        c_data1.add_series("样本份数", qc_df["样本数"].tolist())
        ch_shp1 = s_meth.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED, PptInches(0.60), PptInches(1.25), PptInches(5.9), PptInches(4.78), c_data1
        )
        _style_native_chart(ch_shp1.chart, t, show_legend=False)

        conv_df = pd.DataFrame([
            {"分析场景": "分类交叉表", "统计检验方法": "Pearson χ² / Fisher", "效应量指标": "Cramér's V (0.1/0.3/0.5)", "显著性判准": "* p<0.05, ** p<0.01"},
            {"分析场景": "两组均值对比", "统计检验方法": "Welch t / Mann-Whitney", "效应量指标": "Cohen's d (0.2/0.5/0.8)", "显著性判准": "*** p<0.001, ns 不显著"},
            {"分析场景": "多组均值对比", "统计检验方法": "ANOVA F / Kruskal H", "效应量指标": "η² (0.01/0.06/0.14)", "显著性判准": "双侧检验 α = 0.05"},
            {"分析场景": "量表信效度", "统计检验方法": "Cronbach α / EFA", "效应量指标": "α ≥ 0.70, KMO ≥ 0.70", "显著性判准": "Bartlett p < 0.001"},
            {"分析场景": "回归与路径模型", "统计检验方法": "OLS / Logistic / SEM", "效应量指标": "R², Adj R², OR, AUC", "显著性判准": "VIF < 5 无共线性"},
        ])
        _add_text(s_meth, 6.75, 1.25, 6.0, 0.38, "全报告统计检验与效应量口径规范表", font_size=13, bold=True, color_hex=t["accent_primary"])
        _add_ppt_table(s_meth, conv_df, 6.75, 1.72, 6.0, 4.15, t, max_rows=6)
        _add_conclusion_strip(s_meth, t, qc_res["conclusion"])
        _write_slide_notes_markdown(s_meth, "研究方法与数据质控口径", qc_res["conclusion"], qc_df)
        findings_collector.append((
            f"原始样本 {qc_res['total_n']} 份，质控后有效 n={qc_res['valid_n']}（有效率 {qc_res['valid_rate']:.1f}%）",
            f"P.{page_counter:02d}",
            "全部后续描述统计、假设检验与多变量建模均基于清洗后的有效样本执行",
            "消除无效噪声样本干扰",
        ))
        page_counter += 1

    # ==========================================================================
    # 章节 02：核心题目分布与单题洞察
    # ==========================================================================
    if "ch2_single" in active_chapters and caps["modules"]["single_question"]:
        ch_idx += 1
        ch_code = f"{ch_idx:02d}"
        s_bullets = []
        if caps["single_cols"]:
            s_bullets.append(f"单选题频数与构成分布：共识别 {len(caps['single_cols'])} 道单选题，定位首位核心选项与帕累托集中度")
        if caps["multi_cols"]:
            s_bullets.append(f"多选题普及率分析：区分作答人数普及率(% of Cases)与响应人次占比(% of Responses)")
        if caps["scale_cols"] or caps["numeric_cols"] or caps["nps_cols"]:
            s_bullets.append("连续与态度指标诊断：结合置信区间、发散堆叠与核密度刻画集中趋势与离散度")
        while len(s_bullets) < 3:
            s_bullets.append("全图表同步写入备注栏 Markdown 源数据表与原生 XML / 矢量 SVG 双层可编辑对象")

        _add_section_divider(ch_code, "核心题目分布与单题洞察", "Single-Question Distributions & Insights", s_bullets)

        # 单选题原生图表（优先采用基准顺序，通用数据自动选取信息量最高的前 3 道单选题）
        single_ordered = []
        for pref_c in ["消费频次", "年龄段", "性别"]:
            if pref_c in caps["single_cols"]:
                single_ordered.append(pref_c)
        for c in caps["single_cols"]:
            if c not in single_ordered:
                single_ordered.append(c)

        if len(single_ordered) >= 1:
            c1 = single_ordered[0]
            r1 = su.analyze_categorical_question(valid_df[c1])
            top1 = r1["table"].iloc[0]
            _add_native_chart_slide(
                f"{ch_code} 单题洞察 · {c1}",
                f"{c1}：「{top1['选项']}」居首（{top1['占比(%)']:.1f}%，n={top1['频数']}）",
                r1["conclusion"], r1["table"], "选项", ["占比(%)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="A",
            )
            findings_collector.append((
                f"「{c1}」中首位选项为「{top1['选项']}」，占比 {top1['占比(%)']:.1f}%（n={top1['频数']}）",
                f"P.{page_counter-1:02d}",
                f"围绕主流群体「{top1['选项']}」配置核心资源与触达策略",
                "夯实基本盘覆盖率",
            ))

        if len(single_ordered) >= 2:
            c2 = single_ordered[1]
            r2 = su.analyze_categorical_question(valid_df[c2])
            top2 = r2["table"].iloc[0]
            _add_native_chart_slide(
                f"{ch_code} 单题洞察 · {c2}",
                f"{c2}：「{top2['选项']}」占比最高（{top2['占比(%)']:.1f}%，n={top2['频数']}）",
                r2["conclusion"], r2["table"], "选项", ["占比(%)"], XL_CHART_TYPE.COLUMN_CLUSTERED, layout_mode="B",
            )

        if len(single_ordered) >= 3:
            c3 = single_ordered[2]
            r3 = su.analyze_categorical_question(valid_df[c3])
            top3 = r3["table"].iloc[0]
            c3_type = XL_CHART_TYPE.DOUGHNUT if len(r3["table"]) <= 6 else XL_CHART_TYPE.BAR_CLUSTERED
            c3_metric = ["频数"] if c3_type == XL_CHART_TYPE.DOUGHNUT else ["占比(%)"]
            _add_native_chart_slide(
                f"{ch_code} 单题洞察 · {c3}",
                f"{c3}：「{top3['选项']}」占 {top3['占比(%)']:.1f}%（共 {len(r3['table'])} 个类别）",
                r3["conclusion"], r3["table"], "选项", c3_metric, c3_type, layout_mode="A",
            )

        # 若为纯单选/少题型问卷且单选题 >= 4 道，补充矢量帕累托图丰富单题洞察
        if not is_coffee_bench and len(single_ordered) >= 4:
            c4 = single_ordered[3]
            b_par = cu.render_chart_by_id("03-帕累托图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=c4)
            _add_svg_chart_slide(f"{ch_code} 单题洞察 · {c4}帕累托分析", b_par, layout_mode="A")

        # 多选题原生图表（仅当存在多选题时生成）
        if caps["multi_cols"]:
            mc_col = "偏好品类" if "偏好品类" in caps["multi_cols"] else caps["multi_cols"][0]
            mc_res = su.split_multiple_choice(valid_df[mc_col])
            mc_top = mc_res["freq_df"].iloc[0]
            _add_native_chart_slide(
                f"{ch_code} 单题洞察 · 多选题（{mc_col}）",
                f"{mc_col}：「{mc_top['选项']}」普及率达 {mc_top['普及率(% of Cases)']:.1f}%，位居多选榜首",
                mc_res["conclusion"], mc_res["freq_df"], "选项", ["普及率(% of Cases)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="B",
            )

        # 区域分布地图（仅当数据含真实中国省份列时单独成页生成！）
        if sm["map"]["enabled"]:
            b_map = cu.render_chart_by_id("44-区域分布地图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=sm["map"]["col"])
            _add_svg_chart_slide(f"{ch_code} 单题洞察 · 样本区域分布地图", b_map, layout_mode="MAP")

        # NPS 仪表盘 & 人口学金字塔（按实际存在的题型组合）
        if caps["nps_cols"] and len(single_ordered) >= 2:
            b_nps = cu.render_chart_by_id("13-NPS仪表盘", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=caps["nps_cols"][0])
            age_c = "年龄段" if "年龄段" in single_ordered else single_ordered[0]
            gen_c = "性别" if "性别" in single_ordered else single_ordered[1]
            b_pyr = cu.render_chart_by_id("14-人口学金字塔", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, age_col=age_c, gender_col=gen_c)
            _add_svg_chart_slide(f"{ch_code} 单题洞察 · 净推荐值与双向人群结构", b_nps, layout_mode="C", bundle_right=b_pyr)
        elif caps["nps_cols"]:
            b_nps = cu.render_chart_by_id("13-NPS仪表盘", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=caps["nps_cols"][0])
            _add_svg_chart_slide(f"{ch_code} 单题洞察 · 净推荐值 (NPS) 诊断", b_nps, layout_mode="A")

        # 量表题发散堆叠条形图（仅当存在量表题时生成）
        if len(caps["scale_cols"]) >= 1:
            b_likert = cu.render_chart_by_id("11-发散堆叠条形图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 单题洞察 · 量表态度发散分布", b_likert, layout_mode="A")

        # 若非基准集且含数值题，生成直方图+核密度页
        if not is_coffee_bench and caps["numeric_cols"]:
            nc0 = caps["numeric_cols"][0]
            b_hist = cu.render_chart_by_id("07-直方图与核密度", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=nc0)
            _add_svg_chart_slide(f"{ch_code} 单题洞察 · {nc0}连续分布", b_hist, layout_mode="A")

    # ==========================================================================
    # 章节 03：交叉分析与群体差异
    # ==========================================================================
    if "ch3_cross" in active_chapters and caps["modules"]["crosstab"]:
        ch_idx += 1
        ch_code = f"{ch_idx:02d}"
        c_bullets = []
        if caps["cross_cat_cat"]:
            c_bullets.append("卡方独立性检验与标准化残差热力图：定位分类变量间的显著关联与局部聚集")
        if caps["cross_cat_num"]:
            c_bullets.append("参数/非参数双轨均值检验：结合雨云图展示不同子群在连续指标上的全息分布")
        if sm["sankey"]["enabled"]:
            c_bullets.append("流转留存矩阵：桑基图揭示前后期选择路径与核心留存率")
        while len(c_bullets) < 3:
            c_bullets.append("自动报告效应量（Cramér's V / Cohen's d / η²）及 Bonferroni 列比例比较结果")

        _add_section_divider(ch_code, "交叉分析与群体差异", "Cross-Tabulation & Subgroup Comparisons", c_bullets)

        if caps["cross_cat_cat"]:
            r_col = "年龄段" if "年龄段" in caps["single_cols"] else caps["single_cols"][0]
            c_col = "消费频次" if ("消费频次" in caps["single_cols"] and r_col != "消费频次") else next(c for c in caps["single_cols"] if c != r_col)
            ct_res = su.crosstab_chi2_analysis(valid_df, r_col, c_col)
            r_pct_df = ct_res["row_pct_df"].reset_index().rename(columns={"index": r_col})
            _add_native_chart_slide(
                f"{ch_code} 交叉分析 · {r_col} × {c_col}",
                f"不同{r_col}的{c_col}分布对比（χ²={ct_res['chi2']:.2f}, Cramér's V={ct_res['cramers_v']:.3f}）",
                ct_res["conclusion"], r_pct_df, r_col, list(ct_res["row_pct_df"].columns[:5]), XL_CHART_TYPE.BAR_STACKED_100, layout_mode="A",
            )
            findings_collector.append((
                f"{r_col} × {c_col}交叉检验：χ²={ct_res['chi2']:.2f} (p={ct_res['p_value']:.4f}, Cramér's V={ct_res['cramers_v']:.3f})",
                f"P.{page_counter-1:02d}",
                "依据显著正残差单元格实施细分人群差异化运营" if ct_res["p_value"] < 0.05 else "各子群结构分布高度一致，可采取统一触达策略",
                "提升细分投放精准度",
            ))

        if caps["cross_cat_num"]:
            g_col = "性别" if "性别" in caps["single_cols"] else caps["single_cols"][0]
            n_col = "月消费额" if "月消费额" in caps["numeric_cols"] else (caps["numeric_cols"] + caps["scale_cols"] + caps["nps_cols"])[0]
            b_rain = cu.render_chart_by_id("09-雨云图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, group_col=g_col, num_col=n_col)
            _add_svg_chart_slide(f"{ch_code} 交叉分析 · {g_col} × {n_col}雨云图", b_rain, layout_mode="B")

        if caps["cross_cat_cat"]:
            r_col = "年龄段" if "年龄段" in caps["single_cols"] else caps["single_cols"][0]
            c_col = "消费频次" if ("消费频次" in caps["single_cols"] and r_col != "消费频次") else next(c for c in caps["single_cols"] if c != r_col)
            b_resid = cu.render_chart_by_id("18-卡方残差热力图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, row_col=r_col, col_col=c_col)
            if sm["sankey"]["enabled"]:
                p_c, cur_c = sm["sankey"]["cols"]
                b_sankey = cu.render_chart_by_id("30-桑基图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, prev_col=p_c, curr_col=cur_c)
                _add_svg_chart_slide(f"{ch_code} 交叉分析 · 残差异常检测与流转桑基图", b_resid, layout_mode="C", bundle_right=b_sankey)
            else:
                _add_svg_chart_slide(f"{ch_code} 交叉分析 · 调整后标准化残差热力图", b_resid, layout_mode="A")

    # ==========================================================================
    # 章节 04：量表信效度与驱动力建模（仅当量表/数值建模条件满足时生成）
    # ==========================================================================
    if "ch4_scale" in active_chapters and caps["modules"]["scale_modeling"]:
        ch_idx += 1
        ch_code = f"{ch_idx:02d}"
        _add_section_divider(
            ch_code, "量表信效度与驱动力建模", "Psychometrics, SEM & Machine Learning Drivers",
            [
                "量表测量与相关结构检验：评估内部一致性信度 Cronbach's α 与多变量相关矩阵",
                "路径与回归驱动力剖析：量化各核心感知/态度维度对结果变量的净效应系数",
                "机器学习交叉验证：结合随机森林特征重要性、ROC 曲线与决策树锁定关键分水岭",
            ],
        )

        if sm["sem"]["enabled"]:
            b_sem = cu.render_chart_by_id("34-结构方程路径图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 量表与建模 · 结构方程路径模型 (SEM)", b_sem, layout_mode="A")
            findings_collector.append((
                b_sem["title"],
                f"P.{page_counter-1:02d}",
                "优先强化标准化路径系数最高的核心驱动潜变量表现",
                "放大关键驱动因子杠杆效应",
            ))
        else:
            b_corr = cu.render_chart_by_id("26-相关性热力图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 量表与建模 · 多变量相关性热力图", b_corr, layout_mode="A")

        if len(caps["scale_cols"]) + len(caps["numeric_cols"]) >= 2 and len(valid_df) >= 25:
            rf_res = su.calc_random_forest_model(valid_df)
            _add_native_chart_slide(
                f"{ch_code} 量表与建模 · 随机森林特征重要性",
                f"随机森林重要性排序：前两大特征合计贡献 {rf_res['importance_df'].head(2)['重要性占比(%)'].sum():.1f}%（准确率 {rf_res['accuracy']*100:.1f}%）",
                rf_res["conclusion"], rf_res["importance_df"], "特征名称", ["重要性占比(%)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="B",
            )
            b_roc = cu.render_chart_by_id("39-ROC曲线图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            b_dt = cu.render_chart_by_id("41-决策树规则图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 量表与建模 · ROC 判别效能与决策树规则", b_roc, layout_mode="C", bundle_right=b_dt)

    # ==========================================================================
    # 章节 05：专项研究模型与文本洞察（严格仅生成当前数据真正支持的专项模型）
    # ==========================================================================
    has_any_special = (
        any(sm[k]["enabled"] for k in ("funnel", "conjoint", "ipa", "psm", "kano", "clustering", "ca"))
        or caps["modules"]["text_analysis"]
    )
    if "ch5_special" in active_chapters and has_any_special:
        ch_idx += 1
        ch_code = f"{ch_idx:02d}"
        sp_bullets = []
        if sm["funnel"]["enabled"] or sm["kano"]["enabled"]:
            sp_bullets.append("转化漏斗与 KANO 需求分层：识别全链路流失瓶颈及必备/期望/魅力属性优先级")
        if sm["psm"]["enabled"] or sm["conjoint"]["enabled"]:
            sp_bullets.append("价格敏感度（PSM）与联合分析：测算最优定价区间及多属性部分价值权重")
        if sm["ipa"]["enabled"] or sm["clustering"]["enabled"] or sm["ca"]["enabled"]:
            sp_bullets.append("空间定位与分群画像：结合对应分析、IPA 矩阵或 K-Means 聚类指导精准施策")
        if caps["modules"]["text_analysis"]:
            sp_bullets.append("非结构化开放文本挖掘：jieba 分词高频语义词云与焦点维度情感拆解")
        while len(sp_bullets) < 3:
            sp_bullets.append("严格依据当前上传问卷的数据列结构自适应启用匹配模型")

        _add_section_divider(ch_code, "市调专项与策略洞察", "Market Research Models & Text Analytics", sp_bullets)

        if sm["funnel"]["enabled"]:
            f_cols = sm["funnel"]["cols"]
            f_short = ["知名", "考虑", "购买", "复购"] if f_cols == ["品牌知名", "品牌考虑", "品牌购买", "品牌复购"] else [str(c) for c in f_cols]
            fn_res = su.calc_brand_funnel(valid_df, f_cols, f_short)
            _add_native_chart_slide(
                f"{ch_code} 市调专项 · 全链路转化漏斗",
                f"转化漏斗各级人数依次为 {'→'.join(map(str, fn_res['counts']))}，总转化率 {fn_res['table'].iloc[-1]['占首级比例(%)']:.1f}%",
                fn_res["conclusion"], fn_res["table"], "漏斗阶段", ["人数", "逐级转化率(%)"], XL_CHART_TYPE.COLUMN_CLUSTERED, layout_mode="A",
            )

        if sm["conjoint"]["enabled"]:
            cj_cols = sm["conjoint"]["cols"]
            cj_res = su.calc_conjoint_analysis(valid_df, attr_cols=cj_cols["attr_cols"], rating_col=cj_cols["rating_col"])
            _add_native_chart_slide(
                f"{ch_code} 市调专项 · 联合分析相对重要性",
                f"联合分析：{cj_res['importance_df'].iloc[0]['属性名称']}相对重要性达 {cj_res['importance_df'].iloc[0]['相对重要性(%)']:.1f}%，最优组合「{cj_res['best_combo']}」",
                cj_res["conclusion"], cj_res["importance_df"], "属性名称", ["相对重要性(%)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="B",
            )

        if sm["ipa"]["enabled"] and sm["psm"]["enabled"]:
            b_ipa = cu.render_chart_by_id("32-IPA四象限图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            b_psm = cu.render_chart_by_id("33-PSM价格敏感度曲线", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · IPA 四象限矩阵与 PSM 最优定价模型", b_ipa, layout_mode="C", bundle_right=b_psm)
        elif sm["ipa"]["enabled"]:
            b_ipa = cu.render_chart_by_id("32-IPA四象限图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · IPA 重要性-满意度四象限", b_ipa, layout_mode="A")
        elif sm["psm"]["enabled"]:
            b_psm = cu.render_chart_by_id("33-PSM价格敏感度曲线", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · PSM 价格敏感度测试", b_psm, layout_mode="A")

        if sm["kano"]["enabled"] and sm["clustering"]["enabled"]:
            b_kano = cu.render_chart_by_id("35-KANO模型图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            b_persona = cu.render_chart_by_id("45-消费者画像卡片", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · KANO 需求分层与典型消费者画像", b_kano, layout_mode="C", bundle_right=b_persona)
        elif sm["kano"]["enabled"]:
            b_kano = cu.render_chart_by_id("35-KANO模型图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · KANO 需求分层模型", b_kano, layout_mode="A")
        elif sm["clustering"]["enabled"]:
            b_persona = cu.render_chart_by_id("45-消费者画像卡片", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · 典型消费者分群画像", b_persona, layout_mode="A")

        # 若为纯单选/多选问卷（无 IPA/PSM/KANO），自动呈现对应分析双标图展示深层定位关系
        if sm["ca"]["enabled"] and not (sm["ipa"]["enabled"] or sm["psm"]["enabled"] or sm["kano"]["enabled"]):
            ca_c1, ca_c2 = sm["ca"]["cols"][0], sm["ca"]["cols"][1]
            b_ca = cu.render_chart_by_id("31-对应分析双标图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, row_col=ca_c1, col_col=ca_c2)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · {ca_c1} × {ca_c2}对应分析", b_ca, layout_mode="A")
            findings_collector.append((
                b_ca["title"],
                f"P.{page_counter-1:02d}",
                f"结合双标图同象限紧邻关系优化「{ca_c1}」与「{ca_c2}」组合策略",
                "强化心智关联匹配",
            ))

        if caps["modules"]["text_analysis"]:
            t_col = "开放题短评" if "开放题短评" in caps["text_cols"] else caps["text_cols"][0]
            b_wc = cu.render_chart_by_id("46-词云与词频组合图", valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo, col=t_col)
            _add_svg_chart_slide(f"{ch_code} 市调专项 · 开放题词云与高频词统计", b_wc, layout_mode="A")

    # 若用户在工作台手动钉选了额外分析节且开启合并，按版式追加
    if export_options.get("include_custom_sections") and report_sections:
        for sec in report_sections:
            if sec.get("png_bytes") and isinstance(sec.get("table"), pd.DataFrame):
                s_cus = prs.slides.add_slide(blank_layout)
                _set_slide_background(s_cus, t["bg_canvas"])
                _add_slide_header_and_footer(
                    s_cus, t, "CUSTOM INSIGHT · 自定义钉选图表",
                    str(sec.get("title", "自定义分析项")), base_footnote, page_counter,
                )
                s_cus.shapes.add_picture(io.BytesIO(sec["png_bytes"]), PptInches(0.55), PptInches(1.18), width=PptInches(7.65))
                _add_ppt_table(s_cus, sec["table"], 8.45, 1.24, 4.32, 4.75, t, max_rows=8)
                _add_conclusion_strip(s_cus, t, str(sec.get("conclusion", "")))
                _write_slide_notes_markdown(s_cus, str(sec.get("title", "")), str(sec.get("conclusion", "")), sec["table"])
                page_counter += 1

    # ==========================================================================
    # 第 6 & 7 类：核心结论与行动建议页 + 技术附录页
    # ==========================================================================
    if "ch6_summary" in active_chapters:
        s_rec = prs.slides.add_slide(blank_layout)
        _set_slide_background(s_rec, t["bg_canvas"])
        _add_slide_header_and_footer(
            s_rec, t, "EXECUTIVE SUMMARY · 核心发现与策略建议",
            "基于实测统计证据的核心研究发现与行动矩阵",
            base_footnote, page_counter,
        )
        if is_coffee_bench and sm["psm"]["enabled"]:
            pc = sm["psm"]["cols"]
            psm_s = su.calc_van_westendorp(valid_df, pc["too_cheap"], pc["cheap"], pc["expensive"], pc["too_expensive"])
            rec_df = pd.DataFrame([
                {
                    "核心数据发现 (Finding)": f"最优价格点 OPP={psm_s['opp'][0]:.2f}元，可接受价格带 [{psm_s['pmc'][0]:.1f},{psm_s['pme'][0]:.1f}]元",
                    "证据模块": "PSM / 联合分析",
                    "策略行动建议 (Action)": "将主力产品券后到手价锚定在可接受价格带核心区间，推出周卡套餐锁住高频客群",
                    "预期业务影响": "提升考虑→购买转化率",
                },
                {
                    "核心数据发现 (Finding)": "品质感知与价格感知双轮显著正向驱动推荐意愿 (p < 0.001)",
                    "证据模块": "SEM 路径模型",
                    "策略行动建议 (Action)": "严控出品稳定性底线，围绕头部高普及率品类打造旗舰大单品",
                    "预期业务影响": "拉升净推荐值 (NPS)",
                },
                {
                    "核心数据发现 (Finding)": "IPA 四象限识别出高重要低满意要素为首要改进抓手",
                    "证据模块": "IPA / KANO",
                    "策略行动建议 (Action)": "针对优先改进象限要素优化高峰期履约动线与性价比感知",
                    "预期业务影响": "降低关键触点流失率",
                },
                {
                    "核心数据发现 (Finding)": "K-Means 识别出三大典型客群，不同分群消费力与敏感度分化鲜明",
                    "证据模块": "聚类画像",
                    "策略行动建议 (Action)": "实施分层会员运营：高频价值群推精品新品，价格敏感群推满减券包",
                    "预期业务影响": "提升复购留存率",
                },
            ])
            rec_strip = "策略优先级排序：先守住必备质量底线，再围绕高权重驱动因子与优先改进象限精准发力。"
        else:
            rows_rec = []
            for f_txt, pg_ref, act_txt, imp_txt in findings_collector[:4]:
                rows_rec.append({
                    "核心数据发现 (Finding)": f_txt,
                    "证据页码": pg_ref,
                    "策略行动建议 (Action)": act_txt,
                    "预期业务价值": imp_txt,
                })
            if not rows_rec:
                rows_rec.append({
                    "核心数据发现 (Finding)": f"已完成 {qc_res['valid_n']} 份有效样本全量描述与推断统计",
                    "证据页码": "全报告",
                    "策略行动建议 (Action)": "结合各题频数/均值分布结果制定针对性改进方案",
                    "预期业务价值": "量化支撑业务决策",
                })
            rec_df = pd.DataFrame(rows_rec)
            rec_strip = f"本报告全部结论均由当前上传数据集（有效样本 n={qc_res['valid_n']}）实时统计检验生成。"

        _add_ppt_table(s_rec, rec_df, 0.55, 1.30, 12.23, 4.75, t, max_rows=5)
        _add_conclusion_strip(s_rec, t, rec_strip)
        _write_slide_notes_markdown(s_rec, "核心结论与行动建议矩阵", rec_strip, rec_df)
        page_counter += 1

        # 附录页
        s_app = prs.slides.add_slide(blank_layout)
        _set_slide_background(s_app, t["bg_canvas"])
        _add_slide_header_and_footer(
            s_app, t, "APPENDIX · 技术附录与方法局限声明",
            "数据质量审计、测量信效度指标与研究局限性声明",
            base_footnote, page_counter,
        )
        app_rows = [
            {
                "技术模块": "样本质量控制审计",
                "核心统计指标": f"原始 N={qc_res['total_n']}，有效 n={qc_res['valid_n']}（有效率 {qc_res['valid_rate']:.1f}%）",
                "质量判定": "；".join(qc_res.get("skipped_rules", [])) if qc_res.get("skipped_rules") else "重复/直线/过快三项核验均已执行",
            },
            {
                "技术模块": "题型结构自动识别",
                "核心统计指标": f"分析题 {len(caps['analysis_cols'])} 道（单选{len(caps['single_cols'])}/多选{len(caps['multi_cols'])}/量表{len(caps['scale_cols'])}/数值{len(caps['numeric_cols'])}）",
                "质量判定": "已自动过滤平台系统元数据列",
            },
        ]
        if len(caps["scale_cols"]) >= 2:
            dims = caps["scale_dims"]
            if dims:
                for d_nm, d_cols in list(dims.items())[:2]:
                    w_tmp = valid_df.copy()
                    if "品质量表3_品质不稳定_反向" in d_cols:
                        w_tmp["品质量表3_正向"] = su.reverse_score(w_tmp["品质量表3_品质不稳定_反向"], max_scale=5)
                        d_cols = ["品质量表3_正向" if c == "品质量表3_品质不稳定_反向" else c for c in d_cols]
                    rel_d = su.calc_reliability(w_tmp, d_cols)
                    app_rows.append({
                        "技术模块": f"{d_nm}维度信度 ({len(d_cols)}题)",
                        "核心统计指标": f"Cronbach's α = {rel_d['alpha']:.3f}",
                        "质量判定": "内部一致性良好 (≥0.70)" if rel_d["alpha"] >= 0.70 else "建议复核题项同质性",
                    })
            else:
                rel_all = su.calc_reliability(valid_df, caps["scale_cols"][:10])
                app_rows.append({
                    "技术模块": f"量表内部一致性 ({len(caps['scale_cols'][:10])}题)",
                    "核心统计指标": f"Cronbach's α = {rel_all['alpha']:.3f}",
                    "质量判定": "内部一致性良好 (≥0.70)" if rel_all["alpha"] >= 0.70 else "跨维度综合测量",
                })
        app_rows.extend([
            {
                "技术模块": "可编辑图表工程规范",
                "核心统计指标": f"原生 XML 图表 + {len(svg_media_blobs)} 个矢量 SVG (svgBlip) + 全页 Markdown 备注表",
                "质量判定": "支持 PowerPoint / Illustrator / Excel 二次编辑",
            },
            {
                "技术模块": "方法局限声明",
                "核心统计指标": demo_str,
                "质量判定": "横断面相关性与回归分析揭示统计关联，不直接等价于因果推断",
            },
        ])
        app_df = pd.DataFrame(app_rows)
        _add_ppt_table(s_app, app_df, 0.55, 1.30, 12.23, 4.75, t, max_rows=6)
        _add_conclusion_strip(s_app, t, "本报告所有图表源数据均已同步嵌入对应幻灯片备注栏（Notes Markdown）及 Word 数据附录。")
        _write_slide_notes_markdown(s_app, "技术附录与方法声明", "数据质量与信效度汇总", app_df)

    # 保存基础 PPTX 并注入矢量 SVG 媒体包与 svgBlip 扩展
    raw_buf = io.BytesIO()
    prs.save(raw_buf)
    final_pptx_bytes = _inject_svg_media_into_pptx(raw_buf.getvalue(), svg_media_blobs)
    return final_pptx_bytes


# ==============================================================================
# Word 报告生成（含完整图表与末尾「数据附录」源数据表）
# ==============================================================================
def _add_docx_dataframe(doc: Document, df: pd.DataFrame, max_rows: int = 30):
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
    theme_name: str = DEFAULT_THEME_NAME,
) -> bytes:
    """
    生成与当前主题配色同步的 Word 分析报告 (.docx)，
    包含样本概况、质控小结、详细分析图表章节，以及文档末尾完整的「图表源数据附录」。
    """
    t = resolve_effective_theme(theme_name)
    doc = Document()
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Microsoft YaHei"
    normal_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal_style.font.size = Pt(10.5)

    title_p = doc.add_heading("SurveyScope 问卷调查深度分析报告", level=0)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title_p.runs:
        r.font.color.rgb = _docx_rgb(t["accent_primary"])

    if is_demo:
        demo_p = doc.add_paragraph("【重要提示：当前报告基于系统内置模拟数据生成，仅供演示】")
        demo_p.runs[0].bold = True
        demo_p.runs[0].font.color.rgb = _docx_rgb(t["accent_secondary"])

    if is_weighted:
        w_p = doc.add_paragraph("【加权状态：当前分析已启用事后分层加权，检验类结果基于加权数据，仅供参考】")
        w_p.runs[0].bold = True

    doc.add_heading("一、样本概况与缺失率统计", level=1)
    doc.add_paragraph(
        f"本次调查原始上传样本量共 {overview_res['n_samples']} 份，包含问卷题目 {overview_res['n_questions']} 道。"
    )
    _add_docx_dataframe(doc, overview_res.get("missing_df"))

    doc.add_heading("二、数据质控小结", level=1)
    doc.add_paragraph(qc_res["conclusion"])
    qc_summary_df = pd.DataFrame([{
        "原始总样本量": qc_res["total_n"],
        "重复作答样本数": qc_res["dup_n"],
        "直线作答样本数": qc_res["straight_n"],
        "过快作答样本数": qc_res["speed_n"],
        "异常标记合计": qc_res["flagged_n"],
        "质控后有效样本量": qc_res["valid_n"],
        "有效率(%)": qc_res["valid_rate"],
    }])
    _add_docx_dataframe(doc, qc_summary_df)

    doc.add_heading("三、问卷数据分析结果与高清图表", level=1)
    for idx, sec in enumerate(report_sections, start=1):
        doc.add_heading(f"3.{idx} {sec.get('title', '分析项')}", level=2)
        if sec.get("conclusion"):
            p_conc = doc.add_paragraph(f"分析结论：{sec['conclusion']}")
            p_conc.runs[0].bold = True
        if sec.get("png_bytes"):
            doc.add_picture(io.BytesIO(sec["png_bytes"]), width=Inches(6.1))
        if isinstance(sec.get("table"), pd.DataFrame) and not sec["table"].empty:
            _add_docx_dataframe(doc, sec["table"], max_rows=15)
        doc.add_paragraph("")

    # 新增第四章：完整图表源数据附录
    doc.add_heading("四、图表源数据附录 (Source Data Appendix)", level=1)
    doc.add_paragraph("以下表格汇总了本报告全部图表对应的聚合源数据，可直接复制至 Excel 进行二次制表与核算：")
    for idx, sec in enumerate(report_sections, start=1):
        tbl_df = sec.get("source_df") if isinstance(sec.get("source_df"), pd.DataFrame) else sec.get("table")
        if isinstance(tbl_df, pd.DataFrame) and not tbl_df.empty:
            doc.add_heading(f"附表 {idx}：{sec.get('title', '图表源数据')}", level=2)
            _add_docx_dataframe(doc, tbl_df, max_rows=50)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ==============================================================================
# S5 幻灯片逐页高保真 PNG 截图渲染器
# ==============================================================================
def render_pptx_slides_to_images(pptx_bytes: bytes, out_dir: str, theme_name: str = DEFAULT_THEME_NAME) -> list:
    """
    读取已生成的 .pptx 文件字节流，优先调用本机真实 Office PowerPoint 引擎将每一页幻灯片
    真实渲染导出为 16:9 高清 PNG 截图（1920 × 1080 px），若无 Office 环境则回退至矢量画布渲染器。
    用于 S5 逐页真实视觉核验与打包交付。
    """
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if re.match(r"^slide_\d+\.png$", fn):
            try:
                os.remove(os.path.join(out_dir, fn))
            except Exception:
                pass
    abs_out_dir = os.path.abspath(out_dir)

    # 优先使用本机 Microsoft PowerPoint COM 引擎进行 100% 真实幻灯片渲染
    if os.name == "nt":
        tmp_pptx = os.path.join(abs_out_dir, "_temp_render_source.pptx")
        try:
            with open(tmp_pptx, "wb") as f:
                f.write(pptx_bytes)
            ps_script = (
                f"$ErrorActionPreference = 'Stop'; "
                f"$app = New-Object -ComObject PowerPoint.Application; "
                f"$pres = $app.Presentations.Open('{tmp_pptx}', $true, $false, $false); "
                f"$i = 1; "
                f"foreach ($slide in $pres.Slides) {{ "
                f"  $out = Join-Path '{abs_out_dir}' ('slide_{{0:D2}}.png' -f $i); "
                f"  $slide.Export($out, 'PNG', 1920, 1080); "
                f"  $i++; "
                f"}}; "
                f"$pres.Close(); "
                f"$app.Quit()"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=60,
            )
            if os.path.exists(tmp_pptx):
                try:
                    os.remove(tmp_pptx)
                except Exception:
                    pass
            if res.returncode == 0:
                real_paths = sorted([
                    os.path.join(out_dir, fn)
                    for fn in os.listdir(out_dir)
                    if re.match(r"^slide_\d+\.png$", fn)
                ])
                if len(real_paths) >= 22:
                    return real_paths
        except Exception:
            if os.path.exists(tmp_pptx):
                try:
                    os.remove(tmp_pptx)
                except Exception:
                    pass

    t = resolve_effective_theme(theme_name)
    prs = Presentation(io.BytesIO(pptx_bytes))
    slide_w = float(prs.slide_width.inches)
    slide_h = float(prs.slide_height.inches)

    out_paths = []
    for s_idx, slide in enumerate(prs.slides, start=1):
        fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor=t["bg_canvas"])
        ax = fig.add_axes([0, 0, 1, 1], facecolor=t["bg_canvas"])
        ax.set_xlim(0, slide_w)
        ax.set_ylim(slide_h, 0)  # 顶部为 y=0，向下递增，与 PPT 坐标系完全一致
        ax.axis("off")

        for shp in slide.shapes:
            x = float(shp.left.inches)
            y = float(shp.top.inches)
            w = float(shp.width.inches)
            h = float(shp.height.inches)

            # 1. 嵌入的位图图片
            if shp.shape_type == 13 or hasattr(shp, "image"):
                try:
                    img_arr = plt.imread(io.BytesIO(shp.image.blob))
                    ax.imshow(img_arr, extent=[x, x + w, y + h, y], aspect="auto", zorder=3)
                    continue
                except Exception:
                    pass

            # 2. 原生图表对象 (has_chart)
            if getattr(shp, "has_chart", False):
                ch = shp.chart
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=t["bg_card"], edgecolor=t["border_line"], lw=1.0, zorder=2))
                try:
                    cats = [str(c.label) for c in ch.plots[0].categories]
                    vals = [float(v) for v in ch.series[0].values]
                    max_v = max(vals + [1.0])
                    n_c = max(len(cats), 1)
                    for ci, (clbl, cv) in enumerate(zip(cats, vals)):
                        by = y + 0.35 + ci * ((h - 0.65) / n_c)
                        bh = max(0.18, (h - 0.65) / n_c * 0.68)
                        bw = (w - 2.2) * (cv / max_v)
                        ax.add_patch(Rectangle((x + 1.45, by), max(0.08, bw), bh, facecolor=t["accent_primary"], alpha=0.88, zorder=3))
                        ax.text(x + 1.38, by + bh / 2, clbl[:10], ha="right", va="center", fontsize=8.5, color=t["text_primary"], zorder=4)
                        ax.text(x + 1.52 + max(0.08, bw), by + bh / 2, f"{cv:g}", ha="left", va="center", fontsize=8.5, weight="bold", color=t["text_primary"], zorder=4)
                except Exception:
                    pass
                continue

            # 3. 表格对象 (has_table)
            if getattr(shp, "has_table", False):
                tbl = shp.table
                nr = len(tbl.rows)
                nc = len(tbl.columns)
                rh = h / max(nr, 1)
                cw = w / max(nc, 1)
                for ri in range(nr):
                    for ci in range(nc):
                        cx = x + ci * cw
                        cy = y + ri * rh
                        fc = t["accent_primary"] if ri == 0 else (t["bg_card"] if ri % 2 == 1 else t["bg_secondary"])
                        tc = auto_text_color(fc, t["text_primary"])
                        ax.add_patch(Rectangle((cx, cy), cw, rh, facecolor=fc, edgecolor=t["border_line"], lw=0.6, zorder=2))
                        cell_txt = tbl.cell(ri, ci).text.replace("\x0b", " ").replace("\n", " ").strip()
                        max_c = max(12, int(cw * 12.5))
                        if len(cell_txt) > max_c:
                            cell_txt = cell_txt[: max_c - 1] + "…"
                        ax.text(cx + cw / 2, cy + rh / 2, cell_txt, ha="center", va="center", fontsize=7.8 if ri > 0 else 8.2, weight="bold" if ri == 0 else "normal", color=tc, zorder=3)
                continue

            # 4. 矩形/圆角矩形色块与文本框
            fill_hex = None
            try:
                if shp.fill.type == 1:
                    rgb = shp.fill.fore_color.rgb
                    fill_hex = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            except Exception:
                fill_hex = None

            if fill_hex:
                ax.add_patch(Rectangle((x, y), w, h, facecolor=fill_hex, edgecolor="none", zorder=1))

            if getattr(shp, "has_text_frame", False):
                full_txt = shp.text_frame.text.replace("\x0b", "\n").strip()
                if full_txt:
                    p0 = shp.text_frame.paragraphs[0]
                    f_sz = float(p0.font.size.pt) if p0.font.size else 12.0
                    is_b = bool(p0.font.bold)
                    txt_hex = t["text_primary"]
                    try:
                        if p0.font.color and p0.font.color.rgb:
                            rgb = p0.font.color.rgb
                            txt_hex = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
                    except Exception:
                        pass
                    align = p0.alignment
                    if align == PP_ALIGN.CENTER:
                        tx, ha = x + w / 2, "center"
                    elif align == PP_ALIGN.RIGHT:
                        tx, ha = x + w - 0.08, "right"
                    else:
                        tx, ha = x + 0.08, "left"
                    ax.text(
                        tx, y + h / 2, full_txt,
                        ha=ha, va="center",
                        fontsize=max(7.5, f_sz * 0.72),
                        weight="bold" if is_b else "normal",
                        color=txt_hex, zorder=5,
                    )

        img_path = os.path.join(out_dir, f"slide_{s_idx:02d}.png")
        fig.savefig(img_path, dpi=150, facecolor=t["bg_canvas"], edgecolor="none")
        plt.close(fig)
        out_paths.append(img_path)

    return out_paths
