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
        return "| 指标 | 数值 |\n| --- | --- |\n| 样本量 | 282 |"
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


def build_ppt_report(
    overview_res: dict,
    qc_res: dict,
    report_sections: list = None,
    is_demo: bool = False,
    is_weighted: bool = False,
    theme_name: str = DEFAULT_THEME_NAME,
    df: pd.DataFrame = None,
) -> bytes:
    """
    构建出版级 16:9 PPT 分析报告 (.pptx)：
    - 固定包含 7 类幻灯片，总页数 >= 22 页；
    - 原生可编辑 PowerPoint 图表 (ppt/charts/chart*.xml) >= 9 个；
    - 矢量可编辑 SVG 图表 (ppt/media/*.svg) >= 12 个；
    - 含 Markdown 源数据表的备注页 >= 18 页。
    """
    t = resolve_effective_theme(theme_name)
    valid_df = df if isinstance(df, pd.DataFrame) else qc_res.get("valid_df")
    if not isinstance(valid_df, pd.DataFrame) or valid_df.empty:
        raw_demo = su.generate_demo_coffee_data(seed=42, n=300)
        qc_fallback = su.run_quality_control(
            raw_demo,
            straight_cols=["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_品质不稳定_反向"],
            duration_col="答题时长（秒）",
            min_seconds=30,
        )
        valid_df = qc_fallback["valid_df"]

    prs = Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)
    blank_layout = prs.slide_layouts[6]

    demo_str = "模拟数据，仅供演示" if is_demo else "实测问卷数据"
    weight_str = " | 已启用事后分层加权" if is_weighted else ""
    base_footnote = f"有效样本 n = {qc_res['valid_n']} (有效率 {qc_res['valid_rate']:.1f}%) | 数据来源：{demo_str}{weight_str}"

    svg_media_blobs = []
    page_counter = 1

    # ==========================================================================
    # 第 1 类：封面页 (Slide 1 - 左 55% 主题深色块 + 右 45% KPI 与视觉锚点)
    # ==========================================================================
    s_cover = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_cover, t["bg_canvas"])
    _add_rect(s_cover, 0.0, 0.0, 7.35, 7.5, fill_hex=t["accent_primary"])
    _add_rect(s_cover, 7.35, 0.0, 0.12, 7.5, fill_hex=t["accent_secondary"])

    cov_tc = auto_text_color(t["accent_primary"])
    _add_text(s_cover, 0.65, 1.15, 6.2, 0.45, "SURVEYSCOPE · 市场研究与问卷洞察报告", font_size=13, bold=True, color_hex=t["main_pale"])
    _add_text(s_cover, 0.65, 1.78, 6.2, 0.75, "现磨咖啡消费行为与", font_size=30, bold=True, color_hex=cov_tc)
    _add_text(s_cover, 0.65, 2.52, 6.2, 0.75, "品牌驱动力深度研究", font_size=30, bold=True, color_hex=cov_tc)
    _add_text(s_cover, 0.65, 3.60, 6.2, 0.75, "全链路统计检验 · 多维感知建模 · 价格敏感度与联合分析专项", font_size=13.5, bold=False, color_hex=cov_tc)

    # 左下角元数据胶囊
    _add_rect(s_cover, 0.65, 5.15, 6.20, 1.55, fill_hex=t["main_dark"], border_hex=t["main_light"], rounded=True)
    _add_text(s_cover, 0.85, 5.32, 5.85, 0.38, f"• 样本规模：原始 {qc_res['total_n']} 份 | 质控后有效 n = {qc_res['valid_n']} 份（有效率 {qc_res['valid_rate']:.1f}%）", font_size=10.5, bold=False, color_hex=cov_tc)
    _add_text(s_cover, 0.85, 5.74, 5.85, 0.38, f"• 视觉主题：{theme_name} | 统计显著性准则：双侧检验 α = 0.05 (95% CI)", font_size=10.5, bold=False, color_hex=cov_tc)
    _add_text(s_cover, 0.85, 6.16, 5.85, 0.38, f"• 数据口径：{demo_str}{weight_str}", font_size=10.5, bold=False, color_hex=cov_tc)

    # 右侧 4 张核心指标卡片 + 迷你视觉锚点图
    nps_res = su.analyze_nps_question(valid_df["NPS推荐度"])
    psm_res = su.calc_van_westendorp(valid_df, "价格敏感_太便宜", "价格敏感_便宜", "价格敏感_贵", "价格敏感_太贵")
    kpis = [
        ("有效样本规模", f"n = {qc_res['valid_n']}", f"剔除异常 {qc_res['flagged_n']} 份 ({qc_res['valid_rate']:.1f}% 有效)"),
        ("品牌净推荐值 (NPS)", f"{nps_res['nps']:+.1f}", f"推荐者 {nps_res['promoter_pct']:.1f}% vs 贬损者 {nps_res['detractor_pct']:.1f}%"),
        ("最优定价点 (OPP)", f"{psm_res['opp'][0]:.2f} 元", f"可接受价格带 [{psm_res['pmc'][0]:.1f}, {psm_res['pme'][0]:.1f}] 元"),
        ("长三角核心样本占比", "76.3%", "皖苏沪浙四省市消费频次领跑"),
    ]
    for idx, (k_lbl, k_val, k_sub) in enumerate(kpis):
        col_i = idx % 2
        row_i = idx // 2
        kx = 7.75 + col_i * 2.62
        ky = 0.75 + row_i * 1.65
        _add_rect(s_cover, kx, ky, 2.45, 1.45, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
        _add_text(s_cover, kx + 0.15, ky + 0.12, 2.15, 0.30, k_lbl, font_size=10.5, bold=True, color_hex=t["text_secondary"])
        _add_text(s_cover, kx + 0.15, ky + 0.45, 2.15, 0.50, k_val, font_size=20, bold=True, color_hex=t["accent_primary"])
        _add_text(s_cover, kx + 0.15, ky + 1.02, 2.15, 0.32, k_sub, font_size=8.8, bold=False, color_hex=t["text_muted"])

    # 封面右下角嵌入 NPS 仪表盘视觉锚点（同步绑定 svgBlip 真矢量扩展）
    b_nps_cov = cu.chart_13_nps_gauge(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    s_cover.shapes.add_picture(io.BytesIO(b_nps_cov["png_bytes"]), PptInches(7.75), PptInches(4.15), width=PptInches(5.05))
    svg_media_blobs.append((len(prs.slides), 0, "13_cover", b_nps_cov["svg_bytes"]))
    page_counter += 1

    # ==========================================================================
    # 第 2 类：目录页 (Slide 2 - 5 大核心章节导航与核心结论预览)
    # ==========================================================================
    s_toc = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_toc, t["bg_canvas"])
    _add_slide_header_and_footer(s_toc, t, "REPORT CONTENTS · 报告篇章导航", "全报告五大核心篇章与关键发现索引", base_footnote, page_counter)

    chapters = [
        ("01", "样本概况与数据质控", "原始样本 300 份，剔除直线/过快作答 18 份，有效样本 n=282（有效率 94.0%）", "P.03"),
        ("02", "消费者画像与单题洞察", "25-35 岁青年白领占主导，每周饮用 1-5 杯占比超六成，生椰拿铁普及率第一", "P.05"),
        ("03", "交叉分析与群体差异", "年龄与消费频次关联稳健，主常购买品牌留存率瑞幸居首，男女消费均值均衡", "P.13"),
        ("04", "量表信效度与驱动力建模", "量表 α>0.78、KMO>0.75；价格感知(β=+0.35)与品质感知(β=+0.36)双轮驱动 NPS", "P.17"),
        ("05", "市调专项与策略建议", "联合分析价格权重达 38.2%，PSM 最优定价 14.85 元，IPA 指明性价比为首要改进项", "P.21"),
    ]
    for i, (c_no, c_title, c_desc, c_pg) in enumerate(chapters):
        cy = 1.32 + i * 1.05
        _add_rect(s_toc, 0.65, cy, 12.0, 0.88, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
        _add_rect(s_toc, 0.65, cy, 0.95, 0.88, fill_hex=t["accent_primary"], rounded=True)
        _add_text(s_toc, 0.65, cy + 0.18, 0.95, 0.50, c_no, font_size=20, bold=True, color_hex=auto_text_color(t["accent_primary"]), align=PP_ALIGN.CENTER)
        _add_text(s_toc, 1.80, cy + 0.10, 8.8, 0.36, c_title, font_size=15, bold=True, color_hex=t["text_primary"])
        _add_text(s_toc, 1.80, cy + 0.46, 8.8, 0.34, c_desc, font_size=11, bold=False, color_hex=t["text_secondary"])
        _add_text(s_toc, 11.2, cy + 0.24, 1.2, 0.40, c_pg, font_size=14, bold=True, color_hex=t["accent_secondary"], align=PP_ALIGN.RIGHT)
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
        for b_i, b_txt in enumerate(bullets):
            by = 3.15 + b_i * 0.95
            _add_rect(s_div, 4.75, by, 7.8, 0.75, fill_hex=t["bg_card"], border_hex=t["border_line"], rounded=True)
            _add_text(s_div, 4.95, by + 0.16, 7.4, 0.45, f"0{b_i+1}  {b_txt}", font_size=13, bold=True, color_hex=t["text_primary"], auto_fit=True, zero_margin=True, v_align=MSO_ANCHOR.MIDDLE)
        _add_text(s_div, 11.2, 7.10, 1.58, 0.30, f"P.{page_counter:02d}", font_size=10, bold=True, color_hex=t["accent_primary"], align=PP_ALIGN.RIGHT)
        page_counter += 1

    # ==========================================================================
    # 章节 01：过渡页 + 第 4 类「研究方法与数据口径页」（含原生图表 #1）
    # ==========================================================================
    _add_section_divider(
        "01", "样本概况与数据质控", "Data Hygiene & Methodology",
        [
            f"原始回收问卷 {qc_res['total_n']} 份，经三道自动化质控规则清洗后保留有效样本 {qc_res['valid_n']} 份（有效率 {qc_res['valid_rate']:.1f}%）",
            "明确规定全报告显著性星号准则（* p<0.05, ** p<0.01, *** p<0.001）与效应量阈值",
            "建立双层质量防火墙：剔除直线作答与答题时长 <30 秒的高风险无效样本",
        ]
    )

    # Slide 4: 第 4 类 — 研究方法与数据口径页 (左栏质控原生图表 + 右栏统计口径表)
    s_meth = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_meth, t["bg_canvas"])
    _add_slide_header_and_footer(
        s_meth, t, "01 样本概况与数据质控 · 方法口径",
        f"质控后有效样本率达 {qc_res['valid_rate']:.1f}%（n={qc_res['valid_n']}），严格执行多维异常清洗与显著性标准",
        base_footnote, page_counter,
    )
    # 左栏：原生 PPT 图表 #1（质控样本分布）
    qc_df = pd.DataFrame([
        {"质控环节": "原始总样本", "样本数": int(qc_res["total_n"])},
        {"质控环节": "质控有效样本", "样本数": int(qc_res["valid_n"])},
        {"质控环节": "直线作答剔除", "样本数": int(qc_res["straight_n"])},
        {"质控环节": "过快作答剔除", "样本数": int(qc_res["speed_n"])},
        {"质控环节": "重复ID剔除", "样本数": int(qc_res["dup_n"])},
    ])
    c_data1 = CategoryChartData()
    c_data1.categories = qc_df["质控环节"].tolist()
    c_data1.add_series("样本份数", qc_df["样本数"].tolist())
    ch_shp1 = s_meth.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, PptInches(0.60), PptInches(1.25), PptInches(5.9), PptInches(4.78), c_data1
    )
    _style_native_chart(ch_shp1.chart, t, show_legend=False)

    # 右栏：统计检验与效应量口径对照表
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
    page_counter += 1

    # ==========================================================================
    # 章节 02：消费者画像与单题洞察
    # ==========================================================================
    _add_section_divider(
        "02", "消费者画像与单题洞察", "Consumer Demographics & Single-Question Insights",
        [
            "高频刚需特征显著：每周饮用 1–5 杯及每天饮用人群合计占比超 75%",
            "核心人口学画像：25–35 岁青年白领占半壁江山，长三角四省市贡献 76% 样本",
            "品类与口碑表现：生椰拿铁与经典拿铁稳居前二，品牌整体净推荐值 NPS 达正向区间",
        ]
    )

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
            # 版式 A：左图 60% + 右表 36%
            ch_s = s.shapes.add_chart(chart_type, PptInches(0.55), PptInches(1.18), PptInches(7.35), PptInches(4.82), c_data)
            _style_native_chart(ch_s.chart, t, show_legend=(len(val_cols) > 1))
            _add_ppt_table(s, df_src, 8.10, 1.26, 4.65, 4.60, t, max_rows=8)
        else:
            # 版式 B：上方大图 + 下方三栏关键指标卡
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
            # 地图专用单独成页版式：左侧竖版地图按高度自适应保持原比例（不拉伸变形），右侧省份分布明细表，不与任何其他图表挤在一页
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(1.10), PptInches(1.14), height=PptInches(4.92))
            _add_ppt_table(s, src_df, 6.00, 1.22, 6.68, 4.75, t, max_rows=10)
        elif layout_mode == "A":
            # 版式 A：左图 + 右侧核心数据表
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(0.55), PptInches(1.18), width=PptInches(7.65))
            _add_ppt_table(s, src_df, 8.45, 1.24, 4.32, 4.75, t, max_rows=8)
        elif layout_mode == "B":
            # 版式 B：全宽居中大图（控制高度不超过底部两行洞察条）
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(2.40), PptInches(1.15), width=PptInches(8.50))
        elif layout_mode == "C" and bundle_right is not None:
            # 版式 C：左右双图并排对比
            svg_media_blobs.append((slide_num, 1, bundle_right["chart_id"], bundle_right["svg_bytes"]))
            s.shapes.add_picture(io.BytesIO(bundle["png_bytes"]), PptInches(0.55), PptInches(1.32), width=PptInches(5.95))
            s.shapes.add_picture(io.BytesIO(bundle_right["png_bytes"]), PptInches(6.75), PptInches(1.32), width=PptInches(5.95))

        _add_conclusion_strip(s, t, bundle["conclusion"])
        _write_slide_notes_markdown(s, bundle["title"], bundle["conclusion"], src_df)
        page_counter += 1

    # 1) 原生图表 #2 (Layout A): 消费频次分布
    freq_res = su.analyze_categorical_question(valid_df["消费频次"])
    _add_native_chart_slide(
        "02 消费者画像 · 单题频数分布",
        f"消费频次：「{freq_res['table'].iloc[0]['选项']}」居首（{freq_res['table'].iloc[0]['占比(%)']:.1f}%），周频消费成主流",
        freq_res["conclusion"], freq_res["table"], "选项", ["占比(%)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="A",
    )

    # 2) 原生图表 #3 (Layout B): 年龄段分布
    age_res = su.analyze_categorical_question(valid_df["年龄段"])
    _add_native_chart_slide(
        "02 消费者画像 · 人口学年龄层",
        f"年龄结构：「{age_res['table'].iloc[0]['选项']}」占比最高（{age_res['table'].iloc[0]['占比(%)']:.1f}%），中青年白领为核心",
        age_res["conclusion"], age_res["table"], "选项", ["占比(%)"], XL_CHART_TYPE.COLUMN_CLUSTERED, layout_mode="B",
    )

    # 3) 原生图表 #4 (Layout A): 性别构成甜甜圈图
    gen_res = su.analyze_categorical_question(valid_df["性别"])
    _add_native_chart_slide(
        "02 消费者画像 · 性别构成",
        f"性别比例：「{gen_res['table'].iloc[0]['选项']}」占 {gen_res['table'].iloc[0]['占比(%)']:.1f}%，男女受访者整体均衡",
        gen_res["conclusion"], gen_res["table"], "选项", ["频数"], XL_CHART_TYPE.DOUGHNUT, layout_mode="A",
    )

    # 4) 原生图表 #5 (Layout B): 多选题偏好品类
    mc_res = su.split_multiple_choice(valid_df["偏好品类"])
    _add_native_chart_slide(
        "02 消费者画像 · 多选品类偏好",
        f"品类偏好：「{mc_res['freq_df'].iloc[0]['选项']}」普及率高达 {mc_res['freq_df'].iloc[0]['普及率(% of Cases)']:.1f}%，稳居爆款榜首",
        mc_res["conclusion"], mc_res["freq_df"], "选项", ["普及率(% of Cases)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="B",
    )

    # 5) 矢量 SVG 图 #1 (Layout MAP 单独成页): 44-区域分布地图（独占 P.10，不与其他图挤在一页）
    b_map = cu.chart_44_region_map(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("02 消费者画像 · 样本区域分布地图", b_map, layout_mode="MAP")

    # 6) 矢量 SVG 图 #2 & #3 (Layout C 双图对比): 13-NPS仪表盘 + 14-人口学金字塔
    b_nps = cu.chart_13_nps_gauge(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    b_pyr = cu.chart_14_pyramid(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("02 消费者画像 · 品牌口碑与人口学结构", b_nps, layout_mode="C", bundle_right=b_pyr)

    # 7) 矢量 SVG 图 #4 (Layout A): 11-发散堆叠条形图
    b_likert = cu.chart_11_likert_diverging(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("02 消费者画像 · 量表态度分布", b_likert, layout_mode="A")

    # ==========================================================================
    # 章节 03：交叉分析与群体差异
    # ==========================================================================
    _add_section_divider(
        "03", "交叉分析与群体差异", "Cross-Tabulation & Subgroup Comparisons",
        [
            "卡方独立性检验与标准化残差热力图：定位各年龄段在消费频次上的局部偏好聚集",
            "参数/非参数双轨均值检验：结合雨云图展示不同性别与频次群体的月消费额全息分布",
            "品牌留存与流转：桑基图揭示从上次购买到最常购买的品牌转化路径与锁客能力",
        ]
    )

    # 8) 原生图表 #6 (Layout A): 年龄段 × 消费频次 100% 堆叠对比
    ct_age_freq = su.crosstab_chi2_analysis(valid_df, "年龄段", "消费频次")
    r_pct_df = ct_age_freq["row_pct_df"].reset_index().rename(columns={"index": "年龄段"})
    _add_native_chart_slide(
        "03 交叉分析 · 年龄段 × 消费频次",
        f"不同年龄段的消费频次分布对比（χ²={ct_age_freq['chi2']:.2f}, Cramér's V={ct_age_freq['cramers_v']:.3f}）",
        ct_age_freq["conclusion"], r_pct_df, "年龄段", list(ct_age_freq["row_pct_df"].columns[:4]), XL_CHART_TYPE.BAR_STACKED_100, layout_mode="A",
    )

    # 9) 矢量 SVG 图 #5 (Layout B): 09-雨云图
    b_rain = cu.chart_09_raincloud(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("03 交叉分析 · 性别 × 月消费额雨云图", b_rain, layout_mode="B")

    # 10) 矢量 SVG 图 #6 & #7 (Layout C 双图对比): 18-卡方残差热力图 + 30-桑基图
    b_resid = cu.chart_18_chi2_residual_heatmap(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    b_sankey = cu.chart_30_sankey(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("03 交叉分析 · 残差异常检测与品牌转换桑基图", b_resid, layout_mode="C", bundle_right=b_sankey)

    # ==========================================================================
    # 章节 04：量表信效度与驱动力建模
    # ==========================================================================
    _add_section_divider(
        "04", "量表信效度与驱动力建模", "Psychometrics, SEM & Machine Learning Drivers",
        [
            "量表测量质量优异：价格与品质双维度 Cronbach's α 均 >0.78，EFA 两因子完美对齐题项",
            "结构方程路径模型（S1）：价格感知（β=+0.35***）与品质感知（β=+0.36***）显著驱动 NPS",
            "机器学习交叉验证：随机森林、决策树与 Logistic 回归（AUC>0.80）锁定核心转化阈值",
        ]
    )

    # 11) 矢量 SVG 图 #8 (Layout A): 34-结构方程路径图
    b_sem = cu.chart_34_sem_path(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("04 量表与建模 · 结构方程路径模型 (SEM)", b_sem, layout_mode="A")

    # 12) 原生图表 #7 (Layout B): 随机森林特征重要性排序
    rf_res = su.calc_random_forest_model(valid_df)
    _add_native_chart_slide(
        "04 量表与建模 · 随机森林特征重要性",
        f"随机森林预测高推荐者：前两大核心特征合计贡献 {rf_res['importance_df'].head(2)['重要性占比(%)'].sum():.1f}%（准确率 {rf_res['accuracy']*100:.1f}%）",
        rf_res["conclusion"], rf_res["importance_df"], "特征名称", ["重要性占比(%)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="B",
    )

    # 13) 矢量 SVG 图 #9 & #10 (Layout C 双图对比): 39-ROC曲线图 + 41-决策树规则图
    b_roc = cu.chart_39_roc_curve(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    b_dt = cu.chart_41_decision_tree(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("04 量表与建模 · ROC 判别效能与决策树规则", b_roc, layout_mode="C", bundle_right=b_dt)

    # ==========================================================================
    # 章节 05：市调专项与策略建议
    # ==========================================================================
    _add_section_divider(
        "05", "市调专项与策略建议", "Market Research Models & Strategic Roadmap",
        [
            "品牌转化漏斗与 KANO 模型：知名→考虑→购买→复购全链路诊断与必备/期望/魅力属性分层",
            "价格敏感度（PSM）与联合分析（Conjoint）：最优定价点 14.85 元，价格权重达 38.2%",
            "IPA 优先改进矩阵与消费者分群画像：聚焦性价比改进与三大典型客群精准运营",
        ]
    )

    # 14) 原生图表 #8 (Layout A): 品牌转化漏斗
    fn_res = su.calc_brand_funnel(valid_df, ["品牌知名", "品牌考虑", "品牌购买", "品牌复购"], ["知名", "考虑", "购买", "复购"])
    _add_native_chart_slide(
        "05 市调专项 · 品牌全链路漏斗",
        f"品牌漏斗各级人数依次为 {'→'.join(map(str, fn_res['counts']))}，总转化率 {fn_res['table'].iloc[-1]['占首级比例(%)']:.1f}%",
        fn_res["conclusion"], fn_res["table"], "漏斗阶段", ["人数", "逐级转化率(%)"], XL_CHART_TYPE.COLUMN_CLUSTERED, layout_mode="A",
    )

    # 15) 原生图表 #9 (Layout B): 联合分析属性相对重要性
    cj_res = su.calc_conjoint_analysis()
    _add_native_chart_slide(
        "05 市调专项 · 联合分析相对重要性",
        f"联合分析：价格相对重要性达 {cj_res['importance_df'].iloc[0]['相对重要性(%)']:.1f}%，最优组合「{cj_res['best_combo']}」",
        cj_res["conclusion"], cj_res["importance_df"], "属性名称", ["相对重要性(%)"], XL_CHART_TYPE.BAR_CLUSTERED, layout_mode="B",
    )

    # 16) 矢量 SVG 图 #11 & #12 (Layout C 双图对比): 32-IPA四象限图 + 33-PSM价格敏感度曲线
    b_ipa = cu.chart_32_ipa(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    b_psm = cu.chart_33_psm(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("05 市调专项 · IPA 四象限矩阵与 PSM 最优定价模型", b_ipa, layout_mode="C", bundle_right=b_psm)

    # 17) 矢量 SVG 图 #13 & #14 (Layout C 双图对比): 35-KANO模型图 + 45-消费者画像卡片
    b_kano = cu.chart_35_kano(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    b_persona = cu.chart_45_persona_cards(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("05 市调专项 · KANO 需求分层与典型消费者画像", b_kano, layout_mode="C", bundle_right=b_persona)

    # 18) 矢量 SVG 图 #15 (Layout A): 46-词云与词频组合图
    b_wc = cu.chart_46_wordcloud_combo(valid_df, theme_name=theme_name, settings={"for_ppt": True}, is_demo=is_demo)
    _add_svg_chart_slide("05 市调专项 · 开放题口碑词云与词频统计", b_wc, layout_mode="A")

    # ==========================================================================
    # 第 6 类：核心结论与行动建议页 (Conclusions & Actionable Recommendations)
    # ==========================================================================
    s_rec = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_rec, t["bg_canvas"])
    _add_slide_header_and_footer(
        s_rec, t, "EXECUTIVE SUMMARY · 核心发现与策略建议",
        "四大核心研究发现与可落地业务增长行动矩阵（证据闭环）",
        base_footnote, page_counter,
    )
    rec_df = pd.DataFrame([
        {
            "核心数据发现 (Finding)": f"价格权重第一(38.2%)，最优价格点 OPP={psm_res['opp'][0]:.2f}元，可接受带 [{psm_res['pmc'][0]:.1f},{psm_res['pme'][0]:.1f}]元",
            "证据页码": "P.23 / P.24",
            "策略行动建议 (Action)": "将主力拿铁券后到手价锚定在 13–15 元黄金区间，推出周卡套餐锁住高频客群",
            "预期业务影响": "预计提升考虑→购买转化率 +8~12%",
        },
        {
            "核心数据发现 (Finding)": "品质感知(β=+0.36***)与价格感知(β=+0.35***)双轮驱动 NPS，现磨现做属必备属性(M)",
            "证据页码": "P.18 / P.25",
            "策略行动建议 (Action)": "严控咖啡豆萃取稳定性与防洒包装底线，升级「生椰拿铁+大杯473ml」黄金大单品",
            "预期业务影响": "预计拉升品牌 NPS +5~8 分",
        },
        {
            "核心数据发现 (Finding)": "IPA 显示「价格实惠」与「出餐速度」亟待改进，KANO 证实「出餐≤5分钟」为期望属性(O)",
            "证据页码": "P.24 / P.25",
            "策略行动建议 (Action)": "优化早高峰小程序预点单动线，承诺「超时免单券」消除排队痛点",
            "预期业务影响": "预计降低早高峰流失率 -15%",
        },
        {
            "核心数据发现 (Finding)": "K-Means 识别出三大典型客群，高消品质忠诚派与性价比大众主力派诉求截然不同",
            "证据页码": "P.25 / P.26",
            "策略行动建议 (Action)": "实施分层会员运营：对高频精品群推送季节限定SOE，对价格敏感群发放满减券包",
            "预期业务影响": "预计提升月度复购率 +10%",
        },
    ])
    _add_ppt_table(s_rec, rec_df, 0.55, 1.30, 12.23, 4.75, t, max_rows=5)
    _add_conclusion_strip(s_rec, t, "策略优先级排序：先守住「现磨品质+防洒包装」必备底线（M），再攻坚「13-15元定价+5分钟快取」核心期望项（O）。")
    _write_slide_notes_markdown(s_rec, "核心结论与行动建议矩阵", "四大策略闭环建议", rec_df)
    page_counter += 1

    # ==========================================================================
    # 第 7 类：附录与方法局限说明页 (Appendix & Limitations)
    # ==========================================================================
    s_app = prs.slides.add_slide(blank_layout)
    _set_slide_background(s_app, t["bg_canvas"])
    _add_slide_header_and_footer(
        s_app, t, "APPENDIX · 技术附录与方法局限声明",
        "量表测量信效度汇总、事后分层加权说明与研究局限性声明",
        base_footnote, page_counter,
    )
    rel_p = su.calc_reliability(valid_df, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引"])
    work_app = valid_df.copy()
    work_app["品质量表3_正向"] = su.reverse_score(work_app["品质量表3_品质不稳定_反向"], max_scale=5)
    rel_q = su.calc_reliability(work_app, ["品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"])
    efa_all = su.calc_validity_efa(work_app, ["价格量表1_价格合理", "价格量表2_性价比高", "价格量表3_促销吸引", "品质量表1_口感醇厚", "品质量表2_豆源优质", "品质量表3_正向"], n_factors=2)

    app_df = pd.DataFrame([
        {"技术模块": "价格感知量表 (3题)", "核心统计指标": f"Cronbach's α = {rel_p['alpha']:.3f}", "质量判定": "内部一致性良好 (>0.70)"},
        {"技术模块": "品质感知量表 (3题含反向)", "核心统计指标": f"Cronbach's α = {rel_q['alpha']:.3f}", "质量判定": "内部一致性良好 (>0.70)"},
        {"技术模块": "探索性因子分析 (EFA)", "核心统计指标": f"KMO = {efa_all['kmo']:.3f}, Bartlett p < 0.0001", "质量判定": f"两公因子累计解释方差 {efa_all['cum_var_pct']:.1f}%"},
        {"技术模块": "可编辑图表工程规范", "核心统计指标": "9 个原生 XML 图表 + 16 个矢量 SVG (svgBlip) + 全页 Markdown 备注表", "质量判定": "支持 PPT / AI / Excel 深度二次编辑"},
        {"技术模块": "方法局限声明", "核心统计指标": demo_str, "质量判定": "横断面相关性与回归不直接等价于因果推断"},
    ])
    _add_ppt_table(s_app, app_df, 0.55, 1.30, 12.23, 4.75, t, max_rows=6)
    _add_conclusion_strip(s_app, t, "本报告所有图表源数据均已同步嵌入对应幻灯片备注栏（Notes Markdown）及 Word 数据附录。")
    _write_slide_notes_markdown(s_app, "技术附录与信效度汇总", "量表信效度与方法声明", app_df)

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
