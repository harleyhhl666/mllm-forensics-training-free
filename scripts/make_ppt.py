#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成中文汇报 PPT。数字从 figure_data.json（由 make_figures.py 从分析产物解析）读取。"""
import json, os
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

D = json.load(open("reports/figures/figure_data.json", encoding="utf-8"))
A, B = D["A"], D["B"]
FIG = "reports/figures"
OUT = "reports/ppt/TRAINING_FREE_FINAL_PRESENTATION.pptx"

W, H = Cm(33.87), Cm(19.05)          # 16:9
INK = RGBColor(0x22, 0x30, 0x3C)
SUB = RGBColor(0x5A, 0x66, 0x72)
BLUE = RGBColor(0x2F, 0x6F, 0xAF)
ORANGE = RGBColor(0xC4, 0x62, 0x2D)
GREEN = RGBColor(0x3E, 0x8E, 0x5A)
BG = RGBColor(0xFA, 0xFB, 0xFC)
LINE = RGBColor(0xD8, 0xDE, 0xE4)
FONT = "Microsoft YaHei"

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def tb(s, x, y, w, h, text, size=18, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space=1.0, anchor=MSO_ANCHOR.TOP):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = space
        if i:
            p.space_before = Pt(2)
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FONT
    return box


def title(s, t, sub=None):
    tb(s, Cm(1.9), Cm(1.15), Cm(30), Cm(1.5), t, size=29, bold=True)
    bar = s.shapes.add_shape(1, Cm(1.9), Cm(2.62), Cm(3.1), Cm(0.11))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    bar.shadow.inherit = False
    if sub:
        tb(s, Cm(1.9), Cm(3.0), Cm(30), Cm(1.0), sub, size=15, color=SUB)


def pic(s, name, x, y, w, max_h=None):
    """按真实宽高比放置图片；给定 max_h 时自动缩放以免压到下方元素。"""
    from PIL import Image
    p = os.path.join(FIG, name)
    iw, ih = Image.open(p).size
    h = Cm(w.cm * ih / iw)
    if max_h is not None and h > max_h:
        h = max_h
        w = Cm(h.cm * iw / ih)
    return s.shapes.add_picture(p, x, y, width=w, height=h)


def bullets(s, items, x=Cm(1.9), y=Cm(4.1), w=Cm(30), size=18, gap=Cm(1.42),
            dot=BLUE):
    for i, it in enumerate(items):
        yy = y + gap * i
        d = s.shapes.add_shape(9, x, yy + Cm(0.22), Cm(0.26), Cm(0.26))
        d.fill.solid()
        d.fill.fore_color.rgb = dot
        d.line.fill.background()
        d.shadow.inherit = False
        tb(s, x + Cm(0.72), yy, w - Cm(0.72), gap, it, size=size)


def card(s, x, y, w, h, head, val, note=None, color=BLUE):
    bx = s.shapes.add_shape(1, x, y, w, h)
    bx.fill.solid()
    bx.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bx.line.color.rgb = LINE
    bx.line.width = Pt(0.75)
    bx.shadow.inherit = False
    tb(s, x, y + Cm(0.42), w, Cm(0.8), head, size=13, color=SUB,
       align=PP_ALIGN.CENTER)
    tb(s, x, y + Cm(1.15), w, Cm(1.4), val, size=30, bold=True, color=color,
       align=PP_ALIGN.CENTER)
    if note:
        tb(s, x, y + h - Cm(1.0), w, Cm(0.8), note, size=12, color=SUB,
           align=PP_ALIGN.CENTER)


def table(s, rows, x, y, w, colw=None, size=15, head_bg=BLUE, rowh=Cm(1.02)):
    nr, nc = len(rows), len(rows[0])
    t = s.shapes.add_table(nr, nc, x, y, w, rowh * nr).table
    if colw:
        tot = sum(colw)
        for i, cw in enumerate(colw):
            t.columns[i].width = Cm(round(w.cm * cw / tot, 2))
    for r in range(nr):
        t.rows[r].height = rowh
        for c in range(nc):
            cell = t.cell(r, c)
            cell.text = ""
            cell.margin_left = cell.margin_right = Cm(0.28)
            cell.margin_top = cell.margin_bottom = Cm(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (head_bg if r == 0 else
                                        (RGBColor(0xFF, 0xFF, 0xFF) if r % 2
                                         else RGBColor(0xF2, 0xF5, 0xF8)))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
            run = p.add_run()
            txt = str(rows[r][c])
            bold = txt.startswith("*")
            run.text = txt.lstrip("*")
            run.font.size = Pt(size)
            run.font.bold = (r == 0) or bold
            run.font.name = FONT
            run.font.color.rgb = (RGBColor(0xFF, 0xFF, 0xFF) if r == 0
                                  else (ORANGE if bold else INK))
    return t


def note(s, text, color=SUB, size=14):
    tb(s, Cm(1.9), H - Cm(1.75), Cm(30), Cm(1.1), text, size=size, color=color)


def concl(s, text, y=None, color=GREEN):
    y = y or (H - Cm(2.9))
    bx = s.shapes.add_shape(1, Cm(1.9), y, Cm(30.1), Cm(1.85))
    bx.fill.solid()
    bx.fill.fore_color.rgb = RGBColor(0xED, 0xF5, 0xEF)
    bx.line.fill.background()
    bx.shadow.inherit = False
    lb = s.shapes.add_shape(1, Cm(1.9), y, Cm(0.13), Cm(1.85))
    lb.fill.solid()
    lb.fill.fore_color.rgb = color
    lb.line.fill.background()
    lb.shadow.inherit = False
    tb(s, Cm(2.5), y, Cm(29), Cm(1.85), text, size=17, bold=True, color=INK,
       anchor=MSO_ANCHOR.MIDDLE)


# ============ 1 标题 ============
s = slide()
band = s.shapes.add_shape(1, 0, Cm(5.4), W, Cm(0.16))
band.fill.solid()
band.fill.fore_color.rgb = BLUE
band.line.fill.background()
band.shadow.inherit = False
tb(s, Cm(2.6), Cm(6.2), Cm(28.6), Cm(2.4),
   "基于外部取证工具与多模态大模型的图像鉴伪探索", size=36, bold=True,
   align=PP_ALIGN.CENTER)
tb(s, Cm(2.6), Cm(9.0), Cm(28.6), Cm(1.2), "—— Training-Free 阶段实验总结",
   size=22, color=SUB, align=PP_ALIGN.CENTER)
tb(s, Cm(2.6), Cm(11.6), Cm(28.6), Cm(1.0),
   "ELA / TruFor  ＋  Qwen2.5-VL 7B / 32B", size=17, color=BLUE,
   align=PP_ALIGN.CENTER)
tb(s, Cm(2.6), Cm(15.3), Cm(28.6), Cm(0.9),
   "26 个正式实验 ｜ 约 26300 次推理 ｜ 全部提示与样本实验前冻结", size=13,
   color=SUB, align=PP_ALIGN.CENTER)

# ============ 2 问题 ============
s = slide()
title(s, "我们想解决什么问题")
bullets(s, [
    "传统取证工具能找到可疑区域，但通常不会用自然语言解释为什么可疑。",
    "多模态大模型会解释图像，但它是否真的能正确利用专业取证证据？",
    "因此先不训练模型，直接把工具输出交给大模型，看它能做到什么。",
], y=Cm(3.9), gap=Cm(1.5), size=19)
pic(s, "fig1_流程.png", Cm(2.4), Cm(9.6), Cm(29.1), max_h=Cm(7.6))

# ============ 3 路线 ============
s = slide()
title(s, "整个实验路线")
steps = [("ELA", "经典方法，不需要训练：能给一点位置线索，整图判断偏弱"),
         ("TruFor", "训练过的取证模型：先单独验证，确认工具本身足够强"),
         ("分数 ＋ 热力图", "整图分数很有用，热力图没有带来预期提升"),
         ("结构化空间证据", "用固定程序把热力图转成文字位置描述"),
         ("7B / 32B", "同样的提示词，两种规模行为完全不同"),
         ("解释正确性检查", "位置说对了，理由是否也说对了")]
y = Cm(3.9)
for i, (h, d) in enumerate(steps):
    box = s.shapes.add_shape(1, Cm(1.9), y, Cm(7.2), Cm(1.62))
    box.fill.solid()
    box.fill.fore_color.rgb = BLUE if i % 2 == 0 else RGBColor(0x4A, 0x88, 0xC4)
    box.line.fill.background()
    box.shadow.inherit = False
    tb(s, Cm(1.9), y, Cm(7.2), Cm(1.62), h, size=17, bold=True,
       color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
       anchor=MSO_ANCHOR.MIDDLE)
    tb(s, Cm(9.6), y, Cm(22), Cm(1.62), d, size=16, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(steps) - 1:
        ar = s.shapes.add_shape(1, Cm(5.4), y + Cm(1.62), Cm(0.1), Cm(0.42))
        ar.fill.solid()
        ar.fill.fore_color.rgb = LINE
        ar.line.fill.background()
        ar.shadow.inherit = False
    y += Cm(2.04)
note(s, "每一次转向都是被一个否定结果推动的，不是按计划推进。")

# ============ 4 ELA ============
s = slide()
title(s, "ELA：能定位，但不适合做最终判断")
bullets(s, [
    "原理一句话：把图像重新压缩一次，观察不同区域对压缩的反应差异，反应异常的地方可能被编辑过。",
    "在 TGIF 数据上，对局部改动有一定的定位效果。",
    "但作为「整张图是真还是假」的依据明显偏弱。",
    "换成更大的模型也没有解决，说明问题不在模型规模。",
], y=Cm(4.0), gap=Cm(2.0), size=18)
concl(s, "ELA 可以作为一个弱的空间提示，不足以作为可靠的最终判据 → 转向 TruFor")

# ============ 5 TruFor ============
s = slide()
title(s, "TruFor：先确认工具本身足够强", "200 组图像，每组一张假图与一张配对真图")
tf = D["trufor"]
vals = list(tf.values())
card(s, Cm(1.9), Cm(4.0), Cm(7.0), Cm(3.4), "整图判别 AUROC",
     f"{vals[0]:.4f}", "接近上限", BLUE)
card(s, Cm(9.55), Cm(4.0), Cm(7.0), Cm(3.4), "低误报下检出率",
     f"{vals[1]:.3f}", "误报 ≤5% 时", BLUE)
card(s, Cm(17.2), Cm(4.0), Cm(7.0), Cm(3.4), "像素级 AUROC",
     f"{vals[2]:.3f}", "逐像素定位", GREEN)
card(s, Cm(24.85), Cm(4.0), Cm(7.0), Cm(3.4), "区域重合度 IoU",
     f"{vals[3]:.3f}", "与真实区域重叠", GREEN)
pic(s, "fig2_TruFor工具性能.png", Cm(9.9), Cm(7.7), Cm(14.0), max_h=Cm(6.2))
tb(s, Cm(1.9), Cm(14.15), Cm(30.1), Cm(0.9),
   "另有像素级 F1 = 0.846。蓝色为整图判断能力，绿色为区域定位能力。", size=13,
   color=SUB)
concl(s, "工具本身足够强，因此后续重点研究大模型怎么使用它，而不是怀疑工具太差",
      y=Cm(15.5))

# ============ 6 分数有用 ============
s = slide()
title(s, "7B：分数有用，热力图未必有用")
card(s, Cm(4.0), Cm(4.3), Cm(10.6), Cm(4.2), "只给整图篡改分数",
     f"{A['A0'][2]:.3f}", "综合判别指标", BLUE)
tb(s, Cm(15.2), Cm(6.0), Cm(2.4), Cm(1.2), "→", size=34, color=SUB,
   align=PP_ALIGN.CENTER)
card(s, Cm(18.2), Cm(4.3), Cm(10.6), Cm(4.2), "再加空间热力图",
     "−0.228", "综合指标反而下降", ORANGE)
tb(s, Cm(1.9), Cm(9.1), Cm(30.1), Cm(1.2),
   "病例层面：56 张图从「只给分数时判对为假」变成「加热力图后判成真」，反方向 0 例。",
   size=17, color=SUB)
bullets(s, [
    "给出整图分数后真假判断明显变好 —— 整图分数是强信号。",
    "但在已有分数的情况下再给热力图，判断没有进一步改善，反而下降。",
], y=Cm(10.8), gap=Cm(1.45), size=18)
concl(s, "提出问题：模型到底有没有在看热力图里的位置？")

# ============ 7 平移热力图 ============
s = slide()
title(s, "平移热力图实验", "把热力图整体平移，让它和真实篡改区域错位")
tb(s, Cm(1.9), Cm(4.3), Cm(14.6), Cm(3.4),
   "做法\n热力图里的数值一个都没变\n（数值多重集逐个元素相同），\n只破坏了对齐关系。",
   size=17)
card(s, Cm(18.0), Cm(4.2), Cm(13.9), Cm(3.6), "平移热力图后，真假判断的变化",
     "−0.027", "置信区间包含 0，统计上看不出差别", ORANGE)
concl(s, "7B 没有真正检查热力图与原图的位置是否对应：它看到了第二张图，\n"
         "但没有利用其中的空间对应关系", y=Cm(9.0))
tb(s, Cm(1.9), Cm(12.6), Cm(30), Cm(2.4),
   "补充：32B 上重复同一实验，结果同样是 −0.027，置信区间同样包含 0。\n"
   "单纯增加参数规模，并不能解决「模型是否核对空间证据」的问题。", size=17,
   color=SUB)

# ============ 8 结构化 ============
s = slide()
title(s, "改成结构化空间证据", "不再让模型自己看热力图，先用固定程序整理成文字")
bx = s.shapes.add_shape(1, Cm(1.9), Cm(4.3), Cm(13.4), Cm(5.4))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xF8)
bx.line.color.rgb = LINE
bx.shadow.inherit = False
tb(s, Cm(2.6), Cm(4.8), Cm(12), Cm(4.6),
   "区域 1\n    位置：右下\n    面积占比：3.2%\n    平均可疑度：0.87\n"
   "    最高可疑度：0.99", size=17, space=1.3)
rows = [["指标", "直接给热力图", "转成文字描述"],
        ["解释与工具证据一致", f"{D['raw'][1]:.3f}", f"*{D['st'][1]:.3f}"],
        ["编造不存在的区域", "—", f"*{0.0:.3f}"]]
table(s, rows, Cm(16.6), Cm(4.3), Cm(15.3), colw=[6, 4, 4], size=15)
concl(s, "整理过程完全由程序完成，没有引入任何新模型，规则在实验前已冻结", y=Cm(9.9))
tb(s, Cm(1.9), Cm(13.4), Cm(30), Cm(1.6),
   "结果：模型终于能够稳定理解「工具认为哪里有问题」，并且从不编造证据里不存在的区域。",
   size=18)

# ============ 9 崩了 ============
s = slide()
title(s, "但 7B 的真假判断突然崩了")
card(s, Cm(3.4), Cm(4.8), Cm(10.6), Cm(4.6), "只给整图分数",
     f"{A['A0'][2]:.3f}", "综合判别指标", BLUE)
tb(s, Cm(14.6), Cm(6.5), Cm(4.6), Cm(1.4), "→", size=40, color=ORANGE,
   align=PP_ALIGN.CENTER)
card(s, Cm(19.8), Cm(4.8), Cm(10.6), Cm(4.6), "加结构化证据",
     f"{A['A2'][2]:.3f}", "综合判别指标", ORANGE)
bx = s.shapes.add_shape(1, Cm(1.9), Cm(11.4), Cm(30.1), Cm(2.5))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xFA, 0xF0, 0xE9)
bx.line.fill.background()
bx.shadow.inherit = False
tb(s, Cm(1.9), Cm(11.4), Cm(30.1), Cm(2.5),
   "为什么信息更清楚了，判断反而更差？", size=25, bold=True, color=ORANGE,
   align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ============ 10 提示方式 ============
s = slide()
title(s, "真正的问题是提示方式", "我们先后排除了几种解释")
rows = [["曾经的猜测", "结论", "依据"],
        ["是「面积」字段让模型觉得改动太小", "否", "各字段伤害相近；只给位置再加面积，表现反而变好"],
        ["是字段太多、结构太复杂", "基本否", "同样提示词但载荷完全为空时，判断就已经崩了"],
        ["需要把「是否篡改」和「范围多大」拆开", "否", "拆开后误报率约 0.95，变成过度检出"]]
table(s, rows, Cm(1.9), Cm(4.2), Cm(30.1), colw=[10, 3, 13], size=14,
      rowh=Cm(1.24))
tb(s, Cm(1.9), Cm(9.6), Cm(30.1), Cm(1.8),
   "定位到的原因：提示里强调「局部区域的取证信息」之后，\n"
   "7B 容易得出「虽然局部有修改，但整张图仍然算真实」。", size=17)
bx = s.shapes.add_shape(1, Cm(1.9), Cm(11.6), Cm(19.4), Cm(2.3))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xEC, 0xF2, 0xF9)
bx.line.fill.background()
bx.shadow.inherit = False
tb(s, Cm(2.5), Cm(11.6), Cm(18.4), Cm(2.3),
   "加一句规则：只要确实存在真实篡改，\n无论多小，都判为假。", size=17, bold=True,
   anchor=MSO_ANCHOR.MIDDLE)
card(s, Cm(22.4), Cm(11.4), Cm(9.5), Cm(2.7), "综合判别指标",
     f"{A['A2'][2]:.3f} → {A['A3'][2]:.3f}", None, GREEN)

# ============ 11 四格 ============
s = slide()
title(s, "严格对照后发现：不是空间证据让它更准")
rows = [["7B", "无结构化证据", "有结构化证据"],
        ["无明确规则", f"A0　{A['A0'][2]:.3f}", f"A2　{A['A2'][2]:.3f}"],
        ["有明确规则", f"*A1　{A['A1'][2]:.3f}", f"*A3　{A['A3'][2]:.3f}"]]
table(s, rows, Cm(1.9), Cm(4.2), Cm(14.2), colw=[5, 5, 5], size=16,
      rowh=Cm(1.35))
pic(s, "fig3_7B四格.png", Cm(17.2), Cm(4.1), Cm(14.6), max_h=Cm(8.4))
bx = s.shapes.add_shape(1, Cm(1.9), Cm(9.0), Cm(14.2), Cm(2.6))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xFA, 0xF0, 0xE9)
bx.line.fill.background()
bx.shadow.inherit = False
tb(s, Cm(2.4), Cm(9.0), Cm(13.4), Cm(2.6),
   f"A1 与 A3 基本一样\n差异 +0.005，置信区间包含 0", size=17, bold=True,
   color=ORANGE, anchor=MSO_ANCHOR.MIDDLE)
concl(s, "真正带来判断恢复的是把任务规则讲清楚，不是结构化空间证据")

# ============ 12 32B ============
s = slide()
title(s, "32B 的行为完全不同", "提示词与 7B 逐字节相同，样本一致，唯一变量是模型规模")
rows = [["32B 条件", "综合判别指标", "真图误报率"],
        ["仅分数、无规则", f"*{B['B0'][2]:.3f}", f"{B['B0'][1]:.3f}"],
        ["仅分数、加规则", f"{B['B1'][2]:.3f}", f"{B['B1'][1]:.3f}"],
        ["加证据、无规则", f"{B['B2'][2]:.3f}", f"{B['B2'][1]:.3f}"],
        ["加证据、加规则", f"*{B['B3'][2]:.3f}", f"*{B['B3'][1]:.3f}"]]
table(s, rows, Cm(1.9), Cm(4.2), Cm(14.2), colw=[6, 4.6, 4.6], size=15,
      rowh=Cm(1.15))
pic(s, "fig4_32B四格.png", Cm(17.2), Cm(4.1), Cm(14.6), max_h=Cm(7.0))
bullets(s, [
    "32B 不会出现 7B 那种崩塌：加证据后召回率差异精确为 0.000。",
    "再加规则反而容易误判真图：误报率升到 0.337，三分之一真图被判假。",
], y=Cm(11.4), gap=Cm(1.35), size=17)
concl(s, "32B 最好的配置是「只给整图分数」；因此 7B 上的这两条机制与模型规模有关，\n"
         "不能说成这一类模型的普遍规律", color=ORANGE)

# ============ 13 空间证据的价值 ============
s = slide()
title(s, "那么空间证据真正带来的是什么")
bx = s.shapes.add_shape(1, Cm(1.9), Cm(3.9), Cm(30.1), Cm(1.9))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xED, 0xF5, 0xEF)
bx.line.fill.background()
bx.shadow.inherit = False
tb(s, Cm(1.9), Cm(3.9), Cm(30.1), Cm(1.9),
   "不是更高的真假判断准确率，而是更可靠的位置解释", size=22, bold=True,
   color=GREEN, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rows = [["指标", "直接给热力图", "转成文字位置描述"],
        ["解释与工具证据一致", f"{D['raw'][1]:.3f}", f"*{D['st'][1]:.3f}"],
        ["解释命中真实篡改区域", f"{D['raw'][2]:.3f}", f"*{D['st'][2]:.3f}"],
        ["解释的位置准确度", f"{D['raw'][3]:.3f}", f"*{D['st'][3]:.3f}"]]
table(s, rows, Cm(1.9), Cm(6.3), Cm(15.0), colw=[7, 4, 4.6], size=14,
      rowh=Cm(1.16))
pic(s, "fig6_热力图对结构化.png", Cm(17.6), Cm(6.3), Cm(14.3), max_h=Cm(7.2))
note(s, "提醒：命中率容易受标注区域横跨多个网格影响（184 张里 54 张跨 4 格以上），"
        "要连位置准确度和覆盖完整度一起看 —— 工具覆盖完整度只有 0.531，模型解释也继承了这个缺口。")

# ============ 14 镜像 ============
s = slide()
title(s, "模型很听工具的话，但不会自己纠错",
      "把结构化证据左右镜像：位置描述变错，强度数值和原图都不动")
rows = [["", "正确证据", "镜像之后"],
        ["工具说", "右下", "*左下（已改错）"],
        ["模型解释说", "右下", "*左下"],
        ["真实篡改区域", "右下", "右下"]]
table(s, rows, Cm(1.9), Cm(4.6), Cm(12.6), colw=[5, 4, 5], size=15,
      rowh=Cm(1.2))
tb(s, Cm(1.9), Cm(9.6), Cm(12.6), Cm(3.0),
   f"镜像之后：\n跟随输入证据 {D['mirror']['7B']['mir'][0]:.3f}（不变）\n"
   f"命中真实区域 7B {D['mirror']['7B']['mir'][1]:.3f} ／ "
   f"32B {D['mirror']['32B']['mir'][1]:.3f}", size=16, color=SUB)
pic(s, "fig7_镜像证据.png", Cm(15.4), Cm(4.4), Cm(16.5), max_h=Cm(8.2))
concl(s, "工具错在哪里，模型通常也跟着错到哪里 —— 结构化证据带来的「解释准确」\n"
         "是借来的准确，来自工具正确，不是模型自己有核对能力")

# ============ 15 语义解释 ============
s = slide()
title(s, "位置说对，不代表理由说对", "对假图解释的逐条审查，各 100 条，同一批图像")
m7, m32 = D["x3"]["m7"], D["x3"]["m32"]
rows = [["对假图的解释", "7B", "32B"],
        ["完全有图像依据", f"{m7[0]*100:.0f}%", f"*{m32[0]*100:.0f}%"],
        ["部分有依据", f"{m7[1]*100:.0f}%", f"{m32[1]*100:.0f}%"],
        ["没有依据", f"*{m7[2]*100:.0f}%", f"{m32[2]*100:.0f}%"],
        ["「被篡改」说法没有依据", "*79.1%", "4.4%"]]
table(s, rows, Cm(1.9), Cm(4.3), Cm(14.2), colw=[7, 3.6, 3.6], size=14,
      rowh=Cm(1.12))
pic(s, "fig8_解释支持度.png", Cm(16.9), Cm(4.7), Cm(15.0), max_h=Cm(7.9))
bx = s.shapes.add_shape(1, Cm(1.9), Cm(10.2), Cm(14.2), Cm(2.7))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xFA, 0xF0, 0xE9)
bx.line.fill.background()
bx.shadow.inherit = False
tb(s, Cm(2.4), Cm(10.2), Cm(13.4), Cm(2.7),
   "7B 在这 100 张图上位置全部说对，\n却有 61% 的解释没有依据 ——\n"
   "地方找对了，原因说错了。", size=16, bold=True, anchor=MSO_ANCHOR.MIDDLE)
note(s, "两个模型虚构物体的情况都很少（3% 与 1%），问题不是「看见不存在的东西」。"
        "语义解释结果来自盲化辅助标注，用于内部分析，不是独立人工真值，仍需后续更严格验证。")

# ============ 16 结论 ============
s = slide()
title(s, "这阶段最终得到什么")
items = [
    ("1", "TruFor 的整图篡改分数是当前最可靠的真假判断信号。", BLUE),
    ("2", "直接加入空间证据并不会稳定提高真假判断。\n"
          "7B 上差异置信区间包含 0，32B 上甚至变差。", ORANGE),
    ("3", "把空间证据结构化后，可以明显提高模型的位置解释能力。\n"
          "与证据一致率 1.000，虚报率 0.000。", GREEN),
    ("4", "模型能忠实跟随工具，但未必会独立验证工具，\n"
          "也未必能稳定解释「为什么这里是假」。", ORANGE),
]
y = Cm(4.0)
for num, txt, col in items:
    n = s.shapes.add_shape(9, Cm(1.9), y + Cm(0.28), Cm(1.15), Cm(1.15))
    n.fill.solid()
    n.fill.fore_color.rgb = col
    n.line.fill.background()
    n.shadow.inherit = False
    tb(s, Cm(1.9), y + Cm(0.28), Cm(1.15), Cm(1.15), num, size=17, bold=True,
       color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
       anchor=MSO_ANCHOR.MIDDLE)
    tb(s, Cm(3.6), y, Cm(28.2), Cm(2.6), txt, size=18, anchor=MSO_ANCHOR.MIDDLE)
    y += Cm(2.85)
note(s, "限制：仅在 TGIF sd2-sp 一个数据集、一类篡改上验证；报告阈值是评测工作点，"
        "不是可直接部署的阈值。")

# ============ 17 下一阶段 ============
s = slide()
title(s, "下一阶段：从「告诉它哪里可疑」到「让它学会为什么可疑」")
tb(s, Cm(1.9), Cm(4.0), Cm(30), Cm(1.4),
   "Training-Free 阶段已经回答了「工具 ＋ 大模型直接推理能做到什么」。", size=19)
bullets(s, [
    "构造区域级的解释数据；",
    "微调 / 后训练；",
    "教模型识别纹理、边缘、光照、几何等具体异常；",
    "验证解释是否真正正确，而不只是变流畅。",
], y=Cm(5.9), gap=Cm(1.4), size=18)
bx = s.shapes.add_shape(1, Cm(1.9), Cm(11.9), Cm(30.1), Cm(2.6))
bx.fill.solid()
bx.fill.fore_color.rgb = RGBColor(0xEC, 0xF2, 0xF9)
bx.line.fill.background()
bx.shadow.inherit = False
tb(s, Cm(2.5), Cm(11.9), Cm(29), Cm(2.6),
   "两个现成条件：真假图只在篡改区域内有差别，天然提供对照；\n"
   "本阶段建立的解释审查标准可以直接用于衡量训练效果。", size=17,
   anchor=MSO_ANCHOR.MIDDLE)
note(s, "本阶段代码与结果已冻结，标签 training-free-v1。")

os.makedirs("reports/ppt", exist_ok=True)
prs.save(OUT)
print(f"已生成 {OUT}，共 {len(prs.slides.__iter__.__self__._sldIdLst)} 页")
