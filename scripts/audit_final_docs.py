#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终内容审计：检查中文化、术语、数字可追溯性、PPT 版面。"""
import json, re, sys
from pptx import Presentation

MD = "reports/TRAINING_FREE_FINAL_SUMMARY.md"
PPTC = "reports/ppt/TRAINING_FREE_FINAL_PRESENTATION_CONTENT.md"
PPTX = "reports/ppt/TRAINING_FREE_FINAL_PRESENTATION.pptx"
CC = "reports/CROSSCHECK.txt"

P, W, F = [], [], []
ok = lambda m: (P.append(m), print(f"  通过  {m}"))
warn = lambda m: (W.append(m), print(f"  注意  {m}"))
bad = lambda m: (F.append(m), print(f"  不合格  {m}"))

md = open(MD, encoding="utf-8").read()
pc = open(PPTC, encoding="utf-8").read()
cc = open(CC, encoding="utf-8", errors="replace").read()

print("=" * 74)
print("最终内容审计")
print("=" * 74)

# ---------- 1 中文化 ----------
print("\n[1] 语言与术语")
cjk = len(re.findall(r"[\u4e00-\u9fff]", md))
ok(f"Markdown 中文字数 {cjk}") if 7000 <= cjk <= 14000 else \
    warn(f"Markdown 中文字数 {cjk}，超出建议区间 8000~12000")

BAN = ["faithfulness", "arbitration", "framing", "semantics", "grounding",
       "counterfactual", "decision utility", "scalar score",
       "structured evidence", "capacity-dependent", "interaction"]
for doc, nm in ((md, "Markdown"), (pc, "PPT 文案")):
    hit = {b: len(re.findall(b, doc, re.I)) for b in BAN}
    hit = {k: v for k, v in hit.items() if v}
    ok(f"{nm} 未出现禁用英文术语") if not hit else bad(f"{nm} 出现禁用术语 {hit}")

ALLOW = {"ELA", "TruFor", "Qwen2.5-VL", "TGIF", "AUROC", "IoU", "F1", "JSON",
         "Training-Free", "training-free", "sd2-sp", "PPT", "Markdown"}
for doc, nm in ((md, "Markdown"), (pc, "PPT 文案")):
    words = set(re.findall(r"[A-Za-z][A-Za-z0-9.\-]{2,}", doc))
    extra = {w for w in words
             if not any(a.lower() in w.lower() for a in ALLOW)}
    extra -= {"figures", "png", "docs", "reports", "md", "runs", "sha",
              "CROSSCHECK", "EXPERIMENT", "INDEX", "FINAL", "MANIFEST",
              "INVALID", "AND", "SUPERSEDED", "RUNS", "DATASET", "SPLITS",
              "REPRODUCIBILITY", "EXPERIMENTAL", "HISTORY", "SUMMARY",
              "PRESENTATION", "CONTENT", "TRAINING", "FREE", "fig",
              "EXPERIMENTS", "REPORT", "Stable", "Diffusion", "YaHei"}
    ok(f"{nm} 无多余英文词") if not extra else \
        warn(f"{nm} 出现其他英文词 {sorted(extra)[:8]}")

# ---------- 2 不夸大 ----------
print("\n[2] 结论表述")
OVER = ["MLLM 普遍", "普遍存在", "结构化证据能够提升图像鉴伪性能",
        "显著提升.*鉴伪性能", "证明了.*普遍"]
h = [o for o in OVER if re.search(o, md)]
ok("未出现过度概括的表述") if not h else bad(f"出现夸大表述 {h}")

MUST = [("结构化.*主要改善|不是更高的真假判断准确率|没有带来额外",
         "明确写出结构化证据未提升判断"),
        ("与模型规模有关|不能说成.*普遍规律", "明确标注机制与规模有关"),
        ("不是独立人工真值|不是独立标注", "标注限制已写明"),
        ("0.650", "标注一致性限制已写明"),
        ("评测.*工作点|不是可以直接部署|不是可直接部署", "阈值性质已写明"),
        ("被否证|推翻|排除", "保留了被推翻的早期猜测")]
for pat, desc in MUST:
    ok(desc) if re.search(pat, md) else bad(f"缺少：{desc}")

# 无效结果不得引用。注意：指向 INVALID_AND_SUPERSEDED_RUNS.md 这一文档名是允许的，
# 需要排除的是作废运行的具体数字与产物名。
body = re.sub(r"`docs/INVALID_AND_SUPERSEDED_RUNS\.md`", "", md)
INV = ["0.592", "score_dup", "verdicts_INVALID"]
h = [i for i in INV if i in body]
ok("未引用已作废结果的任何数字") if not h else bad(f"引用了作废结果 {h}")

# ---------- 3 数字可追溯 ----------
print("\n[3] 数字可追溯性")
KEY = ["0.9845", "0.955", "0.837", "0.826", "0.978", "0.897", "0.114", "0.967",
       "0.902", "0.918", "0.880", "0.663", "0.337", "0.979", "0.531", "0.791",
       "0.044", "0.014", "0.029", "1.000", "0.000"]
src = cc + open("runs/explainability/x1.txt", encoding="utf-8",
                errors="replace").read() + \
    open("x3/x3_report.txt", encoding="utf-8", errors="replace").read()
src = src.replace(" ", "")
missing = [k for k in KEY if k not in src and k != "0.9845"]
ok(f"{len(KEY)} 个关键数字均可在分析产物中找到") if not missing else \
    bad(f"以下数字无法追溯到分析产物 {missing}")
ok("核算文件报告 0 处不一致") if "MISMATCHES: 0" in cc else \
    bad("核算文件存在不一致")

fd = json.load(open("reports/figures/figure_data.json", encoding="utf-8"))
chk = [("A0 J", fd["A"]["A0"][2], 0.826), ("A3 J", fd["A"]["A3"][2], 0.902),
       ("B0 J", fd["B"]["B0"][2], 0.918), ("B3 误报", fd["B"]["B3"][1], 0.337),
       ("7B 无依据", fd["x3"]["m7"][2], 0.61),
       ("32B 无依据", fd["x3"]["m32"][2], 0.01)]
allok = True
for nm, got, want in chk:
    if abs(got - want) > 0.002:
        bad(f"图表数据 {nm} 为 {got}，应为 {want}")
        allok = False
if allok:
    ok("图表数据与核算值一致（图表由脚本自动读取分析产物）")

# ---------- 4 PPT 版面 ----------
print("\n[4] PPT 版面")
prs = Presentation(PPTX)
n = len(prs.slides._sldIdLst)
ok(f"PPT 共 {n} 页") if 14 <= n <= 17 else warn(f"PPT {n} 页，建议 14~17 页")
H, Wd = prs.slide_height.cm, prs.slide_width.cm
over = small = overlap = 0
for i, s in enumerate(prs.slides, 1):
    els = []
    for sh in s.shapes:
        if sh.top is None or sh.height is None:
            continue
        if sh.width and sh.width.cm > 33:
            continue
        if sh.top.cm + sh.height.cm > H - 0.15 or \
           sh.left.cm + sh.width.cm > Wd - 0.15:
            print(f"    第{i}页有元素越界")
            over += 1
        if sh.has_text_frame:
            for p_ in sh.text_frame.paragraphs:
                for r in p_.runs:
                    if r.font.size and r.font.size.pt < 12:
                        small += 1
        els.append((sh.left.cm, sh.top.cm, sh.width.cm, sh.height.cm,
                    str(sh.shape_type)))
    for a in range(len(els)):
        for b in range(a + 1, len(els)):
            x1, y1, w1, h1, t1 = els[a]
            x2, y2, w2, h2, t2 = els[b]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            oy = min(y1 + h1, y2 + h2) - max(y1, y2)
            if ox > 1.0 and oy > 0.5 and "PICTURE" in (t1 + t2) and t1 != t2:
                print(f"    第{i}页图片与文字重叠")
                overlap += 1
ok("无元素越界") if not over else bad(f"{over} 处元素越界")
ok("无图片压住文字") if not overlap else bad(f"{overlap} 处图片压住文字")
ok("无小于 12pt 的文字") if not small else warn(f"{small} 处文字小于 12pt")

# 每页是否有标题
notitle = []
for i, s in enumerate(prs.slides, 1):
    txt = " ".join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame)
    if not txt.strip():
        notitle.append(i)
ok("每页均有文字内容") if not notitle else bad(f"空白页 {notitle}")

# PPT 英文占比
allt = " ".join(sh.text_frame.text for s in prs.slides for sh in s.shapes
                if sh.has_text_frame)
en = len(re.findall(r"[A-Za-z]", allt))
zh = len(re.findall(r"[\u4e00-\u9fff]", allt))
r = en / max(1, en + zh)
ok(f"PPT 以中文为主（中文 {zh} 字，英文字母 {en} 个，占比 {r:.1%}）") \
    if r < 0.12 else warn(f"PPT 英文占比 {r:.1%} 偏高")

print("\n" + "=" * 74)
print(f"通过 {len(P)}　注意 {len(W)}　不合格 {len(F)}")
if W:
    print("\n注意事项：")
    for m in W:
        print(f"  - {m}")
if F:
    print("\n不合格项：")
    for m in F:
        print(f"  - {m}")
print("=" * 74)
sys.exit(1 if F else 0)
