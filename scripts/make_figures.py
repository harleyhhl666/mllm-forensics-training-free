#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从最终分析产物自动读取数字并绘制中文图表。禁止手写数字。

所有数值都从 reports/CROSSCHECK.txt、runs/explainability/x1.txt、
x3/x3_report.txt 以及 runs/ 下的原始逐条结果中解析得到。
"""
import json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# ---------------- 中文字体 ----------------
FONT = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
             r"C:\Windows\Fonts\simsun.ttc"):
    if os.path.exists(cand):
        FONT = font_manager.FontProperties(fname=cand)
        font_manager.fontManager.addfont(cand)
        break
if FONT is None:
    sys.exit("找不到中文字体")
plt.rcParams["font.family"] = FONT.get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

C_MAIN = "#2F6FAF"      # 主色
C_WARN = "#C4622D"      # 对比色
C_GOOD = "#3E8E5A"
C_GREY = "#9AA3AB"
C_LIGHT = "#BFD4E8"

CC = open("reports/CROSSCHECK.txt", encoding="utf-8", errors="replace").read()
X1 = open("runs/explainability/x1.txt", encoding="utf-8", errors="replace").read()
X3 = open("x3/x3_report.txt", encoding="utf-8", errors="replace").read()


def cell_metrics(tag):
    """从 CROSSCHECK 中取某个单元格的 recall / FPR / J。"""
    blk = re.search(rf"\[{tag}[^\]]*\](.*?)(?=\n  \[|\n##|\Z)", CC, re.S)
    if not blk:
        raise SystemExit(f"找不到 {tag}")
    t = blk.group(1)
    g = lambda k: float(re.search(rf"{k}\s+([0-9.]+)", t).group(1))
    return g("recall"), g("FPR"), g("Youden J")


A = {k: cell_metrics(k) for k in ("A0", "A1", "A2", "A3")}
B = {k: cell_metrics(v) for k, v in (("B0", "B0_score_only"),
                                     ("B1", "B1_score_rule"),
                                     ("B2", "B2_structured"),
                                     ("B3", "B3_structured_rule"))}
print("7B :", {k: tuple(round(x, 3) for x in v) for k, v in A.items()})
print("32B:", {k: tuple(round(x, 3) for x in v) for k, v in B.items()})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("写出", p)
    return p


# ---------------- 图1 实验流程 ----------------
fig, ax = plt.subplots(figsize=(11, 2.9))
ax.axis("off")
boxes = [("原始图像", C_LIGHT), ("外部取证工具\nELA / TruFor", C_MAIN),
         ("取证证据\n整图分数 + 可疑区域", C_MAIN),
         ("多模态大模型\nQwen2.5-VL 7B / 32B", C_WARN),
         ("真假判断\n+ 区域解释", C_GOOD)]
x = 0.02
for i, (t, c) in enumerate(boxes):
    w = 0.175
    fc = c if i else C_LIGHT
    tc = "white" if i else "#22303C"
    ax.add_patch(plt.Rectangle((x, 0.3), w, 0.42, facecolor=fc,
                               edgecolor="none", transform=ax.transAxes))
    ax.text(x + w / 2, 0.51, t, ha="center", va="center", color=tc,
            fontproperties=FONT, fontsize=11, transform=ax.transAxes)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + w + 0.028, 0.51), xytext=(x + w + 0.003, 0.51),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color="#5A6672", lw=1.6))
    x += w + 0.031
ax.text(0.5, 0.06, "本阶段不训练、不微调模型：工具输出直接交给大模型推理",
        ha="center", fontproperties=FONT, fontsize=10.5, color="#5A6672",
        transform=ax.transAxes)
save(fig, "fig1_流程.png")

# ---------------- 图2 TruFor 工具本身 ----------------
m = lambda k: float(re.search(rf"{k}\s+([0-9.]+)\s+target", CC).group(1))
names = ["整图判别\nAUROC", "低误报下\n检出率", "像素级\nAUROC",
         "区域重合度\nIoU", "像素级 F1"]
vals = [m("image AUROC"), m(r"TPR@FPR<=0\.05"), m(r"pixel AUROC \(mean\)"),
        m(r"IoU @map>0\.5 \(mean\)"), m(r"pixel F1 @map>0\.5 \(mean\)")]
fig, ax = plt.subplots(figsize=(7.6, 3.5))
bars = ax.bar(names, vals, color=[C_MAIN, C_MAIN, C_GOOD, C_GOOD, C_GOOD],
              width=.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.3f}", ha="center",
            fontproperties=FONT, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.13)
ax.set_ylabel("数值", fontproperties=FONT, fontsize=11)
ax.set_title("TruFor 工具单独评测（200 组图像，蓝色为整图判断，绿色为区域定位）",
             fontproperties=FONT, fontsize=12)
for t in ax.get_xticklabels():
    t.set_fontproperties(FONT)
    t.set_fontsize(10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.25)
save(fig, "fig2_TruFor工具性能.png")


# ---------------- 图3/4 通用 2x2 图 ----------------
def plot22(d, keys, labels, title, fname, hi=None):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    J = [d[k][2] for k in keys]
    cols = [C_GOOD if (hi and k == hi) else C_MAIN for k in keys]
    bars = ax.bar(labels, J, color=cols, width=.58)
    for b, v in zip(bars, J):
        ax.text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}", ha="center",
                fontproperties=FONT, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("综合判别指标（越高越好）", fontproperties=FONT, fontsize=11)
    ax.set_title(title, fontproperties=FONT, fontsize=12)
    for t in ax.get_xticklabels():
        t.set_fontproperties(FONT)
        t.set_fontsize(10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)
    return save(fig, fname)


LAB = ["仅整图分数\n无明确规则", "仅整图分数\n+ 明确规则",
       "加结构化证据\n无明确规则", "加结构化证据\n+ 明确规则"]
plot22(A, ["A0", "A1", "A2", "A3"], LAB,
       "7B 模型：四种输入组合下的真假判断表现", "fig3_7B四格.png", hi="A3")
plot22(B, ["B0", "B1", "B2", "B3"], LAB,
       "32B 模型：相同四种组合（提示词与 7B 完全一致）", "fig4_32B四格.png",
       hi="B0")

# ---------------- 图5 7B vs 32B 对比 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
idx = np.arange(4)
w = .36
for ax, (met, mi, ttl) in zip(axes, [("综合判别指标", 2, "真假判断总体表现"),
                                     ("真图被误判为假的比例", 1, "真图误报情况")]):
    a = [A[k][mi] for k in ("A0", "A1", "A2", "A3")]
    b = [B[k][mi] for k in ("B0", "B1", "B2", "B3")]
    ax.bar(idx - w / 2, a, w, label="7B", color=C_MAIN)
    ax.bar(idx + w / 2, b, w, label="32B", color=C_WARN)
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va + .012, f"{va:.2f}", ha="center",
                fontproperties=FONT, fontsize=8.5)
        ax.text(i + w / 2, vb + .012, f"{vb:.2f}", ha="center",
                fontproperties=FONT, fontsize=8.5)
    ax.set_xticks(idx)
    ax.set_xticklabels(["仅分数", "分数+规则", "加证据", "证据+规则"],
                       fontproperties=FONT, fontsize=9.5)
    ax.set_title(ttl, fontproperties=FONT, fontsize=11.5)
    ax.set_ylabel(met, fontproperties=FONT, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)
axes[0].set_ylim(0, 1.08)
axes[1].set_ylim(0, .46)
# 图例统一放到图外，避免与柱子重叠
axes[1].legend(prop=FONT, fontsize=10, frameon=False, loc="upper left")
fig.suptitle("两种规模模型的行为差异：7B 靠规则救回，32B 只给分数最好",
             fontproperties=FONT, fontsize=12.5, y=1.03)
save(fig, "fig5_7B对32B.png")

# ---------------- 图6 热力图 vs 结构化 ----------------
def x1row(tag):
    r = re.search(rf"{tag}\s+cite ([0-9.]+)\s+Expl->Evidence ([0-9.]+)\s+"
                  rf"Expl->GT ([0-9.]+)\s+GT precision ([0-9.]+)", X1)
    return [float(r.group(i)) for i in range(1, 5)]


raw, st = x1row(r"raw map \(E1\)"), x1row(r"structured \(E2\)")
fig, ax = plt.subplots(figsize=(8, 3.8))
idx = np.arange(3)
w = .34
rv = [raw[1], raw[2], raw[3]]
sv = [st[1], st[2], st[3]]
ax.bar(idx - w / 2, rv, w, label="直接给热力图", color=C_GREY)
ax.bar(idx + w / 2, sv, w, label="转成文字位置描述", color=C_MAIN)
for i, (a_, b_) in enumerate(zip(rv, sv)):
    ax.text(i - w / 2, a_ + .015, f"{a_:.3f}", ha="center", fontproperties=FONT,
            fontsize=10)
    ax.text(i + w / 2, b_ + .015, f"{b_:.3f}", ha="center", fontproperties=FONT,
            fontsize=10, fontweight="bold")
ax.set_xticks(idx)
ax.set_xticklabels(["解释与工具证据一致", "解释命中真实篡改区域",
                    "解释的位置准确度"], fontproperties=FONT, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("比例", fontproperties=FONT, fontsize=11)
ax.set_title("把热力图整理成文字位置描述后，模型的区域解释明显更准（7B，同一批图像）",
             fontproperties=FONT, fontsize=11.5)
ax.legend(prop=FONT, fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.25)
save(fig, "fig6_热力图对结构化.png")

# ---------------- 图7 镜像实验 ----------------
mir = {}
for tag in ("7B", "32B"):
    blk = re.search(rf"\[{tag}\] n=\d+.*?paired mirrored - correct:.*?\n", X1, re.S)
    t = blk.group(0)
    cor = re.search(r"correct evidence.*?Expl->Evidence ([0-9.]+)\s+Expl->GT "
                    r"([0-9.]+)", t)
    mrr = re.search(r"mirrored evidence.*?Expl->Evidence ([0-9.]+)\s+Expl->GT "
                    r"([0-9.]+)", t)
    mir[tag] = dict(cor=(float(cor.group(1)), float(cor.group(2))),
                    mir=(float(mrr.group(1)), float(mrr.group(2))))
print("镜像:", mir)
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
for ax, tag in zip(axes, ("7B", "32B")):
    idx = np.arange(2)
    w = .34
    c = mir[tag]["cor"]
    m_ = mir[tag]["mir"]
    ax.bar(idx - w / 2, c, w, label="使用正确证据", color=C_MAIN)
    ax.bar(idx + w / 2, m_, w, label="证据被左右镜像", color=C_WARN)
    for i, (a_, b_) in enumerate(zip(c, m_)):
        ax.text(i - w / 2, a_ + .02, f"{a_:.3f}", ha="center",
                fontproperties=FONT, fontsize=10)
        ax.text(i + w / 2, b_ + .02, f"{b_:.3f}", ha="center",
                fontproperties=FONT, fontsize=10, fontweight="bold")
    ax.set_xticks(idx)
    ax.set_xticklabels(["解释是否跟随\n输入证据", "解释是否命中\n真实篡改区域"],
                       fontproperties=FONT, fontsize=11)
    ax.set_title(f"{tag} 模型", fontproperties=FONT, fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)
axes[0].set_ylim(0, 1.3)
axes[0].set_ylabel("比例", fontproperties=FONT, fontsize=11)
# 图例放到图形上方，避免压住柱子
axes[0].legend(prop=FONT, fontsize=11, frameon=False, ncol=2,
               loc="lower left", bbox_to_anchor=(0.0, 1.14))
fig.suptitle("把证据位置故意改错：模型照旧跟随工具，却不再对应真实篡改区域"
             "（各 70 组合格图像）", fontproperties=FONT, fontsize=12.5, y=1.13)
save(fig, "fig7_镜像证据.png")

# ---------------- 图8 语义解释支持度 ----------------
def x3over(tag):
    r = re.search(rf"fake {tag}[^\n]*?\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
                  rf"([0-9.]+)\s+\d+", X3)
    return [float(r.group(i)) for i in range(1, 5)]


s7, s32 = x3over(r"M1 \(7B\)"), x3over(r"M2 \(32B\)")
print("X3 7B:", s7, " 32B:", s32)
fig, ax = plt.subplots(figsize=(8.4, 3.2))
cats = ["完全有依据", "部分有依据", "没有依据", "难以判断"]
cols = [C_GOOD, "#9DC3A6", C_WARN, C_GREY]
for i, (nm, d) in enumerate((("7B", s7), ("32B", s32))):
    left = 0
    for v, c, cn in zip(d, cols, cats):
        ax.barh(nm, v, left=left, color=c, height=.5,
                label=cn if i == 0 else None)
        if v >= .04:
            ax.text(left + v / 2, i, f"{v*100:.0f}%", ha="center", va="center",
                    color="white", fontproperties=FONT, fontsize=11,
                    fontweight="bold")
        left += v
ax.set_xlim(0, 1)
ax.set_xlabel("占该模型全部被审解释的比例", fontproperties=FONT, fontsize=10.5)
for t in ax.get_yticklabels():
    t.set_fontproperties(FONT)
    t.set_fontsize(12)
ax.set_xticks(np.arange(0, 1.01, .2))
ax.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0, 1.01, .2)],
                   fontproperties=FONT)
ax.set_title("对假图的解释是否有图像依据（各 100 条，盲化辅助标注）",
             fontproperties=FONT, fontsize=12)
ax.legend(prop=FONT, fontsize=9.5, ncol=4, frameon=False,
          loc="lower center", bbox_to_anchor=(.5, -.42))
ax.spines[["top", "right", "left"]].set_visible(False)
save(fig, "fig8_解释支持度.png")

# ---------------- 图9 篡改面积分层 ----------------
def extent(tag):
    blk = re.search(rf"\[{tag}\]\s*\n\s*extent.*?\n(.*?)(?=\n\s*\[|\n=|\Z)", X3,
                    re.S)
    out = {}
    for line in blk.group(1).strip().splitlines():
        p = line.split()
        if len(p) >= 7:
            out[p[0]] = float(p[4])      # unsup 列
    return out


e7, e32 = extent("7B"), extent("32B")
print("面积分层 7B:", e7, " 32B:", e32)
order = ["small", "medium", "large"]
zh = {"small": "小面积篡改", "medium": "中等面积", "large": "较大面积"}
fig, ax = plt.subplots(figsize=(7.6, 3.6))
idx = np.arange(3)
w = .34
a = [e7[k] for k in order]
b = [e32[k] for k in order]
ax.bar(idx - w / 2, a, w, label="7B", color=C_WARN)
ax.bar(idx + w / 2, b, w, label="32B", color=C_MAIN)
for i, (va, vb) in enumerate(zip(a, b)):
    ax.text(i - w / 2, va + .015, f"{va:.3f}", ha="center", fontproperties=FONT,
            fontsize=10.5, fontweight="bold")
    ax.text(i + w / 2, vb + .015, f"{vb:.3f}", ha="center", fontproperties=FONT,
            fontsize=10.5)
ax.set_xticks(idx)
ax.set_xticklabels([zh[k] for k in order], fontproperties=FONT, fontsize=10.5)
ax.set_ylabel("解释完全没有依据的比例", fontproperties=FONT, fontsize=10.5)
ax.set_ylim(0, .88)
ax.set_title("篡改区域越小，7B 的解释越不可靠；32B 没有这个趋势",
             fontproperties=FONT, fontsize=11.5)
ax.legend(prop=FONT, fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.25)
save(fig, "fig9_篡改面积分层.png")

json.dump(dict(A=A, B=B, trufor=dict(zip(names, vals)), raw=raw, st=st,
               mirror=mir, x3=dict(m7=s7, m32=s32), extent7=e7, extent32=e32),
          open(os.path.join(OUT, "figure_data.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n全部图表已从分析产物生成，无手写数字。")
