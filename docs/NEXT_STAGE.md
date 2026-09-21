# NEXT_STAGE.md

# Post-Training / Fine-Tuning Stage

**Scope note, stated first:** nothing in this file is a result. This is a plan. No
conclusion here belongs to the training-free repository, and none of it has been tested.

---

## The question this stage inherits

The training-free stage answered "can a pretrained MLLM *use* external forensic evidence"
and produced a clean split:

- **decisions** — solved, and not by the MLLM. An external detector's scalar score already
  carries it (32B score-only J 0.918). Adding evidence structure does not help and can hurt.
- **spatial explanation** — solved, but borrowed. Structured evidence yields near-perfect
  citation of the tool's regions (agreement 1.000), yet mirror the evidence and the
  explanation follows it to the wrong place (Explanation→GT → 0.014).
- **semantic explanation** — **not** solved. Across 300 audited explanations, evaluable
  texture / lighting / boundary / geometry / physics claims numbered **0–3 each**. Both
  models name a location and assert manipulation; they rarely say *why* in a way that
  survives audit.

So the open question is:

> How do we make an MLLM explain **why** a particular region is forged — with claims that
> are checkable against the image?

The training-free evidence says this is a **supervision** problem, not a prompting one.
Three rounds of prompt intervention (field ablation, framing control, task semantics) moved
decisions substantially and fine-grained explanation content essentially not at all.

---

## Candidate directions

Ordered roughly by cost. None is committed.

### 1. Region-level explanation supervision

Train on `(region, evidence-type, justification)` rather than on verdicts. The point is to
make the model produce *auditable* claims, not more confident ones.

### 2. Paired real/fake supervision

The dataset's strongest and least-used property: TGIF `sd2-sp` is a pixel-exact local
splice, so a fake and its paired real differ **only** inside the mask. That is a
ready-made contrastive signal for "what actually changed here", and the X3 audit already
showed annotators rely on exactly this comparison.

### 3. A forensic rationale dataset

The scarcity finding is the design brief: we need training data whose rationales are
*about* texture, boundary, lighting, geometry and physical plausibility, because the base
models do not produce those categories on their own.

### 4. LoRA / small-scale SFT

Consistent with the resource envelope (1–2× 24 GB). Explicitly excluded during the
training-free stage to keep that stage's claims clean; now in scope.

### 5. Annotation of the five weak claim types

Currently unmeasurable — 0–3 evaluable claims each. Any progress here needs annotation
before it needs modelling.

---

## Constraints carried forward

These are not negotiable, and they are why the training-free results are trustworthy:

1. **Pre-register before running.** Protocol, splits, thresholds, sample IDs and metrics
   frozen to disk first. A freeze script refuses to overwrite.
2. **Never reuse a discovery set for confirmation.**
3. **No post-hoc metric or definition changes.** "Strengthen the definition to raise the
   hit rate" is disqualifying.
4. **Report negative results plainly.** Most of this project's value came from them.
5. **Guard the pipeline, not just the analysis.** Assert rendered prompt and payload
   hashes against the freeze *before* inference. A payload bug once voided 1472 runs;
   the hash guard that now blocks it was added in response.
6. **Keep decision utility and explanation faithfulness in separate sections.** A
   performance gain must never be reported as improved spatial reasoning when the
   evidence's summary statistics could leak the label.
7. **Do not fabricate an independent annotator.** If a genuinely independent annotation
   endpoint is unavailable, say so, as the X3 limitation does.

## Data boundaries

677 distinct `coco_id` are consumed. The TGIF training split's sd2 pool (1558 `coco_id`)
has **zero** intersection with them and is therefore available — but note that the three
TGIF generators share the same 2242 `coco_id`, so switching generator does not buy fresh
sources. If this stage needs more data it needs a **different dataset**, and that decision
should be explicit rather than incidental.

## Repository boundary

The training-free stage is frozen at this commit. If the post-training work continues in
this repository it should be a **new branch**, so the training-free results stay
reconstructible at their frozen state. Its findings must not be merged into
`docs/CURRENT_FINDINGS.md`, which describes the training-free stage only.

## Open questions worth settling early

- Does supervision produce *correct* fine-grained claims, or just more fluent ones? The
  X3 rubric already exists and can measure this — reuse it rather than inventing a score.
- Does a fine-tuned model start checking spatial correspondence, which neither 7B nor 32B
  does (shift effect −0.027, CI containing zero at both scales)?
- Can explanation quality improve without re-introducing the framing sensitivity that
  wrecked 7B decisions?
- Since the score already wins on decisions, is the right target an **explanation-only**
  model that never renders a verdict?
