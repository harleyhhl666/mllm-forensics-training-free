"""Two-stage MLLM probe: Stage A (evidence perception) then Stage B (arbitration).

Protocol rules hard-coded here:
  * Stage A is FORBIDDEN from producing a real/fake verdict. The prompt says so,
    and any verdict-like token in Stage A output is recorded as a protocol
    violation rather than silently used.
  * Stage B receives the image, the tool visualizations, and Stage A's OWN
    summary verbatim. Stage B cannot edit Stage A's record: stage_a is written
    to disk before Stage B is called, and is never mutated afterwards.
  * The no_tools condition (Control 1) runs the SAME Stage A prompt minus tool
    images, so localization with/without tools is comparable.
  * Generation config comes entirely from the YAML. Nothing is hardcoded.
"""
import json, os, re, sys
import numpy as np
import torch
from PIL import Image


# ------------------------------------------------------------------ prompts

STAGE_A_WITH_TOOLS = """You are assisting a digital image forensics analysis.

You are given:
1. An image under examination.
2. Outputs of automated forensic tools applied to that image, with metadata.

The 3x3 grid names for locations are exactly:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

TOOL METADATA:
{tool_meta}

Your task in THIS step is ONLY to report what the forensic tool outputs show.
Do NOT decide whether the image is real or fake. Do NOT mention authenticity.
That decision happens in a later, separate step.

Report strictly as JSON with these keys:
{{
  "evidence_detected": true or false,
  "evidence_sources": [list of tool names showing an anomaly, from: {tool_names}],
  "predicted_regions": [list of 3x3 grid cell names where the anomaly is strongest],
  "evidence_description": "one or two sentences describing the anomaly"
}}
Output JSON only."""

STAGE_A_NO_TOOLS = """You are assisting a digital image forensics analysis.

You are given only an image under examination. No automated tool outputs are provided.

The 3x3 grid names for locations are exactly:
top-left, top-center, top-right, center-left, center, center-right,
bottom-left, bottom-center, bottom-right.

Your task in THIS step is ONLY to report whether any region of the image shows
signs of local inconsistency (in texture, noise, lighting, edges, or detail
level). Do NOT decide whether the image is real or fake. Do NOT mention
authenticity. That decision happens in a later, separate step.

Report strictly as JSON with these keys:
{{
  "evidence_detected": true or false,
  "evidence_sources": ["visual_inspection"],
  "predicted_regions": [list of 3x3 grid cell names where the inconsistency is strongest],
  "evidence_description": "one or two sentences describing what you observe"
}}
Output JSON only."""

STAGE_B = """You are completing a digital image forensics analysis.

You are given:
1. The image under examination.
2. Outputs of automated forensic tools applied to that image, with metadata.
3. The evidence summary that was produced in the previous analysis step.

TOOL METADATA:
{tool_meta}

EVIDENCE SUMMARY FROM THE PREVIOUS STEP (do not revise it, treat it as given):
{stage_a_summary}

Now make the final determination for this image.

Report strictly as JSON with these keys:
{{
  "final_verdict": "real" or "fake",
  "reason": "one or two sentences justifying the verdict"
}}
Output JSON only."""


# Stage A is told not to decide authenticity. Detecting a violation must NOT flag
# ordinary anomaly vocabulary: describing a region as showing signs of tampering
# is exactly what Stage A is asked to do. Only CONCLUSIVE verdict language counts,
# i.e. an assertion that the image as a whole is real/fake.
VERDICT_WORDS = re.compile(
    r"(?:\b(?:image|photo|photograph|picture)\b[^.]{0,40}?\b(?:is|appears|seems|looks)\b"
    r"[^.]{0,20}?\b(?:real|fake|authentic|genuine|pristine|unaltered|forged|manipulated)\b"
    r"|\b(?:is|it['\u2019]s)\s+(?:a\s+)?(?:real|fake|authentic|genuine|pristine)\b"
    r"|\b(?:conclude|verdict|therefore)\b[^.]{0,30}?\b(?:real|fake)\b)", re.I)


# ------------------------------------------------------------------ parsing

def extract_json(text):
    """Pull the first JSON object out of model output. Returns (obj, raw, ok)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start = t.find("{")
    if start < 0:
        return None, text, False
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1]), text, True
                except json.JSONDecodeError:
                    return None, text, False
    return None, text, False


def normalize_stage_a(obj, valid_tool_names):
    """Coerce Stage A output into the fixed schema. Records what was coerced."""
    CELLS = {"top-left", "top-center", "top-right", "center-left", "center",
             "center-right", "bottom-left", "bottom-center", "bottom-right"}
    notes = []
    if not isinstance(obj, dict):
        return None, ["not_a_dict"]
    ed = obj.get("evidence_detected")
    if isinstance(ed, str):
        ed = ed.strip().lower() in ("true", "yes")
        notes.append("evidence_detected_coerced_from_string")
    if not isinstance(ed, bool):
        notes.append("evidence_detected_missing_or_invalid")
        ed = None
    srcs = obj.get("evidence_sources") or []
    if isinstance(srcs, str):
        srcs = [srcs]
    srcs_clean = [s for s in srcs if isinstance(s, str) and s.strip().lower()
                  in {v.lower() for v in valid_tool_names}]
    if len(srcs_clean) != len(srcs):
        notes.append("unknown_evidence_source_dropped")
    regs = obj.get("predicted_regions") or []
    if isinstance(regs, str):
        regs = [regs]
    regs_clean = [r.strip().lower() for r in regs
                  if isinstance(r, str) and r.strip().lower() in CELLS]
    if len(regs_clean) != len(regs):
        notes.append("invalid_region_name_dropped")
    desc = obj.get("evidence_description")
    desc = desc if isinstance(desc, str) else ""
    # protocol check: did Stage A leak a verdict despite being told not to?
    leak = bool(VERDICT_WORDS.search(desc))
    return dict(evidence_detected=ed, evidence_sources=srcs_clean,
                predicted_regions=regs_clean, evidence_description=desc,
                verdict_leak_in_stage_a=leak), notes


def normalize_stage_b(obj):
    if not isinstance(obj, dict):
        return None, ["not_a_dict"]
    v = obj.get("final_verdict")
    notes = []
    if isinstance(v, str):
        vl = v.strip().lower()
        if vl in ("real", "authentic", "genuine", "pristine"):
            v = "real"
        elif vl in ("fake", "manipulated", "tampered", "forged", "edited"):
            v = "fake"
        else:
            notes.append(f"unrecognized_verdict:{vl[:30]}")
            v = None
    else:
        notes.append("verdict_missing")
        v = None
    r = obj.get("reason")
    return dict(final_verdict=v, reason=r if isinstance(r, str) else ""), notes


# ------------------------------------------------------------------ the model

class QwenVLProbe:
    def __init__(self, cfg):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        mc = cfg["model"]
        path = mc["path"]
        dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16)[mc["dtype"]]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=dtype, device_map=mc["device_map"])
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(
            path, min_pixels=mc["min_pixels"], max_pixels=mc["max_pixels"])
        self.gen = dict(mc["generation"])
        self.cfg = cfg

    def _build(self, prompt, images):
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images, return_tensors="pt", padding=True)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def ask(self, prompt, images, gen_override=None):
        inputs = self._build(prompt, images)
        g = dict(self.gen)
        if gen_override:
            g.update(gen_override)
        kw = dict(max_new_tokens=g["max_new_tokens"],
                  repetition_penalty=g.get("repetition_penalty", 1.0))
        if g.get("do_sample"):
            kw.update(do_sample=True, temperature=g["temperature"],
                      top_p=g.get("top_p", 1.0))
        else:
            kw.update(do_sample=False)
        out = self.model.generate(**inputs, **kw)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        txt = self.proc.tokenizer.decode(trimmed, skip_special_tokens=True)
        n_vis = int(inputs["image_grid_thw"].prod(dim=1).sum().item() // 4) \
            if "image_grid_thw" in inputs else None
        return txt, dict(n_visual_tokens=n_vis,
                         n_prompt_tokens=int(inputs["input_ids"].shape[1]))
