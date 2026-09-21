"""Multi-GPU-safe subclass of the 7B QwenVLProbe, for the 32B capacity ablation.

Only DEVICE PLACEMENT changes. Prompt construction, generation kwargs, JSON
parsing and the visual-token budget are inherited unchanged from QwenVLProbe so
that the 7B/32B comparison varies model size and nothing else.

Two placement bugs the 7B code would hit with a sharded model:
  * `.to(self.model.device)` — a model split by accelerate has no single device;
    inputs must go to the device holding the FIRST module (the embedding), and
    accelerate's hooks move activations from there.
  * calling `model.to(...)` after `device_map` — would undo the sharding.
Neither is done here; the base class's `_build` is overridden for the first.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mllm_probe import QwenVLProbe, extract_json


class QwenVLProbeMultiGPU(QwenVLProbe):
    def __init__(self, cfg):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        mc = cfg["model"]
        path = mc["path"]
        dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16)[mc["dtype"]]
        dm = mc["device_map"]
        mm = mc.get("max_memory")
        kw = {}
        if mm:
            # keys may arrive as strings from YAML/JSON; accelerate wants ints for GPUs
            kw["max_memory"] = {(int(k) if str(k).isdigit() else k): v
                                for k, v in mm.items()}
        t0 = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=dtype, device_map=dm, **kw)
        self.model.eval()
        # NOTE: deliberately NO model.to(...) — that would break device_map sharding.
        self.load_time_s = time.time() - t0
        self.proc = AutoProcessor.from_pretrained(
            path, min_pixels=mc["min_pixels"], max_pixels=mc["max_pixels"])
        self.gen = dict(mc["generation"])
        self.cfg = cfg
        self.device_map_resolved = getattr(self.model, "hf_device_map", None)
        # the device that must receive input_ids
        self.input_device = self._first_device()

    def _first_device(self):
        dm = self.device_map_resolved
        if not dm:
            return next(self.model.parameters()).device
        for mod, dev in dm.items():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev not in ("cpu", "disk"):
                return torch.device(dev)
        return next(self.model.parameters()).device

    def _build(self, prompt, images):
        """Same as the base class, but inputs land on the embedding's device."""
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        # an empty image list must become images=None: the processor indexes
        # images[0] to pick a device and would raise IndexError on []
        inputs = self.proc(text=[text], images=(images or None),
                           return_tensors="pt", padding=True)
        return {k: (v.to(self.input_device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    def placement_report(self):
        dm = self.device_map_resolved or {}
        per = {}
        for mod, dev in dm.items():
            per.setdefault(str(dev), []).append(mod)
        offload = [d for d in per if d in ("cpu", "disk")]
        return dict(input_device=str(self.input_device),
                    devices=sorted(per, key=str),
                    n_modules_per_device={d: len(m) for d, m in per.items()},
                    cpu_or_disk_offload=offload,
                    has_offload=bool(offload))


def gpu_mem():
    out = {}
    for i in range(torch.cuda.device_count()):
        out[i] = dict(
            alloc_GB=round(torch.cuda.memory_allocated(i) / 1e9, 2),
            reserved_GB=round(torch.cuda.memory_reserved(i) / 1e9, 2),
            peak_GB=round(torch.cuda.max_memory_allocated(i) / 1e9, 2))
    return out


def nvidia_used():
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return {int(l.split(",")[0]): int(l.split(",")[1])
            for l in r.stdout.strip().splitlines() if l.strip()}
