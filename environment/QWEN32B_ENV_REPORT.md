# QWEN32B_ENV_REPORT.md

Environment preparation for the **Qwen2.5-VL-32B-Instruct capacity ablation**.
Question under test: *is the 7B Stage-V self-verification collapse driven by model
capacity?* The only intended experimental variable is model size (7B → 32B).

Status: **environment ready, smoke tests passed, formal 300-inference ablation NOT run.**

Machine-readable twin: `/mnt/disk3/borui/fevi/runs/qwen32b/qwen32b_env_frozen.json`
(sha1 `5a669edd8d76e50abf5d8948b500465a0afb6410`)

---

## 1. GPU status

All 7 × RTX 3090 (24576 MiB each) were **idle** before loading — only Xorg,
4 MiB per card. No other user's process was touched or terminated.

| GPU | before | after 32B load | Stage-V peak | free |
|---|---|---|---|---|
| 0 | 16 MiB | 15.61 GB | 16.30 GB | ~8 GB |
| 1 | 16 MiB | 17.55 GB | 18.21 GB | ~6 GB |
| 2 | 16 MiB | 17.55 GB | 18.21 GB | ~6 GB |
| 3 | 16 MiB | 16.19 GB | 16.81 GB | ~7.5 GB |
| 4–6 | 16 MiB | unused | unused | reserved |

- Driver **535.183.01**, CUDA driver API **12.2**, torch CUDA runtime **12.1**
- `nvcc` on PATH is 10.1 — stale and **irrelevant** (no CUDA extension is compiled;
  flash-attn is not installed)
- BF16 supported, compute capability 8.6

Peak stays **≤ 18.2 GB of 24 GB**, i.e. ~6 GB headroom on the tightest card —
inside the requested 2–3 GB safety margin.

## 2. Environment

Created by **cloning** the verified 7B env, so versions are identical by
construction rather than by reinstallation:

```
conda create -n qwen_vl_32b --clone qwen_vl
```

| package | version |
|---|---|
| python | 3.10.13 |
| torch | 2.5.1 (cuda 12.1, cudnn 90100) |
| torchvision | 0.20.1 |
| transformers | 5.14.1 |
| accelerate | 1.14.0 |
| qwen-vl-utils | 0.0.14 |
| pillow | 9.4.0 |
| safetensors | 0.8.0 |
| numpy | 2.2.6 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.24.0 |
| flash-attn | **not installed** (attn left at library default, same as 7B) |
| bitsandbytes | 0.49.2 present but **unused** |

The 7B env `qwen_vl` was **not modified**. Nothing was upgraded for the 32B run.

**Version note:** the 32B checkpoint's `config.json` declares
`transformers_version: 4.49.0`; the installed 5.14.1 loads it unmodified. No
minimum-version conflict was hit, so no silent upgrade was performed.

## 3. Model snapshot

| field | value |
|---|---|
| repo | `Qwen/Qwen2.5-VL-32B-Instruct` |
| revision (pinned) | `7cfb30d71a1f4f49a57592323337a4a4727301da` |
| local path | `/mnt/disk3/borui/hf_cache/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d7…` |
| size | **68.28 GB**, 18 shards |
| integrity | all 18 shards present, all safetensors headers valid |
| license | apache-2.0 |
| download | `HF_ENDPOINT=https://hf-mirror.com` (direct `huggingface.co` unreachable) |

Commit is pinned; `main` is never used. Config hashes: `config.json`
`7e64c1bf…`, `preprocessor_config.json` `2bf9a853…`, `generation_config.json`
`5f3464e0…`, `chat_template.json` `da5373cb…`.

Cache placed on `/mnt/disk3` (237 G free) rather than `/` (169 G free, 90 % used),
since the existing HF cache is already 128 G.

## 4. Loading strategy — BF16, 4 GPUs, no quantization

```yaml
dtype: bfloat16
device_map: auto
max_memory: {0: 20GiB, 1: 20GiB, 2: 20GiB, 3: 20GiB}
CUDA_VISIBLE_DEVICES: 0,1,2,3
```

- **No quantization.** BF16 as required for a clean size ablation.
- **No CPU/disk offload** — verified from the resolved `hf_device_map`.
- Module split 15/18/18/18 across the 4 cards; inputs go to `cuda:0`.
- Load time **28.7 s** warm (95.4 s cold, first read from disk).
- GPUs 4–6 left free.

**Multi-GPU code changes** (in a new subclass, `src/mllm_probe_mgpu.py` — the 7B
`QwenVLProbe` is subclassed, never edited):

1. Inputs go to the device holding the **embedding layer**, not `self.model.device`
   (a sharded model has no single device). Accelerate's hooks move activations.
2. **No `model.to(...)`** after `device_map` — that would undo the sharding.
3. Empty image list → `images=None`; the processor indexes `images[0]` to choose a
   device and raises `IndexError` on `[]` (this was the only real bug found, caught
   by smoke 1).

Prompt construction, generation kwargs, and JSON parsing are **inherited
unchanged**.

## 5. Smoke tests — all passed

| level | input | time | result |
|---|---|---|---|
| 1 | text `hello` | 2.3 s | `"Hello! How can I assist you today? 😊"` |
| 2 | synthetic image (not an experiment sample) | 2.1 s | correct description; 368 visual tokens; vision path OK across shards |
| 3 | one Stage-V pair, frozen prompt | **10.8 s** | valid JSON, all 4 schema fields present |

**Smoke 3 detail.** Development sample `airplane_163746_bbox_2` — already consumed
by the 7B M1 run, deliberately *not* an untouched source. Stage-V prompt sha1
`cfa6908aaf02a60f71ad74153a0a3df773252b45`, **verified identical** to the frozen 7B
prompt (asserted at runtime).

32B output on that sample:
```json
{"image_consistency": "matched", "forensic_support": "insufficient",
 "strongest_region": "top-center",
 "reason": "...located at the top-center, corresponding to the airplanes in the sky.
  These objects have distinct edges and details that naturally produce strong
  responses..."}
```
7B on the same sample: `mismatched` / `insufficient` / `top-center` → State C.

No parser change was needed; the 7B `extract_json` handled the ` ```json ` fence.
**The prompt was not touched**, per the standing rule that only parser-level
issues may be fixed.

## 6. Parity with the 7B pipeline

| item | 7B | 32B |
|---|---|---|
| Stage-V prompt sha1 | `cfa6908a…` | `cfa6908a…` (identical) |
| min_pixels / max_pixels | 200704 / 802816 | 200704 / 802816 |
| **visual tokens (same sample)** | **1944** | **1944** |
| **prompt tokens (same sample)** | **2364** | **2364** |
| decoding | `do_sample=false`, `max_new_tokens=320`, `rep_penalty=1.0` | identical |
| forensic tool config | unchanged | unchanged |
| tool/parsing code | `forensic_tools`, `mllm_probe` | imported, not reimplemented |

Token counts match **exactly** on the shared sample, and the 32B count (1944) sits
inside the 7B Stage-V range (1344–1998, mean 1811.7 over 400 runs). Visual
information content is therefore held constant; resolution was **not** raised for
the larger model.

## 7. Planned ablation — prepared, not executed

- 100 × fake + correct ELA, 100 × fake + donor ELA, 100 × real + own ELA
- **300 inferences**, ≈ **54 min** at 10.8 s each
- Reuses the frozen M1 Stage-V inputs (`mitigation/mitigation_frozen.json`), so the
  comparison is **paired per sample** against `mitigation_run/stage_v.jsonl`
- Metrics: `P(matched|·)`, `P(supported|·)`, `P(mismatched|donor)`,
  `P(insufficient|real-own)`, plus
  `ΔMatch = P(matched|correct) − P(matched|donor)` and
  `ΔSupport = P(supported|fake-correct) − P(supported|real-own)`
- 7B baseline to beat: **supported 0/400**, State A **0**, ΔMatch **+0.020**

---

## Artifacts

| file | purpose |
|---|---|
| `runs/qwen32b/qwen32b_env_frozen.json` | machine-readable freeze record |
| `runs/qwen32b/smoke_report.json` | raw smoke-test measurements |
| `configs/exp01_32b.yaml` | 32B config (only model block differs from `exp01.yaml`) |
| `src/mllm_probe_mgpu.py` | multi-GPU subclass |
| `src/smoke32b.py` | three-level smoke test |
| `src/freeze_32b_env.py` | freeze-record writer |

Not done, per instruction: the 300-inference ablation, the full M1 re-run, TruFor
deployment, and any Stage-V prompt change.
