# TRUFOR_ENV_REPORT.md

Deployment and interface verification for **TruFor** as a candidate external
forensic evidence source. This stage answers only *does it run here, and what does
it actually output?* — **not** whether it beats ELA.

Status: **deployed, interfaces verified, formal feasibility NOT run.**

Machine-readable twin: `/mnt/disk3/borui/fevi/trufor_smoke/trufor_env_frozen.json`
(sha1 `6d23c8b1365eca88d951b0523498d174ee02be96`)

---

## 1. Repository

| field | value |
|---|---|
| url | `https://github.com/grip-unina/TruFor` |
| path | `/home/borui/haolin/fevi/external_tools/TruFor` |
| commit | `ae54475df6f41a491d7615100feb19263dec13f7` (2025-05-29) |
| local modifications | **NONE** (`git status --porcelain` empty) |
| patches | none |

Official core code was **not edited**. All adaptation lives in a separate wrapper.

## 2. License note

`test_docker/LICENSE.txt`, holder **GRIP-UNINA, University Federico II of Naples**.
Recorded verbatim, no legal interpretation:

- **(i)** use, reproduction and modification "only for informational and nonprofit
  purposes"; use for "industrial or profit-oriented activities is expressly
  prohibited"
- **(ii)** reproductions must retain all original/copyright notices
- **(iii)** "reference to the original authors is given whenever results … are made
  public"

| question | recorded answer |
|---|---|
| academic / nonprofit | permitted under (i) |
| report benchmark results in a paper | permitted, and (iii) **requires citing** Guillaro et al., CVPR 2023 |
| commercial use | expressly prohibited under (i) |
| attribution required | **yes** |

A second license, `LICENSE_CMX.txt`, covers the CMX component.

## 3. Official requirements audit (read before installing)

| question | answer |
|---|---|
| recommended Python | not pinned; Docker base is `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime` |
| PyTorch / CUDA | officially torch 1.11.0 + CUDA 11.3 |
| Docker required? | **no** — Docker is offered, but inference is plain Python |
| bare-Python inference | **yes**: `python trufor_test.py -gpu 0 -in … -out …` |
| declared deps | `tqdm`, `yacs>=0.1.8`, `timm>=0.5.4`, `numpy==1.21.5` (that is all) |
| official checkpoints | **1** for inference: `trufor.pth.tar` |
| official threshold | **none in the inference code** |

**Checkpoint contents (one file, whole model):** Noiseprint++ DnCNN extractor +
SegFormer-B2 dual encoder + localization decoder + confidence decoder + `confpool`
detection head. The `pretrained_models/` weights in the repo are for **training
only** and are unused at inference.

## 4. Environment — isolated

New env `trufor`; `qwen_vl` and `qwen_vl_32b` were **not touched** (verified: `timm`
and `yacs` are still absent from `qwen_vl`).

| package | version |
|---|---|
| python | 3.10.20 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| timm | 1.0.29 |
| yacs | 0.1.8 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| opencv-python-headless | 4.11.0 |
| pillow | 12.3.0 |
| tqdm, matplotlib | 4.70.1, installed |

**Compatibility deviation, recorded not hidden:** the host driver is 535.183.01 /
CUDA 12.2, so the official torch 1.11/cu113 pin is not usable. We used
torch 2.5.1+cu121. The official README explicitly warns that *"score values can
slightly change when a different version of python, pytorch, cuda, cudnn, or other
libraries changes."* Nothing was tuned to compensate.

## 5. Server resources

All 7 × RTX 3090 idle (16 MiB each) before and after; **no other process touched**.
Inference used **one** GPU (`CUDA_VISIBLE_DEVICES=5`); multi-GPU is not required.

Disk: `/` 167 G free, `/mnt/disk3` 174 G free. Code under
`/home/borui/haolin/fevi/external_tools/TruFor`, outputs under
`/mnt/disk3/borui/fevi/trufor_smoke`.

**Peak GPU: 2.55 GB** — trivial next to the Qwen runs.

## 6. Checkpoint

| field | value |
|---|---|
| source | `https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip` |
| reachability | **direct HTTP 200** — no mirror, no Google Drive needed |
| zip size | 260 878 690 B |
| **zip MD5** | `7bee48f3476c75616c3c5721ab256ff8` |
| **matches official MD5** | **yes** (README states the same value) |
| zip SHA256 | `953f1f7eda0dd2c5ece322ae9c185ba1079c1265aa5fdf319ef5a20604d206d8` |
| extracted | `weights/trufor.pth.tar`, 281 496 429 B |
| MD5 / SHA256 | `55d7075dd1ff945e9c0f9437c5df9495` / `953f…` (see frozen JSON) |

## 7. Preprocessing (official, unchanged)

`PIL.open().convert("RGB")` → HWC numpy → CHW float tensor **÷ 256.0** (not 255.0).
**No resize, no crop** — `batch_size=1` is used deliberately "to allow arbitrary
input sizes". ImageNet mean/std normalisation happens **inside** the model
(`preprc_imagenet_torch`), not in the data loader.

## 8. The three outputs — traced through source, not guessed

### 8.1 `map` — pixel-level localization
Raw head emits 2-class logits; the official script takes
`F.softmax(pred, dim=0)[1]` → **per-pixel probability of the manipulated class**,
range **[0, 1]**, already at input resolution (no resize needed).
**No official binarisation threshold.**

### 8.2 `score` — whole-image integrity/detection score ✅ **it genuinely exists**
```
builder_np_conf.py: f1 = weighted_statistics_pooling(conf)
                    f2 = weighted_statistics_pooling(out[:,1:2]-out[:,0:1], logsigmoid(conf))
                    det = self.detection(cat(f1, f2))       # confpool head, 8->128->1
trufor_test.py:     det_sig = torch.sigmoid(det).item()
```
- variable: `det` → saved as `score`
- **already sigmoid-applied → a probability, not a logit**
- range **[0, 1]** (matches the README)
- **direction: HIGHER = MORE LIKELY MANIPULATED** (`higher_is_fake`)
- **no official decision threshold anywhere in the inference code**

### 8.3 `conf` — reliability/confidence map ⚠️ **read this carefully**
`torch.sigmoid(conf)`, range **[0, 1]**, pixel resolution, produced by a **second
decoder head** (`decode_head_conf`).

Its true semantics: **where the LOCALIZATION prediction is expected to be
reliable** — a *pixel-level* confidence map. It is **NOT** an image-level "is this
image trustworthy" score. The only way it enters an image-level quantity is as
pooled input (`f1`) to the detection head.

I am stating this plainly rather than mapping it onto our research narrative: we do
**not** get a free image-level reliability signal from TruFor. We get one
image-level number (`score`) and two pixel-level maps.

## 9. Adapter

`src/trufor_adapter.py` — calls the official model with the official config and
copies the official pre/post-processing verbatim.

**Bit-exact equivalence with the official script on all 4 smoke images:**
`|Δscore| = 0.000e+00`, `|Δmap|max = 0.000e+00`, `|Δconf|max = 0.000e+00`.

No image-level score was invented; `score` comes straight from the official
detection head.

## 10. Smoke test — interface only

4 images: 2 fake + 2 paired real, all **development samples already consumed by the
7B M1 run**. No new data downloaded.

| image | GT | score | map>0.5 px | conf mean |
|---|---|---|---|---|
| `airplane_163746_bbox_2` | fake | **0.9985** | 0.669 % | 0.9940 |
| `airplane_1761_segm_0` | fake | **0.8727** | 0.440 % | 0.9826 |
| `airplane_163746_bbox_2` | real | **0.2963** | 1.187 % | 0.9743 |
| `airplane_1761_segm_0` | real | **0.2914** | 0.298 % | 0.9833 |

All four ran; scores exist and are in-range; maps and conf maps generated for both
classes. **Real images do produce non-zero localization response** — the real
`163746` actually has a *larger* `map>0.5` area (1.187 %) than either fake, which is
exactly why image-level AUROC, not eyeballing, is needed.

**No performance conclusion is drawn.** Four images cannot support one, and the
ordering here could easily be luck.

Runtime: model load 3.4 s; per-image 0.18–0.82 s; official script 8.16 s wall for
4 images including load. Peak GPU 2.55 GB.

Visualizations: `/mnt/disk3/borui/fevi/trufor_smoke/vis/*_panel.png` —
input | localization | reliability | GT mask. Maps are drawn on a **fixed 0–1
scale** (`inferno` / `viridis`), **not** min-max stretched, so displayed intensity
equals the model probability. The **GT mask appears only in its own panel for human
inspection**; it is never a model input and never alters any output.

## 11. No tuning declaration

No threshold tuning, no preprocessing changes, no checkpoint shopping, no
localization post-processing, no cutoff fitting. The single official checkpoint with
official preprocessing and official inference logic. There is **no** official
threshold to record. The formal feasibility metric will be **threshold-free AUROC**.

## 12. Known issues

1. Official `trufor_test.py` wraps inference in a bare `try/except` that swallows
   **every** exception and continues — a failed image silently produces no output
   file. The adapter does not swallow errors.
2. The official script **skips** an image whose `.npz` already exists, so stale
   outputs are never refreshed.
3. `torch.load(weights_only=False)` raises a FutureWarning on torch 2.5.1; the
   checkpoint is the official one with verified MD5.
4. `timm.models.layers` deprecation warning — harmless.
5. Library versions differ from the official Docker pin, so last-decimal score
   differences from published numbers are expected (officially acknowledged).

## 13. Verdict

| question | answer |
|---|---|
| deployed successfully | **yes** |
| image-level score genuinely exists | **yes** — `score`, sigmoid probability in [0,1] |
| score direction | **higher = more likely fake** |
| localization map normal | **yes**, [0,1] at input resolution |
| reliability map semantics | **pixel-level localization confidence — NOT image-level trust** |
| ready for formal feasibility | **yes** |

None of stopping conditions A–D was triggered: weights obtained from the official
URL with matching MD5, dependencies resolved in isolation, a real whole-image score
exists, and the reliability map's true meaning is documented rather than
reinterpreted.

---

## Artifacts

| path | content |
|---|---|
| `trufor_smoke/trufor_env_frozen.json` | machine-readable freeze record |
| `trufor_smoke/adapter_out/trufor_results.json` | adapter output + semantics |
| `trufor_smoke/output/*.npz` | official-script outputs (equivalence reference) |
| `trufor_smoke/smoke_summary.json` | smoke rows + colormap/GT notes |
| `trufor_smoke/vis/*_panel.png` | visual panels |
| `src/trufor_adapter.py` | wrapper |
| `src/trufor_smoke_verify.py` | equivalence check + rendering |
| `src/freeze_trufor_env.py` | freeze-record writer |

Not done, per instruction: the formal 150–200 source feasibility, any MLLM
integration, any new TGIF download, any threshold tuning.
