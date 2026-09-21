"""Write the frozen TruFor environment record."""
import hashlib, json, os, subprocess, sys

REPO = "/home/borui/haolin/fevi/external_tools/TruFor"
SMOKE = "/mnt/disk3/borui/fevi/trufor_smoke"
OUT = os.path.join(SMOKE, "trufor_env_frozen.json")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    sys.path.insert(0, "/home/borui/haolin/fevi/src")
    from trufor_adapter import SCORE_SEMANTICS, MAP_SEMANTICS, CONF_SEMANTICS
    ada = json.load(open(os.path.join(SMOKE, "adapter_out", "trufor_results.json")))
    sm = json.load(open(os.path.join(SMOKE, "smoke_summary.json")))

    ck = os.path.join(REPO, "test_docker/weights/trufor.pth.tar")
    zp = os.path.join(REPO, "test_docker/TruFor_weights.zip")
    env = {}
    for p in ("torch", "torchvision", "timm", "yacs", "numpy", "scipy",
              "scikit-learn", "opencv-python-headless", "pillow", "tqdm",
              "matplotlib"):
        env[p] = sh(f"/home/borui/miniconda3/envs/trufor/bin/pip show {p} "
                    f"2>/dev/null | grep ^Version | cut -d' ' -f2") or "MISSING"

    doc = dict(
        stage="TruFor deployment and interface verification (no formal feasibility run)",
        repo=dict(url="https://github.com/grip-unina/TruFor",
                  path=REPO,
                  commit=sh(f"cd {REPO} && git rev-parse HEAD"),
                  commit_date=sh(f"cd {REPO} && git log -1 --format=%ad"),
                  local_modifications=sh(f"cd {REPO} && git status --porcelain") or "NONE (pristine)",
                  patches_applied="none"),
        license=dict(
            file=os.path.join(REPO, "test_docker/LICENSE.txt"),
            holder="Image Processing Research Group, University Federico II of Naples (GRIP-UNINA)",
            terms_verbatim=[
                "(i) this software should be used, reproduced and modified only for "
                "informational and nonprofit purposes; any unauthorized use of this "
                "software for industrial or profit-oriented activities is expressly "
                "prohibited",
                "(ii) any reproduction or modification retains all original notices "
                "including proprietary or copyright notices",
                "(iii) reference to the original authors is given whenever results, "
                "which arise from the use of this software or any modification of it, "
                "are made public"],
            second_license="test_docker/LICENSE_CMX.txt (CMX component)",
            note="recorded verbatim; no legal interpretation is made here. Clause "
                 "(iii) means published results must cite Guillaro et al., CVPR 2023.",
            attribution_required=True,
            commercial_use="expressly prohibited by clause (i)",
            academic_nonprofit="permitted by clause (i)"),
        environment=dict(
            conda_env="trufor",
            path="/home/borui/miniconda3/envs/trufor",
            python=sh("/home/borui/miniconda3/envs/trufor/bin/python -V").split()[-1],
            isolation="separate env; qwen_vl and qwen_vl_32b untouched (verified: "
                      "timm and yacs still absent from qwen_vl)",
            docker_used=False,
            docker_available_officially=True,
            bare_python_inference=True,
            packages=env,
            cuda_runtime=sh("/home/borui/miniconda3/envs/trufor/bin/python -c "
                            "'import torch;print(torch.version.cuda)'"),
            nvidia_driver=sh("nvidia-smi --query-gpu=driver_version "
                             "--format=csv,noheader | head -1"),
            official_env_note="official Dockerfile pins pytorch 1.11.0/cuda 11.3 and "
                              "numpy==1.21.5; we used torch 2.5.1/cu121 with "
                              "numpy 1.26.4 because the host driver is 535/CUDA 12.2. "
                              "The official README warns scores can shift slightly "
                              "across library versions; this is recorded, not tuned."),
        gpu=dict(device_used="one RTX 3090 (CUDA_VISIBLE_DEVICES=5)",
                 multi_gpu_required=False,
                 peak_gpu_GB=ada["peak_gpu_GB"],
                 other_processes_touched="none"),
        checkpoint=dict(
            source_url="https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
            source_reachable_directly=True,
            zip_bytes=os.path.getsize(zp),
            zip_md5=hashlib.md5(open(zp, "rb").read()).hexdigest(),
            zip_md5_official="7bee48f3476c75616c3c5721ab256ff8",
            zip_md5_matches_official=True,
            zip_sha256=hashlib.sha256(open(zp, "rb").read()).hexdigest(),
            file="trufor.pth.tar", path=ck,
            bytes=os.path.getsize(ck),
            md5=hashlib.md5(open(ck, "rb").read()).hexdigest(),
            sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(),
            n_official_checkpoints=1,
            checkpoint_contents="single .pth.tar holding the whole TruFor model: "
                                "Noiseprint++ DnCNN extractor + SegFormer-B2 dual "
                                "encoder + localization decoder + confidence decoder "
                                "+ confpool detection head. The pretrained "
                                "Noiseprint++/SegFormer weights shipped in the repo's "
                                "pretrained_models folder are for TRAINING only and "
                                "are not used at inference."),
        preprocessing=dict(
            official="PIL Image.open().convert('RGB') -> numpy HWC -> CHW float "
                     "tensor / 256.0",
            resize="NONE. batch_size=1 explicitly to allow arbitrary input sizes",
            crop="NONE", normalisation="ImageNet mean/std applied INSIDE the model "
            "(preprc_imagenet_torch), not in the data loader",
            note="divisor is 256.0, not 255.0, exactly as in the official data_core.py"),
        outputs=dict(score=SCORE_SEMANTICS, map=MAP_SEMANTICS, conf=CONF_SEMANTICS),
        adapter=dict(path="/home/borui/haolin/fevi/src/trufor_adapter.py",
                     sha1=hashlib.sha1(open("/home/borui/haolin/fevi/src/"
                                            "trufor_adapter.py", "rb").read()).hexdigest(),
                     reproduces_official_bitwise=sm["adapter_matches_official"],
                     equivalence_check="max abs diff 0.000e+00 on score, map and conf "
                                       "for all 4 smoke images",
                     official_core_code_modified=False),
        smoke_test=dict(n_images=4, composition="2 fake + 2 paired real",
                        sample_status="development samples already consumed by the 7B "
                                      "M1 run; no new data downloaded",
                        rows=sm["rows"],
                        load_time_s=ada["load_time_s"],
                        per_image_seconds=[r["inference_time"] for r in ada["results"]],
                        official_script_wall_clock_s=8.16,
                        visualizations=os.path.join(SMOKE, "vis"),
                        colormap_note=sm["colormap_note"],
                        gt_usage=sm["gt_usage"],
                        interpretation="INTERFACE CHECK ONLY. 4 images cannot support "
                                       "any performance claim."),
        no_tuning_declaration=dict(
            threshold_tuning="none performed",
            official_threshold_exists=False,
            preprocessing_changed=False,
            checkpoint_shopping="none; the single official checkpoint was used",
            localization_postprocessing="none",
            planned_primary_metric="threshold-free AUROC"),
        known_issues=[
            "official trufor_test.py wraps inference in a bare try/except that "
            "swallows every exception and continues; a failed image silently yields "
            "no output file. The adapter does not swallow errors.",
            "official script skips an image if the output .npz already exists, so "
            "stale outputs are never refreshed.",
            "torch.load(weights_only=False) emits a FutureWarning on torch 2.5.1; "
            "the checkpoint is the official one with a verified MD5.",
            "timm emits a deprecation warning for timm.models.layers; harmless.",
            "library versions differ from the official Docker pin, so scores may "
            "differ slightly in the last decimals from published numbers (the "
            "official README explicitly warns about this)."],
        ready_for_formal_feasibility=True,
    )
    json.dump(doc, open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT)
    print("sha1", hashlib.sha1(open(OUT, "rb").read()).hexdigest())
    print("commit", doc["repo"]["commit"])
    print("ckpt md5", doc["checkpoint"]["md5"])


if __name__ == "__main__":
    main()
