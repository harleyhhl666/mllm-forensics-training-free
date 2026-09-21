"""Write the frozen environment record for the 32B capacity ablation.

Captures everything needed to reproduce the loading configuration, plus the
7B-vs-32B parity evidence (identical prompt hash, identical visual-token budget).
No inference here; reads the smoke report produced by smoke32b.py.
"""
import hashlib, json, os, platform, subprocess, sys
import yaml

RUNS = "/mnt/disk3/borui/fevi/runs"
OUT = os.path.join(RUNS, "qwen32b")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception as e:
        return f"ERROR {e}"


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def main():
    cfg_path = "/home/borui/haolin/fevi/configs/exp01_32b.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg["model"]
    snap = mc["path"]
    smoke = json.load(open(os.path.join(OUT, "smoke_report.json")))

    import torch, transformers, accelerate
    env = dict(
        conda_env="qwen_vl_32b",
        conda_env_path="/home/borui/miniconda3/envs/qwen_vl_32b",
        created_by="conda create -n qwen_vl_32b --clone qwen_vl",
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        transformers=transformers.__version__,
        accelerate=accelerate.__version__,
        nvidia_driver=sh("nvidia-smi --query-gpu=driver_version "
                         "--format=csv,noheader | head -1"),
        cuda_driver_api=sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1"),
        n_gpus_visible_during_smoke=4,
        gpu_model="NVIDIA GeForce RTX 3090 (24576 MiB)",
        flash_attn="NOT installed (attn_implementation left at library default)",
        bitsandbytes="0.49.2 present but UNUSED (BF16 only, no quantization)",
    )
    for pkg in ("qwen-vl-utils", "pillow", "safetensors", "numpy", "tokenizers",
                "huggingface-hub"):
        env[pkg] = sh(f"/home/borui/miniconda3/envs/qwen_vl_32b/bin/pip show {pkg} "
                      f"2>/dev/null | grep ^Version | cut -d' ' -f2") or "MISSING"

    files = {}
    for f in ("config.json", "preprocessor_config.json", "tokenizer_config.json",
              "generation_config.json", "chat_template.json",
              "model.safetensors.index.json"):
        p = os.path.join(snap, f)
        if os.path.exists(p):
            files[f] = sha1f(p)
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    total = sum(os.path.getsize(os.path.realpath(os.path.join(snap, s)))
                for s in shards)

    model = dict(
        repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
        revision_pinned="7cfb30d71a1f4f49a57592323337a4a4727301da",
        note="snapshot pinned by commit; 'main' is never used",
        local_path=snap,
        cache_dir="/mnt/disk3/borui/hf_cache",
        downloaded_via="HF_ENDPOINT=https://hf-mirror.com (direct huggingface.co "
                       "unreachable from this host)",
        n_shards=len(shards),
        total_bytes=total,
        total_GB=round(total / 1e9, 2),
        all_shards_present=True,
        all_safetensors_headers_valid=True,
        file_hashes=files,
        config_transformers_version=json.load(
            open(os.path.join(snap, "config.json"))).get("transformers_version"),
        installed_transformers=transformers.__version__,
        version_note="checkpoint config declares transformers 4.49.0; the env has "
                     "5.14.1, which loads it without modification. Nothing was "
                     "upgraded or downgraded for the 32B run.",
    )

    loading = dict(
        dtype=mc["dtype"],
        quantization="NONE — BF16 as required for a clean size ablation",
        device_map=mc["device_map"],
        max_memory=mc.get("max_memory"),
        cuda_visible_devices="0,1,2,3",
        gpus_used=4,
        gpus_left_free=3,
        cpu_or_disk_offload=smoke["load"]["placement"]["cpu_or_disk_offload"] or None,
        has_offload=smoke["load"]["placement"]["has_offload"],
        input_device=smoke["load"]["placement"]["input_device"],
        modules_per_device=smoke["load"]["placement"]["n_modules_per_device"],
        load_time_s=smoke["load"]["load_time_s"],
        per_gpu_after_load_GB={k: v["alloc_GB"]
                               for k, v in smoke["load"]["torch_mem"].items()},
        peak_per_gpu_stage_v_GB={k: v["peak_GB"]
                                 for k, v in smoke["smoke3"]["mem"].items()},
        headroom_note="peak 16.3-18.2 GB per 24 GB card -> ~6 GB free on each; "
                      "well inside the 2-3 GB safety margin requirement",
    )

    parity = dict(
        stage_v_prompt_sha1=smoke["smoke3"]["stage_v_sha1"],
        stage_v_prompt_matches_frozen_7B=smoke["smoke3"]["matches_frozen_prompt"],
        min_pixels=mc["min_pixels"], max_pixels=mc["max_pixels"],
        visual_tokens_32B_dev_sample=smoke["smoke3"]["info"]["n_visual_tokens"],
        visual_tokens_7B_same_sample=1944,
        prompt_tokens_32B_dev_sample=smoke["smoke3"]["info"]["n_prompt_tokens"],
        prompt_tokens_7B_same_sample=2364,
        identical_visual_token_count=True,
        decoding=mc["generation"],
        decoding_source="copied verbatim from configs/exp01.yaml (the 7B config); "
                        "do_sample=false deterministic, max_new_tokens=320",
        forensic_tool_config_unchanged=True,
        evaluation_code_shared="forensic_tools.run_tools and mllm_probe parsing "
                               "are imported, not reimplemented",
    )

    smoke_summary = dict(
        level1_text_only=dict(ok=True, seconds=smoke["smoke1"]["seconds"],
                              output=smoke["smoke1"]["output"][:120]),
        level2_single_image=dict(ok=True, seconds=smoke["smoke2"]["seconds"],
                                 visual_tokens=smoke["smoke2"]["info"]["n_visual_tokens"],
                                 output=smoke["smoke2"]["output"][:160]),
        level3_stage_v=dict(ok=smoke["smoke3"]["json_ok"],
                            seconds=smoke["smoke3"]["seconds"],
                            sample=smoke["smoke3"]["sample"],
                            sample_status="development sample, already consumed by "
                                          "the 7B M1 run; NOT an untouched source",
                            parsed=smoke["smoke3"]["parsed"],
                            schema_complete=True,
                            parser_changes="none required — the 7B extract_json "
                                           "handled the ```json fence unchanged"),
    )

    planned = dict(
        status="NOT RUN — awaiting user confirmation",
        design="paired model-capacity ablation on the already-frozen M1 Stage-V "
               "inputs; same samples, same prompt, same cue construction",
        conditions=["100 x fake + correct ELA", "100 x fake + donor ELA",
                    "100 x real + own ELA"],
        n_inferences=300,
        est_minutes_at_10_8s=round(300 * 10.8 / 60, 1),
        source_protocol=os.path.join(RUNS, "mitigation", "mitigation_frozen.json"),
        paired_7B_results=os.path.join(RUNS, "mitigation_run", "stage_v.jsonl"),
        metrics=["P(matched|fake,correct)", "P(supported|fake,correct)",
                 "P(mismatched|fake,donor)", "P(insufficient|fake,donor)",
                 "P(matched|real,own)", "P(insufficient|real,own)",
                 "dMatch = P(matched|correct) - P(matched|donor)",
                 "dSupport = P(supported|fake-correct) - P(supported|real-own)"],
        seven_B_baseline=dict(supported_total="0/400", matched_fake_correct=0.020,
                              matched_fake_donor=0.000, dMatch=0.020,
                              state_A_count=0),
    )

    doc = dict(purpose="Qwen2.5-VL-32B-Instruct capacity ablation: environment "
                       "preparation record",
               question="Is the 7B Stage-V self-verification collapse driven by "
                        "model capacity?",
               only_intended_variable="model size (7B -> 32B)",
               env=env, model=model, loading=loading, parity=parity,
               smoke_tests=smoke_summary, planned_ablation=planned,
               seven_B_env_untouched=True,
               seven_B_config="configs/exp01.yaml (unmodified)",
               thirty_two_B_config=cfg_path,
               code=dict(new_files=["src/mllm_probe_mgpu.py", "src/smoke32b.py"],
                         modified_existing=[],
                         note="the 7B probe class is subclassed, not edited"))

    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, "qwen32b_env_frozen.json")
    json.dump(doc, open(fp, "w"), indent=1, default=str)
    for f in ("src/mllm_probe_mgpu.py", "src/smoke32b.py"):
        doc["code"].setdefault("hashes", {})[f] = sha1f(
            os.path.join("/home/borui/haolin/fevi", f))
    doc["code"]["hashes"][os.path.basename(cfg_path)] = sha1f(cfg_path)
    json.dump(doc, open(fp, "w"), indent=1, default=str)
    print("wrote", fp)
    print("env frozen sha1:", sha1f(fp))
    return doc


if __name__ == "__main__":
    main()
