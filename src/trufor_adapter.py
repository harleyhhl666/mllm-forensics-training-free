"""Minimal wrapper around the official TruFor inference path.

Calls the official model with the official config, preprocessing and postprocessing
-- the transformations are copied verbatim from test_docker/src/trufor_test.py, not
reinvented. Nothing is thresholded, rescaled or recalibrated here.

Field semantics, traced through the official source (not inferred from names):

  score : image-level DETECTION score.
          builder_np_conf.py encode_decode() -> det = self.detection(cat(f1,f2))
          where f1/f2 are weighted_statistics_pooling of the confidence map and of
          the localization logit difference. trufor_test.py then applies
          torch.sigmoid(det).item(), so it is a SIGMOID PROBABILITY in [0,1].
          HIGHER = MORE LIKELY MANIPULATED (the detection head is trained on the
          manipulated class; README calls the range [0,1]).
          There is NO official decision threshold in the inference code.

  map   : pixel-level localization. Raw head output is 2-class logits; the official
          script takes F.softmax(pred, dim=0)[1], i.e. the per-pixel probability of
          the MANIPULATED class, in [0,1], already at input resolution.
          No official binarisation threshold is applied.

  conf  : pixel-level CONFIDENCE/reliability map, torch.sigmoid(conf) in [0,1].
          It is the output of a SECOND decoder head (decode_head_conf) at pixel
          resolution. It expresses where the LOCALIZATION prediction is expected to
          be reliable. It is NOT an image-level trust score; the only place it
          contributes to an image-level quantity is as pooled input to the
          detection head.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

TRUFOR_SRC = "/home/borui/haolin/fevi/external_tools/TruFor/test_docker/src"

SCORE_SEMANTICS = dict(
    field="score",
    definition="sigmoid(det) where det is the confpool detection head output",
    range=[0.0, 1.0],
    type="sigmoid probability (not a logit)",
    direction="higher_is_fake",
    official_threshold=None,
    traced_in=["test_docker/src/trufor_test.py:140,150",
               "test_docker/src/models/cmx/builder_np_conf.py:encode_decode detection branch"],
)
MAP_SEMANTICS = dict(
    field="map",
    definition="softmax(pred, dim=0)[1] = per-pixel probability of the manipulated class",
    range=[0.0, 1.0], resolution="input resolution (no resize needed)",
    official_threshold=None,
    traced_in=["test_docker/src/trufor_test.py:142-144"],
)
CONF_SEMANTICS = dict(
    field="conf",
    definition="sigmoid of a second decoder head (decode_head_conf); pixel-level "
               "confidence that the LOCALIZATION prediction is reliable",
    range=[0.0, 1.0], resolution="input resolution",
    is_image_level_trust_score=False,
    note="NOT an image-level reliability score. Used at image level only as pooled "
         "input to the detection head.",
    traced_in=["test_docker/src/trufor_test.py:130-133",
               "test_docker/src/models/cmx/builder_np_conf.py:decode_head_conf"],
)


class TruForAdapter:
    def __init__(self, gpu=0, src=TRUFOR_SRC):
        self.src = src
        cwd = os.getcwd()
        if src not in sys.path:
            sys.path.insert(0, src)
        parent = os.path.join(src, "..")
        if parent not in sys.path:
            sys.path.insert(0, parent)
        os.chdir(src)                      # official config.py reads ./trufor.yaml
        try:
            import torch
            from config import _C as config
            config.defrost()
            config.merge_from_file("trufor.yaml")
            config.freeze()
            self.config = config
            self.device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = config.CUDNN.BENCHMARK
            cudnn.deterministic = config.CUDNN.DETERMINISTIC
            cudnn.enabled = config.CUDNN.ENABLED
            ckpt_path = config.TEST.MODEL_FILE
            self.ckpt_path = os.path.abspath(ckpt_path)
            ck = torch.load(ckpt_path, map_location=torch.device(self.device))
            assert config.MODEL.NAME == "detconfcmx", config.MODEL.NAME
            from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
            t0 = time.time()
            self.model = confcmx(cfg=config)
            self.model.load_state_dict(ck["state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.load_time = time.time() - t0
            self.torch = torch
        finally:
            os.chdir(cwd)

    def run(self, image_path):
        """Official preprocessing: RGB, CHW, float/256.0 -- no resize, no crop."""
        import torch
        from torch.nn import functional as F
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
        rgb = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float) / 256.0
        rgb = rgb.unsqueeze(0).to(self.device)
        t0 = time.time()
        with torch.no_grad():
            pred, conf, det, npp = self.model(rgb)
            det_sig = torch.sigmoid(det).item() if det is not None else None
            if conf is not None:
                conf = torch.sigmoid(torch.squeeze(conf, 0))[0].cpu().numpy()
            pred = torch.squeeze(pred, 0)
            loc = F.softmax(pred, dim=0)[1].cpu().numpy()
        dt = time.time() - t0
        return dict(integrity_score=det_sig, localization_map=loc,
                    reliability_map=conf,
                    input_size=[int(img.shape[0]), int(img.shape[1])],
                    inference_time=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ad = TruForAdapter(gpu=args.gpu)
    import torch
    print(f"model loaded in {ad.load_time:.1f}s from {ad.ckpt_path}")
    recs = []
    torch.cuda.reset_peak_memory_stats(0)
    for p in args.images:
        r = ad.run(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        npz = os.path.join(args.out, stem + "_trufor.npz")
        np.savez_compressed(npz, map=r["localization_map"],
                            conf=r["reliability_map"],
                            score=np.array(r["integrity_score"]))
        rec = dict(image_path=os.path.abspath(p),
                   integrity_score=r["integrity_score"],
                   score_direction=SCORE_SEMANTICS["direction"],
                   score_type=SCORE_SEMANTICS["type"],
                   official_threshold=None,
                   localization_map_path=npz + "::map",
                   reliability_map_path=npz + "::conf",
                   localization_map_range=[float(r["localization_map"].min()),
                                           float(r["localization_map"].max())],
                   reliability_map_range=[float(r["reliability_map"].min()),
                                          float(r["reliability_map"].max())],
                   input_size=r["input_size"],
                   inference_time=round(r["inference_time"], 3))
        recs.append(rec)
        print(f"  {os.path.basename(p):<46} score={r['integrity_score']:.6f} "
              f"{r['inference_time']:.2f}s  size={r['input_size']}")
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    out = dict(adapter="trufor_adapter.py",
               checkpoint=ad.ckpt_path,
               checkpoint_md5=hashlib.md5(open(ad.ckpt_path, "rb").read()).hexdigest(),
               semantics=dict(score=SCORE_SEMANTICS, map=MAP_SEMANTICS,
                              conf=CONF_SEMANTICS),
               preprocessing="PIL convert('RGB') -> CHW float / 256.0; NO resize, "
                             "NO crop, NO normalisation here (imagenet norm is "
                             "applied inside the model)",
               peak_gpu_GB=round(peak, 2), load_time_s=round(ad.load_time, 1),
               results=recs)
    jp = os.path.join(args.out, "trufor_results.json")
    json.dump(out, open(jp, "w"), indent=1)
    print(f"peak GPU {peak:.2f} GB -> {jp}")


if __name__ == "__main__":
    main()
