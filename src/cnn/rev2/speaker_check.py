"""화자 일관성 독립 재검증.

논문은 화자 내/간 거리를 데이터를 만든 Base 모델 자신의 내부 임베딩으로 쟀다.
[21] x-vector 를 인용해 놓고 실제로는 쓰지 않았으므로, 독립 화자 검증 모델
(ECAPA-TDNN, VoxCeleb 학습)로 다시 측정한다.

usage: python speaker_check.py [--per-speaker 40]
"""
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

os.environ.setdefault("HF_HOME", "/data/hye0n/hf")

import common as _c  # noqa: E402
import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/data/hye0n/hf/ecapa",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})

    m = _c.build_manifest(drop_outliers=True)
    rng = np.random.default_rng(args.seed)
    embs = {}
    for sid, g in m.groupby("speaker_id"):
        pick = g.sample(n=min(args.per_speaker, len(g)), random_state=args.seed)
        vecs = []
        for fp in pick.filepath:
            y, sr = sf.read(fp, dtype="float32")
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
            e = enc.encode_batch(wav).squeeze().detach().cpu().numpy()
            vecs.append(e / (np.linalg.norm(e) + 1e-9))
        embs[sid] = np.stack(vecs)
        print(f"  {sid}: {len(vecs)}개")

    within = []
    for sid, V in embs.items():
        d = [1 - float(np.dot(V[i], V[j])) for i, j in itertools.combinations(range(len(V)), 2)]
        within.append(float(np.mean(d)))
    centroids = {s: V.mean(axis=0) / np.linalg.norm(V.mean(axis=0)) for s, V in embs.items()}
    between = [(a, b, 1 - float(np.dot(centroids[a], centroids[b])))
               for a, b in itertools.combinations(centroids, 2)]
    bd = np.array([x[2] for x in between])
    wmean = float(np.mean(within))
    closest = min(between, key=lambda x: x[2])

    res = {"encoder": "speechbrain/spkrec-ecapa-voxceleb",
           "n_speakers": len(embs), "per_speaker": args.per_speaker,
           "within_mean": wmean,
           "between_mean": float(bd.mean()), "between_min": float(bd.min()),
           "between_over_within": float(bd.mean() / wmean) if wmean > 0 else None,
           "closest_pair": {"a": closest[0], "b": closest[1], "distance": closest[2]},
           "note": "논문 §3.4 의 2.29 는 Base 모델 자체 임베딩 기준. 여기 값은 독립 검증 모델 기준."}
    print("\n화자 내 평균 거리 %.4f | 화자 간 평균 %.4f (최소 %.4f) | between/within = %.2f" % (
        wmean, bd.mean(), bd.min(), res["between_over_within"]))
    print("가장 가까운 쌍: %s-%s = %.4f" % (closest[0], closest[1], closest[2]))
    _c.save_json("speaker_check_ecapa.json", res)


if __name__ == "__main__":
    main()
