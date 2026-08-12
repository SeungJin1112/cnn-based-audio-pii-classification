"""자릿수 프로브셋 채점 — 모델 점수 대 자릿수 길이/그룹 구조 곡선.

data_probe 의 통제 자극(PII 의미 없는 순수 난수 자릿수열)에 학습된 모델을 적용해
결정 함수가 무엇의 함수인지(총 자릿수 / 최장 연속 런 / 문틀)를 특정한다.

usage:
  python probe_eval.py --ckpt out/ckpt_indist_s42.pt --ckpt out/ckpt_indist_s43.pt ...
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402
from engine import get_device, predict_clip_probs  # noqa: E402
from models import build_model  # noqa: E402

PROBE_DIR = os.path.join(os.path.dirname(HERE), "data_probe")


def load_probe():
    df = pd.read_csv(os.path.join(PROBE_DIR, "metadata.csv"))
    df["filepath"] = [os.path.normpath(os.path.join(PROBE_DIR, str(fp).replace("\\", "/")))
                      for fp in df["filepath"]]
    df["label"] = 1                      # 사용하지 않지만 Dataset 인터페이스가 요구
    return df


def logistic_fit(x, y):
    """점수 곡선에 로지스틱을 적합해 반포화점(결정 경계 자릿수)을 추정한다."""
    from scipy.optimize import curve_fit

    def f(t, x0, k):
        return 1.0 / (1.0 + np.exp(-k * (t - x0)))

    try:
        popt, _ = curve_fit(f, np.asarray(x, float), np.asarray(y, float),
                            p0=[10.0, 1.0], maxfev=20000)
        return {"x50": float(popt[0]), "slope": float(popt[1])}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--model", default="simple_cnn",
                    help="체크포인트를 만든 구조. 순서 헤드 변형 채점 시 지정한다.")
    ap.add_argument("--out", default="probe_curve.json")
    args = ap.parse_args()

    _c.set_determinism(42)
    if "gru" in args.model:
        torch.backends.cudnn.enabled = False
    device = get_device()
    df = load_probe()
    print("probe n=%d | axis=%s" % (len(df), dict(df.axis.value_counts())))

    all_probs = []
    for ck in args.ckpt:
        path = ck if os.path.isabs(ck) else os.path.join(HERE, ck)
        model, ch = build_model(args.model)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(device).eval()
        p = predict_clip_probs(model, df, ch, device)
        all_probs.append(np.asarray(p))
        print(f"  scored with {os.path.basename(path)}: mean={p.mean():.3f}")

    P = np.stack(all_probs)                      # [n_ckpt, n_clip]
    df["prob_mean"] = P.mean(axis=0)
    df["prob_std"] = P.std(axis=0)
    for i, ck in enumerate(args.ckpt):
        df[f"prob_{os.path.basename(ck).replace('.pt', '')}"] = P[i]
    _c.save_scores(args.out.replace(".json", "_scores.csv"), df, df["prob_mean"].values)

    res = {"n": int(len(df)), "model": args.model, "ckpts": args.ckpt, "curves": {}}
    for (axis, frame), g in df.groupby(["axis", "frame"]):
        key = f"{axis}/{frame}"
        rows = []
        for n, gg in g.groupby("n_digits"):
            rows.append({
                "n_digits": int(n), "n": int(len(gg)),
                "mean_prob": float(gg.prob_mean.mean()),
                "median_prob": float(gg.prob_mean.median()),
                "rate_ge_0.5": float((gg.prob_mean >= 0.5).mean()),
                "seed_std": float(gg.prob_std.mean()),
            })
        res["curves"][key] = rows
        xs = [r["n_digits"] for r in rows]
        ys = [r["rate_ge_0.5"] for r in rows]
        res["curves"][key + "/fit"] = logistic_fit(xs, ys)
        print(f"\n== {key} ==")
        print(pd.DataFrame(rows).to_string(index=False))
        print("  로지스틱 적합:", res["curves"][key + "/fit"])

    # 총 자릿수를 맞추고 그룹 구조만 바꾼 대조 (연속 N vs 4자리 그룹 N)
    cmp_rows = []
    cont = df[(df.axis == "continuous") & (df.frame == "sentence")]
    grp = df[df.axis == "grouped4"]
    for n in sorted(set(grp.n_digits)):
        a = cont[cont.n_digits == n]
        b = grp[grp.n_digits == n]
        if len(a) and len(b):
            cmp_rows.append({"n_digits": int(n),
                             "continuous_rate": float((a.prob_mean >= 0.5).mean()),
                             "grouped4_rate": float((b.prob_mean >= 0.5).mean()),
                             "continuous_mean": float(a.prob_mean.mean()),
                             "grouped4_mean": float(b.prob_mean.mean())})
    res["grouping_contrast"] = cmp_rows
    print("\n== 총 자릿수 고정, 그룹 구조 대조 ==")
    print(pd.DataFrame(cmp_rows).to_string(index=False))

    # 길이와 자릿수의 분리: 문틀이 없으면 같은 자릿수라도 클립이 1초가량 짧아지므로,
    # '맨숫자 N자리'와 '문장 M자리'가 길이는 비슷하고 자릿수는 다른 쌍이 생긴다.
    dur = df.groupby(["frame", "n_digits"]).agg(
        dur=("duration", "mean"), rate=("prob_mean", lambda s: float((s >= 0.5).mean())),
        prob=("prob_mean", "mean")).reset_index()
    res["duration_vs_digits"] = dur.to_dict("records")
    pairs = []
    bare = dur[dur.frame == "bare"]
    sent = dur[dur.frame == "sentence"]
    for _, b in bare.iterrows():
        j = (sent.dur - b.dur).abs().idxmin()
        srow = sent.loc[j]
        if abs(srow.dur - b.dur) < 0.25 and srow.n_digits != b.n_digits:
            pairs.append({"bare_digits": int(b.n_digits), "bare_dur": float(b.dur),
                          "bare_rate": float(b.rate),
                          "sent_digits": int(srow.n_digits), "sent_dur": float(srow.dur),
                          "sent_rate": float(srow.rate)})
    res["duration_matched_pairs"] = pairs
    print("\n== 길이는 비슷하고 자릿수가 다른 쌍 (길이 대 자릿수 분리) ==")
    if pairs:
        print(pd.DataFrame(pairs).to_string(index=False))
    else:
        print("  (0.25초 이내로 짝지어지는 쌍 없음)")

    # 회귀로도 확인: 자릿수와 길이를 함께 넣었을 때 각각의 기여
    try:
        import numpy as _np
        from sklearn.linear_model import LogisticRegression
        X = _np.column_stack([df.n_digits.values, df.duration.values])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = (df.prob_mean.values >= 0.5).astype(int)
        if len(set(y)) == 2:
            lr = LogisticRegression(max_iter=2000).fit(X, y)
            res["logit_coef"] = {"n_digits": float(lr.coef_[0][0]),
                                 "duration": float(lr.coef_[0][1])}
            print("\n표준화 로지스틱 계수 — 자릿수 %.3f / 길이 %.3f"
                  % (lr.coef_[0][0], lr.coef_[0][1]))
    except Exception as e:
        print("회귀 실패:", e)

    _c.save_json(args.out, res)


if __name__ == "__main__":
    main()
