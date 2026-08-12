"""저장된 클립별 점수에 대한 사후 분석.

  agg        3시드 in-dist / held-out-PII 결과를 평균±표준편차로 집계
  carrier    캐리어 유무별 조건부 오탐률 (남아 있던 confound 의 직접 검정)
  digits     자릿수 구조(총 자릿수·최장 런) 대 점수 — 관측 기반 곡선
  bootstrap  템플릿 그룹 부트스트랩 신뢰구간 (지배적 불확실성 축)

usage: python analyze.py --what agg carrier digits bootstrap [--suffix ""]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402


def _ms(vals):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


# ---------------------------------------------------------------- agg
def agg(suffix):
    out = {}
    files = sorted(glob.glob(os.path.join(_c.OUT_DIR, f"indist{suffix}_s*.json")))
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    if runs:
        keys = ["f1", "recall", "precision", "roc_auc", "pr_auc", "hard_neg_fpr@0.5"]
        out["indist"] = {"seeds": [r["_meta"]["seed"] for r in runs],
                         **{k: _ms([r[k] for r in runs]) for k in keys}}
        out["indist"]["per_pii_recall"] = {
            pt: _ms([r["per_pii_recall@0.5"][pt]["recall"] for r in runs
                     if pt in r["per_pii_recall@0.5"]])
            for pt in ["phone", "rrn", "account", "card"]}
        cats = sorted({c for r in runs for c in r["per_category_fpr@0.5"]})
        out["indist"]["per_category_fpr"] = {
            c: {"n": runs[0]["per_category_fpr@0.5"][c]["n"],
                **_ms([r["per_category_fpr@0.5"][c]["fpr"] for r in runs
                       if c in r["per_category_fpr@0.5"]])}
            for c in cats}
        tgts = [row["target_recall"] for row in runs[0]["prefilter"]]
        out["indist"]["prefilter"] = [
            {"target_recall": t,
             **{k: _ms([row[k] for r in runs for row in r["prefilter"]
                        if row["target_recall"] == t])
                for k in ["threshold", "recall", "precision", "hard_neg_fpr", "stt_call_rate"]}}
            for t in tgts]
        out["indist"]["calibration"] = {
            k: _ms([r["calibration"][k] for r in runs])
            for k in ["ece_raw", "temperature", "ece_temp", "ece_platt"]}
        out["indist"]["split"] = runs[0]["split"]
        out["indist"]["epochs_run"] = _ms([r["train_info"]["epochs_run"] for r in runs])

    held = {}
    for f in sorted(glob.glob(os.path.join(_c.OUT_DIR, f"heldpii{suffix}_*_s*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        held.setdefault(r["held"], []).append(r["held_recall@0.5"])
    if held:
        out["heldpii"] = {k: {"n": None, **_ms(v)} for k, v in held.items()}

    _c.save_json(f"agg{suffix}.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False)[:4000])
    return out


# ---------------------------------------------------------------- carrier
def carrier(suffix):
    rows = []
    for f in sorted(glob.glob(os.path.join(_c.OUT_DIR, f"scores_indist{suffix}_s*_test.csv"))):
        df = pd.read_csv(f)
        df = _c.digit_features(df)
        seed = os.path.basename(f).split("_s")[-1].split("_")[0]
        hard = df[df.source_type == "negative_hard"]
        pos = df[df.source_type == "positive"]
        for grp, sub in [("hard_carrier", hard[hard.has_carrier]),
                         ("hard_no_carrier", hard[~hard.has_carrier]),
                         ("pos_carrier", pos[pos.has_carrier]),
                         ("pos_no_carrier", pos[~pos.has_carrier])]:
            if len(sub) == 0:
                continue
            rows.append({"seed": seed, "group": grp, "n": len(sub),
                         "rate_ge_0.5": float((sub.prob >= 0.5).mean()),
                         "mean_prob": float(sub.prob.mean())})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    summ = (t.groupby("group").agg(n=("n", "first"),
                                   rate_mean=("rate_ge_0.5", "mean"),
                                   rate_std=("rate_ge_0.5", "std"),
                                   prob_mean=("mean_prob", "mean"))
            .reset_index().to_dict("records"))
    print("\n== 캐리어 유무별 집계 ==")
    print(pd.DataFrame(summ).to_string(index=False))
    _c.save_json(f"carrier_conditional{suffix}.json",
                 {"per_seed": rows, "summary": summ})
    return summ


# ---------------------------------------------------------------- digits
def digits(suffix):
    frames = []
    for f in sorted(glob.glob(os.path.join(_c.OUT_DIR, f"scores_indist{suffix}_s*_test.csv"))):
        df = _c.digit_features(pd.read_csv(f))
        df["seed"] = os.path.basename(f).split("_s")[-1].split("_")[0]
        frames.append(df)
    if not frames:
        print("점수 파일이 없다")
        return None
    df = pd.concat(frames, ignore_index=True)

    by_total = (df.groupby("total_digits")
                  .agg(n=("prob", "size"), mean_prob=("prob", "mean"),
                       rate=("prob", lambda s: float((s >= 0.5).mean())),
                       label=("label", "mean"))
                  .reset_index())
    by_run = (df.groupby("max_run")
                .agg(n=("prob", "size"), mean_prob=("prob", "mean"),
                     rate=("prob", lambda s: float((s >= 0.5).mean())),
                     label=("label", "mean"))
                .reset_index())
    print("== 총 자릿수별 ==")
    print(by_total.to_string(index=False))
    print("\n== 최장 연속 런별 ==")
    print(by_run.to_string(index=False))

    # 어느 변수가 점수를 더 잘 설명하는가 — 단변량 상관과 AUC
    from sklearn.metrics import roc_auc_score
    y = df.label.values
    res = {"by_total_digits": by_total.to_dict("records"),
           "by_max_run": by_run.to_dict("records"),
           "spearman": {
               "total_digits": float(pd.Series(df.total_digits).corr(df.prob, method="spearman")),
               "max_run": float(pd.Series(df.max_run).corr(df.prob, method="spearman")),
               "duration": float(pd.Series(df.duration).corr(df.prob, method="spearman"))},
           "auc_as_predictor": {
               "total_digits": float(roc_auc_score(y, df.total_digits)),
               "max_run": float(roc_auc_score(y, df.max_run)),
               "duration": float(roc_auc_score(y, df.duration)),
               "cnn_prob": float(roc_auc_score(y, df.prob))}}
    print("\n== 점수와의 스피어만 상관 ==", res["spearman"])
    print("== 라벨 예측력(AUC) ==", res["auc_as_predictor"])
    _c.save_json(f"digit_structure{suffix}.json", res)
    return res


# ---------------------------------------------------------------- bootstrap
def bootstrap(suffix, n_boot=2000, seed=0):
    """템플릿 그룹 단위 재표집으로 주요 지표의 95% 구간을 낸다."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in sorted(glob.glob(os.path.join(_c.OUT_DIR, f"scores_indist{suffix}_s*_test.csv"))):
        df = pd.read_csv(f)
        tag = os.path.basename(f)
        groups = df.template_id.unique()
        stats = {"f1": [], "precision": [], "recall": [], "hard_neg_fpr": []}
        for _ in range(n_boot):
            pick = rng.choice(groups, size=len(groups), replace=True)
            sub = pd.concat([df[df.template_id == g] for g in pick], ignore_index=True)
            if sub.label.nunique() < 2:
                continue
            m = classification_metrics(sub.label.values, sub.prob.values, 0.5)
            stats["f1"].append(m["f1"])
            stats["precision"].append(m["precision"])
            stats["recall"].append(m["recall"])
            stats["hard_neg_fpr"].append(hard_neg_fpr(sub.prob.values, sub.source_type.values, 0.5))
        out[tag] = {k: {"mean": float(np.nanmean(v)),
                        "lo95": float(np.nanpercentile(v, 2.5)),
                        "hi95": float(np.nanpercentile(v, 97.5))}
                    for k, v in stats.items()}
        print(f"== {tag} ==")
        print(json.dumps(out[tag], indent=2))
    _c.save_json(f"bootstrap_template{suffix}.json", out)
    return out


# ---------------------------------------------------------------- calib
def _ece(y, prob, n_bins=10):
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(e)


def calib(suffix):
    """보정 파라미터를 검증 분할에서 적합하고 test 에서 평가한다.

    run_v2_seeds 안의 즉석 계산은 test 에서 온도를 골랐으므로 낙관적이다.
    저장된 클립별 점수만으로 재학습 없이 바로잡을 수 있다.
    """
    from sklearn.linear_model import LogisticRegression
    rows = []
    for f in sorted(glob.glob(os.path.join(_c.OUT_DIR, f"scores_indist{suffix}_s*_test.csv"))):
        seed = os.path.basename(f).split("_s")[-1].split("_")[0]
        vf = f.replace("_test.csv", "_val.csv")
        if not os.path.exists(vf):
            continue
        va, te = pd.read_csv(vf), pd.read_csv(f)
        clip = lambda p: np.clip(p.astype(float), 1e-6, 1 - 1e-6)  # noqa: E731
        lv, lt = np.log(clip(va.prob) / (1 - clip(va.prob))), np.log(clip(te.prob) / (1 - clip(te.prob)))
        # 온도: 검증 NLL 최소화
        best_T, best_nll = 1.0, np.inf
        for T in np.arange(0.3, 5.01, 0.01):
            p = 1 / (1 + np.exp(-lv / T))
            nll = -np.mean(va.label * np.log(p + 1e-12) + (1 - va.label) * np.log(1 - p + 1e-12))
            if nll < best_nll:
                best_T, best_nll = float(T), nll
        pl = LogisticRegression().fit(lv.values.reshape(-1, 1), va.label.values)
        rows.append({
            "seed": seed,
            "ece_raw_test": _ece(te.label, te.prob),
            "temperature": best_T,
            "ece_temp_test": _ece(te.label, 1 / (1 + np.exp(-lt / best_T))),
            "ece_platt_test": _ece(te.label, pl.predict_proba(lt.values.reshape(-1, 1))[:, 1]),
            "ece_raw_val": _ece(va.label, va.prob),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    summ = {k: _ms(t[k]) for k in t.columns if k != "seed"}
    print("\n== 집계 ==")
    for k, v in summ.items():
        print(f"  {k:16s} {v['mean']:.4f} ± {v['std']:.4f}")
    _c.save_json(f"calibration_val_fit{suffix}.json", {"per_seed": rows, "summary": summ})
    return summ


# ---------------------------------------------------------------- heldcat
def heldcat(suffix):
    """카테고리별 미학습 오탐률을 자릿수 구조와 함께 정리한다."""
    m = _c.digit_features(_c.build_manifest(drop_outliers=True))
    struct = (m[m.source_type == "negative_hard"].groupby("template_id")
              .agg(total_digits=("total_digits", "median"),
                   max_run=("max_run", "median"),
                   n_groups=("n_groups", "median"),
                   runs=("runs", lambda s: s.mode().iloc[0])).to_dict("index"))

    per = {}
    for f in sorted(glob.glob(os.path.join(_c.OUT_DIR, f"heldcat{suffix}_*_s*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        per.setdefault(r["category"], {"n": r["n"], "fprs": [], "probs": []})
        per[r["category"]]["fprs"].append(r["held_fpr@0.5"])
        per[r["category"]]["probs"].append(r["mean_prob"])

    rows = []
    for cat, d in per.items():
        s = struct.get(cat, {})
        rows.append({"category": cat, "n": d["n"],
                     "total_digits": int(s.get("total_digits", 0)),
                     "max_run": int(s.get("max_run", 0)),
                     "runs": s.get("runs", ""),
                     "fpr": _ms(d["fprs"]), "mean_prob": _ms(d["probs"])})
    rows.sort(key=lambda r: r["total_digits"])
    flat = pd.DataFrame([{"category": r["category"], "n": r["n"],
                          "총자릿수": r["total_digits"], "최장런": r["max_run"],
                          "구조": r["runs"], "FPR": r["fpr"]["mean"],
                          "std": r["fpr"]["std"], "시드": r["fpr"]["n"]} for r in rows])
    print(flat.to_string(index=False))

    # 어떤 구조 변수가 카테고리별 오탐률을 설명하는가
    from scipy.stats import spearmanr
    y = [r["fpr"]["mean"] for r in rows]
    corr = {"total_digits": float(spearmanr([r["total_digits"] for r in rows], y).statistic),
            "max_run": float(spearmanr([r["max_run"] for r in rows], y).statistic)}
    print("\n오탐률과의 스피어만 상관:", corr)
    _c.save_json(f"heldcat_summary{suffix}.json", {"categories": rows, "spearman": corr})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+",
                    default=["agg", "carrier", "digits", "bootstrap", "heldcat"])
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    fn = {"agg": agg, "carrier": carrier, "digits": digits, "bootstrap": bootstrap,
          "heldcat": heldcat, "calib": calib}
    for w in args.what:
        print(f"\n{'=' * 30} {w} {'=' * 30}")
        fn[w](args.suffix)


if __name__ == "__main__":
    main()
