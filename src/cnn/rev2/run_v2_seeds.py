"""v2 결정성 재학습 드라이버 — 한 (mode, seed) 조합을 학습하고 산출물을 남긴다.

논문 4.4절의 "표마다 다른 실행" 문제를 없애기 위해, v2 계열의 모든 수치를
동일한 결정성 절차·동일 분할에서 시드만 바꿔 재생산한다.

  분할 시드는 42 로 고정하고 학습 시드만 42/43/44 로 바꾼다.
  → 평가셋(v2-HN)이 시드 간 동일하므로 ± 는 순수하게 학습 분산을 뜻한다.

usage:
  python run_v2_seeds.py --mode indist  --seed 42 [--epochs 40]
  python run_v2_seeds.py --mode heldpii --held rrn --seed 42
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402  (rev2/common.py)
from engine import get_device, predict_clip_probs  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from models import build_model  # noqa: E402
from splits import split_manifest  # noqa: E402

SPLIT_SEED = 42


# ---------------------------------------------------------------- 학습 루프
def train_deterministic(model, tr_df, va_df, channels, device, epochs, seed,
                        lr=1e-4, weight_decay=1e-4, batch_size=32, patience=8):
    """engine.train_model 과 동일한 절차에 DataLoader 결정성만 추가한 버전."""
    import copy
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from dataset import AudioPIIDataset

    model.to(device)
    ds = AudioPIIDataset(tr_df, mode="train", channels=channels, seed=seed, augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4,
                        drop_last=False, worker_init_fn=_c.seed_worker,
                        generator=_c.loader_generator(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()

    best_f1, best_state, bad = -1.0, None, 0
    history = []
    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
        sched.step()
        vp = predict_clip_probs(model, va_df, channels, device)
        vf1 = classification_metrics(va_df.label.values, vp)["f1"]
        history.append({"epoch": ep, "train_loss": run_loss / len(ds), "val_f1": vf1})
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_f1": best_f1, "history": history, "epochs_run": len(history)}


# ---------------------------------------------------------------- 지표
def per_pii_recall(df, probs, thr=0.5):
    pred = (np.asarray(probs) >= thr).astype(int)
    out = {}
    for pt in ["phone", "rrn", "account", "card"]:
        m = (df.pii_type.values == pt) & (df.label.values == 1)
        if m.sum():
            out[pt] = {"n": int(m.sum()), "recall": float(pred[m].mean())}
    return out


def per_category_fpr(df, probs, thr=0.5):
    pred = (np.asarray(probs) >= thr).astype(int)
    out = {}
    hard = df.source_type.values == "negative_hard"
    for cat in sorted(set(df.template_id.values[hard])):
        m = hard & (df.template_id.values == cat)
        out[cat] = {"n": int(m.sum()), "fpr": float(pred[m].mean())}
    return out


def prefilter_points(va_df, vp, te_df, tp, targets=(0.99, 0.95, 0.90)):
    """검증 분할 positive 확률의 분위수로 임계값을 정하고 test 에서 평가한다."""
    vpos = np.asarray(vp)[va_df.label.values == 1]
    y, p = te_df.label.values, np.asarray(tp)
    rows = []
    for t in targets:
        thr = float(np.quantile(vpos, 1.0 - t)) if len(vpos) else 0.5
        m = classification_metrics(y, p, thr)
        rows.append({
            "target_recall": t, "threshold": round(thr, 4),
            "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
            "hard_neg_fpr": hard_neg_fpr(p, te_df.source_type.values, thr),
            "stt_call_rate": float((p >= thr).mean()),
        })
    return rows


def calibration(y, p, n_bins=10):
    """ECE + 온도 스케일링 + Platt 스케일링."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)

    def ece(prob):
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.sum():
                e += m.mean() * abs(prob[m].mean() - y[m].mean())
        return float(e)

    logit = np.log(p / (1 - p))
    best_T, best_e = 1.0, ece(p)
    for T in np.arange(0.5, 5.01, 0.05):
        e = ece(1 / (1 + np.exp(-logit / T)))
        if e < best_e:
            best_T, best_e = float(T), e
    platt = LogisticRegression().fit(logit.reshape(-1, 1), y)
    p_platt = platt.predict_proba(logit.reshape(-1, 1))[:, 1]
    return {"ece_raw": ece(p), "temperature": best_T, "ece_temp": best_e,
            "ece_platt": ece(p_platt)}


# ---------------------------------------------------------------- 모드
def run_indist(m, device, epochs, seed, tag):
    tr, va, te, _ = split_manifest(m, seed=SPLIT_SEED)
    model, ch = build_model("simple_cnn")
    info = train_deterministic(model, tr, va, ch, device, epochs, seed)
    vp = predict_clip_probs(model, va, ch, device)
    tp = predict_clip_probs(model, te, ch, device)

    y = te.label.values
    res = classification_metrics(y, tp, 0.5)
    res["hard_neg_fpr@0.5"] = hard_neg_fpr(tp, te.source_type.values, 0.5)
    res["per_pii_recall@0.5"] = per_pii_recall(te, tp, 0.5)
    res["per_category_fpr@0.5"] = per_category_fpr(te, tp, 0.5)
    res["prefilter"] = prefilter_points(va, vp, te, tp)
    res["calibration"] = calibration(y, tp)
    res["split"] = {
        s: {"n": int(len(d)),
            "positive": int((d.source_type == "positive").sum()),
            "hard": int((d.source_type == "negative_hard").sum()),
            "easy": int((d.source_type == "negative_easy").sum())}
        for s, d in [("train", tr), ("val", va), ("test", te)]
    }
    res["train_info"] = {k: info[k] for k in ["best_val_f1", "epochs_run"]}

    ckpt = os.path.join(_c.OUT_DIR, f"ckpt_{tag}.pt")
    os.makedirs(_c.OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    _c.save_scores(f"scores_{tag}_test.csv", te, tp)
    _c.save_scores(f"scores_{tag}_val.csv", va, vp)
    print("[indist %s] F1=%.3f recall=%.3f prec=%.3f hardFPR=%.3f roc=%.3f" % (
        tag, res["f1"], res["recall"], res["precision"], res["hard_neg_fpr@0.5"], res["roc_auc"]))
    return res


def run_loso(m, device, epochs, seed, speaker, tag):
    """화자 한 명을 통째로 test 로 두고 나머지로 학습하는 폴드 하나."""
    te = m[m.speaker_id == speaker].reset_index(drop=True)
    rest = m[m.speaker_id != speaker].reset_index(drop=True)
    tr, va, te2, _ = split_manifest(rest, seed=SPLIT_SEED)
    tr = pd.concat([tr, te2], ignore_index=True)

    model, ch = build_model("simple_cnn")
    info = train_deterministic(model, tr, va, ch, device, epochs, seed)
    vp = predict_clip_probs(model, va, ch, device)
    tp = predict_clip_probs(model, te, ch, device)

    # 검증 분할에서 F1 을 최대화하는 임계값(0.05~0.95, 0.05 간격)
    grid = np.linspace(0.05, 0.95, 19)
    thr = float(grid[int(np.argmax([classification_metrics(va.label.values, vp, t)["f1"]
                                    for t in grid]))])
    y = te.label.values
    m05 = classification_metrics(y, tp, 0.5)
    mt = classification_metrics(y, tp, thr)
    _c.save_scores(f"scores_{tag}.csv", te, tp)
    print("[loso %s %s] thr=%.2f F1@0.5=%.3f F1@tuned=%.3f roc=%.3f" % (
        speaker, tag, thr, m05["f1"], mt["f1"], m05["roc_auc"]))
    return {"speaker": speaker, "n_test": int(len(te)), "thr_tuned": thr,
            "f1@0.5": m05["f1"], "recall@0.5": m05["recall"], "precision@0.5": m05["precision"],
            "roc_auc": m05["roc_auc"], "f1@tuned": mt["f1"], "recall@tuned": mt["recall"],
            "hard_neg_fpr@0.5": hard_neg_fpr(tp, te.source_type.values, 0.5)}


def run_heldcat(m, device, epochs, seed, cat, tag):
    """hard-negative 카테고리 하나를 학습에서 완전히 제외하고 그 카테고리의 오탐률을 잰다.

    템플릿 그룹 분할은 열 개 카테고리 중 세 개만 평가 분할에 남긴다. 그 결과
    카드와 충돌하도록 설계한 제품 일련번호처럼 가장 진단적인 통제가 평가에서 빠진다.
    이 절차는 카테고리마다 '학습에 없던 비PII 숫자열'에 대한 오탐률을 주므로,
    열 카테고리 전부를 누수 없이 평가할 수 있다.
    """
    held = m[(m.source_type == "negative_hard") & (m.template_id == cat)].reset_index(drop=True)
    pool = m[~((m.source_type == "negative_hard") & (m.template_id == cat))].reset_index(drop=True)
    tr, va, te_extra, _ = split_manifest(pool, seed=SPLIT_SEED)
    tr = pd.concat([tr, te_extra], ignore_index=True)

    model, ch = build_model("simple_cnn")
    info = train_deterministic(model, tr, va, ch, device, epochs, seed)
    hp = predict_clip_probs(model, held, ch, device)
    fpr = float((np.asarray(hp) >= 0.5).mean())
    _c.save_scores(f"scores_{tag}.csv", held, hp)
    print("[heldcat %s %s] n=%d FPR@0.5=%.3f" % (cat, tag, len(held), fpr))
    return {"category": cat, "n": int(len(held)), "held_fpr@0.5": fpr,
            "mean_prob": float(np.mean(hp)),
            "best_val_f1": info["best_val_f1"], "epochs_run": info["epochs_run"]}


def run_heldpii(m, device, epochs, seed, held, tag):
    neg, pos = m[m.label == 0], m[m.label == 1]
    train_pool = pd.concat([neg, pos[pos.pii_type != held]], ignore_index=True)
    tr, va, te_extra, _ = split_manifest(train_pool, seed=SPLIT_SEED)
    tr = pd.concat([tr, te_extra], ignore_index=True)
    test = pos[pos.pii_type == held].reset_index(drop=True)

    model, ch = build_model("simple_cnn")
    info = train_deterministic(model, tr, va, ch, device, epochs, seed)
    tp = predict_clip_probs(model, test, ch, device)
    recall = float((np.asarray(tp) >= 0.5).mean())
    _c.save_scores(f"scores_{tag}.csv", test, tp)
    print("[heldpii %s %s] n=%d recall@0.5=%.3f" % (held, tag, len(test), recall))
    return {"held": held, "n": int(len(test)), "held_recall@0.5": recall,
            "best_val_f1": info["best_val_f1"], "epochs_run": info["epochs_run"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["indist", "heldpii", "heldcat", "loso"])
    ap.add_argument("--held", default="rrn")
    ap.add_argument("--cat", default="serial")
    ap.add_argument("--speaker", default="spk01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--data", default=None, help="기본은 data_qwen (v3 실험 시 경로 지정)")
    ap.add_argument("--suffix", default="", help="산출물 파일명 접미사 (예: _v3)")
    ap.add_argument("--keep-outliers", action="store_true",
                    help="655초 반복 생성 클립을 남긴다(논문 기존 분할 재현용)")
    args = ap.parse_args()

    _c.set_determinism(args.seed)
    device = get_device()
    m = _c.build_manifest(root=args.data, drop_outliers=not args.keep_outliers)
    print("manifest n=%d | speakers=%d | %s" % (
        len(m), m.speaker_id.nunique(), dict(m.source_type.value_counts())))

    if args.mode == "indist":
        tag = f"indist{args.suffix}_s{args.seed}"
        res = run_indist(m, device, args.epochs, args.seed, tag)
    elif args.mode == "heldcat":
        tag = f"heldcat{args.suffix}_{args.cat}_s{args.seed}"
        res = run_heldcat(m, device, args.epochs, args.seed, args.cat, tag)
    elif args.mode == "loso":
        tag = f"loso{args.suffix}_{args.speaker}_s{args.seed}"
        res = run_loso(m, device, args.epochs, args.seed, args.speaker, tag)
    else:
        tag = f"heldpii{args.suffix}_{args.held}_s{args.seed}"
        res = run_heldpii(m, device, args.epochs, args.seed, args.held, tag)

    res["_meta"] = {"mode": args.mode, "seed": args.seed, "split_seed": SPLIT_SEED,
                    "epochs": args.epochs, "data": args.data or "data_qwen",
                    "keep_outliers": bool(args.keep_outliers), "n_manifest": int(len(m))}
    _c.save_json(f"{tag}.json", res)


if __name__ == "__main__":
    main()
