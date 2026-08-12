"""feedback2 대응 신규 그림 — rev2/out/*.json 실측치 기반. 출력: paper/figs/*.pdf"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
FIGS = os.path.join(os.path.dirname(HERE), "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

# 본문이 한국어이므로 축 라벨·범례도 한국어로 쓴다. 폰트가 없으면 경고만 남긴다.
from matplotlib import font_manager as _fm  # noqa: E402
for _cand in ("UnDotum", "Baekmuk Dotum", "NanumGothic"):
    if any(f.name == _cand for f in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _cand
        break
else:
    print("경고: 한글 폰트를 찾지 못했다. 라벨이 깨질 수 있다.")
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C_MAIN, C_ALT, C_BAD, C_GREY = "#2b6cb0", "#dd8452", "#c44e52", "#8c8c8c"


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", name)


# ---- 자릿수 길이 대 점수 곡선 ----
def fig_digit_curve():
    d = load("probe_curve.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    style = [("continuous/sentence", C_MAIN, "o", "-", "연속 자릿수열 (문장 문틀)"),
             ("grouped4/sentence", C_ALT, "s", "--", "4자리 그룹 (4-4-4-4)"),
             ("continuous/bare", C_GREY, "^", ":", "연속 자릿수열 (문틀 없음)")]
    for key, col, mk, ls, lab in style:
        rows = d["curves"].get(key)
        if not rows:
            continue
        x = [r["n_digits"] for r in rows]
        y = [r["rate_ge_0.5"] for r in rows]
        e = [r["seed_std"] for r in rows]
        ax.errorbar(x, y, yerr=e, marker=mk, ls=ls, color=col, capsize=3, label=lab)
    fit = d["curves"].get("continuous/sentence/fit", {})
    if "x50" in fit:
        ax.axvline(fit["x50"], color=C_BAD, lw=1.2, ls="-.")
        # 주석은 곡선이 지나지 않는 우하단에 두고, 범례는 비어 있는 좌상단으로 옮긴다.
        # (예전에는 둘 다 우하단이라 범례가 주석을 덮었다.)
        ax.text(fit["x50"] + 0.3, 0.04, f"반포화점 $\\approx$ {fit['x50']:.1f}자리",
                color=C_BAD, fontsize=9)
    ax.set_xlabel("총 자릿수"); ax.set_ylabel("판정 비율 $P(\\hat{p} \\geq 0.5)$")
    ax.set_ylim(-0.03, 1.14); ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    save(fig, "digit_curve.pdf")


# ---- 카테고리별 미학습 오탐률 ----
def fig_heldcat():
    d = load("heldcat_summary.json")
    if not d:
        return
    rows = sorted(d["categories"], key=lambda r: r["total_digits"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = [C_BAD if r["fpr"]["mean"] >= 0.5 else C_MAIN for r in rows]
    ax.bar(x, [r["fpr"]["mean"] for r in rows], yerr=[r["fpr"]["std"] for r in rows],
           capsize=3, color=cols)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['category']}\n({r['total_digits']}d)" for r in rows],
                       fontsize=8)
    ax.set_ylabel("unseen-category FPR"); ax.set_ylim(0, 1.05)
    save(fig, "heldcat_fpr.pdf")


# ---- 열화 조건 ----
def fig_degrade():
    """degrade_s4*.json(기본) / degrade_na_s4*.json(잡음 증강) / whisper_configs.json 에서 직접 집계."""
    import glob
    def agg(pat):
        rows = {}
        for f in sorted(glob.glob(os.path.join(OUT, pat))):
            for r in json.load(open(f))["rows"]:
                rows.setdefault(r["condition"], []).append(r)
        return {c: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                    for k in ["f1", "recall"]} for c, rs in rows.items()}
    base, na = agg("degrade_s4*.json"), agg("degrade_na_s4*.json")
    w = load("whisper_configs.json")
    if not base or not na or not w:
        return
    conds = ["g711", "g711_bp20", "g711_bp10", "g711_bp05"]
    lbl = ["G.711\n(no noise)", "+20 dB", "+10 dB", "+5 dB"]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.errorbar(x, [base[c]["recall"][0] for c in conds], yerr=[base[c]["recall"][1] for c in conds],
                marker="o", ls="-", color=C_MAIN, capsize=3, label="CNN baseline")
    ax.errorbar(x, [na[c]["recall"][0] for c in conds], yerr=[na[c]["recall"][1] for c in conds],
                marker="s", ls="--", color=C_ALT, capsize=3, label="CNN + noise aug.")
    ax.plot(x, [w["conditions"][c]["default"]["format_context"]["recall"] for c in conds],
            marker="^", ls=":", color=C_GREY, label="STT+NLP")
    fb = ["g711", "g711_snr20", "g711_snr10"]          # 전대역 잡음(참고, 5 dB 조건 없음)
    ax.plot([0, 1, 2], [base[c]["recall"][0] for c in fb], marker="x", ls="-.", color=C_BAD,
            alpha=0.7, label="CNN, full-band noise (unrealistic)")
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=9)
    ax.set_xlabel("telephone passband noise (in-band SNR)"); ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.12); ax.legend(fontsize=8, ncol=2, loc="lower left")
    save(fig, "degradation.pdf")


# ---- LOSO: v1(3화자) 대 v2(14화자) ----
def fig_loso():
    d = load("loso_summary.json")
    if not d:
        return
    old = json.load(open(os.path.join(os.path.dirname(HERE), "runs",
                                      "loso_simple_cnn.json")))["aggregate"]
    labels = ["F1@0.5", "Recall@0.5", "ROC-AUC"]
    v1m = [old["test_f1"]["mean"], old["test_recall"]["mean"], old["test_roc_auc"]["mean"]]
    v1s = [old["test_f1"]["std"], old["test_recall"]["std"], old["test_roc_auc"]["std"]]
    a = d["aggregate"]
    v2m = [a["f1@0.5"]["mean"], a["recall@0.5"]["mean"], a["roc_auc"]["mean"]]
    v2s = [a["f1@0.5"]["std"], a["recall@0.5"]["std"], a["roc_auc"]["std"]]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(x - w / 2, v1m, w, yerr=v1s, capsize=4, color=C_GREY, label="v1 (화자 3명)")
    ax.bar(x + w / 2, v2m, w, yerr=v2s, capsize=4, color=C_MAIN, label="v2 (화자 14명)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("점수"); ax.set_ylim(0, 1.32); ax.legend(fontsize=9)
    save(fig, "loso_v1_v2.pdf")

    per = d["per_speaker"]
    sp = sorted(per, key=lambda s: per[s]["f1@0.5"])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(np.arange(len(sp)), [per[s]["f1@0.5"] for s in sp], color=C_MAIN)
    ax.axhline(a["f1@0.5"]["mean"], color=C_BAD, ls="--", lw=1.2,
               label=f"평균 {a['f1@0.5']['mean']:.3f}")
    ax.set_xticks(np.arange(len(sp)))
    ax.set_xticklabels(sp, rotation=60, fontsize=8)
    ax.set_xlabel("화자 폴드 (F1 오름차순)", fontsize=9)
    ax.set_ylabel("F1@0.5"); ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    save(fig, "loso_folds.pdf")


# ---- CNN 대 STT+NLP ----
def fig_vs():
    w = load("whisper_configs.json")
    a = load("agg.json")
    if not (w and a):
        return
    clean = w["conditions"].get("clean", {})
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.scatter([a["indist"]["recall"]["mean"]], [a["indist"]["precision"]["mean"]],
               s=90, color=C_MAIN, zorder=3, label="CNN (acoustic)")
    ax.errorbar([a["indist"]["recall"]["mean"]], [a["indist"]["precision"]["mean"]],
                xerr=[a["indist"]["recall"]["std"]], yerr=[a["indist"]["precision"]["std"]],
                color=C_MAIN, capsize=3, zorder=2)
    mk = ["o", "s", "^", "D"]
    for i, (cn, b) in enumerate(clean.items()):
        f = b["format_context"]
        ax.scatter([f["recall"]], [f["precision"]], s=70, marker=mk[i % 4],
                   color=C_ALT, zorder=3, label=f"STT+NLP ({cn})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0.6, 1.03); ax.set_ylim(0.6, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "cnn_vs_sttnlp.pdf")


if __name__ == "__main__":
    fig_digit_curve()
    fig_heldcat()
    fig_degrade()
    fig_loso()
    fig_vs()
