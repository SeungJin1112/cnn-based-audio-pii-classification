"""논문 그림 생성 — runs/*.json 실측치 기반. 출력: paper/figs/*.pdf (벡터).

주의(2026-08-11): 2차 피드백 반영으로 v2 계열 실험을 재실행했으므로
`loso_v1_v2.pdf`, `loso_folds.pdf`, `cnn_vs_sttnlp.pdf`는 이 스크립트가 아니라
`src/cnn/rev2/make_figs2.py`가 생성한 것이 최신이다. `precision_by_category.pdf`는
카테고리별 미학습 오탐률 그림(`heldcat_fpr.pdf`)으로 대체되어 더 이상 쓰지 않는다.
이 스크립트를 다시 돌리면 최신 그림을 덮어쓰므로, 재생성이 필요하면 `make_figs2.py`를
이어서 실행할 것.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
# colorblind-safe
C_MAIN, C_ALT, C_BAD, C_GREY = "#2b6cb0", "#dd8452", "#c44e52", "#8c8c8c"


def load(name):
    return json.load(open(os.path.join(RUNS, name)))


# ---- Fig 1: LOSO v1 vs v2 (speaker generalization stabilizes) ----
def fig_loso():
    v1 = load("loso_simple_cnn.json")["aggregate"]
    v2 = load("qwen_experiment.json")["loso"]["aggregate"]
    labels = ["F1@0.5", "Recall@0.5", "ROC-AUC"]
    v1m = [v1["test_f1"]["mean"], v1["test_recall"]["mean"], v1["test_roc_auc"]["mean"]]
    v1s = [v1["test_f1"]["std"], v1["test_recall"]["std"], v1["test_roc_auc"]["std"]]
    v2m = [v2["f1@0.5"]["mean"], v2["recall@0.5"]["mean"], v2["roc"]["mean"]]
    v2s = [v2["f1@0.5"]["std"], v2["recall@0.5"]["std"], v2["roc"]["std"]]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(x - w/2, v1m, w, yerr=v1s, capsize=4, color=C_GREY, label="v1 (3 speakers)")
    ax.bar(x + w/2, v2m, w, yerr=v2s, capsize=4, color=C_MAIN, label="v2 (14 speakers)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.32)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    for i, (m, s) in enumerate(zip(v1m, v1s)):
        ax.text(i - w/2, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    for i, (m, s) in enumerate(zip(v2m, v2s)):
        ax.text(i + w/2, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIGS, "loso_v1_v2.pdf")); plt.close(fig)


# ---- Fig 2: per-fold v2 LOSO F1 (14 unseen speakers) ----
def fig_loso_folds():
    folds = load("qwen_experiment.json")["loso"]["folds"]
    agg = load("qwen_experiment.json")["loso"]["aggregate"]["f1@0.5"]
    names = [f["held"].replace("spk", "S") for f in folds]
    f1 = [f["f1@0.5"] for f in folds]
    order = np.argsort(f1)
    names = [names[i] for i in order]; f1 = [f1[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cols = [C_BAD if v < 0.7 else C_MAIN for v in f1]
    ax.bar(names, f1, color=cols)
    ax.axhline(agg["mean"], color="k", ls="--", lw=1,
               label=f"mean {agg['mean']:.2f} ± {agg['std']:.2f}")
    ax.set_ylabel("F1@0.5 (held-out speaker)"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(axis="x", labelrotation=0)
    fig.savefig(os.path.join(FIGS, "loso_folds.pdf")); plt.close(fig)


# ---- Fig 3: hard-neg FPR by category (precision diagnosis) ----
def fig_precision():
    d = load("precision_diagnosis.json")["hard_neg_fpr_by_category"]
    # 라벨 주의: count 카테고리는 "하나 둘 셋..." 고유어 수사로, 아라비아 숫자가 없다.
    # (이전 라벨 "count 9-11d" 는 오기였음)
    cats = {"seat": "seat no.\n(short)", "count": "counting\n(no digits)",
            "order": "order no.\n(9-11 digits)"}
    ks = ["seat", "count", "order"]
    fpr = [d[k]["fpr"] for k in ks]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    cols = [C_MAIN, C_ALT, C_BAD]
    bars = ax.bar([cats[k] for k in ks], fpr, color=cols)
    ax.set_ylabel("Hard-neg FPR @0.5"); ax.set_ylim(0, 0.85)
    ax.tick_params(axis="x", labelsize=9)
    for b, v in zip(bars, fpr):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("Hard-negative FPR by category", fontsize=9)
    fig.savefig(os.path.join(FIGS, "precision_by_category.pdf")); plt.close(fig)


# ---- Fig 4: CNN vs STT+NLP tradeoff ----
def fig_vs():
    d = load("stt_nlp_eval.json")
    rows = [
        ("CNN (acoustic)", d["cnn_indist"]["recall"], d["cnn_indist"]["precision"], C_MAIN),
        ("STT+NLP (format only)", d["stt_nlp_format_only"]["recall"], d["stt_nlp_format_only"]["precision"], C_GREY),
        ("STT+NLP (format+context)", d["stt_nlp_format_context"]["recall"], d["stt_nlp_format_context"]["precision"], C_ALT),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for name, r, p, c in rows:
        ax.scatter(r, p, s=140, color=c, zorder=3, edgecolor="k", linewidth=0.6)
        ax.annotate(name, (r, p), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0.85, 1.02); ax.set_ylim(0.65, 1.0)
    ax.set_title("CNN: high recall / STT+NLP+context: high precision", fontsize=9)
    fig.savefig(os.path.join(FIGS, "cnn_vs_sttnlp.pdf")); plt.close(fig)


# ---- Fig 5: efficiency (params vs F1) ----
def fig_efficiency():
    models = [
        ("simple_cnn", 60706, 0.955, C_MAIN),
        ("mobilenet_v3_small", 1519906, 0.776, C_GREY),
        ("efficientnet_b0", 4010110, 0.867, C_GREY),
        ("convnext_tiny", 27821666, 0.804, C_GREY),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, p, f1, c in models:
        ax.scatter(p, f1, s=160, color=c, zorder=3, edgecolor="k", linewidth=0.6)
        dy = 0.012 if name != "convnext_tiny" else -0.028
        ax.annotate(name, (p, f1), xytext=(0, 8 if dy > 0 else -14),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Controlled F1"); ax.set_ylim(0.72, 1.0)
    ax.set_title("60K custom CNN beats 28M ImageNet backbone", fontsize=9)
    fig.savefig(os.path.join(FIGS, "efficiency.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_loso(); fig_loso_folds(); fig_precision(); fig_vs(); fig_efficiency()
    print("figures written to", FIGS)
    print(os.listdir(FIGS))
