import argparse
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ALPHA = 0.05
EFFECT_MIN = 0.10
GAIN_MIN = 0.01
TAU = 0.30
TEST_SIZE = 0.40
BACKGROUND_SIZE = 100
LIME_ROWS_PER_GROUP = 100
LIME_SAMPLES = 600
MI_NEIGHBORS = 5
N_MI_NULL = 50
N_DELTA_NULL = 200
MIN_CELL = 5
PROXY_LOGIT_SD = 0.8
DECOY_RHO = 0.7

SCENARIOS = {
    "Контроль": {"n_used": 0, "rho_used": 0.0, "n_decoy": 0, "n_noise": 2},
    "Слабая связь": {"n_used": 2, "rho_used": 0.30, "n_decoy": 2, "n_noise": 2},
    "Умеренная связь": {"n_used": 3, "rho_used": 0.55, "n_decoy": 2, "n_noise": 2},
    "Сильная связь": {"n_used": 3, "rho_used": 0.85, "n_decoy": 2, "n_noise": 2},
}
METHODS = ["Корреляция (MI)", "SHAP", "LIME", "SHAP ∩ LIME"]


@dataclass
class Encoder:
    num: list
    cat: list
    levels: list

    @classmethod
    def fit(cls, X):
        num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat = [c for c in X.columns if c not in num]
        levels = [sorted(X[c].astype(str).unique()) for c in cat]
        return cls(num, cat, levels)

    @property
    def features(self):
        return self.num + self.cat

    @property
    def groups(self):
        return np.array(self.num + [c for c, lv in zip(self.cat, self.levels) for _ in lv])

    @property
    def columns(self):
        return self.num + [f"{c}={v}" for c, lv in zip(self.cat, self.levels) for v in lv]

    def to_ordinal(self, X):
        cols = [X[c].to_numpy(dtype=float) for c in self.num]
        for c, lv in zip(self.cat, self.levels):
            cols.append(pd.Categorical(X[c].astype(str), categories=lv).codes.astype(float))
        return np.column_stack(cols)

    def one_hot(self, arr):
        parts = [arr[:, : len(self.num)]]
        for j, lv in enumerate(self.levels):
            codes = np.clip(np.rint(arr[:, len(self.num) + j]).astype(int), 0, len(lv) - 1)
            parts.append(np.eye(len(lv))[codes])
        return np.hstack(parts)

    def transform(self, X):
        return self.one_hot(self.to_ordinal(X))


def load_german_credit():
    frame = fetch_openml("credit-g", version=1, as_frame=True).frame.copy()
    a = frame["personal_status"].astype(str).str.contains("female").astype(int).to_numpy()
    y = (frame["class"].astype(str) == "bad").astype(int).to_numpy()
    X = frame.drop(columns=["class", "personal_status"])
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(str)
    return X.reset_index(drop=True), a, y


def base_logit(X, y):
    enc = Encoder.fit(X)
    M = StandardScaler().fit_transform(enc.transform(X))
    return LogisticRegression(C=0.3, max_iter=3000).fit(M, y).decision_function(M)


def make_strata(X):
    amount = pd.qcut(X["credit_amount"], 3, labels=False, duplicates="drop").to_numpy()
    duration = pd.qcut(X["duration"], 3, labels=False, duplicates="drop").to_numpy()
    return amount * 3 + duration


def build_scenario(X, a, cfg, rng):
    Xs = X.copy()
    n = len(Xs)
    z = (a - a.mean()) / a.std()
    used, decoys = [], []

    def proxy(rho):
        return rho * z + np.sqrt(1 - rho**2) * rng.standard_normal(n)

    for i in range(cfg["n_used"]):
        Xs[f"proxy_used_{i + 1}"] = proxy(cfg["rho_used"])
        used.append(f"proxy_used_{i + 1}")
    for i in range(cfg["n_decoy"]):
        Xs[f"proxy_decoy_{i + 1}"] = proxy(DECOY_RHO)
        decoys.append(f"proxy_decoy_{i + 1}")
    for i in range(cfg["n_noise"]):
        Xs[f"noise_{i + 1}"] = rng.standard_normal(n)
    return Xs, used, decoys


def synthesize_labels(base, Xs, used, rng):
    logit = base.copy()
    if used:
        s = Xs[used].sum(axis=1).to_numpy()
        logit = logit + PROXY_LOGIT_SD * (s - s.mean()) / s.std()
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(int)


def mask_decoys(M, enc, decoys):
    if not decoys:
        return M
    out = M.copy()
    out[:, np.isin(enc.groups, decoys)] = 0.0
    return out


def fit_model(M, y, seed, kind):
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=4,
            eval_metric="logloss",
            verbosity=0,
        ).fit(M, y)
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed
    ).fit(M, y)


def margin(model, M):
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        return model.predict(M, output_margin=True)
    return model.decision_function(M)


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(ranked, 0, 1)
    return out


def approval_gap(approved, a, strata):
    gaps, weights = [], []
    for s in np.unique(strata):
        m = strata == s
        m1, m0 = m & (a == 1), m & (a == 0)
        if m1.sum() >= MIN_CELL and m0.sum() >= MIN_CELL:
            gaps.append(abs(approved[m1].mean() - approved[m0].mean()))
            weights.append(m.sum())
    return np.average(gaps, weights=weights) if gaps else np.nan


def measure(approved, a, y, strata, rng):
    b_r = approval_gap(approved, a, strata)
    null = [approval_gap(approved, rng.permutation(a), strata) for _ in range(N_DELTA_NULL)]
    delta = np.nanquantile(null, 0.95)
    rate = lambda m: approved[m].mean()
    return {
        "B_R": b_r,
        "delta": delta,
        "DP": abs(rate(a == 1) - rate(a == 0)),
        "EO_good": abs(rate((a == 1) & (y == 0)) - rate((a == 0) & (y == 0))),
        "EO_bad": abs(rate((a == 1) & (y == 1)) - rate((a == 0) & (y == 1))),
        "signal": bool(b_r > delta),
    }


def proxy_filter(ord_all, a, enc, seed):
    discrete = np.array([False] * len(enc.num) + [True] * len(enc.cat))
    mi = mutual_info_classif(
        ord_all, a, discrete_features=discrete, n_neighbors=MI_NEIGHBORS, random_state=seed
    )
    rng = np.random.default_rng(seed)
    null_max = [
        mutual_info_classif(
            ord_all,
            rng.permutation(a),
            discrete_features=discrete,
            n_neighbors=MI_NEIGHBORS,
            random_state=seed,
        ).max()
        for _ in range(N_MI_NULL)
    ]
    eps = float(np.quantile(null_max, 0.95))
    return mi > eps, mi, eps


def aggregate(vals, groups, features):
    return np.column_stack([vals[:, groups == f].sum(axis=1) for f in features])


def shap_attributions(model, M_bg, M_audit, enc):
    explainer = shap.TreeExplainer(
        model, data=M_bg, feature_perturbation="interventional", model_output="raw"
    )
    vals = explainer.shap_values(M_audit)
    if isinstance(vals, list):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[..., -1]
    base_val = np.ravel(explainer.expected_value)[-1]
    gap = np.abs(base_val + vals.sum(axis=1) - margin(model, M_audit)).max()
    return aggregate(vals, enc.groups, enc.features), float(gap)


def lime_attributions(model, enc, ord_train, ord_rows, seed):
    n_num = len(enc.num)
    names = {n_num + j: list(lv) for j, lv in enumerate(enc.levels)}
    explainer = LimeTabularExplainer(
        ord_train,
        feature_names=enc.features,
        categorical_features=list(range(n_num, len(enc.features))),
        categorical_names=names,
        class_names=["good", "bad"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )
    predict = lambda arr: model.predict_proba(enc.one_hot(arr))
    out = np.zeros((len(ord_rows), len(enc.features)))
    for i, row in enumerate(ord_rows):
        exp = explainer.explain_instance(
            row, predict, num_features=len(enc.features), num_samples=LIME_SAMPLES, labels=(1,)
        )
        for j, w in exp.as_map()[1]:
            out[i, j] = w
    return out


def group_flags(attr, a):
    g0, g1 = attr[a == 0], attr[a == 1]
    n_feat = attr.shape[1]
    p_ks, p_mw, effect = np.ones(n_feat), np.ones(n_feat), np.zeros(n_feat)
    for j in range(n_feat):
        if np.ptp(attr[:, j]) == 0:
            continue
        p_ks[j] = ks_2samp(g0[:, j], g1[:, j]).pvalue
        res = mannwhitneyu(g0[:, j], g1[:, j])
        p_mw[j] = res.pvalue
        effect[j] = abs(res.statistic / (len(g0) * len(g1)) - 0.5)
    return (bh_adjust(p_ks) < ALPHA) & (bh_adjust(p_mw) < ALPHA) & (effect >= EFFECT_MIN)


def feature_gain_shares(model, enc):
    n_cols = len(enc.columns)
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        raw = model.get_booster().get_score(importance_type="total_gain")
        col_gains = np.zeros(n_cols, dtype=float)
        for k, v in raw.items():
            col_gains[int(k[1:])] = float(v)
    else:
        col_gains = np.asarray(model.feature_importances_, dtype=float)
    total = col_gains.sum()
    if total > 0:
        col_gains = col_gains / total
    return {f: float(col_gains[enc.groups == f].sum()) for f in enc.features}


def verified_truth(model, enc, used, decoys):
    shares = feature_gain_shares(model, enc)
    verified = {name for name in used if shares.get(name, 0.0) >= GAIN_MIN}
    max_decoy_gain = max((shares.get(name, 0.0) for name in decoys), default=0.0)
    return verified, max_decoy_gain


def score(pred, truth):
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    return {
        "precision": tp / pred.sum() if pred.sum() else np.nan,
        "recall": tp / truth.sum() if truth.sum() else np.nan,
        "fpr": fp / (~truth).sum(),
        "n_flagged": int(pred.sum()),
    }


def run_once(X, a, base, strata, name, cfg, seed, kind):
    rng = np.random.default_rng(seed)
    Xs, used, decoys = build_scenario(X, a, cfg, rng)
    y = synthesize_labels(base, Xs, used, rng)
    idx = np.arange(len(Xs))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=seed)

    enc = Encoder.fit(Xs)
    M_tr, M_te = enc.transform(Xs.iloc[idx_tr]), enc.transform(Xs.iloc[idx_te])
    M_tr_fit = mask_decoys(M_tr, enc, decoys)
    model = fit_model(M_tr_fit, y[idx_tr], seed, kind)
    p_te = model.predict_proba(M_te)[:, 1]
    approved = (p_te < TAU).astype(int)
    a_te, y_te = a[idx_te], y[idx_te]

    level1 = measure(approved, a_te, y_te, strata[idx_te], rng)
    level1["auc"] = roc_auc_score(y_te, p_te)

    ord_all = enc.to_ordinal(Xs)
    candidates, mi, eps = proxy_filter(ord_all, a, enc, seed)

    bg = M_tr[rng.choice(len(M_tr), BACKGROUND_SIZE, replace=False)]
    attr_shap, gap = shap_attributions(model, bg, M_te, enc)
    level1["shap_additivity_gap"] = gap

    sel = np.concatenate(
        [
            rng.choice(
                np.where(a_te == g)[0],
                min(LIME_ROWS_PER_GROUP, (a_te == g).sum()),
                replace=False,
            )
            for g in (0, 1)
        ]
    )
    ord_te = enc.to_ordinal(Xs.iloc[idx_te])
    attr_lime = lime_attributions(model, enc, enc.to_ordinal(Xs.iloc[idx_tr]), ord_te[sel], seed)

    features = np.array(enc.features)
    verified, max_decoy_gain = verified_truth(model, enc, used, decoys)
    truth = np.isin(features, list(verified))

    flags = {
        "Корреляция (MI)": candidates,
        "SHAP": candidates & group_flags(attr_shap, a_te),
        "LIME": candidates & group_flags(attr_lime, a_te[sel]),
    }
    flags["SHAP ∩ LIME"] = flags["SHAP"] & flags["LIME"]

    rows = [
        {"scenario": name, "seed": seed, "method": m, **score(mask, truth), "flagged": ", ".join(features[mask])}
        for m, mask in flags.items()
    ]
    level1.update(
        {
            "scenario": name,
            "seed": seed,
            "P_star": ", ".join(sorted(verified)),
            "eps_MI": eps,
            "max_decoy_gain": max_decoy_gain,
        }
    )
    detail = {"attr": attr_shap, "a": a_te, "features": features, "truth": truth}
    return rows, level1, detail


def h1_share(df):
    shares = {}
    others = [m for m in METHODS if m != "SHAP ∩ LIME"]
    for name, part in df.groupby("scenario", sort=False):
        if part["recall"].isna().all():
            continue
        wide = part.pivot(index="seed", columns="method", values=["precision", "recall"])
        valid = wide["recall"].notna().all(axis=1)
        if not valid.any():
            continue
        p = wide["precision"].loc[valid].fillna(0.0)
        r = wide["recall"].loc[valid].fillna(0.0)
        ok = (p["SHAP ∩ LIME"] > p[others].max(axis=1)) & (r["SHAP ∩ LIME"] >= 0.9 * r[METHODS].max(axis=1))
        shares[name] = float(ok.mean())
    return shares


def style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def clip_yerr(mean, std, low=0.0, high=None):
    m = mean.fillna(0.0).to_numpy(dtype=float)
    s = std.fillna(0.0).to_numpy(dtype=float)
    err_low = np.minimum(s, np.maximum(0.0, m - low))
    err_high = np.minimum(s, np.maximum(0.0, high - m)) if high is not None else s
    return np.vstack([err_low, err_high])


def plot_metrics(df, path):
    names = list(SCENARIOS)
    palette = sns.color_palette("deep", len(METHODS))
    metrics = [("precision", "Precision"), ("recall", "Recall (TPR)"), ("fpr", "FPR")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), facecolor="white")
    width = 0.2
    handles, labels = None, None
    for ax, (col, title) in zip(axes, metrics):
        for k, method in enumerate(METHODS):
            grouped = df[df["method"] == method].groupby("scenario", sort=False)[col]
            mean = grouped.mean().reindex(names)
            std = grouped.std().reindex(names).fillna(0.0)
            xs = np.arange(len(names)) + (k - 1.5) * width
            yerr = clip_yerr(mean, std, low=0.0, high=1.0)
            ax.bar(
                xs,
                mean.fillna(0.0),
                width,
                yerr=yerr,
                color=palette[k],
                label=method,
                capsize=3,
                error_kw={"linewidth": 1.2},
            )
            for x, m in zip(xs, mean):
                if np.isnan(m):
                    ax.text(x, 0.03, "н/д", ha="center", va="bottom", rotation=90, fontsize=9, color="#555555")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.06)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Обнаружение прокси-признаков: среднее ± σ по повторным прогонам", fontsize=15, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(METHODS), frameon=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_signed(detail, path):
    attr, a, features, truth = detail["attr"], detail["a"], detail["features"], detail["truth"]
    mean_abs = np.abs(attr).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10][::-1]
    labels = [features[i] + (" *" if truth[i] else "") for i in order]
    pos = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor="white")
    axes[0].barh(pos, mean_abs[order], color=sns.color_palette("deep")[0])
    axes[0].set_yticks(pos)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Среднее |SHAP|: глобальная важность")
    axes[0].set_xlabel("лог-шансы дефолта")
    axes[1].barh(
        pos - 0.2,
        attr[a == 0][:, order].mean(axis=0),
        height=0.4,
        label="A = 0",
        color=sns.color_palette("deep")[2],
    )
    axes[1].barh(
        pos + 0.2,
        attr[a == 1][:, order].mean(axis=0),
        height=0.4,
        label="A = 1",
        color=sns.color_palette("deep")[3],
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(pos)
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Средний знаковый SHAP по группам")
    axes[1].set_xlabel("лог-шансы дефолта")
    axes[1].legend(frameon=True)
    fig.suptitle("Сценарий «Сильная связь»: * — функционально используемые прокси", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_level1(level1, path):
    names = list(SCENARIOS)
    grouped = level1.groupby("scenario", sort=False)
    x = np.arange(len(names))
    w = 0.35
    palette = sns.color_palette("deep")
    b_mean, b_std = grouped["B_R"].mean().reindex(names), grouped["B_R"].std().reindex(names)
    d_mean, d_std = grouped["DP"].mean().reindex(names), grouped["DP"].std().reindex(names)
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.bar(
        x - w / 2,
        b_mean,
        w,
        yerr=clip_yerr(b_mean, b_std, low=0.0, high=1.0),
        capsize=4,
        color=palette[0],
        label="B_R(f)",
    )
    ax.bar(
        x + w / 2,
        d_mean,
        w,
        yerr=clip_yerr(d_mean, d_std, low=0.0, high=1.0),
        capsize=4,
        color=palette[1],
        label="Разность DP",
    )
    ax.scatter(
        x - w / 2,
        grouped["delta"].mean().reindex(names),
        marker="D",
        s=95,
        color="black",
        zorder=5,
        label="δ: 95-й перцентиль при перемешанном A",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Разница вероятностей одобрения")
    ax.set_title("Уровень 1: измерение межгрупповых различий")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", choices=["xgb", "gbc"], default="xgb" if XGBClassifier is not None else "gbc")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    style()

    X, a, y_orig = load_german_credit()
    base = base_logit(X, y_orig)
    strata = make_strata(X)

    protocol = {
        "model": args.model,
        "shap": "TreeSHAP, interventional, log-odds, background=%d" % BACKGROUND_SIZE,
        "lime": "discretize_continuous=True, samples=%d, rows per group=%d" % (LIME_SAMPLES, LIME_ROWS_PER_GROUP),
        "tau": TAU,
        "alpha": ALPHA,
        "effect_min": EFFECT_MIN,
        "gain_min": GAIN_MIN,
        "R": "credit_amount (tertiles) x duration (tertiles)",
        "repeats": args.repeats,
    }
    print("ПРОТОКОЛ АУДИТА:", protocol)

    all_rows, all_level1, detail_strong = [], [], None
    for name, cfg in SCENARIOS.items():
        for r in range(args.repeats):
            rows, level1, detail = run_once(X, a, base, strata, name, cfg, args.seed + r, args.model)
            all_rows += rows
            all_level1.append(level1)
            if name == "Сильная связь" and r == 0:
                detail_strong = detail
            print(
                f"{name} | seed {args.seed + r} | AUC={level1['auc']:.3f} | B_R={level1['B_R']:.3f} | delta={level1['delta']:.3f} | P*=[{level1['P_star']}]"
            )

    df, level1_df = pd.DataFrame(all_rows), pd.DataFrame(all_level1)
    df.to_csv("audit_results.csv", index=False)
    level1_df.to_csv("audit_level1.csv", index=False)

    summary = df.groupby(["scenario", "method"], sort=False)[["precision", "recall", "fpr", "n_flagged"]].agg(["mean", "std"]).round(3)
    print(summary.to_string())
    print(level1_df.groupby("scenario", sort=False)[["auc", "B_R", "delta", "DP", "EO_good", "EO_bad", "signal"]].mean().round(3).to_string())
    print("Максимальный разрыв аддитивности SHAP:", level1_df["shap_additivity_gap"].max())
    print("Максимальный gain приманок в модели:", level1_df["max_decoy_gain"].max())
    print("Доля прогонов, где выполнено условие H1:", h1_share(df))

    plot_metrics(df, "fig1_precision_recall_fpr.png")
    plot_signed(detail_strong, "fig2_signed_shap_groups.png")
    plot_level1(level1_df, "fig3_level1_measurement.png")


if __name__ == "__main__":
    main()