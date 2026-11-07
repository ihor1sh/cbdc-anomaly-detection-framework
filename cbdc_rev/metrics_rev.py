"""Metrics for the revision: threshold-free ranking metrics, operating-point metrics, validation-calibrated thresholds."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def curve_metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)

    def rec_at(a):
        m = fpr <= a
        return float(tpr[m].max()) if m.any() else 0.0

    idx = int(np.searchsorted(tpr, 0.9))           # same definition as src/metrics.py (v1.0)
    return dict(auc_pr=float(average_precision_score(y, s)), auc_roc=float(roc_auc_score(y, s)),
                rec_at_fpr1=rec_at(0.01), rec_at_fpr5=rec_at(0.05),
                fpr_at_r90=float(fpr[idx]) if idx < len(fpr) else float("nan"))


def at_threshold(y, s, tau):
    pred = s >= tau
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=p, recall=r, f1=f1, fpr=fp / (fp + tn) if fp + tn else 0.0, n_flagged=int(pred.sum()))


def thr_max_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def thr_fpr(y, s, a=0.05):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.where(fpr <= a)[0].max()
    return float(thr[i])


def full_panel(yv, sv, yt, st, tau_default=0.5, st_pub=None):
    """Everything one run reports. yv/sv: validation labels/scores; yt/st: test labels/scores.
    st_pub: test scores in the published (min-max normalised) form, used only for the tau=0.5 columns of Isolation Forest."""
    out = curve_metrics(yt, st)
    pub = at_threshold(yt, st if st_pub is None else st_pub, tau_default)
    out.update({f"{k}_t05": v for k, v in pub.items()})
    tf = thr_max_f1(yv, sv); m = at_threshold(yt, st, tf)
    out.update({f"{k}_vf1": v for k, v in m.items()}); out["tau_vf1"] = tf
    tr = thr_fpr(yv, sv, 0.05); m = at_threshold(yt, st, tr)
    out.update({f"{k}_vfpr5": v for k, v in m.items()}); out["tau_vfpr5"] = tr
    out["prevalence_test"] = float(yt.mean())
    return out
