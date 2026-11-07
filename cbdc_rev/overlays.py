"""Synthetic KYC overlays.

published_overlay(): faithful re-implementation of src/kyc_overlay.py (v1.0). It is generated
    WITHOUT looking at labels or at the graph, so the five columns carry no information about
    illicit behaviour (this is verified empirically in the experiments).
signal_overlay(): label-conditional overlay with a tunable strength rho = share of illicit
    transactions bound to 'high-risk' identities (rho = 0.10 equals the null overlay).
Both return an (N, 5) float64 array with columns = KYC_COLS and the same raw semantics as the
published code: country_risk in {0: high_risk, 1: increased, 2: standard} (alphabetical category
codes), kyc_tier in {1,2,3}, kyc_age_days in [1,1999], pep_flag in {0,1}, aml_history >= 0.
"""
import numpy as np

COUNTRY = ["high_risk", "increased", "standard"]
BASE = dict(country=[0.04, 0.15, 0.81], tier=[0.30, 0.60, 0.10], pep=0.03, aml_zero=0.92, aml_geom_p=0.4)


def _zipf_weights(n_id, alpha):
    ranks = np.arange(1, n_id + 1)
    w = 1.0 / np.power(ranks, alpha)
    return w / w.sum()


def published_overlay(n, seed, alpha=1.6, ratio=0.35, return_ids=False):
    rng = np.random.default_rng(seed)
    n_id = max(1, int(n * ratio))
    keys = np.array(COUNTRY)
    country = rng.choice(keys, size=n_id, p=np.array(BASE["country"]) / sum(BASE["country"]))
    tier = rng.choice([1, 2, 3], size=n_id, p=np.array(BASE["tier"]) / sum(BASE["tier"])).astype(int)
    age = rng.integers(1, 2000, size=n_id)
    pep = rng.random(n_id) < BASE["pep"]
    zero_mask = rng.random(n_id) < BASE["aml_zero"]
    nonzero = rng.geometric(p=BASE["aml_geom_p"], size=n_id)
    aml = np.where(zero_mask, 0, nonzero)
    tx_to_id = rng.choice(n_id, size=n, p=_zipf_weights(n_id, alpha))
    code = {"high_risk": 0, "increased": 1, "standard": 2}
    c = np.vectorize(code.get)(country)
    out = np.column_stack([c[tx_to_id], tier[tx_to_id], age[tx_to_id], pep[tx_to_id].astype(int), aml[tx_to_id]])
    if return_ids:
        return out.astype(np.float64), tx_to_id
    return out.astype(np.float64)


def signal_overlay(y, seed, rho=0.30, pi_high=0.10):
    """Transaction-level, label-conditional overlay (no identity sharing).
    Each transaction gets a latent risk class h; P(h=1 | illicit) = rho, P(h=1 | licit/unknown) = pi_high.
    Given h, the five attributes are drawn from clearly separated distributions whose pooled marginals equal
    the nominal ones of the published overlay when h ~ Bernoulli(pi_high).  rho == pi_high is the exact null
    (attributes independent of the label); larger rho = stronger identity signal."""
    rng = np.random.default_rng(seed)
    n = len(y)
    h = rng.random(n) < np.where(y == 1, rho, pi_high)
    c_hi = np.array([0.30, 0.40, 0.30]); c_lo = (np.array(BASE["country"]) - pi_high * c_hi) / (1 - pi_high)
    t_hi = np.array([0.55, 0.40, 0.05]); t_lo = (np.array(BASE["tier"]) - pi_high * t_hi) / (1 - pi_high)
    pep_hi = 0.20; pep_lo = (BASE["pep"] - pi_high * pep_hi) / (1 - pi_high)
    z_hi = 0.52; z_lo = (BASE["aml_zero"] - pi_high * z_hi) / (1 - pi_high)

    def draw(p_hi, p_lo, values):
        u = rng.random(n)
        cdf_hi = np.cumsum(p_hi); cdf_lo = np.cumsum(p_lo)
        k = np.where(h, np.searchsorted(cdf_hi, u), np.searchsorted(cdf_lo, u))
        return np.asarray(values)[np.minimum(k, len(values) - 1)]

    country = draw(c_hi, c_lo, [0, 1, 2])
    tier = draw(t_hi, t_lo, [1, 2, 3])
    age = np.where(h, rng.integers(1, 600, size=n), rng.integers(1, 2000, size=n))
    pep = (rng.random(n) < np.where(h, pep_hi, pep_lo)).astype(int)
    zero = rng.random(n) < np.where(h, z_hi, z_lo)
    aml = np.where(zero, 0, rng.geometric(p=BASE["aml_geom_p"], size=n))
    return np.column_stack([country, tier, age, pep, aml]).astype(np.float64)


def standardise(K, fit_mask):
    mu = K[fit_mask].mean(0); sd = K[fit_mask].std(0); sd[sd == 0] = 1.0
    return (K - mu) / sd
