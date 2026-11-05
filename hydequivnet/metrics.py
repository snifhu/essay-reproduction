from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    denom = np.sum((obs - np.mean(obs)) ** 2) + 1e-6
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    r = pearson_r(obs, sim)
    alpha = np.std(sim) / (np.std(obs) + 1e-6)
    beta = np.mean(sim) / (np.mean(obs) + 1e-6)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def pearson_r(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    if len(obs) < 2 or np.std(obs) < 1e-12 or np.std(sim) < 1e-12:
        return float("nan")
    r, _ = pearsonr(obs, sim)
    return float(r)


def peak_f1(obs: np.ndarray, sim: np.ndarray, quantile: float = 0.95) -> float:
    threshold = np.quantile(obs, quantile)
    return float(f1_score((obs > threshold).astype(int), (sim > threshold).astype(int)))


def flow_bias_skill(obs: np.ndarray, sim: np.ndarray, quantile: float, high_flow: bool = True) -> float:
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    threshold = np.quantile(obs, quantile)
    mask = obs >= threshold if high_flow else obs <= threshold
    if not np.any(mask):
        return float("nan")
    bias_pct = 100.0 * (np.mean(sim[mask]) - np.mean(obs[mask])) / (np.mean(obs[mask]) + 1e-6)
    return float(1.0 - abs(bias_pct) / 100.0)

