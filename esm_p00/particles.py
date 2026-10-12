"""Backward path: a resample-move particle filter over the uncertain card values.

A particle is a card set (material, instrument, junction). It also carries its
prior-width coordinates z, one per uncertain value (Param.at): the prior is N(0, 1)
per coordinate, U(-1, 1) for a uniform, and one jump size suits every parameter.
The points of an observation are assimilated one at a time:
    weight    multiply each particle's weight by the likelihood of the new point;
    resample  when the effective number of particles falls below half (systematic);
    move      Metropolis steps in z, scaled by the cloud's own covariance, accepted
              against the prior and all points seen so far.
Cards are rebuilt for every proposal, so their validators keep the physical support.

The caller supplies the log-likelihood of one point for one card set (it runs the
graph), so this module combines no mechanisms. At the end, updated_cards writes the
posterior back as cards; they hold independent marginals, so the correlations are
returned beside them.

Reference: Chopin, Biometrika 89, 539 (2002), iterated batch importance sampling.
"""
from __future__ import annotations

import math

import numpy as np

from esm_p00.cards import Param

ESS_FRACTION = 0.5                    # resample when the effective number falls below this share
JUMP = 2.38                           # random-walk scale, divided by sqrt(dimension)


def names_of(nominal: dict) -> list[tuple[str, str]]:
    """(card, field) of every uncertain value, in a fixed order: the axes of z."""
    return [(key, name) for key, card in nominal.items() for name in card.uncertain()]


def cards_at(nominal: dict, names: list[tuple[str, str]], z: np.ndarray) -> dict:
    """The card set at prior-width coordinates z; ValueError outside the physical support."""
    return {key: card.at({name: zi for (k, name), zi in zip(names, z) if k == key})
            for key, card in nominal.items()}


def log_prior(kinds: list[str], z: np.ndarray) -> float:
    """-z^2 / 2 per normal or lognormal coordinate; a uniform adds 0 inside |z| <= 1, -inf outside."""
    total = 0.0
    for kind, zi in zip(kinds, z):
        if kind == "uniform":
            if abs(zi) > 1.0:
                return -math.inf
        else:
            total -= 0.5 * zi * zi
    return total


def normalized(log_w: np.ndarray) -> np.ndarray:
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def ess(log_w: np.ndarray) -> float:
    """Effective number of particles, 1 / sum(w^2)."""
    w = normalized(log_w)
    return 1.0 / float(w @ w)


def resample(log_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: indices of the particles to keep, copies included."""
    n = len(log_w)
    return np.searchsorted(np.cumsum(normalized(log_w)), (rng.random() + np.arange(n)) / n).clip(0, n - 1)


def quantiles(z: np.ndarray, w: np.ndarray, q=(0.05, 0.5, 0.95)) -> np.ndarray:
    """Weighted quantiles of every coordinate: shape (len(q), dimension)."""
    out = []
    for column in z.T:
        order = np.argsort(column)
        cumulative = np.cumsum(w[order])
        out.append(column[order][np.searchsorted(cumulative, q).clip(0, len(column) - 1)])
    return np.array(out).T


def particle_filter(nominal, steps, data, loglik, n, rng, n_moves=5, record=None):
    """Assimilate the observation point by point.

    loglik(cards, step, observed) -> log-likelihood of one point, -inf when unsupported.
    record(k, z, w), when given, is called after every point. Returns the particle card
    sets, their coordinates z, their weights, the axes (names) and a few statistics.
    """
    names = names_of(nominal)
    kinds = [nominal[key].uncertain()[name].dist for key, name in names]
    dim = len(names)

    z, cards = [], []                                     # prior particles, truncated to the support
    while len(z) < n:
        zi = np.array([rng.uniform(-1.0, 1.0) if kind == "uniform" else rng.standard_normal() for kind in kinds])
        try:
            cards.append(cards_at(nominal, names, zi))
        except ValueError:
            continue
        z.append(zi)
    z, log_w, log_lik = np.array(z), np.zeros(n), np.zeros(n)
    stats = {"resamples": 0, "proposed": 0, "accepted": 0, "min_ess": float(n)}

    for k, (step, observed) in enumerate(zip(steps, data)):
        gain = np.array([loglik(c, step, observed) for c in cards])          # weight
        log_w, log_lik = log_w + gain, log_lik + gain
        if not np.isfinite(log_w).any():
            raise RuntimeError(f"no particle explains point {k}: the model cannot reproduce the observation")
        stats["min_ess"] = min(stats["min_ess"], ess(log_w))
        if ess(log_w) < ESS_FRACTION * n:                                    # resample
            keep = resample(log_w, rng)
            z, cards, log_lik, log_w = z[keep], [cards[i] for i in keep], log_lik[keep], np.zeros(n)
            stats["resamples"] += 1
            seen = list(zip(steps[:k + 1], data[:k + 1]))
            for _ in range(n_moves):                                         # move
                spread = np.cov(z.T).reshape(dim, dim) + 1e-9 * np.eye(dim)
                jump = np.linalg.cholesky(spread) * JUMP / math.sqrt(dim)
                for i in range(n):
                    stats["proposed"] += 1
                    zq = z[i] + jump @ rng.standard_normal(dim)
                    prior_q = log_prior(kinds, zq)
                    if not math.isfinite(prior_q):
                        continue
                    try:
                        cq = cards_at(nominal, names, zq)
                    except ValueError:                                       # outside the physical support
                        continue
                    lik_q = sum(loglik(cq, s, y) for s, y in seen)
                    if math.log(rng.random()) < (lik_q + prior_q) - (log_lik[i] + log_prior(kinds, z[i])):
                        z[i], cards[i], log_lik[i] = zq, cq, lik_q
                        stats["accepted"] += 1
        if record is not None:
            record(k, z.copy(), normalized(log_w))
    return cards, z, normalized(log_w), names, stats


def updated_cards(nominal, names, z, w):
    """The posterior written back as cards, and the correlation matrix the cards cannot hold.

    Each uncertain Param is centred at the weighted mean of its coordinate and its width is
    scaled by the weighted spread: normal and lognormal keep their family, a uniform becomes
    normal. The new card is an ordinary card, usable as the prior of the next experiment.
    """
    mean = w @ z
    cov = (z - mean).T @ ((z - mean) * w[:, None])
    spread = np.sqrt(np.diag(cov))
    new = {}
    for (key, name), m, s in zip(names, mean, spread):
        p = nominal[key].uncertain()[name]
        centre = p.at(m).value
        if p.dist == "lognormal":
            new.setdefault(key, {})[name] = Param(value=centre, dist="lognormal",
                                                  dist_params={"sigma_ln": s * p.dist_params["sigma_ln"]})
        else:
            width = p.dist_params["sigma"] if p.dist == "normal" else p.dist_params["half_width"]
            new.setdefault(key, {})[name] = Param(value=centre, dist="normal", dist_params={"sigma": s * width})
    cards = {key: card.replace(new.get(key, {})) for key, card in nominal.items()}
    return cards, cov / np.outer(spread, spread)
