"""Formal one-sided Lan-DeMets O'Brien-Fleming boundaries.

The nominal final-look z is above the fixed-n one-sided 1.645 boundary:
planned early looks consume some Type-I error even though OBF spending makes
those looks conservative.  Boundaries are solved from canonical joint-normal
repeated-crossing probabilities, not from pointwise cumulative alpha.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache


LOOKS = (0.25, 0.50, 0.75, 1.00)


def _covariance(times):
    import numpy as np

    return np.asarray([
        [math.sqrt(min(left, right) / max(left, right)) for right in times]
        for left in times
    ], dtype=float)


def _crossing_probability(bounds, times, *, integration_seed: int) -> float:
    import numpy as np
    from scipy.stats import multivariate_normal

    dimension = len(bounds)
    rng = np.random.default_rng(integration_seed + dimension)
    no_cross = multivariate_normal.cdf(
        list(bounds), mean=np.zeros(dimension), cov=_covariance(times),
        maxpts=5_000_000, abseps=1e-10, releps=1e-10, rng=rng,
    )
    return 1.0 - float(no_cross)


@lru_cache(maxsize=8)
def _cached_boundaries(total: float, times: tuple[float, ...],
                       integration_seed: int) -> dict:
    import scipy
    from scipy.optimize import brentq
    from scipy.stats import norm

    # Official sfLDOF rho=1 cumulative spending. Alpha remains the total
    # one-sided error; joint calibration below uses an upper boundary only.
    reference = float(norm.ppf(1.0 - total / 2.0))
    spends = [float(2.0 * (1.0 - norm.cdf(reference / math.sqrt(time))))
              for time in times]
    bounds: list[float] = []
    rows = []
    previous = 0.0
    for index, (time, cumulative) in enumerate(zip(times, spends), start=1):
        if index == 1:
            boundary = float(norm.ppf(1.0 - cumulative))
        else:
            prefix_times = times[:index]

            def objective(candidate: float) -> float:
                return _crossing_probability(
                    (*bounds, candidate), prefix_times,
                    integration_seed=integration_seed,
                ) - cumulative

            boundary = float(brentq(objective, 0.0, 8.0,
                                    xtol=1e-10, rtol=1e-10))
        bounds.append(boundary)
        achieved = _crossing_probability(
            tuple(bounds), times[:index], integration_seed=integration_seed,
        )
        rows.append({
            "look": index, "information_fraction": time,
            "cumulative_spend": cumulative,
            "incremental_spend": cumulative - previous,
            "nominal_one_sided_alpha": float(1.0 - norm.cdf(boundary)),
            "z_boundary": boundary,
            "achieved_cumulative_crossing_probability": achieved,
            "integration_absolute_error": abs(achieved - cumulative),
        })
        previous = cumulative
    result = {
        "schema_version": 2,
        "family": "Lan-DeMets O'Brien-Fleming (sfLDOF rho=1)",
        "sidedness": "one-sided upper boundary", "total": total,
        "spending_function": (
            "2*(1-Phi(Phi^-1(1-total/2)/sqrt(t)))"
        ),
        "boundary_interpretation": (
            "canonical joint-normal repeated-crossing inversion; nominal alpha "
            "is 1-Phi(z_k), not cumulative spend"
        ),
        "looks": rows,
        "numeric_source": {
            "scipy_version": scipy.__version__,
            "integration_seed": integration_seed,
            "maxpts": 5_000_000, "abseps": 1e-10, "releps": 1e-10,
            "root": "scipy.optimize.brentq xtol=rtol=1e-10",
        },
        "note": (
            "The final z exceeds fixed-n one-sided z=1.644853626951472 because "
            "early looks consume alpha; direct z=1.645 pointwise inversion is "
            "not a repeated-crossing group-sequential boundary."
        ),
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def obrien_fleming_boundaries(*, total: float = 0.05, looks=LOOKS,
                              integration_seed: int = 4000) -> dict:
    times = tuple(float(value) for value in looks)
    if not times or times[-1] != 1.0 or any(
        left <= 0 or left >= right for left, right in zip(times, times[1:])
    ):
        raise ValueError("looks must be strictly increasing in (0,1] and end at 1")
    # Round-trip through JSON prevents callers mutating the cached authority.
    return json.loads(json.dumps(_cached_boundaries(total, times, integration_seed)))


def e4_sequential_config(*, n_task: int | None = None,
                         look_tasks: tuple[int, ...] | None = None) -> dict:
    """Return the frozen E4 one-sided boundary configuration.

    ``n_task``/``look_tasks`` make the right-sized absolute look schedule part
    of the authority.  The legacy no-argument form deliberately retains the
    preregistration's exact fractional looks for backwards-compatible audits.
    """
    if (n_task is None) != (look_tasks is None):
        raise ValueError("n_task and look_tasks must be supplied together")
    if n_task is None:
        times = LOOKS
        absolute = None
    else:
        if n_task <= 0:
            raise ValueError("n_task must be positive")
        absolute = tuple(int(value) for value in look_tasks or ())
        if (not absolute or absolute[-1] != n_task
                or any(left <= 0 or left >= right
                       for left, right in zip(absolute, absolute[1:]))):
            raise ValueError(
                "look_tasks must be strictly increasing, positive, and end at n_task"
            )
        times = tuple(value / n_task for value in absolute)
    boundary = obrien_fleming_boundaries(total=0.05, looks=times)
    if absolute is not None:
        for row, task_count in zip(boundary["looks"], absolute):
            row["task_count"] = task_count
        boundary["n_task"] = n_task
        boundary["look_tasks"] = list(absolute)
        # The boundary hash above intentionally identifies the numeric
        # calibration.  This outer config hash also freezes the absolute plan.
        boundary["absolute_schedule_sha256"] = hashlib.sha256(
            json.dumps({"n_task": n_task, "look_tasks": absolute},
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    result = {
        "schema_version": 2,
        "efficacy": boundary,
        "conjunction_futility_reversal": json.loads(json.dumps(boundary)),
        "binding": True,
        "arbitrary_peeking": False,
        "note": (
            "Efficacy alpha=0.05 and reversal gamma=0.05 use the same "
            "one-sided canonical boundary in opposite directions."
        ),
        "right_sized_schedule": ({"n_task": n_task,
                                  "look_tasks": list(absolute)}
                                 if absolute is not None else None),
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result
