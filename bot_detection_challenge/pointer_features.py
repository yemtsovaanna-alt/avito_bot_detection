"""Joint XY statistics from event snapshots, not a continuous cursor trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from behavior_features import prepare_events

POINTER_GROUPS = {
    'coverage': ['xy_n_points', 'xy_observed_share'],
    'repetition': ['xy_repeat_share', 'xy_top_point_share'],
    'geometry': ['xy_abs_correlation', 'xy_anisotropy', 'xy_radius_cv'],
    'steps': ['xy_zero_step_share', 'xy_step_median_normalized', 'xy_step_cv'],
    'directions': ['xy_turn_cos_mean', 'xy_repeated_vector_share'],
}


def build_pointer_features(events: pd.DataFrame, meta: pd.DataFrame):
    """Use observable, deduplicated events; retain meta's cookie order.

    Geometry is invariant to translation and uniform positive scaling. Step size
    is divided by the cookie's RMS radius; screen dimensions are not assumed.
    Shape statistics require >=3 points, repetition >=2. Undefined values are NaN.
    Steps require adjacent events to be web events with finite paired XY, unique
    timestamps and 0 < dt <= 1800 s. A missing point or a tied timestamp breaks
    the sequence, as does a long pause or a non-web event.
    """
    ev = prepare_events(events, meta)
    index = pd.Index(meta.cookie_id, name='cookie_id')
    columns = [c for group in POINTER_GROUPS.values() for c in group]
    result = pd.DataFrame(np.nan, index=index, columns=columns)
    web = ev.platform.eq('web')
    paired = web & np.isfinite(ev[['pointer_x', 'pointer_y']]).all(axis=1)
    observed = ev.loc[paired].copy()
    n_points = observed.groupby('cookie_id').size().reindex(index, fill_value=0)
    n_web = web.groupby(ev.cookie_id).sum().reindex(index, fill_value=0)
    result['xy_n_points'] = n_points.astype(float)
    result['xy_observed_share'] = n_points / n_web.replace(0, np.nan)

    point_counts = observed.groupby(['cookie_id', 'pointer_x', 'pointer_y']).size()
    result['xy_repeat_share'] = (1 - point_counts.groupby(level=0).size() / n_points).where(n_points.ge(2))
    result['xy_top_point_share'] = (point_counts.groupby(level=0).max() / n_points).where(n_points.ge(2))

    # Center coordinates before computing covariance; translations cannot become
    # proxies for a particular location on the page.
    observed['cx'] = observed.pointer_x - observed.groupby('cookie_id').pointer_x.transform('mean')
    observed['cy'] = observed.pointer_y - observed.groupby('cookie_id').pointer_y.transform('mean')
    observed['cx2'] = observed.cx**2
    observed['cy2'] = observed.cy**2
    observed['cxcy'] = observed.cx * observed.cy
    observed['radius'] = np.hypot(observed.cx, observed.cy)
    moments = observed.groupby('cookie_id')[['cx2', 'cy2', 'cxcy']].mean()
    trace = moments.cx2 + moments.cy2
    rms_radius = np.sqrt(trace)
    spread_positive = trace.gt(0)
    result['xy_abs_correlation'] = (
        moments.cxcy.abs() / np.sqrt(moments.cx2 * moments.cy2).replace(0, np.nan)
    ).clip(0, 1).where(n_points.ge(3))
    # (lambda_max - lambda_min) / (lambda_max + lambda_min): 1 for a line,
    # 0 for equal spread in all directions. Works also for horizontal lines.
    result['xy_anisotropy'] = (
        np.sqrt((moments.cx2 - moments.cy2)**2 + 4 * moments.cxcy**2)
        / trace.where(spread_positive)
    ).clip(0, 1).where(n_points.ge(3))
    radii = observed.groupby('cookie_id').radius
    result['xy_radius_cv'] = (
        radii.std(ddof=0) / radii.mean().replace(0, np.nan)
    ).where(n_points.ge(3))

    unique_timestamp = ev.groupby(['cookie_id', 'event_ts']).event_ts.transform('size').eq(1)
    point_ok = paired & unique_timestamp
    previous_ok = point_ok.groupby(ev.cookie_id).shift().fillna(False).astype(bool)
    dt = ev.groupby('cookie_id').event_ts.diff().dt.total_seconds()
    ev['dx'] = ev.groupby('cookie_id').pointer_x.diff()
    ev['dy'] = ev.groupby('cookie_id').pointer_y.diff()
    step_ok = point_ok & previous_ok & dt.gt(0) & dt.le(1800)
    ev['step'] = np.hypot(ev.dx, ev.dy).where(step_ok)
    steps = ev.groupby('cookie_id').step
    step_count = steps.count()
    result['xy_zero_step_share'] = ev.step.eq(0).astype(float).where(step_ok).groupby(ev.cookie_id).mean()
    result['xy_step_median_normalized'] = steps.median() / rms_radius.replace(0, np.nan)
    result['xy_step_cv'] = (
        steps.std(ddof=0) / steps.mean().replace(0, np.nan)
    ).where(step_count.ge(2))
    vectors = ev.loc[step_ok].groupby(['cookie_id', 'dx', 'dy']).size()
    result['xy_repeated_vector_share'] = (
        1 - vectors.groupby(level=0).size() / step_count
    ).where(step_count.ge(2))

    previous_dx = ev.groupby('cookie_id').dx.shift()
    previous_dy = ev.groupby('cookie_id').dy.shift()
    previous_step = ev.groupby('cookie_id').step.shift()
    turn_ok = ev.step.gt(0) & previous_step.gt(0)
    cosines = (
        (ev.dx * previous_dx + ev.dy * previous_dy) / (ev.step * previous_step)
    ).clip(-1, 1).where(turn_ok)
    result['xy_turn_cos_mean'] = cosines.groupby(ev.cookie_id).mean()

    audit = {
        'coordinate_events': int(paired.sum()),
        'cookies_with_xy': int(n_points.gt(0).sum()),
        'cookies_with_at_least_3_points': int(n_points.ge(3).sum()),
        'valid_adjacent_steps': int(step_ok.sum()),
        'valid_turns': int(turn_ok.sum()),
        'xy_events_at_ambiguous_timestamps': int((paired & ~unique_timestamp).sum()),
    }
    return result.astype(float), POINTER_GROUPS, audit
