"""Compare joint XY groups with the frozen 30-feature behavior model.

    python bot_detection_challenge/pointer_experiments.py compare
    python bot_detection_challenge/pointer_experiments.py final

Candidate choice uses only validation on April 12–14 and April 15–16. The already
seen April 17–19 period is a final comparison, not a fresh independent test.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from behavior_experiments import (
    DEVELOPMENT_FOLDS, HOLDOUT, PARAMS, ROOT, bootstrap_comparison, evaluate, make_model,
)
from behavior_features import build_features, read_data
from pointer_features import POINTER_GROUPS, build_pointer_features

ARTIFACTS = ROOT / 'artifacts' / 'pointer'
CONFIG = ROOT / 'pointer_selection.json'
BASELINE = 'behavior_30'


def prepare_matrices():
    train, test, events = read_data(ROOT / 'data')
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)
    behavior, _, _ = build_features(events, meta)
    pointer, _, audit = build_pointer_features(events, meta)
    X = behavior.join(pointer, validate='one_to_one')
    if np.isinf(pointer.to_numpy()).any():
        raise ValueError('XY features must be finite or NaN')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return train, test, X.loc[train.cookie_id], X.loc[test.cookie_id]


def candidate_columns(base):
    combinations = {
        BASELINE: [], 'coverage_only': ['coverage'],
        'xy_repetition': ['repetition'], 'xy_geometry': ['geometry'],
        'xy_steps': ['steps'], 'xy_directions': ['directions'],
        'xy_spatial': ['repetition', 'geometry'],
        'xy_spatial_coverage': ['coverage', 'repetition', 'geometry'],
        'xy_geometry_coverage': ['coverage', 'geometry'],
        'xy_motion': ['steps', 'directions'],
        'xy_joint': ['repetition', 'geometry', 'steps', 'directions'],
        'xy_joint_coverage': ['coverage', 'repetition', 'geometry', 'steps', 'directions'],
    }
    return {
        name: base + [c for group in groups for c in POINTER_GROUPS[group]]
        for name, groups in combinations.items()
    }


def compare(train, X, candidate_names=None):
    baseline = json.loads((ROOT / 'selected_features.json').read_text())
    assert baseline['catboost_params'] == PARAMS
    specs = candidate_columns(baseline['features'])
    (ARTIFACTS / 'candidates.json').write_text(json.dumps(specs, indent=2) + '\n')
    rows, predictions = [], []
    previous_predictions = None
    if candidate_names is not None:
        # Additional bounded comparison on development dates only. Reuse prior
        # scores for unchanged candidates instead of needlessly retraining them.
        previous = pd.read_csv(ARTIFACTS / 'development_metrics.csv')
        rows = previous.loc[~previous.model_id.isin(candidate_names)].to_dict('records')
        previous_predictions = pd.read_csv(ARTIFACTS / 'development_predictions.csv')
        to_run = {name: specs[name] for name in candidate_names}
    else:
        to_run = specs
    y = train.target.to_numpy()
    for start, end in DEVELOPMENT_FOLDS:
        fit = train.window_start_ts.lt(start).to_numpy()
        valid = (train.window_start_ts.ge(start) & train.window_start_ts.lt(end)).to_numpy()
        fold_predictions = train.loc[valid, ['cookie_id', 'target', 'window_start_ts']].copy()
        fold_predictions['xy_n_points'] = X.loc[valid, 'xy_n_points'].to_numpy()
        fold_predictions['validation_start'] = start
        if previous_predictions is not None:
            saved = previous_predictions.loc[previous_predictions.validation_start.eq(start)]
            assert saved.cookie_id.tolist() == fold_predictions.cookie_id.tolist()
            for name in specs:
                if name in saved and name not in to_run:
                    fold_predictions[name] = saved[name].to_numpy()
        for name, columns in to_run.items():
            model = make_model(columns)
            model.fit(X.loc[fit, columns], y[fit])
            score = model.predict_proba(X.loc[valid, columns])[:, 1]
            fold_predictions[name] = score
            row = {
                'model_id': name, 'validation_start': start, 'validation_end': end,
                'n_features': len(columns), 'n_train': int(fit.sum()), 'n_valid': int(valid.sum()),
                **evaluate(y[valid], score),
            }
            rows.append(row)
            print(f'{name:22s} {start} n={len(columns):2d} P@R={row["precision_at_recall_70"]:.5f}', flush=True)
            pd.DataFrame(rows).to_csv(ARTIFACTS / 'development_metrics.csv', index=False)
        predictions.append(fold_predictions)
    pd.concat(predictions, ignore_index=True).to_csv(ARTIFACTS / 'development_predictions.csv', index=False)
    result = pd.DataFrame(rows)
    summary = result.groupby('model_id').agg(
        n_features=('n_features', 'first'), mean_precision=('precision_at_recall_70', 'mean'),
        min_precision=('precision_at_recall_70', 'min'), mean_pr_auc=('pr_auc', 'mean'),
    ).sort_values('mean_precision', ascending=False)
    summary.to_csv(ARTIFACTS / 'development_summary.csv')
    print(summary.to_string(), flush=True)

    # Fix the choice before evaluating April 17–19. Among actual XY candidates,
    # prefer the smallest within one percentage point of the best mean precision.
    xy = summary.loc[summary.index.str.startswith('xy_')]
    eligible = xy.loc[xy.mean_precision.ge(xy.mean_precision.max() - 0.01)]
    selected = eligible.sort_values(['n_features', 'mean_precision'], ascending=[True, False]).index[0]
    per_fold = result.pivot(index='validation_start', columns='model_id', values='precision_at_recall_70')
    gain = per_fold[selected] - per_fold[BASELINE]
    # This is a conservative adoption rule, not a significance test.
    supported = bool(gain.mean() >= 0.01 and gain.min() >= -0.01)
    config = {
        'candidate': selected, 'features': specs[selected],
        'baseline_features': baseline['features'], 'catboost_params': PARAMS,
        'selection_rule': 'Smallest XY candidate within 0.01 of best mean development precision',
        'adoption_rule': 'Mean development gain >=0.01 and no fold loses more than 0.01',
        'supported_on_development': supported,
        'development_gains': gain.to_dict(),
        'development_folds': DEVELOPMENT_FOLDS, 'final_comparison_dates': HOLDOUT,
    }
    CONFIG.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Frozen XY candidate: {selected}; development adoption rule passed: {supported}', flush=True)
    return summary


def final_run(train, test, X, Xtest):
    config = json.loads(CONFIG.read_text())
    if config['catboost_params'] != PARAMS:
        raise ValueError('Frozen model parameters no longer match PARAMS')
    fit = train.window_start_ts.lt(HOLDOUT[0]).to_numpy()
    valid = (train.window_start_ts.ge(HOLDOUT[0]) & train.window_start_ts.lt(HOLDOUT[1])).to_numpy()
    y = train.target.to_numpy()
    rows = []
    validation = train.loc[valid, ['cookie_id', 'target', 'window_start_ts']].copy()
    validation['xy_n_points'] = X.loc[valid, 'xy_n_points'].to_numpy()
    for name, columns in {
        BASELINE: config['baseline_features'], 'behavior_xy': config['features'],
    }.items():
        model = make_model(columns)
        model.fit(X.loc[fit, columns], y[fit])
        scores = model.predict_proba(X.loc[valid, columns])[:, 1]
        validation[name] = scores
        rows.append({'model_id': name, 'n_features': len(columns), **evaluate(y[valid], scores)})
        if name == 'behavior_xy':
            pd.DataFrame({'feature': columns, 'importance': model.feature_importances_}).sort_values(
                'importance', ascending=False,
            ).to_csv(ARTIFACTS / 'final_feature_importance.csv', index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ARTIFACTS / 'final_metrics.csv', index=False)
    validation.to_csv(ARTIFACTS / 'final_predictions.csv', index=False)
    bootstrap_comparison(
        validation, baseline_column=BASELINE, selected_column='behavior_xy',
    ).to_csv(ARTIFACTS / 'final_bootstrap.csv')
    print('Final comparison (previously seen dates):\n' + metrics.to_string(index=False), flush=True)

    # Separate experimental submission; the previous model/config remain intact.
    columns = config['features']
    model = make_model(columns)
    model.fit(X[columns], y)
    scores = model.predict_proba(Xtest[columns])[:, 1]
    submission = pd.DataFrame({'cookie_id': test.cookie_id, 'score': scores})
    assert submission.cookie_id.equals(test.cookie_id) and submission.cookie_id.is_unique
    assert len(submission) == len(test) and list(submission.columns) == ['cookie_id', 'score']
    assert np.isfinite(scores).all() and submission.score.between(0, 1).all()
    submission_path = ROOT / 'submissions' / 'catboost_behavior_xy.csv'
    submission.to_csv(submission_path, index=False)
    model.save_model(str(ARTIFACTS / 'catboost_behavior_xy.cbm'))
    print(f'Saved experimental submission: {submission_path}', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['compare', 'refine', 'final'])
    args = parser.parse_args()
    train, test, X, Xtest = prepare_matrices()
    if args.stage == 'compare':
        compare(train, X)
    elif args.stage == 'refine':
        available = set(pd.read_csv(ARTIFACTS / 'development_metrics.csv').model_id)
        additional = [name for name in ['xy_spatial_coverage', 'xy_geometry_coverage'] if name not in available]
        if additional:
            compare(train, X, candidate_names=additional)
        else:
            print('Both compact comparisons are already saved; use compare to rerun all candidates.')
    else:
        final_run(train, test, X, Xtest)


if __name__ == '__main__':
    main()
