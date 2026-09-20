"""Reproducible group ablations on past dates; one final check on April 17–19.

Run from any working directory with the project's Python environment:
    python bot_detection_challenge/behavior_experiments.py compare
    python bot_detection_challenge/behavior_experiments.py refine
    python bot_detection_challenge/behavior_experiments.py final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from behavior_features import SEED, build_features, feature_columns, read_data
from metric import precision_at_recall

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts' / 'behavior'
DEVELOPMENT_FOLDS = [('2026-04-12', '2026-04-15'), ('2026-04-15', '2026-04-17')]
HOLDOUT = ('2026-04-17', '2026-04-20')
PARAMS = dict(
    iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5,
    loss_function='Logloss', auto_class_weights='Balanced',
    random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False,
)
CANDIDATES = {
    'baseline_numeric': ['counts'],
    'baseline_categories': ['counts', 'categories'],
    'rhythm_only': ['rhythm'],
    'counts_rhythm': ['counts', 'rhythm'],
    'counts_search': ['counts', 'search'],
    'counts_diversity': ['counts', 'diversity'],
    'counts_ua': ['counts', 'ua'],
    'counts_rhythm_activity': ['counts', 'rhythm', 'activity'],
    'counts_rhythm_sessions': ['counts', 'rhythm', 'sessions'],
    'counts_rhythm_search': ['counts', 'rhythm', 'search'],
    'counts_rhythm_diversity': ['counts', 'rhythm', 'diversity'],
    'counts_rhythm_actions': ['counts', 'rhythm', 'action_ratios'],
    'counts_rhythm_ua': ['counts', 'rhythm', 'ua'],
    'counts_rhythm_categories': ['counts', 'rhythm', 'categories'],
    'rhythm_search_ua': ['rhythm', 'search', 'ua'],
    'counts_rhythm_search_ua': ['counts', 'rhythm', 'search', 'ua'],
    'counts_rhythm_diversity_ua': ['counts', 'rhythm', 'diversity', 'ua'],
}


def make_model(columns, **overrides):
    return CatBoostClassifier(
        **(PARAMS | overrides), cat_features=[c for c in columns if c.endswith('_mode')],
    )


def evaluate(y, scores):
    p, r, _ = precision_recall_curve(y, scores)
    return {
        'precision_at_recall_70': precision_at_recall(y, scores),
        'pr_auc': auc(r, p), 'roc_auc': roc_auc_score(y, scores),
    }


def bootstrap_comparison(
    validation, repetitions=1000,
    baseline_column='baseline_categories', selected_column='catboost_behavior_selected',
):
    """Paired, stratified cookie bootstrap; does not measure future time drift."""
    rng = np.random.default_rng(SEED)
    y = validation.target.to_numpy()
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    baseline = validation[baseline_column].to_numpy()
    selected = validation[selected_column].to_numpy()
    samples = []
    for _ in range(repetitions):
        ix = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        a, b = precision_at_recall(y[ix], baseline[ix]), precision_at_recall(y[ix], selected[ix])
        samples.append([a, b, b - a])
    return pd.DataFrame(
        np.quantile(samples, [0.025, 0.5, 0.975], axis=0),
        columns=['baseline', 'selected', 'difference'], index=['p025', 'p50', 'p975'],
    )


def compare(train, X, specifications, filename):
    rows = []
    for model_id, columns in specifications.items():
        for start, end in DEVELOPMENT_FOLDS:
            fit_mask = train.window_start_ts.lt(start).to_numpy()
            val_mask = (train.window_start_ts.ge(start) & train.window_start_ts.lt(end)).to_numpy()
            model = make_model(columns)
            model.fit(X.loc[fit_mask, columns], train.target.to_numpy()[fit_mask])
            scores = model.predict_proba(X.loc[val_mask, columns])[:, 1]
            row = {
                'model_id': model_id, 'validation_start': start, 'validation_end': end,
                'n_features': len(columns), 'n_train': int(fit_mask.sum()),
                'n_valid': int(val_mask.sum()), 'n_valid_bots': int(train.target.to_numpy()[val_mask].sum()),
                **evaluate(train.target.to_numpy()[val_mask], scores),
            }
            rows.append(row)
            print(f'{model_id:36s} {start} features={len(columns):2d} P@R={row["precision_at_recall_70"]:.5f}', flush=True)
        pd.DataFrame(rows).to_csv(ARTIFACTS / f'{filename}.csv', index=False)
    result = pd.DataFrame(rows)
    summary = result.groupby('model_id').agg(
        n_features=('n_features', 'first'), mean_precision=('precision_at_recall_70', 'mean'),
        min_precision=('precision_at_recall_70', 'min'), mean_pr_auc=('pr_auc', 'mean'),
    ).sort_values('mean_precision', ascending=False)
    summary.to_csv(ARTIFACTS / f'{filename}_summary.csv')
    print(summary.to_string(), flush=True)
    return summary


def refine(train, X, groups):
    """Rank features only on dates preceding every development validation fold."""
    early = train.window_start_ts.lt(DEVELOPMENT_FOLDS[0][0]).to_numpy()
    base = feature_columns(groups, ['counts', 'rhythm', 'diversity', 'ua'])
    constants = [c for c in base if X.loc[early, c].nunique(dropna=False) <= 1]
    base = [c for c in base if c not in constants]
    ranker = make_model(base)
    ranker.fit(X.loc[early, base], train.target.to_numpy()[early])
    ranking = pd.DataFrame({'feature': base, 'importance': ranker.feature_importances_}).sort_values(
        'importance', ascending=False,
    )
    ranking.to_csv(ARTIFACTS / 'early_train_feature_ranking.csv', index=False)
    print('Constant features in early training:', constants, flush=True)
    specs = {
        'selected_groups': base,
        'selected_plus_sessions': base + groups['sessions'],
        'selected_plus_search': base + groups['search'],
        'selected_top20': ranking.feature.head(20).tolist(),
        'selected_top30': ranking.feature.head(30).tolist(),
        'selected_top40': ranking.feature.head(40).tolist(),
    }
    # Persist every candidate so the selection can be reproduced and reviewed.
    (ARTIFACTS / 'refinement_candidates.json').write_text(json.dumps(specs, indent=2))
    summary = compare(train, X, specs, 'refinement')
    select_features(summary, specs)


def select_features(summary, specs):
    # Prefer a smaller model when its mean precision is within 1.5 percentage
    # points of the best. This rule is fixed before evaluating the final holdout.
    tolerance = 0.015
    eligible = summary.loc[summary.mean_precision.ge(summary.mean_precision.max() - tolerance)]
    chosen = eligible.sort_values(['n_features', 'mean_precision'], ascending=[True, False]).index[0]
    selection = {
        'candidate': chosen, 'features': specs[chosen], 'catboost_params': PARAMS,
        'selection_rule': 'Fewest features within 0.015 of best mean development precision',
        'development_folds': DEVELOPMENT_FOLDS,
        'ranking_train_before': DEVELOPMENT_FOLDS[0][0], 'final_holdout': HOLDOUT,
    }
    (ROOT / 'selected_features.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(f'Frozen selection: {chosen}, {len(specs[chosen])} features', flush=True)


def final_run(train, test, X, Xtest, groups):
    # This frozen configuration is chosen on DEVELOPMENT_FOLDS only.
    selected = json.loads((ROOT / 'selected_features.json').read_text())
    columns = selected['features']
    if selected['catboost_params'] != PARAMS:
        raise ValueError('The frozen parameters must match PARAMS before final evaluation')
    start, end = HOLDOUT
    fit_mask = train.window_start_ts.lt(start).to_numpy()
    val_mask = (train.window_start_ts.ge(start) & train.window_start_ts.lt(end)).to_numpy()
    y = train.target.to_numpy()
    rows, predictions = [], {}
    for name, cols in {
        'baseline_categories': feature_columns(groups, ['counts', 'categories']),
        'catboost_behavior_selected': columns,
    }.items():
        model = make_model(cols)
        model.fit(X.loc[fit_mask, cols], y[fit_mask])
        score = model.predict_proba(X.loc[val_mask, cols])[:, 1]
        predictions[name] = score
        rows.append({'model_id': name, 'n_features': len(cols), **evaluate(y[val_mask], score)})
        if name == 'catboost_behavior_selected':
            pd.DataFrame({'feature': cols, 'importance': model.feature_importances_}).sort_values(
                'importance', ascending=False,
            ).to_csv(ARTIFACTS / 'holdout_feature_importance.csv', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACTS / 'holdout_metrics.csv', index=False)
    print('Final holdout (not used to select features):\n' + results.to_string(index=False), flush=True)
    validation = train.loc[val_mask, ['cookie_id', 'target', 'window_start_ts']].copy()
    for name, values in predictions.items():
        validation[name] = values
    validation.to_csv(ARTIFACTS / 'holdout_predictions.csv', index=False)
    bootstrap_comparison(validation).to_csv(ARTIFACTS / 'holdout_bootstrap.csv')

    model = make_model(columns)
    model.fit(X[columns], y)
    score = model.predict_proba(Xtest[columns])[:, 1]
    submission = pd.DataFrame({'cookie_id': test.cookie_id, 'score': score})
    assert submission.cookie_id.equals(test.cookie_id) and submission.cookie_id.is_unique
    assert len(submission) == len(test)
    assert np.isfinite(score).all() and submission.score.between(0, 1).all()
    submission_path = ROOT / 'submissions' / 'catboost_behavior_selected.csv'
    submission_path.parent.mkdir(exist_ok=True)
    submission.to_csv(submission_path, index=False)
    model.save_model(str(ARTIFACTS / 'catboost_behavior_selected.cbm'))
    print(f'Saved {submission_path} ({len(submission)} rows)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['compare', 'refine', 'final'])
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train, test, events = read_data(ROOT / 'data')
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)
    features, groups, audit = build_features(events, meta)
    (ARTIFACTS / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    (ARTIFACTS / 'feature_groups.json').write_text(json.dumps(groups, indent=2))
    X, Xtest = features.loc[train.cookie_id], features.loc[test.cookie_id]
    if args.stage == 'compare':
        specs = {name: feature_columns(groups, names) for name, names in CANDIDATES.items()}
        compare(train, X, specs, 'group_comparison')
    elif args.stage == 'refine':
        refine(train, X, groups)
    else:
        final_run(train, test, X, Xtest, groups)


if __name__ == '__main__':
    main()
