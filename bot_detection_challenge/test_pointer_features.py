"""Joint geometry, missing-data barriers and timestamp ambiguity checks."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pointer_features import build_pointer_features


class PointerFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({
            'cookie_id': ['cookie', 'empty'],
            'window_start_ts': pd.to_datetime(['2026-04-10'] * 2),
            'window_end_ts': pd.to_datetime(['2026-04-11'] * 2),
        })

    def events(self, xy, seconds=None):
        if seconds is None:
            seconds = np.arange(len(xy)) * 10
        return pd.DataFrame({
            'cookie_id': 'cookie',
            'event_ts': pd.Timestamp('2026-04-10') + pd.to_timedelta(seconds, unit='s'),
            'platform': 'desktop', 'user_agent': 'Mozilla/5.0 Chrome/120.0',
            'pointer_x': [p[0] for p in xy], 'pointer_y': [p[1] for p in xy],
        })

    def test_pairing_changes_geometry_with_same_marginals(self):
        diagonal = self.events([(0, 0), (0, 0), (1, 1), (1, 1)])
        square = self.events([(0, 0), (0, 1), (1, 0), (1, 1)])
        a, _, _ = build_pointer_features(diagonal, self.meta)
        b, _, _ = build_pointer_features(square, self.meta)
        self.assertEqual(a.loc['cookie', 'xy_anisotropy'], 1)
        self.assertEqual(b.loc['cookie', 'xy_anisotropy'], 0)
        self.assertEqual(a.loc['cookie', 'xy_repeat_share'], 0.5)
        self.assertEqual(b.loc['cookie', 'xy_repeat_share'], 0)

    def test_geometry_is_translation_and_uniform_scale_invariant(self):
        ev = self.events([(0, 1), (3, 5), (4, 7), (1, 3)])
        expected, _, _ = build_pointer_features(ev, self.meta)
        changed = ev.copy()
        changed.pointer_x = 7 * changed.pointer_x + 300
        changed.pointer_y = 7 * changed.pointer_y - 90
        actual, _, _ = build_pointer_features(changed, self.meta)
        assert_frame_equal(expected, actual)

    def test_no_steps_across_missing_tied_or_distant_events(self):
        cases = [
            self.events([(0, 0), (np.nan, np.nan), (3, 4)]),
            self.events([(0, 0), (1, 1), (2, 2), (3, 4)], [0, 10, 10, 20]),
            self.events([(0, 0), (3, 4)], [0, 1801]),
        ]
        for ev in cases:
            with self.subTest(events=ev.to_dict('list')):
                f, _, audit = build_pointer_features(ev, self.meta)
                self.assertEqual(audit['valid_adjacent_steps'], 0)
                self.assertTrue(pd.isna(f.loc['cookie', 'xy_zero_step_share']))
                self.assertTrue(pd.isna(f.loc['cookie', 'xy_turn_cos_mean']))

    def test_direction_and_repetition_are_joint(self):
        forward, _, _ = build_pointer_features(self.events([(0, 0), (3, 4), (6, 8)]), self.meta)
        reverse, _, _ = build_pointer_features(self.events([(0, 0), (3, 4), (0, 0)]), self.meta)
        self.assertEqual(forward.loc['cookie', 'xy_turn_cos_mean'], 1)
        self.assertEqual(reverse.loc['cookie', 'xy_turn_cos_mean'], -1)
        self.assertEqual(forward.loc['cookie', 'xy_repeated_vector_share'], 0.5)

    def test_order_duplicates_window_and_labels(self):
        ev = self.events([(0, 0), (1, 1), (2, 3), (4, 5)], [0, 10, 10, 20])
        expected, _, _ = build_pointer_features(ev, self.meta)
        outside = ev.iloc[[0]].assign(event_ts=pd.Timestamp('2026-04-11'), pointer_x=999)
        changed = pd.concat([ev, ev, outside]).sample(frac=1, random_state=0)
        actual, _, _ = build_pointer_features(changed, self.meta.assign(target=[1, 0]))
        assert_frame_equal(expected, actual)

    def test_stationary_single_and_empty_points(self):
        ev = self.events([(5, 7)] * 3)
        f, _, _ = build_pointer_features(ev, self.meta)
        self.assertTrue(all(pd.api.types.is_float_dtype(dtype) for dtype in f.dtypes))
        self.assertEqual(f.loc['cookie', 'xy_zero_step_share'], 1)
        self.assertTrue(pd.isna(f.loc['cookie', 'xy_anisotropy']))
        self.assertTrue(pd.isna(f.loc['cookie', 'xy_turn_cos_mean']))
        self.assertEqual(f.loc['empty', 'xy_n_points'], 0)
        self.assertTrue(f.loc['empty'].drop('xy_n_points').isna().all())
        one, _, _ = build_pointer_features(ev.iloc[:1], self.meta)
        self.assertTrue(pd.isna(one.loc['cookie', 'xy_repeat_share']))
        none, _, _ = build_pointer_features(ev.iloc[:0], self.meta)
        self.assertTrue(none.xy_n_points.eq(0).all())


if __name__ == '__main__':
    unittest.main()
