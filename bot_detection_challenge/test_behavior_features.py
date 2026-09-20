"""Checks for leakage, timestamp ordering and structural missing values."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from behavior_features import build_features, prepare_events


class BehaviorFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({
            'cookie_id': ['seen', 'empty'],
            'window_start_ts': pd.to_datetime(['2026-04-10', '2026-04-10']),
            'window_end_ts': pd.to_datetime(['2026-04-11', '2026-04-11']),
        })
        common = {
            'cookie_id': 'seen', 'eid': 200, 'event_name': 'item_view',
            'platform': ' WEB ', 'user_agent': 'Mozilla/5.0 HeadlessChrome/120.0 Safari/537.36',
            'item_id': 123, 'item_category': 'phones', 'item_location': 'city',
            'seller_type': 'private', 'search_query': np.nan, 'search_page': np.nan,
            'pointer_x': np.nan, 'pointer_y': np.nan,
        }
        first = common | {'event_ts': pd.Timestamp('2026-04-10 00:00:00')}
        second = common | {'event_ts': pd.Timestamp('2026-04-10 00:00:10')}
        tied = second | {'eid': 300, 'event_name': 'photo_swipe'}
        before = common | {'event_ts': pd.Timestamp('2026-04-09 23:59:59')}
        at_end = common | {'event_ts': pd.Timestamp('2026-04-11 00:00:00')}
        self.events = pd.DataFrame([second, before, first, tied, first, at_end])

    def test_window_duplicate_and_tied_events(self):
        features, _, _ = build_features(self.events, self.meta)
        self.assertEqual(features.loc['seen', 'n_events'], 3)
        self.assertEqual(features.loc['seen', 'gap_median'], 5)
        self.assertEqual(features.loc['seen', 'gap_min'], 0)
        self.assertEqual(features.loc['seen', 'active_span_s'], 10)
        self.assertEqual(features.loc['seen', 'ua_headless_share'], 1)
        self.assertEqual(features.loc['seen', 'ua_client_mode'], 'headless_chrome')
        self.assertEqual(features.loc['seen', 'platform_mode'], 'web')
        self.assertEqual(features.loc['empty', 'n_events'], 0)
        self.assertTrue(pd.isna(features.loc['empty', 'gap_median']))

    def test_order_and_duplicates_do_not_change_features(self):
        expected, _, _ = build_features(self.events, self.meta)
        shuffled = pd.concat([self.events, self.events]).sample(frac=1, random_state=7)
        actual, _, _ = build_features(shuffled, self.meta)
        assert_frame_equal(expected, actual)

    def test_labels_and_outside_events_do_not_change_features(self):
        expected, _, _ = build_features(self.events, self.meta)
        future = self.events.iloc[[0]].assign(event_ts=pd.Timestamp('2027-01-01'), item_id=9999)
        actual, _, _ = build_features(
            pd.concat([self.events, future]), self.meta.assign(target=[1, 0]),
        )
        assert_frame_equal(expected, actual)

    def test_structural_missingness_and_denominators(self):
        search = self.events.iloc[[0]].assign(
            event_name='search_results_view', event_ts=pd.Timestamp('2026-04-10 00:01:00'),
            search_query='phone', search_page=5, item_id=np.nan,
        )
        features, _, _ = build_features(pd.concat([self.events, search]), self.meta)
        self.assertEqual(features.loc['seen', 'search_deep_share'], 1)
        self.assertAlmostEqual(features.loc['seen', 'item_unique_share'], 1 / 3)
        self.assertEqual(features.loc['seen', 'search_page_max'], 5)
        self.assertTrue(pd.isna(features.loc['empty', 'search_page_max']))

    def test_window_is_unique_per_cookie(self):
        with self.assertRaises(ValueError):
            prepare_events(self.events, pd.concat([self.meta, self.meta]))

    def test_no_observable_events(self):
        features, _, _ = build_features(self.events.iloc[:0], self.meta)
        self.assertTrue(features.n_events.eq(0).all())
        self.assertTrue(features.gap_median.isna().all())


if __name__ == '__main__':
    unittest.main()
