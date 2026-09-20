"""Cookie-level features computed strictly inside [window_start_ts, window_end_ts)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 0
EVENT_TYPES = [
    'search_results_view', 'item_view', 'photo_swipe', 'seller_page_view',
    'contact_phone_show', 'contact_chat_open', 'contact_message_sent',
    'favorite_add', 'login', 'captcha_shown',
]
CATEGORY_SOURCES = [
    'platform', 'ua_os', 'ua_client', 'item_category', 'item_location', 'seller_type',
]


def prepare_events(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate raw events; normalize platform/UA; keep only observable events."""
    if not meta['cookie_id'].is_unique:
        raise ValueError('Expected one observation window per cookie_id')
    ev = events.drop_duplicates().copy()
    ev['event_ts'] = pd.to_datetime(ev['event_ts'])
    windows = meta[['cookie_id', 'window_start_ts', 'window_end_ts']].copy()
    for column in ['window_start_ts', 'window_end_ts']:
        windows[column] = pd.to_datetime(windows[column])
    ev = ev.merge(windows, on='cookie_id', validate='many_to_one')
    ev = ev.loc[
        ev.event_ts.ge(ev.window_start_ts) & ev.event_ts.lt(ev.window_end_ts)
    ].copy()
    ev['platform'] = (
        ev['platform'].str.strip().str.lower().replace({'iphone': 'ios', 'desktop': 'web'})
    )
    ua = ev['user_agent'].fillna('').str.lower()
    ev['ua_os'] = np.select(
        [ua.str.contains('android'), ua.str.contains(r'iphone|ipad|\bios\b'),
         ua.str.contains('windows'), ua.str.contains(r'macintosh|mac os x'),
         ua.str.contains('linux')],
        ['android', 'ios', 'windows', 'macos', 'linux'], default='unknown',
    )
    ev['ua_http'] = ua.str.match(
        r'^(?:python-requests|python-urllib3|curl|scrapy|go-http-client|node-fetch)/'
    ).astype(float)
    ev['ua_headless'] = ua.str.contains('headlesschrome/').astype(float)
    ev['ua_app'] = ua.str.startswith('avito/').astype(float)
    ev['ua_client'] = np.select(
        [ev.ua_app.eq(1), ev.ua_http.eq(1), ua.str.contains('yabrowser/'),
         ev.ua_headless.eq(1), ua.str.contains('chrome/'), ua.str.contains('firefox/'),
         ua.str.contains('safari/')],
        ['avito_app', 'http_client', 'yandex', 'headless_chrome', 'chrome', 'firefox', 'safari'],
        default='other',
    )
    # Equal timestamps have no observed internal order. No transition features use it.
    return ev.sort_values(['cookie_id', 'event_ts'], kind='stable').reset_index(drop=True)


def build_features(events: pd.DataFrame, meta: pd.DataFrame):
    """Return features (in meta order), named candidate groups, and a data audit.

    Missing statistics stay NaN (CatBoost handles them). Counts for empty cookies
    are zero. No labels, raw IDs, absolute dates, UA versions or coordinates enter X.
    """
    ev = prepare_events(events, meta)
    g = ev.groupby('cookie_id', sort=False)
    index = pd.Index(meta.cookie_id, name='cookie_id')
    f = pd.DataFrame(index=index)
    groups = {}

    def add_group(name, values):
        groups[name] = list(values.columns)
        for column in values:
            f[column] = values[column].reindex(index)

    counts = g.agg(
        n_events=('event_ts', 'size'), event_type_nunique=('event_name', 'nunique'),
        item_nunique=('item_id', 'nunique'), item_category_nunique=('item_category', 'nunique'),
        item_location_nunique=('item_location', 'nunique'),
        search_query_nunique=('search_query', 'nunique'),
    )
    event_counts = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_TYPES, fill_value=0)
    counts = counts.join(event_counts.add_prefix('n_')).reindex(index).fillna(0)
    add_group('counts', counts)

    categories = pd.DataFrame(index=index)
    for column in CATEGORY_SOURCES:
        value_counts = ev.groupby(['cookie_id', column]).size().rename('count').reset_index()
        modes = (
            value_counts.sort_values(['cookie_id', 'count', column], ascending=[True, False, True])
            .drop_duplicates('cookie_id').set_index('cookie_id')[column]
        )
        categories[f'{column}_mode'] = modes.reindex(index).fillna('unknown').astype(str)
    add_group('categories', categories)

    ev['gap'] = g.event_ts.diff().dt.total_seconds()
    gaps = ev.groupby('cookie_id').gap
    rhythm = gaps.agg(['median', 'mean', 'std', 'min', 'max']).add_prefix('gap_')
    rhythm['gap_cv'] = rhythm.gap_std / rhythm.gap_mean.replace(0, np.nan)
    quantiles = gaps.quantile([0.25, 0.75]).unstack().reindex(columns=[0.25, 0.75])
    rhythm['gap_iqr_over_median'] = (quantiles[0.75] - quantiles[0.25]) / (rhythm.gap_median + 1)
    for name, mask in {
        'gap_le_1s_share': ev.gap.le(1), 'gap_le_10s_share': ev.gap.le(10),
        'gap_gt_30m_share': ev.gap.gt(1800),
    }.items():
        rhythm[name] = mask.where(ev.gap.notna()).groupby(ev.cookie_id).mean()
    gap_counts = ev.dropna(subset=['gap']).groupby(['cookie_id', 'gap']).size()
    rhythm['gap_mode_share'] = gap_counts.groupby(level=0).max() / gaps.count()
    rhythm['gap_unique_share'] = gaps.nunique() / gaps.count().replace(0, np.nan)
    add_group('rhythm', rhythm)

    span = g.event_ts.max() - g.event_ts.min()
    activity = pd.DataFrame({'active_span_s': span.dt.total_seconds()})
    activity['events_per_active_minute'] = (counts.n_events - 1).clip(lower=0) / (
        activity.active_span_s / 60 + 1
    )
    ev['hour'] = ev.event_ts.dt.hour
    hours = ev.groupby(['cookie_id', 'hour']).size()
    hour_p = hours / hours.groupby(level=0).transform('sum')
    activity['active_hours'] = hours.groupby(level=0).size()
    activity['hour_entropy'] = (-hour_p * np.log(hour_p)).groupby(level=0).sum()
    activity['max_hour_share'] = hour_p.groupby(level=0).max()
    activity['night_share'] = ev.hour.between(0, 5).groupby(ev.cookie_id).mean()
    # Grid alignment is used only for concentration; never the absolute date.
    ev['minute'] = ev.event_ts.dt.floor('min')
    activity['max_minute_share'] = ev.groupby(['cookie_id', 'minute']).size().groupby(level=0).max() / counts.n_events
    add_group('activity', activity)

    ev['session'] = (ev.gap.isna() | ev.gap.gt(1800)).groupby(ev.cookie_id).cumsum()
    sessions = ev.groupby(['cookie_id', 'session']).agg(
        size=('event_ts', 'size'), start=('event_ts', 'min'), end=('event_ts', 'max'),
    )
    sessions['duration'] = (sessions.end - sessions.start).dt.total_seconds()
    session_features = pd.DataFrame({
        'n_sessions': sessions.groupby(level=0).size(),
        'max_session_event_share': sessions['size'].groupby(level=0).max() / counts.n_events,
        'session_duration_median': sessions.duration.groupby(level=0).median(),
        'session_duration_max': sessions.duration.groupby(level=0).max(),
    })
    add_group('sessions', session_features)

    diversity = pd.DataFrame(index=index)
    for column, short in [('item_id', 'item'), ('item_category', 'category'),
                          ('item_location', 'location'), ('search_query', 'query'),
                          ('event_name', 'event')]:
        frequencies = ev.groupby(['cookie_id', column]).size()
        observed = frequencies.groupby(level=0).sum()
        probabilities = frequencies / frequencies.groupby(level=0).transform('sum')
        diversity[f'{short}_unique_share'] = frequencies.groupby(level=0).size() / observed
        diversity[f'{short}_top_share'] = probabilities.groupby(level=0).max()
        diversity[f'{short}_entropy'] = (-probabilities * np.log(probabilities)).groupby(level=0).sum()
    add_group('diversity', diversity)

    search_ev = ev.loc[ev.event_name.eq('search_results_view')].copy()
    search_groups = search_ev.groupby('cookie_id')
    search = search_groups.search_page.agg(['max', 'mean', 'nunique']).add_prefix('search_page_')
    search['search_deep_share'] = search_ev.search_page.gt(3).where(
        search_ev.search_page.notna()
    ).groupby(search_ev.cookie_id).mean()
    # Breadth per query uses known query/page pairs, without arbitrary ordering of ties.
    pairs = search_ev.dropna(subset=['search_query', 'search_page'])
    pages_per_query = pairs.groupby(['cookie_id', 'search_query']).search_page.nunique()
    search['pages_per_query_max'] = pages_per_query.groupby(level=0).max()
    search['query_page_repeat_share'] = 1 - (
        pairs.drop_duplicates(['cookie_id', 'search_query', 'search_page']).groupby('cookie_id').size()
        / pairs.groupby('cookie_id').size()
    )
    add_group('search', search)

    ratios = pd.DataFrame(index=index)
    denominators = counts.n_events.replace(0, np.nan)
    for event in ['search_results_view', 'item_view', 'photo_swipe', 'seller_page_view',
                  'favorite_add', 'captcha_shown', 'login']:
        ratios[f'share_{event}'] = counts[f'n_{event}'] / denominators
    contact_count = counts[['n_contact_phone_show', 'n_contact_chat_open', 'n_contact_message_sent']].sum(axis=1)
    ratios['share_contacts'] = contact_count / denominators
    ratios['contacts_per_item'] = contact_count / (counts.n_item_view + 1)
    add_group('action_ratios', ratios)

    ua_features = g[['ua_http', 'ua_headless', 'ua_app']].mean().add_suffix('_share')
    ua_features['ua_client_nunique'] = g.ua_client.nunique()
    ua_features['ua_nunique'] = g.user_agent.nunique()
    add_group('ua', ua_features)

    audit = {
        'raw_rows': len(events), 'exact_duplicates': int(events.duplicated().sum()),
        'rows_in_windows': len(ev),
        'rows_outside_windows': len(events.drop_duplicates()) - len(ev),
        'cookies_without_events': int(counts.n_events.eq(0).sum()),
        'adjacent_time_reversals_in_raw_file': int(
            pd.to_datetime(events.event_ts).groupby(events.cookie_id).diff().dt.total_seconds().lt(0).sum()
        ),
        'missing_share': events.isna().mean().to_dict(),
    }
    return f, groups, audit


def feature_columns(groups, names):
    """Stable union of groups; caller chooses which candidates enter a model."""
    return list(dict.fromkeys(column for name in names for column in groups[name]))


def read_data(data_dir):
    date_columns = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
    train = pd.read_csv(data_dir / 'train.csv', parse_dates=date_columns)
    test = pd.read_csv(data_dir / 'test.csv', parse_dates=date_columns)
    events = pd.read_csv(data_dir / 'events.csv.gz', parse_dates=['event_ts'])
    return train, test, events
