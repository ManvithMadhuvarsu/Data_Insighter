import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from data_model_service import join_datasets, profile_key_candidates


def test_profile_key_candidates_prioritizes_unique_id_columns():
    df = pd.DataFrame({
        'customer_id': ['c1', 'c2', 'c3', 'c4'],
        'region': ['East', 'East', 'West', 'West'],
        'sales': [10, 20, 30, 40],
    })

    candidates = profile_key_candidates(df)

    assert candidates
    assert candidates[0]['column'] == 'customer_id'
    assert candidates[0]['role_hint'] == 'primary_key'


def test_join_datasets_normalizes_string_keys_before_merge():
    with TemporaryDirectory() as temp_dir:
        left_path = Path(temp_dir) / 'left.csv'
        right_path = Path(temp_dir) / 'right.csv'

        pd.DataFrame({
            'customer_code': ['A ', 'B ', 'C '],
            'sales': [10, 20, 30],
        }).to_csv(left_path, index=False)
        pd.DataFrame({
            'customer_code': ['A', 'B', 'C'],
            'segment': ['East', 'West', 'North'],
        }).to_csv(right_path, index=False)

        joined = join_datasets(
            {
                'stored_path': str(left_path),
                'source_name': 'left.csv',
            },
            {
                'stored_path': str(right_path),
                'source_name': 'right.csv',
            },
            left_key='customer_code',
            right_key='customer_code',
            join_type='inner',
        )

        assert len(joined) == 3
        assert 'segment' in joined.columns
