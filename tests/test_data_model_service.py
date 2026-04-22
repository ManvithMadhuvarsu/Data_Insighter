import pandas as pd

from data_model_service import profile_key_candidates


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
