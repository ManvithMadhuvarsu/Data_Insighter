import pandas as pd

from access_control_service import (
    access_role,
    apply_row_policies,
    can_edit_record,
    can_view_record,
    with_row_policy,
    with_shared_user,
)


def test_shared_access_roles_are_applied():
    record = {
        'owner': 'owner',
        'metadata': {
            'shared_with': [{'user': 'viewer', 'role': 'viewer'}, {'user': 'editor', 'role': 'editor'}],
        },
    }

    assert access_role(record, 'owner') == 'owner'
    assert access_role(record, 'viewer') == 'viewer'
    assert access_role(record, 'editor') == 'editor'
    assert can_view_record(record, 'viewer') is True
    assert can_edit_record(record, 'viewer') is False
    assert can_edit_record(record, 'editor') is True


def test_row_policies_filter_shared_dataframe():
    df = pd.DataFrame({
        'region': ['North', 'South', 'North'],
        'revenue': [100, 200, 300],
    })
    record = {
        'owner': 'owner',
        'metadata': {
            'row_policies': [{'user': 'viewer', 'column': 'region', 'allowed_values': ['North']}],
        },
    }

    filtered = apply_row_policies(df, record, 'viewer')

    assert filtered['region'].tolist() == ['North', 'North']
    assert filtered['revenue'].tolist() == [100, 300]


def test_updating_share_and_row_policy_metadata():
    record = {'owner': 'owner', 'metadata': {}}

    shared_metadata = with_shared_user(record, 'analyst', 'editor')
    secured_metadata = with_row_policy({'metadata': shared_metadata}, 'analyst', 'region', ['North', 'West'])

    assert shared_metadata['shared_with'][0] == {'user': 'analyst', 'role': 'editor'}
    assert secured_metadata['row_policies'][0]['allowed_values'] == ['North', 'West']
