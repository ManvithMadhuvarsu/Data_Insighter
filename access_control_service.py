from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


VALID_ACCESS_ROLES = {'viewer', 'editor'}


def shared_entries(record: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not record:
        return []
    metadata = record.get('metadata') or {}
    entries = metadata.get('shared_with') or record.get('shared_with') or []
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        user = str(entry.get('user', '')).strip()
        role = str(entry.get('role', 'viewer')).strip().lower() or 'viewer'
        if not user:
            continue
        normalized.append({'user': user, 'role': role if role in VALID_ACCESS_ROLES else 'viewer'})
    return normalized


def row_policies(record: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not record:
        return []
    metadata = record.get('metadata') or {}
    policies = metadata.get('row_policies') or []
    normalized = []
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        user = str(policy.get('user', '')).strip()
        column = str(policy.get('column', '')).strip()
        values = policy.get('allowed_values') or []
        if not user or not column:
            continue
        normalized.append({
            'user': user,
            'column': column,
            'allowed_values': [str(value) for value in values if value is not None],
        })
    return normalized


def access_role(record: Dict[str, Any] | None, username: str) -> str | None:
    if not record or not username:
        return None
    if record.get('owner') == username:
        return 'owner'
    for entry in shared_entries(record):
        if entry['user'] == username:
            return entry['role']
    return None


def can_view_record(record: Dict[str, Any] | None, username: str) -> bool:
    return access_role(record, username) in {'owner', 'viewer', 'editor'}


def can_edit_record(record: Dict[str, Any] | None, username: str) -> bool:
    return access_role(record, username) in {'owner', 'editor'}


def with_shared_user(record: Dict[str, Any], target_user: str, role: str = 'viewer') -> Dict[str, Any]:
    role = role if role in VALID_ACCESS_ROLES else 'viewer'
    metadata = dict(record.get('metadata') or {})
    existing = [entry for entry in shared_entries(record) if entry['user'] != target_user]
    existing.append({'user': target_user, 'role': role})
    metadata['shared_with'] = sorted(existing, key=lambda item: item['user'].lower())
    return metadata


def with_row_policy(record: Dict[str, Any], target_user: str, column: str, allowed_values: List[str]) -> Dict[str, Any]:
    metadata = dict(record.get('metadata') or {})
    remaining = [
        policy
        for policy in row_policies(record)
        if not (policy['user'] == target_user and policy['column'] == column)
    ]
    remaining.append({
        'user': target_user,
        'column': column,
        'allowed_values': [str(value) for value in allowed_values if str(value).strip()],
    })
    metadata['row_policies'] = sorted(remaining, key=lambda item: (item['user'].lower(), item['column'].lower()))
    return metadata


def apply_row_policies(df: pd.DataFrame, record: Dict[str, Any] | None, username: str) -> pd.DataFrame:
    if not record or not username or record.get('owner') == username:
        return df.copy()

    applicable = [policy for policy in row_policies(record) if policy['user'] == username]
    if not applicable:
        return df.copy()

    filtered = df.copy()
    for policy in applicable:
        column = policy['column']
        allowed_values = {str(value) for value in policy['allowed_values']}
        if not allowed_values or column not in filtered.columns:
            continue
        filtered = filtered[filtered[column].astype(str).isin(allowed_values)]
    return filtered.reset_index(drop=True)
