"""Reconstruct historical Twin snapshots from enriched transition records."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import config
from twin_engine.state.state_store import twin_state_store
from twin_engine.state.twin_state import TwinState


def get_state_at(customer_id: str, target_timestamp: str) -> Optional[TwinState]:
    """Return the latest logged state at or before an ISO 8601 timestamp.

    The first enriched record supplies the pre-event baseline for requests
    before the customer's first logged event. Older event-log lines without
    snapshots are ignored; customers with no usable records fall back to the
    live store.
    """
    if not config.EVENT_LOG_PATH.exists():
        return twin_state_store.get(customer_id)

    target_dt = datetime.fromisoformat(target_timestamp)
    best_record = None
    earliest_record = None

    with config.EVENT_LOG_PATH.open() as event_log:
        for line in event_log:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("customer_id") != customer_id:
                continue
            if "prev_state" not in record or "new_state" not in record:
                # Logs written before snapshot enrichment cannot reconstruct state.
                continue

            try:
                occurred_at = datetime.fromisoformat(record["occurred_at"])
            except (KeyError, TypeError, ValueError):
                continue

            if earliest_record is None:
                earliest_record = (occurred_at, record)
            elif occurred_at < earliest_record[0]:
                earliest_record = (occurred_at, record)

            if occurred_at <= target_dt and (best_record is None or occurred_at > best_record[0]):
                best_record = (occurred_at, record)

    if best_record is not None:
        return TwinState.from_dict(best_record[1]["new_state"])
    if earliest_record is not None:
        return TwinState.from_dict(earliest_record[1]["prev_state"])
    return twin_state_store.get(customer_id)