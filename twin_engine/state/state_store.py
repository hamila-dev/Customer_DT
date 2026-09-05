"""Thread-safe persistence for the current Twin state per customer."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from twin_engine.state.twin_state import TwinState

import config


class TwinStateStore:
    """Keep one current `TwinState` per customer in memory and JSON storage."""

    def __init__(self, storage_path: Path = config.TWIN_STORE_PATH):
        self._storage_path = storage_path
        self._lock = threading.RLock()
        self._states: Dict[str, TwinState] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            for customer_id, state_dict in raw.items():
                self._states[customer_id] = TwinState.from_dict(state_dict)

    def _flush_to_disk(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {cid: state.to_dict() for cid, state in self._states.items()}
        tmp_path = self._storage_path.with_suffix(".tmp")
        # Replace the target only after serialization completes so a failed
        # write does not leave a partially written state file.
        tmp_path.write_text(json.dumps(serializable, indent=2, default=str))
        tmp_path.replace(self._storage_path)

    def save(self, state: TwinState) -> None:
        with self._lock:
            self._states[state.customer_id] = state
            self._flush_to_disk()

    def bulk_save(self, states: Iterable[TwinState]) -> None:
        with self._lock:
            for state in states:
                self._states[state.customer_id] = state
            self._flush_to_disk()

    def get(self, customer_id: str) -> Optional[TwinState]:
        with self._lock:
            return self._states.get(customer_id)

    def exists(self, customer_id: str) -> bool:
        with self._lock:
            return customer_id in self._states

    def list_all(self) -> List[TwinState]:
        with self._lock:
            return list(self._states.values())

    def count(self) -> int:
        with self._lock:
            return len(self._states)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._states) == 0


# Shared by API, event generation, and synchronization code.
twin_state_store = TwinStateStore()
