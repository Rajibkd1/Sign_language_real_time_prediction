from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from app.config import Settings


class HistoryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.history_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed(self._read_all()))

    def add(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.enable_history:
            return payload
        item = {
            **payload,
            "id": payload.get("id") or str(uuid4()),
            "created_at": payload.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            items = self._read_all()
            items.append(item)
            items = items[-self.settings.max_history_items :]
            self._write_all(items)
        return item

    def delete(self, item_id: str) -> bool:
        with self._lock:
            items = self._read_all()
            kept = [item for item in items if item.get("id") != item_id]
            self._write_all(kept)
            return len(kept) != len(items)

    def clear(self) -> None:
        with self._lock:
            self._write_all([])

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return []
        return []

    def _write_all(self, items: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

