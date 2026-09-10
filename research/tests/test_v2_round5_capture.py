import base64
import json
from datetime import UTC, datetime

import pytest

from brazil_rv.v2 import round5_capture as module


def test_forward_roster_uses_only_published_exact_unexpired_contracts():
    page = 'futures_app.inner_futures_per_month = {"data":["RB0","RB2701","RB2610","RB1910","RB2701"],"type":1};'
    actual, skipped = module.listed_contracts(
        page, "RB", datetime(2026, 9, 10, tzinfo=UTC)
    )
    assert actual == ["RB2610", "RB2701"]
    assert skipped == ["RB0", "RB1910"]


def test_raw_minute_capture_never_inserts_missing_minutes():
    payload = 'var _brazil_rv=([{"d":"2026-09-10 21:01:00","c":"10"},{"d":"2026-09-10 21:03:00","c":"12"}]);'
    assert len(module.minute_rows(payload)) == 2
    assert module.minute_rows("var _brazil_rv=(null);") == []
    with pytest.raises(ValueError, match="duplicate or unordered"):
        module.minute_rows(
            'callback([{"d":"2026-09-10 21:01:00"},{"d":"2026-09-10 21:01:00"}]);'
        )


def test_index_capture_paginates_and_marks_inactive_preview_even_with_rows(
    tmp_path, monkeypatch
):
    def get(path, url):
        request = json.loads(base64.b64decode(url.rsplit("/", 1)[1]))
        page = request["pageNumber"]
        if "GetConfigurations" in url:
            response = {"preview": [{"has": 0, "version": 0}]}
        else:
            response = {
                "page": {"totalRecords": 2, "totalPages": 2},
                "header": {"date": "10/09/26"},
                "results": [{"cod": f"TEST{page}"}],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response))
        return {"path": str(path), "url": url}

    monkeypatch.setattr(module, "_get", get)
    result = module.capture_index(tmp_path, "IBOV")
    assert result["preview_active_at_capture"] is False
    assert [view["rows"] for view in result["views"]] == [2, 2, 2]
    assert len(result["sources"]) == 7


def test_incomplete_index_pagination_is_not_success(tmp_path, monkeypatch):
    def get(path, url):
        response = (
            {"preview": []}
            if "GetConfigurations" in url
            else {
                "page": {"totalRecords": 2, "totalPages": 1},
                "results": [{"cod": "TEST3"}],
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response))
        return {"path": str(path), "url": url}

    monkeypatch.setattr(module, "_get", get)
    with pytest.raises(ValueError, match="incomplete"):
        module.capture_index(tmp_path, "IBOV")
