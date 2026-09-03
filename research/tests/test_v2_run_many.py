from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import brazil_rv.v2.run_many as launcher
from brazil_rv.v2.config import (
    FULL_PROTOCOL,
    TRIAGE_PROTOCOL,
    load_protocol_preset,
    protocol_preset,
)


def _job(root: Path, index: int) -> launcher.TrajectoryJob:
    run_dir = root / f"job_{index}"
    seed = (11, 29, 47)[index % 3]
    fold = ("F1", "F2", "F3")[index % 3]
    return launcher.TrajectoryJob(
        name=f"job_{index}",
        seed=seed,
        fold=fold,
        run_dir=run_dir,
        command=("fake-train", str(run_dir), str(seed), fold),
        expected_manifest={"configuration": {"kind": "smoke"}},
    )


def _write_completed_manifest(command: list[str]) -> None:
    run_dir = Path(command[1])
    run_dir.joinpath("run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "seed": int(command[2]),
                "fold": command[3],
                "configuration": {"kind": "smoke"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_run_many_caps_concurrency_and_returns_plan_order(
    tmp_path, monkeypatch
) -> None:
    lock = threading.Lock()
    running = 0
    maximum = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal running, maximum
        with lock:
            running += 1
            maximum = max(maximum, running)
        time.sleep(0.02)
        _write_completed_manifest(command)
        with lock:
            running -= 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    jobs = tuple(_job(tmp_path, index) for index in range(5))

    outcomes = launcher.run_many(jobs, max_parallel=2)

    assert maximum == 2
    assert [outcome.name for outcome in outcomes] == [job.name for job in jobs]
    assert all(outcome.status == "completed" for outcome in outcomes)
    assert all(len(outcome.manifest_sha256) == 64 for outcome in outcomes)


def test_completed_trajectory_is_verified_and_skipped(tmp_path, monkeypatch) -> None:
    job = _job(tmp_path, 0)
    job.run_dir.mkdir()
    _write_completed_manifest(list(job.command))

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed trajectory was restarted")

    monkeypatch.setattr(launcher.subprocess, "run", fail_run)
    outcomes = launcher.run_many((job,), max_parallel=1)

    assert outcomes[0].status == "skipped_completed"


def test_incomplete_existing_root_aborts_before_launch(tmp_path, monkeypatch) -> None:
    pending = _job(tmp_path, 0)
    incomplete = _job(tmp_path, 1)
    incomplete.run_dir.mkdir()
    called = False

    def fail_run(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(launcher.subprocess, "run", fail_run)

    with pytest.raises(FileExistsError, match="will not be reused"):
        launcher.run_many((pending, incomplete), max_parallel=2)
    assert called is False
    assert not pending.run_dir.exists()


def test_failed_trajectory_is_not_retried(tmp_path, monkeypatch) -> None:
    calls = 0
    job = _job(tmp_path, 0)

    def fail_run(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(launcher.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="status 17"):
        launcher.run_many(
            (job,),
            max_parallel=1,
            launcher_manifest_path=tmp_path / "launcher.json",
        )
    assert calls == 1
    assert json.loads((tmp_path / "launcher.json").read_text())["status"] == "failed"


def test_plan_loader_and_parallel_limit(tmp_path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "max_parallel": 2,
                "jobs": [
                    {
                        "name": "network_F1_seed11",
                        "seed": 11,
                        "fold": "F1",
                        "run_dir": str(tmp_path / "run"),
                        "command": ["python", "-m", "trainer"],
                        "expected_manifest": {"stage": "F"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    jobs, max_parallel = launcher.load_plan(plan)

    assert max_parallel == 2
    assert jobs[0].seed == 11
    assert jobs[0].expected_manifest == {"stage": "F"}
    with pytest.raises(ValueError, match="between one and six"):
        launcher.run_many(jobs, max_parallel=7)


def test_named_protocol_json_files_match_frozen_presets() -> None:
    assert protocol_preset("triage") == TRIAGE_PROTOCOL
    assert protocol_preset("full") == FULL_PROTOCOL


def test_protocol_loader_rejects_any_config_drift(tmp_path) -> None:
    source = Path("research/configs/v2/triage.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["training"]["epochs"] = 19
    changed = tmp_path / "triage.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the frozen"):
        load_protocol_preset(changed)

    payload["training"]["epochs"] = 20
    payload["unexpected"] = True
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the frozen"):
        load_protocol_preset(changed)
