import errno
import sys

from scripts import run_pipeline_stage


def test_stage_wrapper_keeps_existing_docker_process_group(tmp_path, monkeypatch):
    pid_file = tmp_path / "stage.pgid"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline_stage.py",
            "--pid-file",
            str(pid_file),
            "--",
            "python",
            "stage.py",
        ],
    )
    monkeypatch.setattr(
        run_pipeline_stage.os,
        "setsid",
        lambda: (_ for _ in ()).throw(PermissionError(errno.EPERM, "Operation not permitted")),
    )
    monkeypatch.setattr(run_pipeline_stage.os, "getpid", lambda: 1234)
    monkeypatch.setattr(run_pipeline_stage.os, "getpgrp", lambda: 1234)
    executed: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        run_pipeline_stage.os,
        "execvp",
        lambda executable, arguments: executed.append((executable, arguments)),
    )

    run_pipeline_stage.main()

    assert pid_file.read_text(encoding="utf-8") == "1234\n"
    assert executed == [("python", ["python", "stage.py"])]
