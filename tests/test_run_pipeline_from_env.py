import sys

from scripts.run_pipeline_from_env import pipeline_cmd, upload_cmd


def test_upload_cmd_does_not_echo_raw_service_account_credentials(monkeypatch):
    monkeypatch.setenv(
        "VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS",
        '{"private_key":"secret"}',
    )
    monkeypatch.setenv("VBOGS_GDRIVE_FOLDER_ID", "folder123")

    cmd = upload_cmd("configs/pipeline/portainer.yaml", "drive_sync")

    assert cmd[:4] == [
        sys.executable,
        "scripts/upload_google_drive.py",
        "--config",
        "configs/pipeline/portainer.yaml",
    ]
    assert "--folder-id" in cmd
    assert '{"private_key":"secret"}' not in cmd
    assert "--service-account-credentials" not in cmd


def test_pipeline_cmd_forwards_upload_without_echoing_raw_credentials(monkeypatch):
    monkeypatch.setenv("VBOGS_GDRIVE_UPLOAD", "1")
    monkeypatch.setenv(
        "VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS",
        '{"private_key":"secret"}',
    )
    monkeypatch.setenv("VBOGS_GDRIVE_FOLDER_ID", "folder123")
    monkeypatch.setenv("VBOGS_GDRIVE_DEST", "runs")

    cmd = pipeline_cmd("configs/pipeline/portainer.yaml", "drive_sync")

    assert cmd[:4] == [
        sys.executable,
        "scripts/run_drive_pipeline.py",
        "--config",
        "configs/pipeline/portainer.yaml",
    ]
    assert "--upload-google-drive" in cmd
    assert "--gdrive-folder-id" in cmd
    assert cmd[cmd.index("--gdrive-folder-id") + 1] == "folder123"
    assert "--gdrive-dest" in cmd
    assert "--gdrive-service-account-credentials" not in cmd
    assert '{"private_key":"secret"}' not in cmd
