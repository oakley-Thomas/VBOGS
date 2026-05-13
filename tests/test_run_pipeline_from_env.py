import sys

from scripts.run_pipeline_from_env import upload_cmd


def test_upload_cmd_does_not_echo_raw_service_account_credentials(monkeypatch):
    monkeypatch.setenv("VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS", '{"private_key":"secret"}')
    monkeypatch.setenv("VBOGS_GDRIVE_FOLDER_ID", "folder123")

    cmd = upload_cmd("pipeline_config.portainer.yaml", "drive_sync")

    assert cmd[:4] == [
        sys.executable,
        "scripts/upload_google_drive.py",
        "--config",
        "pipeline_config.portainer.yaml",
    ]
    assert "--folder-id" in cmd
    assert '{"private_key":"secret"}' not in cmd
    assert "--service-account-credentials" not in cmd
