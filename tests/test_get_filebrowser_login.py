from scripts.get_filebrowser_login import (
    FilebrowserLogin,
    find_filebrowser_login,
    format_login,
    generate_password,
    mount_args_for_cli,
)


def test_find_filebrowser_login_parses_s6_generated_password_line():
    logs = """
2025/07/01 14:15:16 Warning: filebrowser.db can't be found.
2025/07/01 14:15:16 Randomly generated password for user 'admin': abc-123_DEF
2025/07/01 14:15:16 Listening on [::]:80
"""

    assert find_filebrowser_login(logs) == FilebrowserLogin(
        username="admin",
        password="abc-123_DEF",
    )


def test_find_filebrowser_login_prefers_latest_password_line():
    logs = """
Randomly generated password for user 'admin': first
Randomly generated password for user 'admin': second
"""

    assert find_filebrowser_login(logs) == FilebrowserLogin(
        username="admin",
        password="second",
    )


def test_format_shell_quotes_secret_values():
    output = format_login(
        FilebrowserLogin(username="admin", password="pass with spaces"),
        container="vbogs-filebrowser-1",
        ports=["0.0.0.0:8088"],
        output_format="shell",
    )

    assert "FILEBROWSER_CONTAINER=vbogs-filebrowser-1" in output
    assert "FILEBROWSER_USERNAME=admin" in output
    assert "FILEBROWSER_PASSWORD='pass with spaces'" in output
    assert "FILEBROWSER_PUBLISHED_PORTS=0.0.0.0:8088" in output


def test_generate_password_is_shell_friendly():
    password = generate_password()

    assert len(password) >= 24
    assert password == password.strip()
    assert "'" not in password
    assert '"' not in password


def test_mount_args_for_cli_uses_database_and_config_mounts():
    inspected = {
        "Mounts": [
            {
                "Type": "volume",
                "Name": "stack_vbogs-filebrowser-database",
                "Destination": "/database",
            },
            {
                "Type": "volume",
                "Name": "stack_vbogs-filebrowser-config",
                "Destination": "/config",
            },
            {
                "Type": "bind",
                "Source": "/repo",
                "Destination": "/srv/project",
            },
        ]
    }

    assert mount_args_for_cli(inspected) == [
        "--mount",
        "type=volume,source=stack_vbogs-filebrowser-config,target=/config",
        "--mount",
        "type=volume,source=stack_vbogs-filebrowser-database,target=/database",
    ]
