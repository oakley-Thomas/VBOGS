from scripts.bootstrap_stack_repo import git_bootstrap_script, normalize_repository_url


def test_normalize_repository_url_accepts_owner_name_and_https_url():
    assert (
        normalize_repository_url("oakley-Thomas/VBOGS")
        == "https://github.com/oakley-Thomas/VBOGS.git"
    )
    assert (
        normalize_repository_url("https://github.com/oakley-Thomas/VBOGS")
        == "https://github.com/oakley-Thomas/VBOGS.git"
    )


def test_git_bootstrap_script_uses_askpass_without_embedding_credentials():
    script = git_bootstrap_script(
        repo_url="https://github.com/oakley-Thomas/VBOGS.git",
        git_ref="main",
        repo_dir="/workspace/VBOGS",
    )

    assert "GIT_ASKPASS" in script
    assert "VBOGS_GITHUB_USER" in script
    assert "VBOGS_GITHUB_TOKEN" in script
    assert "https://github.com/oakley-Thomas/VBOGS.git" in script
    assert "docker exec" not in script
    assert "/var/run/docker.sock" not in script
    assert "@" not in "https://github.com/oakley-Thomas/VBOGS.git".split("//", 1)[1]
