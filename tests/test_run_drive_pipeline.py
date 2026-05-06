import pytest

from scripts import run_drive_pipeline


pytest.importorskip("yaml")


def fit_command(args):
    for step in run_drive_pipeline.build_steps(args):
        if step.name == "fit":
            return step.command
    raise AssertionError("fit step not found")


def test_pipeline_config_fit_keys_map_to_parser_defaults(tmp_path):
    config = tmp_path / "pipeline_config.yaml"
    config.write_text(
        """
pipeline:
  drive: test_drive
  start_at: fit
  stop_after: fit
fit:
  batch_buckets: 64,222
  k_max: 10
  k_growth_min_points: 999
  max_points_per_anchor: 222
""",
        encoding="utf-8",
    )

    args = run_drive_pipeline.parse_args(["--config", str(config)])
    command = fit_command(args)

    assert args.batch_buckets == "64,222"
    assert args.k_max == 10
    assert args.k_growth_min_points == 999
    assert args.max_points_per_anchor == 222
    assert "--batch-buckets" in command
    assert command[command.index("--batch-buckets") + 1] == "64,222"
    assert command[command.index("--k-max") + 1] == "10"
    assert command[command.index("--k-growth-min-points") + 1] == "999"
    assert command[command.index("--max-points-per-anchor") + 1] == "222"


def test_pipeline_cli_overrides_config_fit_keys(tmp_path):
    config = tmp_path / "pipeline_config.yaml"
    config.write_text(
        """
pipeline:
  drive: test_drive
fit:
  batch_buckets: 64,222
  k_max: 40
  k_growth_min_points: 256
  max_points_per_anchor: 222
""",
        encoding="utf-8",
    )

    args = run_drive_pipeline.parse_args(
        [
            "--config",
            str(config),
            "--batch-buckets",
            "64,333",
            "--k-max",
            "20",
            "--k-growth-min-points",
            "777",
            "--max-points-per-anchor",
            "333",
        ]
    )
    command = fit_command(args)

    assert command[command.index("--batch-buckets") + 1] == "64,333"
    assert command[command.index("--k-max") + 1] == "20"
    assert command[command.index("--k-growth-min-points") + 1] == "777"
    assert command[command.index("--max-points-per-anchor") + 1] == "333"


def test_pipeline_defaults_to_bounded_fit_when_config_disabled():
    args = run_drive_pipeline.parse_args(["--config", "", "--drive", "test_drive"])
    command = fit_command(args)

    assert args.max_points_per_anchor == 10000
    assert command[command.index("--max-points-per-anchor") + 1] == "10000"
