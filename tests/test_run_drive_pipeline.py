from scripts.run_drive_pipeline import TORCH_SERVICE, build_parser, build_steps, selected_steps


def test_run_output_root_routes_v1_stage_outputs():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "2013_05_28_drive_0007_sync",
            "--run-output-root",
            "outputs/v1_0",
            "--start-at",
            "map-viz",
            "--stop-after",
            "bundle",
        ]
    )
    steps = selected_steps(build_steps(args), args.start_at, args.stop_after)
    by_name = {step.name: step for step in steps}

    assert list(by_name) == ["map-viz", "render", "nbv", "nbv-viz", "bundle"]
    assert by_name["map-viz"].service == TORCH_SERVICE
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/pointclouds/anchors" in by_name["map-viz"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/views" in by_name["render"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/nbv" in by_name["nbv"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/nbv/viz" in by_name["nbv-viz"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync" in by_name["bundle"].command


def test_run_output_root_keeps_explicit_stage_output_override():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--run-output-root",
            "outputs/v1_0",
            "--render-output-dir",
            "outputs/custom_views",
        ]
    )
    render_step = next(step for step in build_steps(args) if step.name == "render")

    assert "outputs/custom_views" in render_step.command
    assert "outputs/v1_0/drive_sync/views" not in render_step.command
