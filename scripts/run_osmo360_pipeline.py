#!/usr/bin/env python3
"""Run an RGB-only Osmo 360 video reconstruction through Octree-AnyGS.

This program is called by the web scheduler.  It runs heavyweight work in the
dedicated COLMAP and Torch containers, while all hand-offs are ordinary files
under the per-run workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.osmo360 import (
    MAX_UPLOAD_BYTES, PROFILE, STAGES, VideoInfo, image_name, parse_ffprobe,
    profile_manifest, projection_command, sample_timestamps, validate_video_info,
    split_timestamp_groups, virtual_cameras, write_rig_config,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--scene-id", required=True)
    result.add_argument("--gpu", default="0")
    result.add_argument("--start-at", choices=STAGES, default="validate")
    result.add_argument("--stop-after", choices=STAGES, default="bundle")
    result.add_argument("--event-log", type=Path)
    result.add_argument("--cancel-file", type=Path)
    result.add_argument("--sfm-container", default="")
    result.add_argument("--torch-container", default="")
    result.add_argument("--dry-run", action="store_true")
    return result


def emit(path: Path | None, event_type: str, **payload: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": event_type, **payload}, sort_keys=True) + "\n")


def checked(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def service_container(service: str, explicit: str) -> str:
    if explicit:
        return explicit
    filters = ["--filter", f"label=com.docker.compose.service={service}", "--filter", "status=running"]
    project = os.environ.get("VBOGS_COMPOSE_PROJECT", "")
    if project:
        filters += ["--filter", f"label=com.docker.compose.project={project}"]
    result = subprocess.run(["docker", "ps", "-q", *filters], check=True, capture_output=True, text=True)
    values = [value.strip() for value in result.stdout.splitlines() if value.strip()]
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one running {service} container")
    return values[0]


def container_command(container: str, command: list[str], *, gpu: str | None = None) -> list[str]:
    result = ["docker", "exec", "-w", "/workspace/VBOGS"]
    if gpu is not None:
        result += ["-e", f"CUDA_VISIBLE_DEVICES={gpu}"]
    return [*result, container, *command]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_video(workspace: Path) -> Path:
    inputs = workspace / "input"
    files = [path for path in inputs.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}]
    if len(files) != 1:
        raise RuntimeError("Expected exactly one uploaded .mp4 or .mov in workspace/input")
    return files[0]


def ffprobe(video: Path, *, sfm_container: str, dry_run: bool) -> VideoInfo:
    command = container_command(sfm_container, ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)])
    print("+", subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return VideoInfo(4000, 2000, 60.0, "hevc", 1)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_ffprobe(json.loads(result.stdout))


def colmap_version(*, sfm_container: str, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    result = subprocess.run(container_command(sfm_container, ["colmap", "--version"]), check=True, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip() or "unknown"


def run_validate(args: argparse.Namespace, artifacts: Path, sfm_container: str) -> None:
    video = source_video(args.workspace)
    if video.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError("Uploaded video exceeds the 20 GiB maximum")
    info = ffprobe(video, sfm_container=sfm_container, dry_run=args.dry_run)
    validate_video_info(info)
    timestamps = sample_timestamps(info.duration_seconds)
    manifest = profile_manifest(info, timestamps)
    manifest["source"] = {"filename": video.name, "bytes": video.stat().st_size, "sha256": sha256(video) if not args.dry_run else "dry-run"}
    config = REPO_ROOT / "configs" / "pipeline" / "osmo360_balanced.yaml"
    manifest["runtime"] = {
        "colmap_version": colmap_version(sfm_container=sfm_container, dry_run=args.dry_run),
        "config_sha256": sha256(config),
    }
    (artifacts / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(artifacts: Path) -> dict[str, Any]:
    return json.loads((artifacts / "input_manifest.json").read_text(encoding="utf-8"))


def run_project(args: argparse.Namespace, artifacts: Path, sfm_container: str) -> None:
    manifest = read_manifest(artifacts)
    timestamps = [float(value) for value in manifest["projection"]["timestamps"]]
    video = source_video(args.workspace)
    root = artifacts / "projections" / "images"
    for frame_index, timestamp in enumerate(timestamps):
        for camera in virtual_cameras():
            output = root / image_name(frame_index, camera)
            checked(container_command(sfm_container, projection_command(
                ffmpeg="ffmpeg", video=video, timestamp=timestamp, camera=camera, output=output
            )), dry_run=args.dry_run)
    (artifacts / "projections" / "projection_manifest.json").write_text(
        json.dumps({"timestamp_count": len(timestamps), "image_count": len(timestamps) * len(virtual_cameras())}, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_images_txt(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        values = line.split()
        if len(values) < 10:
            raise ValueError(f"Malformed COLMAP image line: {line}")
        result.append({
            "id": int(values[0]), "qvec": [float(value) for value in values[1:5]],
            "tvec": [float(value) for value in values[5:8]], "camera_id": int(values[8]),
            "name": values[9], "points2d": lines[index] if index < len(lines) else "",
        })
        index += 1
    return result


def largest_model(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper did not produce a sparse model")
    return max(candidates, key=lambda path: (path / "images.bin").stat().st_size if (path / "images.bin").exists() else 0)


def run_sfm(args: argparse.Namespace, artifacts: Path, sfm_container: str) -> None:
    images = artifacts / "projections" / "images"
    workspace = artifacts / "sfm"
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    workspace.mkdir(parents=True, exist_ok=True)
    # COLMAP 4 requires the mapper's output root to exist.  It creates the
    # numbered model subdirectory beneath it, which `largest_model` selects.
    sparse.mkdir(parents=True, exist_ok=True)
    common = ["--database_path", str(database), "--image_path", str(images)]
    # COLMAP 4 moved GPU controls to the generic feature-extraction and
    # feature-matching option groups.
    checked(container_command(sfm_container, ["colmap", "feature_extractor", *common, "--ImageReader.single_camera_per_folder", "1", "--FeatureExtraction.use_gpu", "1"], gpu=args.gpu), dry_run=args.dry_run)
    checked(container_command(sfm_container, ["colmap", "sequential_matcher", "--database_path", str(database), "--SequentialMatching.overlap", "6", "--SequentialMatching.loop_detection", "1", "--FeatureMatching.use_gpu", "1"], gpu=args.gpu), dry_run=args.dry_run)
    checked(container_command(sfm_container, ["colmap", "mapper", *common, "--output_path", str(sparse)], gpu=args.gpu), dry_run=args.dry_run)
    selected = largest_model(sparse) if not args.dry_run else sparse / "0"
    text_model = workspace / "model_txt"
    checked(container_command(sfm_container, ["colmap", "model_converter", "--input_path", str(selected), "--output_path", str(text_model), "--output_type", "TXT"]), dry_run=args.dry_run)
    if args.dry_run:
        return
    config = workspace / "rig_config.json"
    write_rig_config(config)
    rigged = workspace / "rigged"
    bundled = workspace / "bundled"
    checked(container_command(sfm_container, ["colmap", "rig_configurator", "--database_path", str(database), "--input_path", str(selected), "--rig_config_path", str(config), "--output_path", str(rigged)]), dry_run=False)
    checked(container_command(sfm_container, ["colmap", "bundle_adjuster", "--input_path", str(rigged), "--output_path", str(bundled)]), dry_run=False)
    final_text = workspace / "final_txt"
    checked(container_command(sfm_container, ["colmap", "model_converter", "--input_path", str(bundled), "--output_path", str(final_text), "--output_type", "TXT"]), dry_run=False)
    validate_sparse_model(final_text)


def validate_sparse_model(model: Path) -> None:
    images = parse_images_txt(model / "images.txt")
    groups = {Path(str(image["name"])).stem.split("_")[1] for image in images}
    point_lines = [line for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if len(groups) < 60:
        raise RuntimeError(f"COLMAP registered only {len(groups)} timestamp groups; need at least 60")
    if len(point_lines) < 5000:
        raise RuntimeError(f"COLMAP produced only {len(point_lines)} sparse points; need at least 5000")
    errors = [float(line.split()[7]) for line in point_lines if len(line.split()) >= 8]
    if not errors or float(np.median(errors)) > 2.0:
        raise RuntimeError("COLMAP sparse model exceeds 2 px median reprojection error")


def qvec_to_c2w(qvec: list[float], tvec: list[float]) -> list[list[float]]:
    qw, qx, qy, qz = qvec
    rotation = np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ], dtype=float)
    c2w = np.eye(4)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ np.asarray(tvec, dtype=float)
    return c2w.tolist()


def write_seed_ply(path: Path, points: Iterable[tuple[list[float], list[int]]]) -> None:
    values = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(values))
        handle.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in values:
            handle.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def run_prepare(args: argparse.Namespace, artifacts: Path) -> None:
    model = artifacts / "sfm" / "final_txt"
    records = parse_images_txt(model / "images.txt")
    frame_ids = sorted({int(Path(str(record["name"])).stem.split("_")[1]) for record in records})
    splits = split_timestamp_groups(frame_ids)
    lookup = {frame_id: split for split, values in splits.items() for frame_id in values}
    train = [record for record in records if lookup[int(Path(str(record["name"])).stem.split("_")[1])] == "train"]
    train_ids = {record["id"] for record in train}
    prepared = artifacts / "colmap" / args.scene_id
    images_dest = prepared / "images"
    sparse = prepared / "sparse" / "0"
    images_dest.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model / "cameras.txt", sparse / "cameras.txt")
    source_images = artifacts / "projections" / "images"
    with (sparse / "images.txt").open("w", encoding="utf-8") as handle:
        for record in train:
            q, t = record["qvec"], record["tvec"]
            handle.write(f"{record['id']} {' '.join(map(str, q))} {' '.join(map(str, t))} {record['camera_id']} {record['name']}\n{record['points2d']}\n")
            target = images_dest / str(record["name"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_images / str(record["name"]), target)
    seed: list[tuple[list[float], list[int]]] = []
    for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        tracks = values[8:]
        observed = {int(tracks[index]) for index in range(0, len(tracks), 2)}
        if observed & train_ids:
            seed.append(([float(values[1]), float(values[2]), float(values[3])], [int(values[4]), int(values[5]), int(values[6])]))
    write_seed_ply(sparse / "points3D.ply", seed)
    intrinsics = {camera.name: {"width": camera.width, "height": camera.height, "fx": camera.focal_length, "fy": camera.focal_length, "cx": (camera.width - 1) / 2, "cy": (camera.height - 1) / 2} for camera in virtual_cameras()}
    frame_records = []
    for frame_id in frame_ids:
        face_records = [record for record in records if int(Path(str(record["name"])).stem.split("_")[1]) == frame_id]
        frame_records.append({"frame_id": frame_id, "split": lookup[frame_id], "images": [{"camera": Path(str(record["name"])).parts[0], "image_name": record["name"], "c2w": qvec_to_c2w(record["qvec"], record["tvec"])} for record in face_records]})
    metadata = {"dataset": "osmo360", "scene_id": args.scene_id, "num_frames": len(frame_ids), "num_images": len(train), "selected_frames": frame_ids, "frame_splits": splits, "split_counts": {key: len(value) for key, value in splits.items()}, "camera_intrinsics": intrinsics, "frame_records": frame_records, "seed_points": len(seed)}
    (prepared / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_record(artifacts: Path) -> Path:
    return artifacts / "train_run.json"


def run_train(args: argparse.Namespace, artifacts: Path, torch_container: str) -> None:
    dataset = artifacts / "colmap" / args.scene_id
    output = artifacts / "octree"
    checked(container_command(torch_container, ["python", "scripts/train_octree_anygs.py", "--dataset-path", str(dataset), "--scene-name", args.scene_id, "--dataset-name", "osmo360", "--output-root", str(output), "--gpu", args.gpu, "--resolution", "4", "--iterations", "15000", "--no-eval", "--gaussian-type", "implicit3D", "--feat-dim", "16", "--base-layer", "9", "--run-record", str(train_record(artifacts))], gpu=args.gpu), dry_run=args.dry_run)


def model_path(artifacts: Path) -> Path:
    record = json.loads(train_record(artifacts).read_text(encoding="utf-8"))
    return Path(str(record["model_path"]))


def run_render(args: argparse.Namespace, artifacts: Path, torch_container: str) -> None:
    model = model_path(artifacts)
    checked(container_command(torch_container, ["python", "Octree-AnyGS/render.py", "-m", str(model), "--skip_test", "--iteration", "-1"], gpu=args.gpu), dry_run=args.dry_run)


def run_bundle(args: argparse.Namespace, artifacts: Path) -> None:
    destination = args.output_root / args.scene_id
    destination.mkdir(parents=True, exist_ok=True)
    model = model_path(artifacts)
    shutil.copytree(model, destination / "model", dirs_exist_ok=True)
    shutil.copytree(artifacts / "colmap" / args.scene_id, destination / "prepared", dirs_exist_ok=True)
    config_path = destination / "model" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise RuntimeError("Octree model config must be a YAML mapping")
    model_params = dict(config.get("model_params") or {})
    model_params["source_path"] = "../prepared"
    config["model_params"] = model_params
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for source in (artifacts / "input_manifest.json", artifacts / "projections" / "projection_manifest.json"):
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    manifest = {"workflow": "osmo360_splat", "scene_id": args.scene_id, "profile": PROFILE, "rgb_only": True, "model_path": "model", "prepared_path": "prepared", "archive": "osmo360_rgb_splat.zip"}
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "VIEWER_COMMANDS.md").write_text(
        "# RGB viewer\n\nRun from this directory with the VBOGS Torch image:\n\n"
        "```bash\npython scripts/view_octree_anygs.py --model-path model --rgb-only --camera-source train\n```\n",
        encoding="utf-8",
    )
    archive = destination / "osmo360_rgb_splat.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path != archive:
                output.write(path, path.relative_to(args.output_root))


def main() -> None:
    args = parser().parse_args()
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise SystemExit("--start-at must not follow --stop-after")
    args.workspace = args.workspace.resolve()
    args.output_root = args.output_root.resolve()
    artifacts = args.workspace / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    sfm_container = service_container("vbogs-sfm", args.sfm_container)
    torch_container = service_container("vbogs-torch", args.torch_container)
    actions = {
        "validate": lambda: run_validate(args, artifacts, sfm_container),
        "project": lambda: run_project(args, artifacts, sfm_container),
        "sfm": lambda: run_sfm(args, artifacts, sfm_container),
        "prepare": lambda: run_prepare(args, artifacts),
        "train": lambda: run_train(args, artifacts, torch_container),
        "render": lambda: run_render(args, artifacts, torch_container),
        "bundle": lambda: run_bundle(args, artifacts),
    }
    emit(args.event_log, "run_started", workflow="osmo360_splat", scene_id=args.scene_id, stages=list(STAGES))
    for stage in STAGES[STAGES.index(args.start_at): STAGES.index(args.stop_after) + 1]:
        if args.cancel_file and args.cancel_file.exists():
            raise SystemExit("Cancellation requested")
        emit(args.event_log, "stage_started", stage=stage)
        actions[stage]()
        emit(args.event_log, "stage_completed", stage=stage)


if __name__ == "__main__":
    main()
