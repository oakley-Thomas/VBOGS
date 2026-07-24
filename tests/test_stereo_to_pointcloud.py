import json

import pytest

from scripts.stereo_to_pointcloud import load_selected_frames


def test_load_selected_frames_filters_train_split(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selected_frames": [0, 1, 2],
                "frame_splits": {
                    "train": [0],
                    "test": [1],
                    "validation": [2],
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_selected_frames(metadata_path, "train") == [0]
    assert load_selected_frames(metadata_path, "all") == [0, 1, 2]


def test_load_selected_frames_rejects_missing_requested_split(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"selected_frames": [0]}), encoding="utf-8")

    with pytest.raises(ValueError, match="train"):
        load_selected_frames(metadata_path, "train")
