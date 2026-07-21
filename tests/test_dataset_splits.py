from vbogs.dataset_splits import frames_for_split, split_frame_indices


def test_uniform_split_counts_and_ordering_for_ten_frames():
    assert split_frame_indices(list(range(10))) == {
        "train": [0, 1, 3, 4, 6, 8, 9],
        "test": [2, 7],
        "validation": [5],
    }


def test_frames_for_split_reads_metadata_frame_splits():
    metadata = {
        "selected_frames": [10, 20, 30, 40],
        "frame_splits": {
            "train": [10, 30],
            "test": [20],
            "validation": [40],
        },
    }

    assert frames_for_split(metadata, "train") == [10, 30]
    assert frames_for_split(metadata, "all") == [10, 20, 30, 40]
