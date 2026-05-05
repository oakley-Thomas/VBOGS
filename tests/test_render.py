import pytest
import torch

from vbogs.render import expand_anchor_scalars_to_gaussians


def test_expand_anchor_scalars_to_gaussians_matches_octree_order():
    per_anchor_scalar = torch.tensor([10.0, 20.0, 30.0, 40.0])
    visible_mask = torch.tensor([False, True, True, False])
    selection_mask = torch.tensor([True, False, True, True, False, True])

    expanded = expand_anchor_scalars_to_gaussians(
        per_anchor_scalar,
        visible_mask,
        selection_mask,
        n_offsets=3,
    )

    assert expanded.shape == (4, 1)
    assert torch.equal(expanded.squeeze(1), torch.tensor([20.0, 20.0, 30.0, 30.0]))


def test_expand_anchor_scalars_to_gaussians_rejects_mismatched_selection_mask():
    with pytest.raises(ValueError, match="selection_mask length"):
        expand_anchor_scalars_to_gaussians(
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
            torch.tensor([True, False]),
            n_offsets=3,
        )
