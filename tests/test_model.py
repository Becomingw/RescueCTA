import torch

from infer_nifti import select_generator_state
from models import build_generator


def test_generator_output_shape():
    model = build_generator(
        "rescuecta",
        in_channels=3,
        out_channels=3,
        num_stages=6,
    ).eval()
    # Six resolution branches require at least 2 × 2 spatial support at the
    # coarsest InstanceNorm layer.
    x = torch.zeros(1, 3, 64, 64)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == x.shape


def test_legacy_generator_alias_has_compatible_keys():
    current = build_generator("rescuecta", in_channels=3, out_channels=3)
    legacy = build_generator("HR", in_channels=3, out_channels=3)
    assert current.state_dict().keys() == legacy.state_dict().keys()


def test_ema_is_selected_by_default():
    raw = {"weight": torch.tensor([1.0])}
    ema = {"weight": torch.tensor([2.0])}
    checkpoint = {
        "G_state_dict": raw,
        "G_ema_state_dict": ema,
    }
    assert select_generator_state(checkpoint) is ema
    assert select_generator_state(checkpoint, "raw") is raw
