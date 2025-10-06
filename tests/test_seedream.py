import pytest

from falimage import cli


def test_seedream_in_registry():
    assert "seedream" in cli.MODEL_REGISTRY
    m = cli.MODEL_REGISTRY["seedream"]
    assert m["endpoint"] == "fal-ai/bytedance/seedream/v4/text-to-image"
    assert m["call"] == "subscribe"
    # Required allowed keys
    for key in ["prompt", "image_size", "num_images", "seed", "enable_safety_checker"]:
        assert key in m["allowed"]


def test_build_arguments_seedream_named_size(monkeypatch):
    # deterministic seed when 0
    monkeypatch.setattr(cli, "randint", lambda a, b: 424242)
    values = {
        "prompt": "A house by the sea",
        "image_size": "portrait_16_9",
        "num_images": 2,
        "seed": 0,
        "enable_safety_checkER": None,  # typo key should be ignored silently via allowed-set
    }
    args = cli.build_arguments("seedream", values)
    assert args["prompt"] == "A house by the sea"
    assert args["image_size"] == "portrait_16_9"
    assert args["num_images"] == 2
    # seed randomized because 0
    assert args["seed"] == 424242


def test_build_arguments_seedream_custom_size():
    values = {
        "prompt": "Mountains at sunrise",
        "image_size": {"width": 1280, "height": 720},
        "num_images": 1,
        "seed": 123,
        "enable_safety_checker": True,
    }
    args = cli.build_arguments("seedream", values)
    assert args["image_size"] == {"width": 1280, "height": 720}
    assert args["seed"] == 123
    assert args["enable_safety_checker"] is True
