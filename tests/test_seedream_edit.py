import os
import pytest

from falimage import cli

def test_seedream_edit_in_registry():
    assert "seedream-edit" in cli.MODEL_REGISTRY
    m = cli.MODEL_REGISTRY["seedream-edit"]
    assert m["endpoint"] == "fal-ai/bytedance/seedream/v4/edit"
    assert m["call"] == "subscribe"
    # Required allowed keys
    for key in [
        "prompt",
        "image_size",
        "num_images",
        "seed",
        "enable_safety_checker",
        "image_urls",
    ]:
        assert key in m["allowed"]


def test_parse_image_urls_mixed():
    # Mixed absolute URLs and names; names get prefix and default extension
    s = "https://example.com/a.jpg,b.png, c"
    urls = cli.parse_image_urls(s)
    pref = os.getenv("SOURCE_IMAGE_URL")
    assert urls == [
        "https://example.com/a.jpg",
        f"{pref}b.png",
        f"{pref}c.jpg",
    ]


def test_build_arguments_seedream_edit(monkeypatch):
    # deterministic seed when 0
    monkeypatch.setattr(cli, "randint", lambda a, b: 424242)

    values = {
        "prompt": "Edit this image",
        "image_size": "portrait_4_3",
        "num_images": 1,
        "seed": 0,
        "enable_safety_checker": False,
        "image_urls": cli.parse_image_urls("foo,https://h/x.png,bar.jpg"),
    }
    args = cli.build_arguments("seedream-edit", values)

    # Arguments are passed through according to allowed set and normalization
    assert args["prompt"] == "Edit this image"
    assert args["image_size"] == "portrait_4_3"
    assert args["num_images"] == 1
    assert args["enable_safety_checker"] is False
    # Seed randomized because 0
    assert args["seed"] == 424242
    # image_urls normalized as per rules
    pref = os.getenv("SOURCE_IMAGE_URL")
    assert args["image_urls"] == [
        f"{pref}foo.jpg",
        "https://h/x.png",
        f"{pref}bar.jpg",
    ]
