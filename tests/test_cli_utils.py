import io
import types
from pathlib import Path
from typing import List

import builtins
import os
import pytest

from falimage import cli

# Prefixes from environment
SAFETENSORS_URL = os.getenv("SAFETENSORS_URL")


def test_split_name_and_ext():
    assert cli._split_name_and_ext("file.jpg") == ("file", ".jpg")
    assert cli._split_name_and_ext("archive.tar.gz") == ("archive.tar", ".gz")
    assert cli._split_name_and_ext("noext") == ("noext", "")
    assert cli._split_name_and_ext(".hidden") == (".hidden", "")


def test_unique_path(tmp_path: Path):
    p = tmp_path / "image.jpg"
    p.write_text("x")
    p2 = cli._unique_path(p)
    assert p2.name == "image-1.jpg"
    p2.write_text("y")
    p3 = cli._unique_path(p)
    assert p3.name == "image-2.jpg"


def test_ext_from_content_type():
    assert cli._ext_from_content_type("image/jpeg") == ".jpg"
    assert cli._ext_from_content_type("image/png") == ".png"
    assert cli._ext_from_content_type("text/html") == ""
    assert cli._ext_from_content_type(None) == ""
    assert cli._ext_from_content_type("image/nonsense; charset=binary") == ""


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("", []),
        ("foo", [{"path": f"{SAFETENSORS_URL}foo.safetensors", "scale": 1.0}]),
        (
            "foo:0.8,bar:1.2,https://ex/lor.safetensors:0.5",
            [
                {"path": f"{SAFETENSORS_URL}foo.safetensors", "scale": 0.8},
                {"path": f"{SAFETENSORS_URL}bar.safetensors", "scale": 1.2},
                {"path": "https://ex/lor.safetensors", "scale": 0.5},
            ],
        ),
        (
            '[{"path": "https://ex/x.safetensors", "scale": 2}]',
            [{"path": "https://ex/x.safetensors", "scale": 2.0}],
        ),
        (
            '["name", {"name": "other", "scale": 0.7}]',
            [
                {"path": f"{SAFETENSORS_URL}name.safetensors", "scale": 1.0},
                {"path": f"{SAFETENSORS_URL}other.safetensors", "scale": 0.7},
            ],
        ),
        (
            "bad:xx",
            [{"path": f"{SAFETENSORS_URL}bad.safetensors", "scale": 1.0}],
        ),
    ],
)
def test_parse_loras(inp, expected):
    assert cli.parse_loras(inp) == expected


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"images": ["u1", {"url": "u2"}]}, ["u1", "u2"]),
        ({"output": ["u3", {"url": "u4"}]}, ["u3", "u4"]),
        ({"image": {"url": "u5"}}, ["u5"]),
        ({"image": ["u6", {"url": "u7"}]}, ["u6", "u7"]),
        ({}, []),
    ],
)
def test_extract_urls(result, expected):
    assert cli.extract_urls(result) == expected


def test_coerce_image_size_named_and_explicit():
    assert cli.coerce_image_size("portrait_4_3", None, None) == "portrait_4_3"
    assert cli.coerce_image_size(None, None, None) is None
    with pytest.raises(Exception):
        cli.coerce_image_size(None, 100, None)
    assert cli.coerce_image_size(None, 640, 480) == {"width": 640, "height": 480}
    with pytest.raises(Exception):
        cli.coerce_image_size("square", 640, 480)


def test_build_arguments_seed_randomized(monkeypatch):
    # Force randint to deterministic value
    monkeypatch.setattr(cli, "randint", lambda a, b: 123)
    vals = {"prompt": "hi", "num_images": 1, "seed": 0}
    args = cli.build_arguments("schnell", vals)
    assert args["seed"] == 123
    assert args["prompt"] == "hi"
    assert args["num_inference_steps"] == 4  # default merged


class FakeResp:
    def __init__(self, content: bytes, ctype: str = "image/jpeg"):
        self._content = content
        self.headers = {"Content-Type": ctype}
        self._iterated = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        # yield in one chunk
        yield self._content

    def close(self):
        pass


class FakeSession:
    def __init__(self, mapping):
        self.mapping = mapping
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, stream=True, timeout=(1, 1)):
        resp = self.mapping[url]
        return resp


def test_save_all_images_download_and_naming(monkeypatch, tmp_path: Path):
    # Map URLs to responses
    mapping = {
        "https://ex/u1.jpg": FakeResp(b"abc", ctype="image/jpeg"),
        # No ext in URL -> infer from content-type
        "https://ex/stream": FakeResp(b"defghi", ctype="image/png"),
        # Non-image skipped
        "https://ex/not": FakeResp(b"<html>", ctype="text/html"),
    }

    # Monkeypatch requests.Session to return our FakeSession
    monkeypatch.setattr(cli.requests, "Session", lambda: FakeSession(mapping))

    out: List[Path] = cli.save_all_images(
        ["https://ex/u1.jpg", "https://ex/stream", "https://ex/not"],
        name_prefix="test",
        savedir=tmp_path,
        open_files=False,
    )

    # Expect two files saved
    assert len(out) == 2
    names = sorted(p.name for p in out)
    assert names[0].startswith("test-")
    assert names[0].endswith(".jpg")
    assert names[1].startswith("test-")
    assert names[1].endswith(".png")
    # Contents
    assert (tmp_path / names[0]).read_bytes() == b"abc"
    assert (tmp_path / names[1]).read_bytes() == b"defghi"
