from pathlib import Path
import io

import piexif
from PIL import Image

from falimage import cli


class FakeResp:
    def __init__(self, content: bytes, ctype: str = "image/jpeg"):
        self._content = content
        self.headers = {"Content-Type": ctype}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
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
        return self.mapping[url]


def _jpeg_bytes(size=(10, 10), color=(0, 255, 0)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_save_all_images_sets_exif(monkeypatch, tmp_path: Path):
    # prepare one JPEG response
    content = _jpeg_bytes()
    mapping = {"https://ex/pic": FakeResp(content, ctype="image/jpeg")}
    monkeypatch.setattr(cli.requests, "Session", lambda: FakeSession(mapping))

    out = cli.save_all_images(["https://ex/pic"], name_prefix="exif", savedir=tmp_path, open_files=False)

    assert len(out) == 1
    p = out[0]

    # Load EXIF and verify at least one of our tags exists (Make)
    exif = piexif.load(str(p))
    assert piexif.ImageIFD.Make in exif["0th"]
