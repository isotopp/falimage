from pathlib import Path
from datetime import datetime
import random

import piexif
from PIL import Image

from falimage.exif import set_exif_data, DEFAULT_CAMERA_MAKE, DEFAULT_CAMERA_MODEL, AUTHOR, SOFTWARE, COPYRIGHT_TEMPLATE


def _make_temp_jpeg(path: Path, size=(8, 8), color=(255, 0, 0)) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format="JPEG")


def test_set_exif_data_writes_expected_tags(tmp_path: Path):
    # Create a temporary JPEG image
    p = tmp_path / "test.jpg"
    _make_temp_jpeg(p)

    # Use fixed datetime and RNG for determinism
    dt = datetime(2024, 1, 2, 3, 4, 5)
    rng = random.Random(12345)

    ok = set_exif_data(p, rng=rng, file_time=dt, quiet=True)
    assert ok is True

    # Load EXIF and verify fields
    exif = piexif.load(str(p))
    print(f"{exif=}")

    assert exif["0th"][piexif.ImageIFD.Make] == DEFAULT_CAMERA_MAKE.encode()
    assert exif["0th"][piexif.ImageIFD.Model] == DEFAULT_CAMERA_MODEL.encode()
    assert exif["0th"][piexif.ImageIFD.Artist] == AUTHOR.encode()
    assert exif["0th"][piexif.ImageIFD.Software] == SOFTWARE.encode()
    assert exif["0th"][piexif.ImageIFD.Copyright] == COPYRIGHT_TEMPLATE.format(year=dt.year,AUTHOR=AUTHOR).encode()
    assert exif["0th"][piexif.ImageIFD.Orientation] == 1

    # Date fields
    expected_date = dt.strftime("%Y:%m:%d %H:%M:%S").encode()
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == expected_date
    assert exif["Exif"][piexif.ExifIFD.DateTimeDigitized] == expected_date

    # Camera settings exist and are of expected types
    assert isinstance(exif["Exif"][piexif.ExifIFD.ExposureTime], tuple)
    assert isinstance(exif["Exif"][piexif.ExifIFD.ISOSpeedRatings], int)
    assert isinstance(exif["Exif"][piexif.ExifIFD.FNumber], tuple)
    assert isinstance(exif["Exif"][piexif.ExifIFD.FocalLength], tuple)
    assert exif["Exif"][piexif.ExifIFD.ExposureProgram] in (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert exif["Exif"][piexif.ExifIFD.Flash] in (0, 1)


def test_set_exif_data_missing_file_returns_false(tmp_path: Path):
    missing = tmp_path / "nope.jpg"
    assert set_exif_data(missing, quiet=True) is False
