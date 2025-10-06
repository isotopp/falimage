import os
import random
from datetime import datetime
from pathlib import Path

import piexif
from PIL import Image

# Constants for EXIF settings
DEFAULT_CAMERA_MAKE = "Apple"
DEFAULT_CAMERA_MODEL = "iPhone 16 pro"
AUTHOR = "John Doe"
SOFTWARE = "falimage"
COPYRIGHT_TEMPLATE = "(C) {year} {AUTHOR}"
ORIENTATION = "Horizontal (normal)"
EXPOSURE_PROGRAM = "Aperture-priority AE"
FLASH_MODE = "Off, Did not fire"

# Realistic ranges for camera settings
EXPOSURE_TIMES = ["1/80", "1/60", "1/125", "1/250", "1/500"]
ISO_VALUES = [100, 125, 200, 400, 800]
APERTURE_VALUES = [1.8, 2.0, 2.8, 4.0, 5.6]
FOCAL_LENGTHS = [35, 50, 85, 100, 135]  # Common portrait focal lengths


def set_exif_data(image_path, *, rng: random.Random | None = None, file_time: datetime | None = None,
                  quiet: bool = True) -> bool:
    """Set fresh EXIF metadata for a given image file.

    Opens an image file, initializes a fresh EXIF dictionary with various metadata
    fields, populates realistic camera settings and date information, and writes
    the EXIF data back to the image file.

    To make this function testable and deterministic, callers may provide
    a random number generator (rng) and a fixed file_time.

    Args:
        image_path (Path | str): Path to the image file to be updated with EXIF data.
        rng: Optional random.Random instance used to select camera settings.
        file_time: Optional datetime used for timestamp fields. If None, the
            file's mtime is used.
        quiet: If True, suppresses print messages. If False, prints status.

    Returns:
        bool: True on success, False on failure (including missing file).
    """
    # Coerce to Path
    p = Path(image_path)

    # Check if the image file exists
    if not p.exists():
        if not quiet:
            print(f"File not found: {p}")
        return False

    try:
        # Open the image file
        img = Image.open(p)
    except Exception as e:
        if not quiet:
            print(f"Can't open image {p}: {e}")
        return False

    # Initialize a fresh EXIF dictionary with various sections
    exif_dict = {
        "0th": {},  # Primary image attributes
        "Exif": {},  # Camera settings and time metadata
        "GPS": {},  # GPS info, if needed
        "Interop": {},  # Interoperability settings
        "1st": {},  # Thumbnail image data
        "thumbnail": None,  # No thumbnail in this case
    }

    # Determine timestamp
    if file_time is None:
        file_time = datetime.fromtimestamp(os.path.getmtime(p))
    formatted_date = file_time.strftime("%Y:%m:%d %H:%M:%S")
    year = file_time.year

    # Populate the primary image attributes section
    exif_dict["0th"][piexif.ImageIFD.Make] = DEFAULT_CAMERA_MAKE.encode()
    exif_dict["0th"][piexif.ImageIFD.Model] = DEFAULT_CAMERA_MODEL.encode()
    exif_dict["0th"][piexif.ImageIFD.Artist] = AUTHOR.encode()
    exif_dict["0th"][piexif.ImageIFD.Software] = SOFTWARE.encode()
    exif_dict["0th"][piexif.ImageIFD.Copyright] = COPYRIGHT_TEMPLATE.format(year=year,AUTHOR=AUTHOR).encode()
    exif_dict["0th"][piexif.ImageIFD.Orientation] = 1  # Horizontal (normal)

    # Populate the date fields in the EXIF section
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = formatted_date.encode()
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = formatted_date.encode()

    # Randomly select realistic camera settings
    r = rng or random
    exposure_time = r.choice(EXPOSURE_TIMES)
    iso_value = r.choice(ISO_VALUES)
    aperture_value = r.choice(APERTURE_VALUES)
    focal_length = r.choice(FOCAL_LENGTHS)

    # Convert exposure time to a rational value
    numerator, denominator = map(int, exposure_time.split('/'))
    exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (numerator, denominator)

    # Populate the camera settings in the EXIF section
    exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = int(iso_value)
    exif_dict["Exif"][piexif.ExifIFD.FNumber] = (int(aperture_value * 10), 10)  # Rational value
    exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (int(focal_length), 1)  # Rational value
    # Note: EXIF ExposureProgram expects a SHORT code; keep a realistic one (3 = Aperture priority)
    exif_dict["Exif"][piexif.ExifIFD.ExposureProgram] = 3
    # Flash tag: 0 = Flash did not fire
    exif_dict["Exif"][piexif.ExifIFD.Flash] = 0

    try:
        # Convert the EXIF dictionary to bytes and save it to the image file
        exif_bytes = piexif.dump(exif_dict)
        img.save(p, exif=exif_bytes)
        if not quiet:
            print(f"Updated EXIF data for {p}")
        return True
    except Exception as e:
        # Handle any errors that occur during the process
        if not quiet:
            print(f"Error updating EXIF data for {p}: {e}")
        return False
