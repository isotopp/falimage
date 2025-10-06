#! /usr/bin/env python

# Unified image generation CLI using fal.ai models and a custom workflow.
#
# Models covered (based on existing img*.py scripts):
# - schnell      -> fal-ai/flux/schnell (simple, fast)
# - dev          -> fal-ai/flux/dev (more controls)
# - realism      -> fal-ai/flux-realism (adds strength, output_format)
# - lora         -> workflows/isotopp/flux-dev-lora (supports LoRA list)
#
# Usage examples:
#   ./falimage.py -m schnell -p "a cat" -# 2 -i landscape_4_3
#   ./falimage.py -m dev -p "a cat" --num-inference-steps 28 --guidance-scale 3.5
#   ./falimage.py -m realism -p "photo of a cat" --strength 1 --output-format jpeg
#   ./falimage.py -m lora -p "a cat" --loras "my_lora:0.8,https://host/x.safetensors:1.2" -n cat
#   ./falimage.py -m lora -f prompt.txt -i portrait_4_3 --dry-run
#
# Requires FAL_KEY in environment (see .env). Uses python-dotenv if present.

from __future__ import annotations

import itertools
import mimetypes
import os
import webbrowser
from pathlib import Path
from pprint import pformat
from random import randint
from typing import Iterable, List, Tuple, Optional
from urllib.parse import urlsplit, unquote

import click
import requests
from dotenv import load_dotenv

from .exif import set_exif_data
from .registry import MODEL_REGISTRY

load_dotenv()

# Allowed named sizes (shared across models in this repository)
ALLOWED_SIZES = {"square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"}

# ------------------------------ Utilities ------------------------------

CHUNK_SIZE = 1024 * 64  # 64 KiB


def _split_name_and_ext(filename: str) -> Tuple[str, str]:
    p = Path(filename)
    return p.stem, p.suffix  # suffix includes leading dot or empty string


def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    for i in itertools.count(1):
        candidate = base.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate


def _ext_from_content_type(content_type: Optional[str]) -> str:
    if not content_type:
        return ""

    # Only trust image/*
    if not content_type.lower().startswith("image/"):
        return ""

    # Try to map to a sensible extension
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    return ext or ""


def save_all_images(
        urls: Iterable[str],
        name_prefix: Optional[str] = None,
        savedir: Path | str = "assets",
        open_files: bool = False,
        timeout: Tuple[float, float] = (5.0, 60.0),  # (connect, read)
) -> List[Path]:
    """
    Download images from URLs into savedir.

    If name_prefix is provided, filenames become:
      "<name_prefix>-<N>-<original_stem><ext>" where N starts at 1.

    Returns a list of saved Paths.

    Security notes:
    - Validates Content-Type is image/* before saving.
    - Never executes files; optional open uses the default system handler.
    - Strips any path segments from URL so filenames cannot escape savedir.
    """
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    with requests.Session() as session:
        session.headers.update({"User-Agent": "image-downloader/1.0"})
        for idx, url in enumerate(urls, start=1):
            try:
                parsed = urlsplit(url)
                # Keep only the final path component, decoded for nicer names
                raw_name = Path(unquote(parsed.path)).name or "download"
                stem, suffix = _split_name_and_ext(raw_name)

                # Prepare request
                resp = session.get(url, stream=True, timeout=timeout)
                resp.raise_for_status()

                ctype = resp.headers.get("Content-Type", "")
                if not ctype.lower().startswith("image/"):
                    print(f"Skip non-image content for {url} (Content-Type={ctype or 'unknown'})")
                    resp.close()
                    continue

                # If no suffix in URL, infer from content type
                if not suffix:
                    inferred = _ext_from_content_type(ctype)
                    suffix = inferred or ".bin"

                if name_prefix:
                    out_name = f"{name_prefix}-{idx}-{stem}{suffix}"
                else:
                    out_name = f"{stem}{suffix}"

                filename = _unique_path(savedir / out_name)

                # Stream to disk
                with open(filename, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)

                # Set EXIF metadata on the saved image (best-effort)
                try:
                    set_exif_data(filename, quiet=False)
                except Exception as e:
                    print(f"Warning: failed to set EXIF for {filename}: {e}")
                    pass

                saved_paths.append(filename)
                print(f"Saved {filename}")

                if open_files:
                    # Uses default handler on macOS/Linux/Windows
                    webbrowser.open(filename.resolve().as_uri())

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")
            except OSError as e:
                print(f"Filesystem error for {url}: {e}")
            except Exception as e:
                # Catch-all to keep batch going
                print(f"Unexpected error for {url}: {e}")

    return saved_paths


def parse_loras(loras: str):
    """Normalize loras input into list of {path, scale} items.

    Accepts JSON list/dict or comma-separated tokens "name[:scale]" or URL[:scale].
    Shorthand names (no slash) are expanded to SAFETENSORS_URL
    """
    import json
    normalized = loras or ""
    result = []
    if not normalized:
        return result

    pref = os.getenv("SAFETENSORS_URL")
    suff = ".safetensors"

    def to_url(token: str) -> str:
        # Treat anything with "://" as a full URL to leave untouched
        return token if "://" in token or "/" in token else f"{pref}{token}{suff}"

    def split_path_scale(token: str) -> tuple[str, Optional[float]]:
        t = token.strip()
        if not t:
            return "", None
        if "://" in t:
            # URL: only treat a colon after the last slash as scale separator
            last_slash = t.rfind('/')
            last_colon = t.rfind(':')
            if last_colon > last_slash:
                path = t[:last_colon]
                try:
                    scale = float(t[last_colon + 1:].strip())
                except ValueError:
                    scale = 1.0
                return path, scale
            return t, None
        # Non-URL: split on the last colon
        if ':' in t:
            path, scale_str = t.rsplit(':', 1)
            try:
                return path.strip(), float(scale_str.strip())
            except ValueError:
                return path.strip(), 1.0
        return t, None

    def to_item(token: str):
        token = token.strip()
        if not token:
            return None
        path_part, scale_val = split_path_scale(token)
        if not path_part:
            return None
        return {"path": to_url(path_part), "scale": (scale_val if scale_val is not None else 1.0)}

    first = normalized.lstrip()
    if first.startswith('[') or first.startswith('{'):
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                for it in parsed:
                    if isinstance(it, str):
                        item = to_item(it)
                        if item:
                            result.append(item)
                    elif isinstance(it, dict):
                        path = it.get('path') or it.get('url') or it.get('name')
                        scale = it.get('scale', 1.0)
                        if path:
                            result.append({"path": to_url(str(path)), "scale": float(scale)})
        except Exception:
            parts = [p.strip() for p in normalized.split(',') if p.strip()]
            result = [to_item(p) for p in parts if to_item(p)]
    else:
        parts = [p.strip() for p in normalized.split(',') if p.strip()]
        result = [to_item(p) for p in parts if to_item(p)]

    return result


def extract_urls(result):
    """Extract a list of image URLs from various result shapes."""
    urls = []
    if isinstance(result, dict):
        if 'images' in result and isinstance(result['images'], list):
            urls = [img.get('url') if isinstance(img, dict) else img for img in result['images']]
        elif 'output' in result and isinstance(result['output'], list):
            urls = [item.get('url') if isinstance(item, dict) else item for item in result['output']]
        elif 'image' in result:
            val = result['image']
            if isinstance(val, list):
                urls = [v.get('url') if isinstance(v, dict) else v for v in val]
            else:
                urls = [val.get('url') if isinstance(val, dict) else val]
    return urls



def parse_image_urls(image_urls: str) -> list[str]:
    """Normalize comma-separated image identifiers to absolute URLs for seedream-edit.

    Rules:
    - Split on commas, trim whitespace, ignore empty parts.
    - If the token contains "://", treat it as a full URL and keep it as-is.
    - Otherwise, treat it as a shorthand name and expand to
      <SOURCE_IMAGE_URL>/<name>", appending ".jpg" if the
      name has no extension (no dot in the last path segment).
    """
    pref = os.getenv("SOURCE_IMAGE_URL")
    out: list[str] = []
    for raw in (image_urls or "").split(','):
        token = raw.strip()
        if not token:
            continue
        if "://" in token:
            out.append(token)
        else:
            # detect extension in last segment
            name = token
            if '.' not in name.split('/')[-1]:
                name = f"{name}.jpg"
            out.append(f"{pref}{name}")
    return out

def coerce_image_size(image_size, width, height):
    """Return the payload value for image_size, validating inputs.

    Rules:
    - If image_size (named) is provided, width/height must NOT be provided.
    - If width/height are used, both must be provided together and image_size must be None.
    - Returns named size string or a dict {width, height}; None if nothing provided.
    """
    # Reject mixing named size with explicit dimensions
    if image_size is not None and (width or height):
        raise click.UsageError("--image-size cannot be combined with --width/--height")

    # Handle explicit width/height
    if width or height:
        if not (width and height):
            raise click.UsageError("--width and --height must be provided together")
        return {"width": int(width), "height": int(height)}

    # Named size or nothing
    if image_size is None:
        return None

    if isinstance(image_size, str):
        if image_size not in ALLOWED_SIZES:
            raise click.UsageError(f"Invalid --image-size. Allowed: {sorted(ALLOWED_SIZES)}")
        return image_size

    return image_size


def build_arguments(model_key, values):
    """Filter and assemble arguments according to model registry and user-provided values.

    values is a dict of all potential options (already normalized where needed).
    """
    model = MODEL_REGISTRY[model_key]
    allowed = model["allowed"]
    args = {}

    # Start with defaults
    args.update(model.get("defaults", {}))

    # Then fill from values if present
    for key, val in values.items():
        if key in allowed and val is not None:
            args[key] = val

    # Special: seed == 0 means randomize across all models for convenience
    if "seed" in allowed and (args.get("seed") is None or int(args.get("seed", 0)) == 0):
        args["seed"] = randint(1, 2 ** 32 - 1)

    return args


def send_request(model_key, arguments, dry_run=False):
    """Send the request to fal.ai for a given model.

    Lazily imports fal_client to keep module import light and make testing easier
    without the dependency present.
    """
    print("*** REQUEST:")
    print(pformat({"model": model_key, "endpoint": MODEL_REGISTRY[model_key]["endpoint"], "arguments": arguments}))
    if dry_run:
        print("*** NOT SENT (dry-run)")
        return {"images": []}

    # Lazy import here to avoid mandatory dependency at import time in tests
    import fal_client  # type: ignore

    model = MODEL_REGISTRY[model_key]
    if model["call"] == "run":
        return fal_client.run(model["endpoint"], arguments=arguments)
    else:
        # subscribe (non-streaming here; we don't hook logs for simplicity)
        return fal_client.subscribe(model["endpoint"], arguments=arguments, with_logs=False)


# ------------------------------ CLI ------------------------------

@click.command()
@click.option('-m', '--model', type=click.Choice(sorted(MODEL_REGISTRY.keys())), default='schnell', show_default=True,
              help='Which model/workflow to use.')
@click.option('--prompt', '-p', type=str, help='Prompt for image generation.')
@click.option('--promptfile', '-f', type=str,
              help='File containing the prompt (looked up under prompts/; .txt implied).')
@click.option('--num-images', '-#', 'num_images', type=int, default=1, show_default=True,
              help='Number of images to generate.')
@click.option('--name', type=str, help='Base name for saved images.')
@click.option('--loras', type=str, default='', show_default=True,
              help='For model "lora": JSON list of {path, scale} or comma list of names/URLs, optionally with :scale. Names are expanded to <SAFETENSORS_URL><name>.safetensors. Default scale=1.')
@click.option('--image-size', '-i', 'image_size', type=click.Choice(sorted(ALLOWED_SIZES)), default='portrait_4_3',
              show_default=True,
              help='Named size. Alternatively use --width/--height to specify exact size.')
@click.option('--width', '-w', type=int, help='Width of generated image (requires --height).')
@click.option('--height', '-h', type=int, help='Height of generated image (requires --width).')
@click.option('--seed', '-s', type=int, default=0, show_default=True, help='Seed (0=random).')
@click.option('--num-inference-steps', type=int, default=None, help='Inference steps (varies by model).')
@click.option('--guidance-scale', type=float, default=None, help='Guidance scale (where supported).')
@click.option('--strength', type=float, default=None, help='For realism model.')
@click.option('--output-format', type=click.Choice(["jpeg", "png"]), default="jpeg",
              help='Output image format (where supported).')
@click.option('--enable-safety-checker/--no-enable-safety-checker', default=None,
              help='Enable or disable safety checker (where supported).')
@click.option('--image-urls', type=str, default='', show_default=True,
              help='Comma-separated list of URLs or names (seedream-edit only). Names are expanded to <SOURCE_IMAGE_URL>/<name>[.jpg]')
@click.option('--dry-run', '-n', is_flag=True, default=False, help='Do not send; print the request and exit.')
def main(model, prompt, promptfile, num_images, name, loras, image_size, width, height, seed,
         num_inference_steps, guidance_scale, strength, output_format, enable_safety_checker, image_urls, dry_run):
    """Unified image generation tool for multiple fal.ai models and a LoRA workflow."""
    # Prompt handling
    prompt_prefix = None
    if promptfile:
        prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
        candidate = str(promptfile)
        if not candidate.endswith(".txt"):
            candidate = f"{candidate}.txt"
        prompt_path = Path(candidate)
        if not prompt_path.is_absolute():
            prompt_path = prompts_dir / prompt_path.name
        if not prompt_path.exists():
            raise click.UsageError(f"Prompt file not found: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        prompt_prefix = prompt_path.stem
    if not prompt:
        raise click.UsageError('Either --prompt or --promptfile must be provided.')

    # Image size normalization
    image_size_param = coerce_image_size(image_size, width, height)

    # LoRAs normalization (only used if model supports it)
    loras_list = parse_loras(loras) if loras else []
    print(f"*** DEBUG: {loras_list=}")

    # Image URLs normalization (only used by seedream-edit)
    image_urls_list = parse_image_urls(image_urls) if image_urls else None

    # Compose value bag
    provided_values = {
        "prompt": prompt,
        "image_size": image_size_param,
        "num_images": num_images,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "output_format": output_format,
        "enable_safety_checker": enable_safety_checker,
        "loras": loras_list,
        "image_urls": image_urls_list,
    }

    arguments = build_arguments(model, provided_values)

    # Warn about ignored options (user provided but not allowed for this model)
    allowed = MODEL_REGISTRY[model]["allowed"]
    ignored = {k: v for k, v in provided_values.items() if v is not None and k not in allowed}
    if ignored:
        click.echo(f"Note: Ignoring unsupported options for model '{model}': {sorted(ignored.keys())}")

    # Send
    result = send_request(model, arguments, dry_run=dry_run)

    # Handle result
    urls = extract_urls(result)
    if urls:
        name_prefix = prompt_prefix or (name if name else None)
        save_all_images(urls, name_prefix=name_prefix, open_files=True)
    else:
        print("No image URLs returned.")


if __name__ == "__main__":
    main()
