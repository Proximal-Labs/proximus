import base64
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Attachment limits: 2000px on the long edge and 4 MB after base64 encoding.
_MAX_DIMENSION = 2000
MAX_ENCODED_BYTES = 4 * 1024 * 1024
_MAX_RAW_BYTES = MAX_ENCODED_BYTES * 3 // 4
_EXIF_ORIENTATION = 0x0112
_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}

EVICTED_NOTE = "[image removed: request too large]"
INVALID_IMAGE_NOTE = "[image removed: provider rejected it]"

type Attachment = tuple[str, str]  # (media_type, base64_data)


def _sniff_media_type(data: bytes) -> str | None:
    """Detect the format from file content, not its extension."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _normalize_image(data: bytes, media_type: str) -> tuple[bytes, str, str]:
    """Apply orientation and size limits; keep original bytes if Pillow is absent."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return data, media_type, ""

    try:
        with Image.open(io.BytesIO(data)) as img:
            # Decode pixels before accepting an otherwise valid header.
            img.load()
            orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
            needs_transpose = orientation != 1
            width, height = img.size
            if (
                not needs_transpose
                and max(width, height) <= _MAX_DIMENSION
                and len(data) <= _MAX_RAW_BYTES
            ):
                return data, media_type, ""
            if needs_transpose:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
            scale = min(1.0, _MAX_DIMENSION / max(width, height))
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            resized = img.convert("RGB") if img.mode not in ("RGB", "L") else img
            if new_size != (width, height):
                resized = resized.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=85)
            notes = []
            if needs_transpose:
                notes.append("EXIF orientation applied")
            if new_size != (width, height):
                notes.append(
                    f"downscaled from {width}x{height} to {new_size[0]}x{new_size[1]}"
                )
            if not notes:
                notes.append("re-encoded as JPEG to fit byte limits")
            note = ", " + ", ".join(notes)
            return buffer.getvalue(), "image/jpeg", note
    except Exception as e:
        raise ValueError(f"image decoder rejected the file: {e}") from e


def prepare(data: bytes, path: str) -> tuple[str, Attachment]:
    """Return an acknowledgment and attachment, or raise a readable ValueError."""
    media_type = _sniff_media_type(data)
    if media_type is None:
        raise ValueError(
            f"{path} is not a recognized image (format is detected from "
            "file content, not the extension). Supported: gif, jpeg, "
            "png, webp."
        )
    try:
        data, media_type, scale_note = _normalize_image(data, media_type)
    except ValueError as e:
        raise ValueError(
            f"Could not decode {path}: {e}. Regenerate or convert it with "
            "a standard image encoder, then call view_image again."
        ) from e
    encoded = base64.b64encode(data).decode()
    if len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError(
            f"{path} is {len(encoded) // 1024} KB encoded after "
            f"downscaling, above the {MAX_ENCODED_BYTES // 1024} KB "
            "attachment limit. Reduce it first (e.g. crop or convert with "
            "PIL via bash)."
        )
    return (
        f"Attached image {path} ({len(data) // 1024} KB{scale_note}).",
        (media_type, encoded),
    )


def user_message(attachments: Sequence[Attachment]) -> dict[str, Any]:
    """Build the user message to append after all tool results."""
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            }
            for media_type, image_b64 in attachments
        ],
    }


def evict_newest(messages: Sequence[dict[str, Any]], note: str | None = None) -> bool:
    """Replace the last image in place to preserve the earlier prompt cache."""
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for i in range(len(content) - 1, -1, -1):
            part = content[i]
            if isinstance(part, dict) and part.get("type") == "image_url":
                content[i] = {"type": "text", "text": note or EVICTED_NOTE}
                return True
    return False


def evict_all(messages: Sequence[dict[str, Any]]) -> int:
    """Replace all attached images in place and return the number removed."""
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                content[i] = {"type": "text", "text": EVICTED_NOTE}
                count += 1
    return count


def save(
    attachment: Attachment,
    logs_dir: Path,
    *,
    turn: int,
    index: int,
    logger: logging.Logger,
) -> str | None:
    """Save an image and return its relative path; a failed write must not end a run."""
    media_type, image_b64 = attachment
    ext = _EXTENSIONS.get(media_type, "bin")
    name = f"turn{turn:04d}-{index:02d}.{ext}"
    try:
        path = logs_dir / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(image_b64))
    except (OSError, ValueError) as e:
        logger.debug(f"Could not save viewed image {name}: {e}")
        return None
    return f"images/{name}"
