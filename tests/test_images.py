"""Offline tests for image preparation, messages, and artifacts."""

import base64
import copy
import io
import logging

import pytest
from PIL import Image

from harbor_proximus import images

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "format, media_type",
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("GIF", "image/gif"),
        ("WEBP", "image/webp"),
    ],
)
def test_prepare_preserves_small_images(format, media_type):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format=format)
    data = buffer.getvalue()

    content, attachment = images.prepare(data, "/app/plot.dat")

    assert content == f"Attached image /app/plot.dat ({len(data) // 1024} KB)."
    assert attachment == (media_type, base64.b64encode(data).decode())


def test_prepare_unknown_format_reports_the_path():
    with pytest.raises(ValueError) as error:
        images.prepare(b"not an image", "/app/plot.png")
    assert str(error.value) == (
        "/app/plot.png is not a recognized image (format is detected from "
        "file content, not the extension). Supported: gif, jpeg, png, webp."
    )


def test_user_message_preserves_attachment_order():
    assert images.user_message([("image/png", "first"), ("image/jpeg", "second")]) == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,first"}},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,second"},
            },
        ],
    }


def test_evict_newest_preserves_earlier_images_and_other_content():
    older = images.user_message([("image/png", "old")])
    recent = images.user_message([("image/png", "first"), ("image/png", "last")])
    text = {"type": "text", "text": "keep this"}
    recent["content"].append(text)
    messages = [older, recent, {"role": "assistant", "content": "done"}]
    before = copy.deepcopy(messages)

    assert images.evict_newest(messages, images.INVALID_IMAGE_NOTE)
    assert older == before[0]
    assert recent["content"] == [
        before[1]["content"][0],
        {"type": "text", "text": "[image removed: provider rejected it]"},
        text,
    ]
    assert messages[-1] == before[-1]

    assert images.evict_newest(messages)
    assert recent["content"][0] == {
        "type": "text",
        "text": "[image removed: request too large]",
    }
    assert older == before[0]
    assert images.evict_newest(messages)
    assert not images.evict_newest(messages)


def test_evict_all_counts_images_and_preserves_other_parts():
    message = images.user_message([("image/png", "first"), ("image/jpeg", "second")])
    other_parts = [{"type": "text", "text": "keep"}, "other", None]
    message["content"].extend(other_parts)
    messages = [
        {"role": "system", "content": "instruction"},
        message,
        {"role": "assistant", "content": None},
    ]

    assert images.evict_all(messages) == 2
    assert messages == [
        {"role": "system", "content": "instruction"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[image removed: request too large]"},
                {"type": "text", "text": "[image removed: request too large]"},
                *other_parts,
            ],
        },
        {"role": "assistant", "content": None},
    ]
    assert images.evict_all(messages) == 0


def test_evict_empty_history():
    assert not images.evict_newest([])
    assert images.evict_all([]) == 0


@pytest.mark.parametrize(
    "media_type, extension",
    [
        ("image/png", "png"),
        ("image/jpeg", "jpg"),
        ("image/gif", "gif"),
        ("image/webp", "webp"),
        ("other", "bin"),
    ],
)
def test_save_uses_relative_paths_and_keeps_files_after_eviction(
    tmp_path, media_type, extension
):
    data = b"image bytes"
    attachment = (media_type, base64.b64encode(data).decode())
    path = images.save(
        attachment, tmp_path, turn=7, index=3, logger=logging.getLogger(__name__)
    )

    assert path == f"images/turn0007-03.{extension}"
    assert images.evict_all([images.user_message([attachment])]) == 1
    assert (tmp_path / path).read_bytes() == data


@pytest.mark.parametrize("failure", ["directory", "decode"])
def test_save_failure_is_nonfatal(tmp_path, caplog, failure):
    if failure == "directory":
        (tmp_path / "images").write_text("not a directory")
    encoded = "a" if failure == "decode" else base64.b64encode(b"image").decode()

    with caplog.at_level(logging.DEBUG):
        path = images.save(
            ("image/png", encoded),
            tmp_path,
            turn=1,
            index=0,
            logger=logging.getLogger(__name__),
        )

    assert path is None
    assert "Could not save viewed image turn0001-00.png" in caplog.text
