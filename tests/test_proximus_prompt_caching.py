"""The second cache breakpoint must land on the newest message, every turn.

Everything after that breakpoint is re-read at full input price on the next
call, so a breakpoint that cannot advance is a pure cost leak -- and it fails
silently, since the run still succeeds and only the bill moves.

The shape that once broke it: an assistant turn that is a pure tool call has
no text, and in a conversation that is only assistant-tool_call -> tool-result
nothing was taggable after the opening instruction, pinning the breakpoint to
``messages[1]`` for the whole run. On the Messages API the tool results form
the newest user turn, so the breakpoint rides on its last tool_result block.
"""

from __future__ import annotations

from harbor_proximus.providers.anthropic import apply_cache_control, to_wire

TTL_1H = {"type": "ephemeral", "ttl": "1h"}


def _wire(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system, wire = to_wire(messages)
    apply_cache_control(system, wire)
    return system, wire


def _marks(wire: list[dict]) -> list[tuple[int, int]]:
    """(message, block) positions carrying a cache breakpoint."""
    return [
        (i, j)
        for i, message in enumerate(wire)
        for j, block in enumerate(message["content"])
        if block.get("cache_control") is not None
    ]


def _bash_turns(n: int) -> list[dict]:
    """A pure tool-calling conversation: no assistant text, no later user turn."""
    messages: list[dict] = [
        {"role": "system", "content": "You are Proximus."},
        {"role": "user", "content": "Optimize the codegen backend."},
    ]
    for i in range(n):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd":"make"}'},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"output {i}"}
        )
    return messages


def test_breakpoint_advances_through_pure_tool_calls():
    """The regression: the anchor must reach the newest message, not messages[1]."""
    system, wire = _wire(_bash_turns(40))

    assert system[-1]["cache_control"] == TTL_1H, "system prompt must be tagged"
    marks = _marks(wire)
    assert marks == [(len(wire) - 1, len(wire[-1]["content"]) - 1)], (
        f"second breakpoint at {marks}; everything after it is re-read at full "
        "price every turn"
    )
    block = wire[-1]["content"][-1]
    assert block["type"] == "tool_result"
    assert block["content"] == "output 39", "tool content must be left untouched"


def test_breakpoints_never_touch_the_history():
    messages = _bash_turns(2)
    _wire(messages)
    assert "cache_control" not in str(messages)


def test_assistant_text_is_an_anchor_when_it_is_newest():
    messages = _bash_turns(2)
    messages.append({"role": "assistant", "content": "Here is the plan."})
    _, wire = _wire(messages)

    assert _marks(wire) == [(len(wire) - 1, 0)]
    assert wire[-1]["content"][0]["text"] == "Here is the plan."


def test_multimodal_breakpoint_tags_text_not_image():
    messages = [
        {"role": "system", "content": "You are Proximus."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]
    _, wire = _wire(messages)
    text, image = wire[-1]["content"]

    assert text["cache_control"] == TTL_1H
    assert image["type"] == "image"
    assert image["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "AAAA",
    }
    assert "cache_control" not in image


def test_tool_results_and_attached_images_share_one_user_turn():
    """Tool results come first in the user turn; the image after them is
    never the anchor, the last tool_result is."""
    messages = _bash_turns(1)
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
        }
    )
    _, wire = _wire(messages)

    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    kinds = [block["type"] for block in wire[-1]["content"]]
    assert kinds == ["tool_result", "image"]
    assert _marks(wire) == [(2, 0)]
