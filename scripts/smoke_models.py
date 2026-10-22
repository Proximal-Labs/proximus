"""Live check of each model API: a tool call, then a turn that replays it.

Usage: uv run python scripts/smoke_models.py anthropic/claude-opus-5-5 \\
           openai/gpt-5.6-sol --reasoning-effort high

The second turn replays the first verbatim (thinking signatures, encrypted
reasoning), which is what a long run depends on.
Uses each provider's usual key variable, or one gateway key for all of them
(--api-key-env with --anthropic-base/--openai-base), and spends a few cents
per model.
"""

import argparse
import asyncio
import json
import logging
import sys

from harbor_proximus import prompts, tools
from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM

TASK = (
    "Use the bash tool to run `echo proximus-smoke`, then reply with exactly "
    "the text it printed."
)


async def check(model: str, args: argparse.Namespace) -> bool:
    base = args.anthropic_base if model.startswith("anthropic/") else args.openai_base
    info = (
        {"max_input_tokens": args.max_input_tokens} if args.max_input_tokens else None
    )
    llm = LLM(
        model,
        ProximusConfig(
            reasoning_effort=args.reasoning_effort,
            model_info=info,
            api_key_env=args.api_key_env,
            api_base=base,
        ),
        remaining_sec=lambda: None,
        logger=logging.getLogger("smoke"),
    )
    llm.session_id = "proximus-smoke"
    schema = [tools.Tool("bash", tools.BASH_DESCRIPTION, tools.BashArgs, None).schema()]
    messages = [
        {"role": "system", "content": prompts.system_prompt(False)},
        {"role": "user", "content": TASK},
    ]
    first = await llm.query(messages, schema)
    if not first.tool_calls:
        print(
            f"FAIL {model} [{type(llm.provider).__name__}]: no tool call: {first.content!r}"
        )
        return False
    messages += llm.assistant_messages(first)
    messages += [
        {"role": "tool", "tool_call_id": call.call_id, "content": "proximus-smoke\n"}
        for call in first.tool_calls
    ]
    second = await llm.query(messages, schema)
    ok = "proximus-smoke" in second.content
    print(
        f"{'OK  ' if ok else 'FAIL'} {model} [{type(llm.provider).__name__}] "
        f"call={first.tool_calls[0].name}"
        f"({json.loads(first.tool_calls[0].raw_arguments or '{}')}) "
        f"reply={second.content.strip()[:60]!r} "
        f"tokens={second.prompt_tokens}/{second.cached_tokens} cached "
        f"cost=${first.cost_usd + second.cost_usd:.4f}"
    )
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="for models missing from the built-in table",
    )
    parser.add_argument("--api-key-env", help="one key variable for every model")
    parser.add_argument("--anthropic-base", help="endpoint for anthropic/ models")
    parser.add_argument("--openai-base", help="endpoint for every other model")
    args = parser.parse_args()
    results = []
    for model in args.models:
        try:
            results.append(await check(model, args))
        except Exception as error:  # noqa: BLE001 - report every model
            print(f"FAIL {model}: {type(error).__name__}: {error}")
            results.append(False)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
