"""Unit tests for the Proximus tool surface and executor."""

import base64
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from harbor_proximus.tools import (
    BASH_DESCRIPTION,
    DEFAULT_TIMEOUT_SEC,
    EXEC_ENV,
    LIST_SUBMISSIONS_DESCRIPTION,
    MAX_TIMEOUT_SEC,
    SUBMIT_DESCRIPTION,
    VIEW_IMAGE_DESCRIPTION,
    BashArgs,
    EnvironmentGoneError,
    NoArgs,
    SubmitArgs,
    Tool,
    ToolExecutor,
    ToolResult,
    ViewImageArgs,
    args_for_trajectory,
    raise_if_environment_gone,
    truncate_middle,
)

pytestmark = pytest.mark.unit


class FakeEnv:
    """Duck-typed environment capturing exec calls."""

    def __init__(self, exec_results: list[ExecResult] | None = None):
        self.exec_calls: list[dict] = []
        self._exec_results = list(exec_results or [])
        self.download_content: str | None = None
        self.download_bytes: bytes | None = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.exec_calls.append(
            {"command": command, "cwd": cwd, "timeout_sec": timeout_sec, "env": env}
        )
        if self._exec_results:
            return self._exec_results.pop(0)
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    async def download_file(self, source_path, target_path):
        if self.download_bytes is not None:
            Path(target_path).write_bytes(self.download_bytes)
            return
        if self.download_content is None:
            raise FileNotFoundError(source_path)
        Path(target_path).write_text(self.download_content)


def make_executor(env: FakeEnv, **kwargs) -> ToolExecutor:
    return ToolExecutor(
        env,  # type: ignore[arg-type]
        **kwargs,
    )


def bash_tool(env: FakeEnv, **kwargs) -> Tool:
    return Tool("bash", BASH_DESCRIPTION, BashArgs, make_executor(env, **kwargs).bash)


def view_image_tool(env: FakeEnv) -> Tool:
    executor = make_executor(env)
    return Tool(
        "view_image", VIEW_IMAGE_DESCRIPTION, ViewImageArgs, executor.view_image
    )


def recording_tool(name: str, args_model: type) -> tuple[Tool, list]:
    seen: list = []

    async def handler(args) -> ToolResult:
        seen.append(args)
        return ToolResult("ok")

    return Tool(name, "test tool", args_model, handler), seen


class TestTruncateMiddle:
    def test_short_text_unchanged(self):
        assert truncate_middle("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 500 + "MIDDLE" + "b" * 500
        result = truncate_middle(text, 100)
        assert len(result) < len(text)
        assert "output truncated" in result
        assert result.startswith("a")
        assert result.endswith("b")


class TestBash:
    async def test_success(self):
        env = FakeEnv([ExecResult(stdout="hello\n", stderr=None, return_code=0)])
        result = await make_executor(env).bash(BashArgs(command="echo hello"))
        assert not result.is_error
        assert result.content.startswith("exit code: 0")
        assert "hello" in result.content
        assert env.exec_calls[0]["command"] == "echo hello"
        assert env.exec_calls[0]["timeout_sec"] == DEFAULT_TIMEOUT_SEC

    async def test_bash_env_carries_no_credentials(self):
        """The agent's own execs get exactly the non-interactive defaults —
        no cloud credential is ever an env var the agent's tool surface can
        read; this pins the env the agent actually sees."""
        env = FakeEnv()
        await make_executor(env).bash(BashArgs(command="env"))
        assert env.exec_calls[0]["env"] == EXEC_ENV

    async def test_nonzero_exit_is_error(self):
        env = FakeEnv([ExecResult(stdout=None, stderr="boom", return_code=2)])
        result = await make_executor(env).bash(BashArgs(command="false"))
        assert result.is_error
        assert "exit code: 2" in result.content
        assert "boom" in result.content

    async def test_timeout_above_ceiling_rejected_with_guidance(self):
        env = FakeEnv()
        result = await make_executor(env).bash(
            BashArgs(command="sleep 1", timeout_sec=999_999)
        )
        assert result.is_error
        assert f"exceeds the {MAX_TIMEOUT_SEC}s foreground limit" in result.content
        assert "background" in result.content
        assert env.exec_calls == []  # rejected before execution

    async def test_timeout_at_ceiling_allowed(self):
        env = FakeEnv()
        await make_executor(env).bash(
            BashArgs(command="make", timeout_sec=MAX_TIMEOUT_SEC)
        )
        assert env.exec_calls[0]["timeout_sec"] == MAX_TIMEOUT_SEC

    async def test_timeout_capped_by_remaining_budget(self):
        # A foreground command must never outlive the time budget: the loop
        # needs control back before the orchestrator's hard kill (the screen
        # lost 4 trials to max-timeout commands started mid-budget).
        env = FakeEnv()
        executor = make_executor(env, remaining_sec=lambda: 1000.0)
        await executor.bash(BashArgs(command="sweep", timeout_sec=MAX_TIMEOUT_SEC))
        assert env.exec_calls[0]["timeout_sec"] == 880  # remaining - 120

        # Near the very end the floor keeps commands runnable.
        executor_late = make_executor(env, remaining_sec=lambda: 90.0)
        await executor_late.bash(BashArgs(command="quick"))
        assert env.exec_calls[1]["timeout_sec"] == 60

        # No budget -> no cap beyond the max.
        env2 = FakeEnv()
        await make_executor(env2).bash(
            BashArgs(command="x", timeout_sec=MAX_TIMEOUT_SEC)
        )
        assert env2.exec_calls[0]["timeout_sec"] == MAX_TIMEOUT_SEC

    async def test_commands_run_in_default_workdir(self):
        env = FakeEnv()
        executor = make_executor(env, default_workdir="/app")
        await executor.bash(BashArgs(command="ls"))
        assert env.exec_calls[0]["cwd"] == "/app"

    async def test_handler_exception_becomes_error_result(self):
        class BrokenEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise ConnectionResetError("transient blip")

        result = await bash_tool(BrokenEnv()).call('{"command": "echo hi"}')
        assert result.is_error
        assert "Tool 'bash' failed: ConnectionResetError: transient blip" in (
            result.content
        )

    async def test_timeout_error_is_reported_explicitly(self):
        class TimeoutEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise TimeoutError()

        result = await make_executor(TimeoutEnv()).bash(
            BashArgs(command="sleep 9999", timeout_sec=60)
        )
        assert result.is_error
        assert "timed out after 60s" in result.content
        # timeout_sec=60 is below the ceiling, so guidance offers both raising
        # it and backgrounding.
        assert "background" in result.content
        assert "timeout_sec" in result.content

    async def test_ceiling_timeout_guidance_omits_raise_hint(self):
        # At the max, "raise timeout_sec" is impossible advice — guidance must
        # point only to backgrounding (PR #2 / pilot audit).
        class TimeoutEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise TimeoutError()

        result = await make_executor(TimeoutEnv()).bash(
            BashArgs(command="sweep", timeout_sec=MAX_TIMEOUT_SEC)
        )
        assert result.is_error
        assert "maximum foreground timeout" in result.content
        assert "background" in result.content

    async def test_binary_output_decode_error_gets_targeted_message(self):
        class BinaryEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise UnicodeDecodeError("utf-8", b"\xc0", 0, 1, "invalid start byte")

        result = await make_executor(BinaryEnv()).bash(
            BashArgs(command="cat binary-file")
        )
        assert result.is_error
        assert "could not be decoded" in result.content
        assert "od/xxd" in result.content
        # Must NOT send the agent down the timeout path.
        assert "timeout_sec" not in result.content

    def test_looks_like_timeout_classification(self):
        looks = ToolExecutor._looks_like_timeout
        # -1 exit after consuming the window -> timeout.
        assert looks(-1, 595.0, 600)
        # -1 exit early in the window -> a real failure, not a timeout.
        assert not looks(-1, 5.0, 600)
        # Consuming the window with a normal exit code -> not a timeout.
        assert not looks(0, 600.0, 600)
        assert not looks(2, 600.0, 600)

    async def test_dash_one_exit_with_full_window_reports_timeout(self, monkeypatch):
        # First call (start) returns 1000, every later call 1599 -> 599s
        # elapsed. Patching time.monotonic would leak into the stdlib module
        # shared by pytest/asyncio, so keep a self-limiting fake.
        ticks = [1000.0]

        def fake_monotonic():
            value = ticks[0]
            ticks[0] = 1599.0
            return value

        monkeypatch.setattr(
            "harbor_proximus.tools.time",
            type("T", (), {"monotonic": staticmethod(fake_monotonic)}),
        )
        env = FakeEnv([ExecResult(stdout="partial", stderr=None, return_code=-1)])
        result = await make_executor(env).bash(BashArgs(command="long-job"))
        assert result.is_error
        assert "timed out after ~599s" in result.content
        assert "partial" in result.content

    async def test_env_gone_error_propagates(self):
        # A dead sandbox must not become just another error result: the loop
        # needs the typed exception to end the run (see proximus.py).
        class GoneEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise EnvironmentGoneError("Modal sandbox is gone")

        with pytest.raises(EnvironmentGoneError):
            await bash_tool(GoneEnv()).call('{"command": "echo hi"}')

    async def test_provider_gone_exception_reraised_from_call(self):
        # Tool.call classifies provider dead-sandbox exceptions by name.
        class GoneEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise _provider_error("SandboxTerminatedError", "modal.exception")

        with pytest.raises(EnvironmentGoneError):
            await bash_tool(GoneEnv()).call('{"command": "echo hi"}')

    async def test_sandbox_timeout_subclass_reraised_as_env_gone(self):
        # Modal's SandboxTimeoutError subclasses builtin TimeoutError, so the
        # executor's command-timeout branch sees it first — it must still
        # classify as a dead sandbox, not a "command timed out" result.
        class GoneEnv(FakeEnv):
            async def exec(self, *args, **kwargs):
                raise _provider_error(
                    "SandboxTimeoutError", "modal.exception", base=TimeoutError
                )

        with pytest.raises(EnvironmentGoneError):
            await bash_tool(GoneEnv()).call('{"command": "echo hi"}')


class TestArgsValidation:
    async def test_unknown_key_rejected(self):
        env = FakeEnv()
        result = await bash_tool(env).call('{"command": "ls", "workdir": "/app"}')
        assert result.is_error
        assert "Invalid arguments for 'bash'" in result.content
        assert "workdir" in result.content
        assert env.exec_calls == []  # rejected before execution

    async def test_wrong_type_rejected(self):
        result = await bash_tool(FakeEnv()).call('{"command": 123}')
        assert result.is_error
        assert "Invalid arguments for 'bash'" in result.content
        assert "command" in result.content

    async def test_missing_required_rejected(self):
        result = await bash_tool(FakeEnv()).call("{}")
        assert result.is_error
        assert "command: Field required" in result.content

    async def test_invalid_json_rejected(self):
        result = await bash_tool(FakeEnv()).call('{"command": ')
        assert result.is_error
        assert "Invalid arguments for 'bash'" in result.content

    async def test_bool_timeout_rejected_by_strict_int(self):
        result = await bash_tool(FakeEnv()).call(
            '{"command": "ls", "timeout_sec": true}'
        )
        assert result.is_error
        assert "timeout_sec" in result.content

    async def test_zero_timeout_rejected(self):
        result = await bash_tool(FakeEnv()).call('{"command": "ls", "timeout_sec": 0}')
        assert result.is_error
        assert "timeout_sec" in result.content

    async def test_empty_arguments_string_uses_defaults(self):
        tool, seen = recording_tool("submit", SubmitArgs)
        result = await tool.call("")
        assert not result.is_error
        assert seen[0].commit == ""

    async def test_submit_blank_commit_means_workspace_snapshot(self):
        tool, seen = recording_tool("submit", SubmitArgs)
        result = await tool.call('{"commit": "   "}')
        assert not result.is_error
        assert seen[0].commit == ""

    async def test_submit_tolerates_stray_keys_and_strips_commit(self):
        tool, seen = recording_tool("submit", SubmitArgs)
        result = await tool.call('{"commit": " abc123 ", "message": "done"}')
        assert not result.is_error
        assert seen[0].commit == "abc123"

    async def test_view_image_requires_path(self):
        result = await view_image_tool(FakeEnv()).call("{}")
        assert result.is_error
        assert "path: Field required" in result.content

    async def test_view_image_rejects_unknown_keys(self):
        result = await view_image_tool(FakeEnv()).call('{"path": "/a.png", "zoom": 2}')
        assert result.is_error
        assert "Invalid arguments for 'view_image'" in result.content

    async def test_view_image_query_accepted_and_ignored(self):
        env = FakeEnv()
        env.download_bytes = make_png(16, 16)
        result = await view_image_tool(env).call(
            '{"path": "/app/plot.png", "query": "what does the curve do?"}'
        )
        assert not result.is_error
        assert result.image is not None


def make_png(width: int, height: int) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg(width: int, height: int, orientation: int | None = None) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    img = Image.new("RGB", (width, height), (30, 30, 200))
    kwargs = {}
    if orientation is not None:
        exif = img.getexif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    img.save(buffer, format="JPEG", **kwargs)
    return buffer.getvalue()


class TestViewImage:
    async def test_attaches_small_image_unchanged(self):
        env = FakeEnv()
        env.download_bytes = make_png(64, 64)
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/app/plot.png")
        )
        assert not result.is_error
        assert result.image is not None
        media_type, b64 = result.image
        assert media_type == "image/png"
        assert base64.b64decode(b64) == env.download_bytes
        assert "Attached image /app/plot.png" in result.content

    async def test_without_pillow_attaches_original_bytes(self, monkeypatch):
        # Pillow is an optional extra (harbor[proximus]); without it the
        # image is attached as-is and downscaling/EXIF handling is skipped.
        import sys

        data = make_png(64, 64)
        monkeypatch.setitem(sys.modules, "PIL", None)
        env = FakeEnv()
        env.download_bytes = data
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/app/plot.png")
        )
        assert not result.is_error
        assert result.image is not None
        media_type, b64 = result.image
        assert media_type == "image/png"
        assert base64.b64decode(b64) == data

    async def test_large_image_is_downscaled(self):
        env = FakeEnv()
        env.download_bytes = make_png(4000, 2000)
        result = await make_executor(env).view_image(ViewImageArgs(path="/app/big.png"))
        assert not result.is_error
        assert result.image is not None
        media_type, b64 = result.image
        assert media_type == "image/jpeg"
        assert "downscaled from 4000x2000" in result.content

        import io

        from PIL import Image

        with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
            assert max(img.size) <= 2000

    async def test_media_type_sniffed_not_extension(self):
        # A PNG mislabeled as .jpg must be sent as image/png: providers
        # reject payloads whose bytes don't match the declared media type.
        env = FakeEnv()
        env.download_bytes = make_png(64, 64)
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/app/photo.jpg")
        )
        assert not result.is_error
        assert result.image is not None
        media_type, b64 = result.image
        assert media_type == "image/png"
        assert base64.b64decode(b64) == env.download_bytes

    async def test_exif_orientation_applied(self):
        env = FakeEnv()
        env.download_bytes = make_jpeg(640, 480, orientation=6)
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/app/rotated.jpg")
        )
        assert not result.is_error
        assert result.image is not None
        media_type, b64 = result.image
        assert media_type == "image/jpeg"
        assert "EXIF orientation applied" in result.content

        import io

        from PIL import Image

        with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
            # Orientation 6 (rotate 90) swaps the stored dimensions and the
            # re-encoded image must carry no orientation tag of its own.
            assert img.size == (480, 640)
            assert img.getexif().get(0x0112, 1) == 1

    async def test_non_image_bytes_rejected(self):
        env = FakeEnv()
        env.download_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/app/audio.png")
        )
        assert result.is_error
        assert "not a recognized image" in result.content

    async def test_malformed_image_is_recoverable_tool_error(self):
        env = FakeEnv()
        # A valid PNG signature/header with no pixel data: sniffing succeeds,
        # but a real decoder must reject it.
        env.download_bytes = make_png(32, 32)[:33]
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/tmp/broken.png")
        )
        assert result.is_error
        assert result.image is None
        assert "Could not decode /tmp/broken.png" in result.content
        assert "call view_image again" in result.content

    async def test_encoded_byte_ceiling_enforced(self, monkeypatch):
        from harbor_proximus import images

        monkeypatch.setattr(images, "MAX_ENCODED_BYTES", 64)
        env = FakeEnv()
        env.download_bytes = make_png(64, 64)
        result = await make_executor(env).view_image(
            ViewImageArgs(path="/app/plot.png")
        )
        assert result.is_error
        assert "attachment limit" in result.content

    async def test_missing_file_reports_error(self):
        result = await make_executor(FakeEnv()).view_image(
            ViewImageArgs(path="/app/nope.png")
        )
        assert result.is_error
        assert "Could not read /app/nope.png" in result.content


def _make_tools() -> list[Tool]:
    async def stub(args) -> ToolResult:
        return ToolResult("ok")

    return [
        Tool("bash", BASH_DESCRIPTION, BashArgs, stub),
        Tool("view_image", VIEW_IMAGE_DESCRIPTION, ViewImageArgs, stub),
        Tool("submit", SUBMIT_DESCRIPTION, SubmitArgs, stub),
        Tool("list_submissions", LIST_SUBMISSIONS_DESCRIPTION, NoArgs, stub),
    ]


def test_tool_schema_shape():
    expected = {
        "bash": (["command", "timeout_sec"], ["command"]),
        "view_image": (["path", "query"], ["path"]),
        "submit": (["commit"], []),
        "list_submissions": ([], []),
    }
    schemas = {tool.name: tool.schema() for tool in _make_tools()}
    for name, (properties, required) in expected.items():
        schema = schemas[name]
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] == name
        assert function["description"]
        params = function["parameters"]
        assert list(params.get("properties", {})) == properties
        assert params.get("required", []) == required
        # Pydantic titles are noise in a provider-facing schema.
        assert "title" not in params
        assert all("title" not in p for p in params.get("properties", {}).values())
    # extra="forbid" surfaces as additionalProperties: false.
    for name in ("bash", "view_image"):
        assert schemas[name]["function"]["parameters"]["additionalProperties"] is False
    # The fixed timeout limits are rendered into the bash descriptions.
    bash = schemas["bash"]["function"]
    assert (
        f"up to {DEFAULT_TIMEOUT_SEC // 60} minutes by default" in bash["description"]
    )
    assert (
        f"up to {MAX_TIMEOUT_SEC} seconds ({MAX_TIMEOUT_SEC // 60} minutes)"
        in bash["description"]
    )
    assert (
        f"(default {DEFAULT_TIMEOUT_SEC}, max {MAX_TIMEOUT_SEC})"
        in bash["parameters"]["properties"]["timeout_sec"]["description"]
    )


def test_args_for_trajectory():
    assert args_for_trajectory('{"command": "ls"}') == {"command": "ls"}
    assert args_for_trajectory("") == {}
    assert args_for_trajectory("{oops") == {"_raw": "{oops"}
    assert args_for_trajectory("[1, 2]") == {"_raw": "[1, 2]"}


def _provider_error(
    name: str,
    module: str,
    base: type[Exception] = Exception,
    message: str = "boom",
) -> Exception:
    """Fabricate an exception shaped like a provider SDK's, without the SDK."""
    return type(name, (base,), {"__module__": module})(message)


class TestRaiseIfEnvironmentGone:
    """Name-based dead-sandbox classification keeps proximus free of
    provider SDK imports and of changes to the environment layer."""

    @pytest.mark.parametrize(
        ("name", "module"),
        [
            ("SandboxTerminatedError", "modal.exception"),
            ("SandboxTimeoutError", "modal.exception"),
            ("NotFoundError", "modal.exception"),
            ("DaytonaNotFoundError", "daytona.common.errors"),
            ("SandboxNotFoundException", "e2b.exceptions"),
        ],
    )
    def test_dead_sandbox_errors_reraise(self, name: str, module: str):
        with pytest.raises(EnvironmentGoneError) as exc_info:
            raise_if_environment_gone(_provider_error(name, module))
        assert name in str(exc_info.value)

    @pytest.mark.parametrize(
        ("name", "module", "base", "message"),
        [
            (
                "ConnectionError",
                "builtins",
                ConnectionError,
                "[Errno -2] Name or service not known",
            ),
            (
                "ConnectionError",
                "builtins",
                ConnectionError,
                "[Errno 111] Connect call failed ('74.248.34.139', 443)",
            ),
            (
                "InternalError",
                "modal.exception",
                Exception,
                "failed to read exec stdio stream: please contact support@modal.com",
            ),
            (
                "RuntimeError",
                "builtins",
                Exception,
                'urpc method "containerManager.WaitPID" failed: EOF',
            ),
            (
                "RuntimeError",
                "builtins",
                Exception,
                "connecting to control server at PID 4: connection refused",
            ),
            (
                "ConflictError",
                "modal.exception",
                Exception,
                "Modal Sandbox is shutting down.",
            ),
        ],
    )
    def test_environment_transport_loss_reraises(
        self,
        name: str,
        module: str,
        base: type[Exception],
        message: str,
    ):
        with pytest.raises(EnvironmentGoneError):
            raise_if_environment_gone(_provider_error(name, module, base, message))

    def test_unrelated_connection_error_passes_through(self):
        raise_if_environment_gone(ConnectionError("task API request failed"))

    @pytest.mark.parametrize(
        ("name", "module"),
        [
            ("RuntimeError", "builtins"),
            # A missing *file* is an ordinary tool failure, not a dead sandbox.
            ("SandboxFilesystemNotFoundError", "modal.exception"),
            # The generic name only counts inside the modal namespace.
            ("NotFoundError", "someothersdk.errors"),
        ],
    )
    def test_ordinary_errors_pass_through(self, name: str, module: str):
        raise_if_environment_gone(_provider_error(name, module))
