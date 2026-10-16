import base64
import json
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.base import BaseEnvironment
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from harbor_proximus.config import BUDGET_MARGIN_SEC


class EnvironmentGoneError(RuntimeError):
    """The sandbox no longer exists or was terminated."""


# Provider SDK exceptions that mean the sandbox itself is gone, matched by
# name so this package needs no provider imports. "NotFoundError" is too
# generic to match alone, so it also requires the modal namespace.
_GONE_EXCEPTION_NAMES = frozenset(
    {
        "SandboxTerminatedError",  # modal: OOM, preemption, kill
        "SandboxTimeoutError",  # modal: sandbox lifetime expired
        "DaytonaNotFoundError",  # daytona: sandbox 404
        "SandboxNotFoundException",  # e2b: deleted or expired
    }
)


def _is_environment_transport_loss(error: Exception) -> bool:
    """Recognize control-plane failures raised by an environment operation.

    A task command's own network error returns inside ``ExecResult``. These
    exceptions instead mean Harbor could not reach the sandbox exec service.
    """
    message = str(error).lower()
    return (
        isinstance(error, ConnectionError)
        and (
            "[errno -2] name or service not known" in message
            or "[errno 111] connect call failed" in message
        )
    ) or (
        "failed to read exec stdio stream" in message
        or "modal sandbox is shutting down" in message
        or ("containermanager.waitpid" in message and "failed: eof" in message)
        or (
            "connecting to control server" in message
            and "connection refused" in message
        )
    )


def raise_if_environment_gone(error: Exception) -> None:
    """Re-raise a dead-sandbox or lost-control-channel provider exception."""
    name = type(error).__name__
    module = type(error).__module__ or ""
    gone = (
        name in _GONE_EXCEPTION_NAMES
        or (name == "NotFoundError" and module.startswith("modal"))
        or _is_environment_transport_loss(error)
    )
    if gone:
        raise EnvironmentGoneError(f"{name}: {error}") from error


# Fixed executor limits. Command timeouts are additionally capped at the
# remaining time budget minus BUDGET_MARGIN_SEC, so the loop regains control
# before the orchestrator's hard kill.
DEFAULT_TIMEOUT_SEC = 600
MAX_TIMEOUT_SEC = 1800
OUTPUT_LIMIT_CHARS = 30_000

# Non-interactive defaults applied to every bash exec
EXEC_ENV: dict[str, str] = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "DEBIAN_FRONTEND": "noninteractive",
}

# Strictest caps across provider vision docs: 2000px long edge, 4MB base64.
_IMAGE_MAX_DIMENSION = 2000
_IMAGE_MAX_ENCODED_BYTES = 4 * 1024 * 1024
_IMAGE_MAX_RAW_BYTES = _IMAGE_MAX_ENCODED_BYTES * 3 // 4  # base64 inflates 4/3
# EXIF tag 274 ("Orientation"); hardcoded to avoid importing PIL at module load.
_EXIF_ORIENTATION = 0x0112


def _sniff_media_type(data: bytes) -> str | None:
    """Detect the image format from magic bytes; extensions are never
    trusted (providers reject mismatched media types)."""
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
    """Apply EXIF orientation and downscale to fit provider limits; returns
    (data, media_type, note). Falls back to the original bytes only when
    Pillow cannot be imported (it is a declared dependency). Invalid image data raises ValueError so callers
    can report a recoverable tool error instead of sending it to a provider."""
    try:
        import io

        from PIL import Image, ImageOps
    except ImportError:
        return data, media_type, ""

    try:
        with Image.open(io.BytesIO(data)) as img:
            # Image.open() is lazy and may accept a valid header around an
            # invalid pixel stream. Force decoding before returning unchanged.
            img.load()
            orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
            needs_transpose = orientation != 1
            width, height = img.size
            if (
                not needs_transpose
                and max(width, height) <= _IMAGE_MAX_DIMENSION
                and len(data) <= _IMAGE_MAX_RAW_BYTES
            ):
                return data, media_type, ""
            if needs_transpose:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
            scale = min(1.0, _IMAGE_MAX_DIMENSION / max(width, height))
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


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # (media_type, base64_data) — attached to the conversation as an image
    # content block by the agent loop (kept out of the ATIF trajectory).
    image: tuple[str, str] | None = None
    # Set only by a confirmed final submission.
    ends_run: bool = False


BASH_DESCRIPTION = (
    "Run a shell command in the task environment and return its "
    "output. Each call runs in a FRESH shell from the task "
    "working directory: environment variables and shell state "
    "do not persist between calls, so prefer absolute paths. "
    "Long output is truncated in the middle. Default timeout "
    f"{DEFAULT_TIMEOUT_SEC}s, max "
    f"{MAX_TIMEOUT_SEC}s. Anything that may run longer "
    "(builds, training runs, benchmarks, servers, parameter "
    "sweeps) must be launched in the background rather than "
    "waited on in the foreground: redirect output to a log "
    "file and append ` &` (e.g. `nohup ./run.sh > /tmp/run.log "
    "2>&1 &`), checkpoint best-so-far results to a file, then "
    "poll the log with later calls. A foreground command that "
    "hits the timeout is killed and its work lost."
)

VIEW_IMAGE_DESCRIPTION = (
    "View an image file (png/jpg/gif/webp; format detected "
    "from content, any extension works) from the task "
    "environment. The image is attached to the conversation "
    "so you can see it. Images consume significant context — "
    "view only what you need."
)

LIST_SUBMISSIONS_DESCRIPTION = (
    "List your recorded submissions (when, id, description) and "
    "which one is currently kept as your result."
)

SUBMIT_DESCRIPTION = (
    "Submit the current workspace as your result. A commit is "
    "optional: omit it and the harness commits the live files for "
    "you (uncommitted work included); or pass a commit hash that "
    "already exists in the workspace repository."
)


class BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(description="The shell command to execute.")
    timeout_sec: int = Field(
        default=DEFAULT_TIMEOUT_SEC,
        strict=True,
        ge=1,
        description=(
            "Seconds before the command is killed (default "
            f"{DEFAULT_TIMEOUT_SEC}, max {MAX_TIMEOUT_SEC})."
        ),
    )


class ViewImageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path to the image file.")
    # Accepted and ignored: models habitually pass one, and the call must
    # never fail on it.
    query: str = Field(
        default="",
        description=(
            "Optional: what to look for in the image. The image is attached "
            "to the conversation for you to view directly."
        ),
    )


class SubmitArgs(BaseModel):
    # Stray extras are tolerated: failing a submission over one is worse.
    model_config = ConfigDict(extra="ignore")

    commit: str = Field(
        default="",
        description=(
            "Optional commit hash or ref. Omit (or pass HEAD) and the "
            "harness commits the current workspace state for you, "
            "uncommitted changes included. An explicit hash must already "
            "exist in the workspace repository."
        ),
    )

    @field_validator("commit")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    handler: Callable[[Any], Awaitable[ToolResult]]

    def schema(self) -> dict[str, Any]:
        parameters = self.args.model_json_schema()
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    async def call(self, raw_arguments: str) -> ToolResult:
        try:
            args = self.args.model_validate_json((raw_arguments or "").strip() or "{}")
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc']) or 'arguments'}: "
                f"{err['msg']}"
                for err in e.errors()
            )
            return ToolResult(
                f"Invalid arguments for '{self.name}': {problems}.", is_error=True
            )
        try:
            return await self.handler(args)
        except EnvironmentGoneError:
            raise
        except Exception as e:
            raise_if_environment_gone(e)
            return ToolResult(
                f"Tool '{self.name}' failed: {type(e).__name__}: {e}", is_error=True
            )


def args_for_trajectory(raw_arguments: str) -> dict[str, Any]:
    """The call's raw arguments as a dict for the ATIF record, valid or not."""
    try:
        value = json.loads((raw_arguments or "").strip() or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_arguments}
    return value if isinstance(value, dict) else {"_raw": raw_arguments}


def truncate_middle(text: str, limit_chars: int) -> str:
    """Truncate the middle of long text, keeping the head and tail."""
    if len(text) <= limit_chars:
        return text
    head = int(limit_chars * 0.6)
    tail = limit_chars - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n[... output truncated: {omitted} characters omitted ...]\n"
        + text[-tail:]
    )


def format_exec_output(stdout: str | None, stderr: str | None) -> str:
    """Join stdout + stderr into one model-readable block, truncated in
    the middle."""
    parts = []
    if stdout:
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    body = "\n".join(parts).strip()
    return truncate_middle(body, OUTPUT_LIMIT_CHARS) if body else "(no output)"


class ToolExecutor:
    """Executes bash and view_image calls against a task environment."""

    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        default_workdir: str | None = None,
        remaining_sec: Callable[[], float | None] | None = None,
    ):
        self._environment = environment
        # Caps command timeouts so a foreground command can't outlive the
        # time budget — the loop must regain control before the hard kill.
        self._remaining_sec = remaining_sec
        self._default_workdir = default_workdir

    async def bash(self, args: BashArgs) -> ToolResult:
        if args.timeout_sec > MAX_TIMEOUT_SEC:
            # Teach instead of silently clamping.
            return ToolResult(
                f"timeout_sec={args.timeout_sec} exceeds the "
                f"{MAX_TIMEOUT_SEC}s foreground limit. Long jobs (builds, "
                "sweeps, training, benchmarks) should run in the background: "
                "redirect output to a log and append ` &`, then poll the log "
                "with later commands — or bound the work to fit the limit.",
                is_error=True,
            )
        timeout = self._clamp_timeout(args.timeout_sec)
        started = time.monotonic()
        try:
            result = await self._environment.exec(
                args.command,
                cwd=self._default_workdir,
                env=dict(EXEC_ENV),
                timeout_sec=timeout,
            )
        except UnicodeDecodeError as e:
            # Binary bytes on stdout/stderr break the transport's text
            # decoding; the whole output is lost.
            return ToolResult(
                f"Command output could not be decoded as text ({e}). Its "
                "output was discarded. Binary data must not reach "
                "stdout/stderr directly: redirect it to a file and inspect "
                "with od/xxd/cat -v/strings, or pipe through such a filter.",
                is_error=True,
            )
        except TimeoutError as e:
            # Modal's SandboxTimeoutError subclasses builtin TimeoutError but
            # means the sandbox died, not the command — classify first.
            raise_if_environment_gone(e)
            return ToolResult(
                f"Command timed out after {timeout}s (timeout_sec={timeout}) "
                f"and was killed. {self._timeout_guidance(timeout)}",
                is_error=True,
            )
        body = format_exec_output(result.stdout, result.stderr)
        elapsed = time.monotonic() - started
        if self._looks_like_timeout(result.return_code, elapsed, timeout):
            return ToolResult(
                f"Command timed out after ~{int(elapsed)}s "
                f"(timeout_sec={timeout}) and was killed; output up to that "
                f"point:\n{body}\n{self._timeout_guidance(timeout)}",
                is_error=True,
            )
        return ToolResult(
            f"exit code: {result.return_code}\n{body}",
            is_error=result.return_code != 0,
        )

    async def view_image(self, args: ViewImageArgs) -> ToolResult:
        path = args.path
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / PurePosixPath(path).name
            try:
                await self._environment.download_file(path, local_path)
                data = local_path.read_bytes()
            except EnvironmentGoneError:
                raise
            except Exception as e:
                raise_if_environment_gone(e)
                return ToolResult(f"Could not read {path}: {e}", is_error=True)

        media_type = _sniff_media_type(data)
        if media_type is None:
            return ToolResult(
                f"{path} is not a recognized image (format is detected from "
                "file content, not the extension). Supported: gif, jpeg, "
                "png, webp.",
                is_error=True,
            )
        try:
            data, media_type, scale_note = _normalize_image(data, media_type)
        except ValueError as e:
            return ToolResult(
                f"Could not decode {path}: {e}. Regenerate or convert it with "
                "a standard image encoder, then call view_image again.",
                is_error=True,
            )
        encoded = base64.b64encode(data).decode()
        if len(encoded) > _IMAGE_MAX_ENCODED_BYTES:
            return ToolResult(
                f"{path} is {len(encoded) // 1024} KB encoded after "
                f"downscaling, above the {_IMAGE_MAX_ENCODED_BYTES // 1024} KB "
                "attachment limit. Reduce it first (e.g. crop or convert with "
                "PIL via bash).",
                is_error=True,
            )
        return ToolResult(
            f"Attached image {path} ({len(data) // 1024} KB{scale_note}).",
            image=(media_type, encoded),
        )

    def _clamp_timeout(self, timeout_sec: int) -> int:
        timeout = min(timeout_sec, MAX_TIMEOUT_SEC)
        if self._remaining_sec is not None:
            remaining = self._remaining_sec()
            if remaining is not None:
                # Return control before budget end so the graceful finalize
                # path wins the race against the hard kill.
                timeout = min(timeout, max(60, int(remaining) - BUDGET_MARGIN_SEC))
        return max(1, timeout)

    @staticmethod
    def _looks_like_timeout(return_code: int, elapsed_sec: float, timeout: int) -> bool:
        """Environments report exec timeouts as a bare -1 exit; infer a
        timeout when the command also consumed the whole window."""
        return return_code == -1 and elapsed_sec >= min(timeout - 1, timeout * 0.9)

    @staticmethod
    def _timeout_guidance(timeout: int) -> str:
        if timeout >= MAX_TIMEOUT_SEC:
            # At the ceiling, "raise timeout_sec" is impossible advice.
            return (
                f"This is the maximum foreground timeout ({MAX_TIMEOUT_SEC}s). "
                "Run this in the background instead: redirect to a log and "
                "append ` &`, then poll the log with later commands."
            )
        return (
            f"Raise timeout_sec (up to {MAX_TIMEOUT_SEC}s), or run it in "
            "the background (redirect to a log, append ` &`, poll the log)."
        )
