"""Extract Answer Engineering rulesets/subruns from notebook JSON and live.

Purpose:
    Parse notebook cells into reproduction-facing ruleset and subrun records,
    including optional system-prompt extraction and markdown normalization.

Architectural role:
    Notebook-ingestion adapter used by reproduction planning workflows.

"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]

_RUN_HEADING_RE = re.compile(r"^##\s*Run(?:\s*:\s*(.*?))?\s*$")
_MODE_HEADING_RE = re.compile(r"^##\s*Mode(?:\s*:\s*(.*?))?\s*$")
_PAPER_ROLE_HEADING_RE = re.compile(r"^##\s*Paper\s*Role(?:\s*:\s*(.*?))?\s*$")
_VARIANT_HEADING_RE = re.compile(r"^##\s*Variant(?:\s*:\s*(.*?))?\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*[-*]\s+(.*?)\s*$")

type GenerationMode = Literal["baseline", "reasoning", "trajectory"]
type PaperRole = Literal["primary", "ablation", "appendix", "exploratory"]

try:
    from google.colab import (  # pyright: ignore[reportMissingModuleSource]
        _message,  # pyright: ignore[reportMissingModuleSource]
    )

    _colab_message_module = _message
except ImportError:
    _colab_message_module = None


class ColabMessageModule(Protocol):
    """Protocol for the subset of Colab messaging used to fetch the live.

    Purpose:
        Abstract the blocking request operation needed to ask a running Colab
        notebook for its current `.ipynb` contents.

    Architectural role:
        Colab runtime capability protocol at the notebook-extraction boundary.

    Inputs (architectural provenance):
        Implemented by the Colab `_message` module when notebook extraction runs
        inside Colab.

    Outputs (downstream usage):
        Consumed by `_extract_notebook_payload_from_colab_runtime()`.

    Invariants/constraints:
        Implementations must support the `get_ipynb` request pattern used by
        this module.

    """

    def blocking_request(
        self, command: str, *, request: str, timeout_sec: int
    ) -> object:
        """Protocol method for Colab's blocking message request API.

        Purpose:
            Describe the external Colab API surface used when notebook
            extraction needs to request frontend-backed data.

        Architectural role:
            Protocol boundary around an optional runtime dependency that is
            unavailable in ordinary local execution.

        Inputs (architectural provenance):
            Receives request names and payloads passed through by notebook
            extraction helpers.

        Outputs (downstream usage):
            Returns Colab-provided response objects consumed by extractor code.

        Invariants/constraints:
            This method documents shape only. Implementations are provided by
            the Colab environment rather than this repository.

        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NotebookRulesetSpec:
    """Parsed specification for one Answer Engineering ruleset cell.

    Parse a marked notebook cell into the ruleset metadata used by reproduction
    planning: display name, authored rules markdown, optional system prompt,
    case-type labels, cell index, and source hint. Notebook users normally get
    these objects indirectly through
    :class:`~ae_paper_reproduction.NotebookSubruns`.

    .. note::
        The source cell must start with the configured Answer Engineering
        marker. Invalid cells fail during extraction, before model generation
        begins.

    Examples:
        ```python
        spec = NotebookRulesetSpec(
            source=cell_source,
            cell_index=3,
            source_hint="notebooks/reproduce.ipynb",
        )
        print(spec.ruleset_name)
        print(spec.rules_markdown)
        ```

    Attributes:
        ruleset_name: Human-readable ruleset name parsed from the cell.
        rules_markdown: Markdown rules body compiled for a subrun.
        system_prompt: Optional system prompt parsed from the cell.
        case_types: Case-type labels requested by the ruleset cell.
        cell_index: Source notebook cell index.
        source_hint: Human-readable notebook/source identifier.

    Runtime behavior:
        Construction normalizes notebook source text, validates the marker,
        extracts run metadata, separates an optional system-prompt section, and
        renders the remaining rules body as markdown.

    Architectural role:
        Notebook-parsing boundary for reproduction planning. It converts raw
        notebook cell text into a typed record consumed by subrun planning.

    Consumes:
        Raw notebook cell source text plus provenance such as cell index and
        source hint.

    Produces:
        Parsed ruleset metadata consumed by
        :class:`~ae_paper_reproduction.SubrunDefinition` and
        :class:`~ae_paper_reproduction.NotebookSubruns`.

    Invariants:
        All fields must describe the same source cell. The parsed markdown and
        metadata should be deterministic for a given source string and parser
        configuration.

    Developer Notes:
        Keep this boundary specific to notebook reproduction planning. Do not
        mix remote notebook connectivity, dataset selection, or runtime
        generation into this parser.

    Todo:
        Improve diagnostics for malformed ruleset cells and expose enough
        context for users editing their own reproduction notebooks.

    See Also:
        :class:`~ae_paper_reproduction.NotebookSubruns`
        :class:`~ae_paper_reproduction.SubrunDefinition`
        :class:`~answer_engineering.CompiledRules`

    """

    ruleset_name: str
    rules_markdown: str
    system_prompt: str | None
    mode: GenerationMode | None
    paper_role: PaperRole | None
    paper_variant: str | None
    case_types: tuple[str, ...]
    cell_index: int
    source_hint: str

    def __init__(
        self,
        source: str,
        *,
        cell_index: int,
        source_hint: str,
        marker: str = "# Answer Engineering Rules",
        default_name_template: str = "notebook-cell-{cell_index}",
    ) -> None:
        """Parse one marked notebook cell into a `NotebookRulesetSpec`.

        Purpose:
            Normalize cell source text, extract run name and case-type bullets
            when present, optionally extract a system-prompt section, and render
            the remaining rules markdown.

        Architectural role:
            Main parsing constructor for notebook-authored rulesets.

        Inputs (architectural provenance):
            Consumes raw cell source text and notebook provenance from
            notebook-loading code.

        Outputs (downstream usage):
            Populates this parsed ruleset specification for subrun extraction
            workflows.

        Invariants/constraints:
            The source must begin with the configured marker for the parse to
            succeed.

        """
        normalized = _normalize_source(source)
        lines = normalized.split("\n")
        if not lines or lines[0].strip() != marker:
            msg = (
                f"Cell {cell_index} in {source_hint} "
                f"does not start with {marker!r}"
            )
            raise ValueError(msg)

        parsed_ruleset_name = default_name_template.format(
            cell_index=cell_index
        )
        parsed_case_types: list[str] = []
        parsed_mode: GenerationMode | None = None
        parsed_paper_role: PaperRole | None = None
        parsed_paper_variant: str | None = None
        body_start = 1

        idx = 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        if idx < len(lines):
            run_match = _RUN_HEADING_RE.match(lines[idx].strip())
            if run_match is not None:
                explicit_name = (run_match.group(1) or "").strip()
                if explicit_name:
                    parsed_ruleset_name = explicit_name
                idx += 1
                while True:
                    extracted_mode, next_idx = _extract_mode(lines, idx)
                    if next_idx != idx:
                        parsed_mode = extracted_mode
                        idx = next_idx
                        continue
                    extracted_role, next_idx = _extract_paper_role(lines, idx)
                    if next_idx != idx:
                        parsed_paper_role = extracted_role
                        idx = next_idx
                        continue
                    extracted_variant, next_idx = _extract_paper_variant(
                        lines, idx
                    )
                    if next_idx != idx:
                        parsed_paper_variant = extracted_variant
                        idx = next_idx
                        continue
                    break
                while idx < len(lines):
                    stripped = lines[idx].strip()
                    if not stripped:
                        idx += 1
                        continue
                    item_match = _LIST_ITEM_RE.match(lines[idx])
                    if item_match is None:
                        break
                    case_type = item_match.group(1).strip()
                    if case_type:
                        parsed_case_types.append(case_type)
                    idx += 1
                body_start = idx

        if parsed_mode is None:
            parsed_mode, body_start = _extract_mode(lines, body_start)
        if parsed_paper_role is None:
            parsed_paper_role, body_start = _extract_paper_role(
                lines, body_start
            )
        if parsed_paper_variant is None:
            parsed_paper_variant, body_start = _extract_paper_variant(
                lines, body_start
            )
        parsed_system_prompt, body_start = _extract_system_prompt(
            lines, body_start
        )
        parsed_rules_markdown = _render_rules_markdown(lines[body_start:])
        object.__setattr__(self, "ruleset_name", parsed_ruleset_name)
        object.__setattr__(self, "rules_markdown", parsed_rules_markdown)
        object.__setattr__(self, "system_prompt", parsed_system_prompt)
        object.__setattr__(self, "mode", parsed_mode)
        object.__setattr__(self, "paper_role", parsed_paper_role)
        object.__setattr__(self, "paper_variant", parsed_paper_variant)
        object.__setattr__(self, "case_types", tuple(parsed_case_types))
        object.__setattr__(self, "cell_index", cell_index)
        object.__setattr__(self, "source_hint", source_hint)


type NotebookSubrun = tuple[NotebookRulesetSpec, str | None]


def _normalize_source(source: str) -> str:
    """Normalize notebook cell source to LF newlines with trailing spaces.

    Purpose:
        Preserve cell content while standardizing newline style and removing
        right-edge whitespace before section parsing.

    """
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(part.rstrip() for part in lines)


def _as_json_object(value: JsonValue) -> JsonObject | None:
    """Return ``value`` as a string-keyed mapping when it is mapping-like."""
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def _cell_source_text(cell: JsonObject) -> str:
    """Extract one notebook cell's source text from string/list JSON forms."""
    source_raw = cell.get("source", "")
    if isinstance(source_raw, list):
        source_parts = [part for part in source_raw if isinstance(part, str)]
        return "".join(source_parts)
    if isinstance(source_raw, str):
        return source_raw
    return str(source_raw)


@dataclass(frozen=True, slots=True, init=False)
class NotebookCellPayload:
    """Minimal parsed notebook cell payload.

    Purpose:
        Hold only the cell type and source text needed by later ruleset
        extraction logic.

    Architectural role:
        Intermediate notebook-representation object inside the extraction
        boundary.

    Inputs (architectural provenance):
        Constructed from raw notebook JSON cell objects.

    Outputs (downstream usage):
        Consumed by `NotebookPayload` and ruleset-extraction functions.

    Invariants/constraints:
        `cell_type` and `source` must correspond to the same original notebook
        cell.

    """

    cell_type: str
    source: str

    def __init__(self, value: JsonValue) -> None:
        """Parse a notebook cell payload from JSON.

        Purpose:
            Validate the basic shape of one cell object and extract the stable
            cell-type and source-text fields needed downstream.

        Architectural role:
            JSON parsing constructor for `NotebookCellPayload`.

        Inputs (architectural provenance):
            Consumes one raw JSON value from the notebook's `cells` array.

        Outputs (downstream usage):
            Populates this cell payload from one raw notebook JSON value.

        Invariants/constraints:
            A returned payload must always have a string `cell_type`.

        """
        cell = _as_json_object(value)
        if cell is None:
            raise ValueError("Notebook cell must be a JSON object.")
        cell_type = cell.get("cell_type")
        if not isinstance(cell_type, str):
            raise ValueError("Notebook cell must include string cell_type.")
        object.__setattr__(self, "cell_type", cell_type)
        object.__setattr__(self, "source", _cell_source_text(cell))


@dataclass(frozen=True, slots=True, init=False)
class NotebookPayload:
    """Parsed notebook payload containing only the cells relevant to extraction.

    Purpose:
        Hold the ordered cell sequence from a notebook in a normalized
        representation that later ruleset-parsing code can traverse.

    Architectural role:
        Notebook-level representation object inside the extraction boundary.

    Inputs (architectural provenance):
        Constructed from notebook JSON loaded from disk or returned by Colab
        runtime APIs.

    Outputs (downstream usage):
        Consumed by ruleset and subrun extraction functions.

    Invariants/constraints:
        Cell order must match the original notebook order.

    """

    cells: tuple[NotebookCellPayload, ...]

    def __init__(self, value: JsonValue) -> None:
        """Parse the notebook payload from notebook JSON.

        Purpose:
            Validate the notebook root shape, parse each cell through
            `NotebookCellPayload(...)`, and preserve only successfully parsed
            cells.

        Architectural role:
            JSON parsing constructor for the notebook-level payload
            representation.

        Inputs (architectural provenance):
            Consumes notebook JSON loaded from disk or returned by Colab runtime
            APIs.

        Outputs (downstream usage):
            Returns a `NotebookPayload` consumed by later ruleset extraction.

        Invariants/constraints:
            Parsed cell order must match the original `cells` array order.

        """
        notebook = _as_json_object(value)
        if notebook is None:
            raise ValueError("Notebook root must be a JSON object.")
        cells_value = notebook.get("cells")
        if not isinstance(cells_value, list):
            object.__setattr__(self, "cells", ())
            return
        parsed_cells: list[NotebookCellPayload] = []
        for raw_cell in cells_value:
            try:
                parsed_cells.append(NotebookCellPayload(raw_cell))
            except ValueError:
                continue
        object.__setattr__(self, "cells", tuple(parsed_cells))


@dataclass(frozen=True, slots=True, init=False)
class ColabIpynbResponse:
    """Parsed Colab runtime response carrying one notebook payload.

    Purpose:
        Wrap the `ipynb` response shape returned by Colab runtime messaging and
        expose the normalized notebook payload.

    Architectural role:
        Colab-response representation object inside notebook extraction.

    Inputs (architectural provenance):
        Constructed from the object returned by Colab's `get_ipynb` request.

    Outputs (downstream usage):
        Consumed by runtime notebook-payload extraction.

    Invariants/constraints:
        A valid response must actually contain a parseable notebook payload.

    """

    notebook: NotebookPayload

    def __init__(self, value: JsonValue) -> None:
        """Parse a Colab runtime response into `ColabIpynbResponse`.

        Purpose:
            Validate that the response contains an `ipynb` field and adapt that
            field into a normalized `NotebookPayload`.

        Architectural role:
            JSON parsing constructor for Colab runtime notebook responses.

        Inputs (architectural provenance):
            Consumes the raw value returned by Colab's `get_ipynb` message
            request.

        Outputs (downstream usage):
            Populates this parsed Colab notebook response for runtime payload
            extraction.

        Invariants/constraints:
            A successful parse must produce a non-`None` notebook payload.

        """
        response = _as_json_object(value)
        if response is None:
            raise ValueError("Colab response must be a JSON object.")
        raw_ipynb = response.get("ipynb")
        if raw_ipynb is None:
            raise ValueError("Colab response must include ipynb.")
        object.__setattr__(self, "notebook", NotebookPayload(raw_ipynb))


def _extract_notebook_payload_from_colab_runtime() -> NotebookPayload | None:
    """Try to read the current notebook payload from a running Colab session.

    Purpose:
        Use Colab's blocking message API to request the live notebook contents
        and adapt the response into the normalized notebook payload
        representation.

    Architectural role:
        Environment-specific notebook-loading helper inside the extraction
        boundary.

    Inputs (architectural provenance):
        Consumes the active Colab runtime modules when notebook extraction runs
        inside Google Colab.

    Outputs (downstream usage):
        Returns a parsed notebook payload for later ruleset extraction, or
        `None` when runtime extraction is unavailable.

    Invariants/constraints:
        Failures in Colab runtime access must be contained and reported as
        `None` rather than crashing local-file fallback.

    """
    colab_module = sys.modules.get("google.colab")
    colab_message_module = getattr(colab_module, "_message", None)
    resolved_colab_message = cast(
        object | None,
        colab_message_module
        if colab_message_module is not None
        else _colab_message_module,
    )
    if resolved_colab_message is None:
        return None
    message_module = cast(ColabMessageModule, resolved_colab_message)

    try:
        response_raw = message_module.blocking_request(
            "get_ipynb", request="", timeout_sec=5
        )
    except (AttributeError, RuntimeError, TimeoutError, TypeError, ValueError):
        return None

    if (
        not isinstance(response_raw, (dict, list, str, int, float, bool))
        and response_raw is not None
    ):
        return None
    try:
        parsed_response = ColabIpynbResponse(cast(JsonValue, response_raw))
    except ValueError:
        return None
    return parsed_response.notebook


def _load_notebook_json(ipynb_path: str | Path) -> tuple[NotebookPayload, str]:
    """Load the notebook payload from Colab runtime or a local `.ipynb` file.

    Purpose:
        Prefer live Colab notebook extraction when available, otherwise read the
        notebook JSON from disk and return the normalized payload plus a source
        hint.

    Architectural role:
        Notebook-loading boundary for ruleset extraction.

    Inputs (architectural provenance):
        Consumes a notebook path provided by reproduction or notebook tooling.

    Outputs (downstream usage):
        Returns the parsed notebook payload and a string describing the source
        used for loading.

    Invariants/constraints:
        Exactly one source path should be chosen for each successful load.

    """
    runtime_payload = _extract_notebook_payload_from_colab_runtime()
    if runtime_payload is not None:
        return runtime_payload, "<colab-runtime>"
    if sys.modules.get("google.colab") is not None:
        msg = (
            "Colab runtime detected but notebook payload "
            "could not be read from runtime."
        )
        raise RuntimeError(msg)

    path = Path(ipynb_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    try:
        notebook = NotebookPayload(loaded)
    except ValueError:
        msg = f"Notebook root must be a JSON object: {path}"
        raise ValueError(msg) from None
    return notebook, str(path)


def _strip_visual_separators(lines: list[str]) -> list[str]:
    """Remove visual separators and collapse repeated blank lines.

    Purpose:
        Normalize markdown cell content before converting notebook-authored rule
        blocks into ruleset text.

    Architectural role:
        Notebook-extraction cleanup helper. It keeps presentation-only
        separators from leaking into the parsed rule domain-specific language.

    Inputs (architectural provenance):
        Receives raw or preselected notebook cell lines from a marked ruleset
        block.

    Outputs (downstream usage):
        Returns cleaned lines consumed by rules markdown rendering and
        system-prompt extraction.

    Invariants/constraints:
        Only standalone `---` separators and repeated blank lines are removed.
        Meaningful authored text and relative line order are preserved.

    """
    stripped_lines = [line for line in lines if line.strip() != "---"]
    collapsed_lines: list[str] = []
    previous_was_blank = False
    for line in stripped_lines:
        is_blank = not line.strip()
        if is_blank and previous_was_blank:
            continue
        collapsed_lines.append(line)
        previous_was_blank = is_blank

    while collapsed_lines and not collapsed_lines[0].strip():
        collapsed_lines.pop(0)
    while collapsed_lines and not collapsed_lines[-1].strip():
        collapsed_lines.pop()
    return collapsed_lines


def _render_rules_markdown(lines: list[str]) -> str:
    """Render cleaned notebook lines into normalized rules markdown.

    Purpose:
        Convert extracted notebook cell lines into the exact text form passed to
        the rules parser or stored in extracted ruleset specs.

    Architectural role:
        Formatting boundary between notebook JSON structure and markdown ruleset
        parsing.

    Inputs (architectural provenance):
        Receives lines collected from a marked notebook cell section.

    Outputs (downstream usage):
        Returns normalized markdown text with one trailing newline when content
        is present, or an empty string for empty sections.

    Invariants/constraints:
        The helper removes presentation separators through
        `_strip_visual_separators` but does not otherwise rewrite authored
        domain-specific language content.

    """
    stripped_lines = _strip_visual_separators(lines)
    text = "\n".join(stripped_lines).rstrip("\n")
    return f"{text}\n" if text else ""


def _extract_system_prompt(
    lines: list[str], body_start: int
) -> tuple[str | None, int]:
    """Extract an optional `## System Prompt` block from notebook lines.

    Purpose:
        Separate notebook-authored system prompt content from the remaining
        ruleset body before ruleset specs are materialized.

    Architectural role:
        Notebook-planning parser helper. It recognizes a small markdown
        convention used by reproduction notebooks without involving the rules
        domain-specific language parser.

    Inputs (architectural provenance):
        Receives notebook cell lines and the index where ruleset body scanning
        should begin.

    Outputs (downstream usage):
        Returns the extracted system prompt text, when present, and the next
        line index where body parsing should continue.

    Invariants/constraints:
        Only a top-level `## System Prompt` heading at the expected body
        position is consumed. Other headings terminate the prompt block but are
        not modified.

    """
    idx = body_start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx >= len(lines) or lines[idx].strip() != "## System Prompt":
        return None, body_start

    idx += 1
    prompt_lines: list[str] = []
    while idx < len(lines):
        line = lines[idx]
        if line.lstrip().startswith("## "):
            break
        prompt_lines.append(line)
        idx += 1

    system_prompt_text = _render_rules_markdown(prompt_lines)
    return system_prompt_text, idx


def _extract_mode(
    lines: list[str], body_start: int
) -> tuple[GenerationMode | None, int]:
    """Extract an optional ``## Mode: ...`` heading from notebook lines."""
    idx = body_start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return None, body_start

    mode_match = _MODE_HEADING_RE.match(lines[idx].strip())
    if mode_match is None:
        return None, body_start
    parsed_mode = (mode_match.group(1) or "").strip().lower()
    if parsed_mode not in {"baseline", "reasoning", "trajectory"}:
        msg = (
            "Mode heading must be one of baseline/reasoning/trajectory; "
            f"got {parsed_mode!r}."
        )
        raise ValueError(msg)
    return cast(GenerationMode, parsed_mode), idx + 1


def _extract_paper_role(
    lines: list[str], body_start: int
) -> tuple[PaperRole | None, int]:
    """Extract an optional ``## Paper Role: ...`` heading."""
    idx = body_start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return None, body_start
    role_match = _PAPER_ROLE_HEADING_RE.match(lines[idx].strip())
    if role_match is None:
        return None, body_start
    parsed_role = (role_match.group(1) or "").strip().lower()
    if parsed_role not in {"primary", "ablation", "appendix", "exploratory"}:
        msg = (
            "Paper Role heading must be one of "
            "primary/ablation/appendix/exploratory; "
            f"got {parsed_role!r}."
        )
        raise ValueError(msg)
    return cast(PaperRole, parsed_role), idx + 1


def _extract_paper_variant(
    lines: list[str], body_start: int
) -> tuple[str | None, int]:
    """Extract an optional ``## Variant: ...`` heading."""
    idx = body_start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return None, body_start
    variant_match = _VARIANT_HEADING_RE.match(lines[idx].strip())
    if variant_match is None:
        return None, body_start
    variant = (variant_match.group(1) or "").strip().lower()
    if not variant:
        raise ValueError("Variant heading must be non-empty.")
    return variant, idx + 1


def extract_answer_engineering_rulesets_from_ipynb(
    ipynb_path: str | Path,
    *,
    marker: str = "# Answer Engineering Rules",
    default_name_template: str = "notebook-cell-{cell_index}",
) -> list[NotebookRulesetSpec]:
    """Extract all marked Answer Engineering rulesets from a notebook.

    Purpose:
        Load a notebook, scan markdown and raw cells for the configured marker,
        and parse each matching cell into a `NotebookRulesetSpec`.

    Architectural role:
        Public notebook-to-ruleset extraction API for reproduction tooling.

    Inputs (architectural provenance):
        Consumes a notebook path plus optional marker and default naming
        template from callers preparing notebook-authored experiments.

    Outputs (downstream usage):
        Returns the ordered list of extracted notebook rulesets consumed by
        subrun planning or direct ruleset loading.

    Invariants/constraints:
        Matching cell order must follow notebook cell order.

    """
    notebook, source_hint = _load_notebook_json(ipynb_path)
    rulesets: list[NotebookRulesetSpec] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type not in {"markdown", "raw"}:
            continue

        source = cell.source

        if not source.lstrip().startswith(marker):
            continue

        rulesets.append(
            NotebookRulesetSpec(
                source,
                cell_index=cell_index,
                source_hint=source_hint,
                marker=marker,
                default_name_template=default_name_template,
            )
        )

    if rulesets:
        return rulesets

    msg = (
        f"Could not find cell starting with marker {marker!r} "
        f"in notebook: {source_hint}"
    )
    raise ValueError(msg)


def extract_answer_engineering_subruns_from_ipynb(
    ipynb_path: str | Path,
    *,
    marker: str = "# Answer Engineering Rules",
    default_name_template: str = "notebook-cell-{cell_index}",
) -> list[NotebookSubrun]:
    """Expand extracted notebook rulesets into notebook subrun pairs.

    Purpose:
        Convert each parsed ruleset into one or more `(ruleset, case_type)`
        subruns, emitting one item per declared case type or a single `None`
        case type when none are listed.

    Architectural role:
        Public notebook-to-subrun extraction API for reproduction planning.

    Inputs (architectural provenance):
        Consumes the same notebook path and marker settings used for ruleset
        extraction.

    Outputs (downstream usage):
        Returns notebook subruns consumed by reproduction planning and execution
        assembly.

    Invariants/constraints:
        Every emitted subrun must refer back to one extracted ruleset from the
        same notebook source.

    """
    rulesets = extract_answer_engineering_rulesets_from_ipynb(
        ipynb_path,
        marker=marker,
        default_name_template=default_name_template,
    )
    subruns: list[NotebookSubrun] = []
    for ruleset in rulesets:
        for case_type in ruleset.case_types or (None,):
            subruns.append((ruleset, case_type))
    return subruns
