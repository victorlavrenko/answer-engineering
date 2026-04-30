"""Build correctness summaries for evaluated answers.

Purpose:
    Turn per-case answer judgments into overall accuracy reports, per-case-type
    breakdowns, and pairwise transition summaries between anchor and candidate
    runs.

Architectural role:
    Primary report-building module in the evaluation boundary.

Inputs (architectural provenance):
    Consumes dataset-backed evaluation results and their correctness flags.

Outputs (downstream usage):
    Accuracy reports and transition summaries used by comparison building,
    telemetry aggregation, and session summaries.

Invariants/constraints:
    Report calculations should operate on already-evaluated answers and should
    not invoke model execution or dataset loading.

"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ae_paper_reproduction.core.evaluation import gold_checks
from ae_paper_reproduction.core.evaluation.result_types import DatasetRow
from answer_engineering import GenerationResult
from answer_engineering.telemetry import RuntimeTelemetrySnapshot


@dataclass(frozen=True, slots=True)
class CaseCount:
    """Store correctness counts for one set of evaluated cases.

    Purpose:
        Represent the number of correct answers and total evaluated answers for
        one slice of results, such as overall accuracy or one case type.

    Architectural role:
        Small counting value object inside the evaluation boundary.

    Inputs (architectural provenance):
        Constructed while summarizing evaluated results.

    Outputs (downstream usage):
        Case counts consumed by accuracy reports and comparison logic.

    Invariants/constraints:
        Correct counts must not exceed total counts.

    """

    correct: int
    total: int


type AccuracyByCase = dict[str, CaseCount]
type AccuracyByCaseInput = dict[str, CaseCount | tuple[int, int]]


@dataclass(frozen=True, slots=True, init=False)
class AccuracyReport:
    """Summarize correctness across a set of evaluated answers.

    Purpose:
        Compute overall and per-case-type correctness counts from evaluated
        results so later comparison and summary code can reuse the same report
        object.

    Architectural role:
        Primary accuracy summary object in the evaluation boundary.

    Inputs (architectural provenance):
        Constructed from a sequence of already-evaluated result rows.

    Outputs (downstream usage):
        Overall and per-case-type correctness metrics consumed by comparison and
        summary builders.

    Invariants/constraints:
        All metrics on the report must be derived from the same evaluated result
        set.

    """

    total: int
    correct: int
    by_case: AccuracyByCase

    @property
    def accuracy(self) -> float:
        """Return the overall accuracy for the evaluated result set.

        Purpose:
            Compute the overall accuracy for the evaluated result set.

        Architectural role:
            Derived metric accessor on the accuracy report.

        Inputs (architectural provenance):
            Reads the stored correctness counts on this report.

        Outputs (downstream usage):
            A float or mapping used by comparison and summary code.

        Invariants/constraints:
            Derived values should stay consistent with the counts stored on the
            report.

        """
        return self.correct / self.total if self.total else 0.0

    def accuracy_by_case(self) -> dict[str, float]:
        """Return per-case-type accuracy values derived from the stored counts.

        Purpose:
            Compute per-case-type accuracy values derived from the stored
            counts.

        Architectural role:
            Derived metric accessor on the accuracy report.

        Inputs (architectural provenance):
            Reads the stored correctness counts on this report.

        Outputs (downstream usage):
            A float or mapping used by comparison and summary code.

        Invariants/constraints:
            Derived values should stay consistent with the counts stored on the
            report.

        """
        return {
            case_type: (counts.correct / counts.total if counts.total else 0.0)
            for case_type, counts in self.by_case.items()
        }

    def __init__(
        self,
        results: Iterable[RulesetEvaluationResult] | None = None,
        *,
        total: int | None = None,
        correct: int | None = None,
        by_case: AccuracyByCaseInput | None = None,
    ) -> None:
        """Build an accuracy report from results or explicit counts.

        Purpose:
            Support the two construction modes used by notebooks and reporting
            code: derive counts from evaluated `RulesetEvaluationResult` rows,
            or recreate a report from already-computed totals and per-case
            buckets.

        Architectural role:
            Rich constructor for the immutable accuracy-report value object. It
            is the normalization point that turns result rows, tuple counts, and
            `CaseCount` values into one canonical `AccuracyReport` shape.

        Inputs (architectural provenance):
            `results` comes from completed subrun evaluation. Explicit `total`,
            `correct`, and `by_case` values come from comparison/reporting code
            that already materialized the counts and needs to preserve them.

        Outputs (downstream usage):
            Stores immutable total, correct, and per-case count fields consumed
            by pairwise comparison, session summaries, telemetry rows, and
            notebooks.

        Invariants/constraints:
            Callers must choose exactly one construction mode. Derived reports
            count every materialized result once, grouped by `case_type`.
            Explicit reports are normalized so downstream code always sees
            `CaseCount` values.

        """
        if results is not None:
            materialized = list(results)
            computed_by_case: dict[str, list[int]] = {}
            computed_correct = 0
            for result in materialized:
                case_counts = computed_by_case.setdefault(
                    result.case_type, [0, 0]
                )
                case_counts[1] += 1
                if result.ok:
                    computed_correct += 1
                    case_counts[0] += 1
            object.__setattr__(self, "total", len(materialized))
            object.__setattr__(self, "correct", computed_correct)
            object.__setattr__(
                self,
                "by_case",
                {
                    case_type: CaseCount(correct=counts[0], total=counts[1])
                    for case_type, counts in computed_by_case.items()
                },
            )
            return

        if total is None or correct is None or by_case is None:
            msg = "Provide either results or total/correct/by_case."
            raise TypeError(msg)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "correct", correct)
        normalized_by_case: AccuracyByCase = {}
        for case_type, counts in by_case.items():
            if isinstance(counts, CaseCount):
                normalized_by_case[case_type] = counts
            else:
                normalized_by_case[case_type] = CaseCount(
                    correct=counts[0], total=counts[1]
                )
        object.__setattr__(self, "by_case", normalized_by_case)


@dataclass(frozen=True, slots=True, init=False)
class RulesetEvaluationResult:
    """Evaluation row for one generated answer.

    Combine a benchmark row with generated answer text, correctness metadata,
    optional Answer Engineering telemetry, and runtime timing. This object is
    the bridge between runtime output and reproduction analysis.

    .. note::
        Pass the full :class:`~answer_engineering.GenerationResult` when
        available, not only answer text, so telemetry and runtime timing remain
        aligned with the judged answer.

    Examples:
        ```python
        result = RulesetEvaluationResult(
            task.row,
            answer="The answer is sudden sensorineural hearing loss.",
            ok=True,
            runtime_sec=0.42,
        )
        ```

        ```python
        answer = runtime.generate(
            GenerationRequest(question=task.question),
            policy=policy,
            rules=task.compiled_rules,
        )
        result = RulesetEvaluationResult(task.row, answer=answer)
        ```

    Attributes:
        row: Source dataset row passed to the constructor.
        answer: Normalized generated answer text.
        id: Dataset question id.
        case_type: Dataset case type used for grouping.
        question: User-facing question text.
        gold: Gold/reference answer.
        ok: Whether the generated answer is judged correct for this row.
        ae_telemetry: Runtime telemetry copied from the generation result when
            available.
        runtime_sec: Runtime duration copied from the generation result when
            available.

    Runtime behavior:
        Construction copies stable row metadata, normalizes either a raw answer
        string or a full ``GenerationResult``, computes correctness when ``ok``
        is omitted, and preserves telemetry/timing when present.

    Architectural role:
        Evaluation boundary between model generation and reporting.

    Consumes:
        Dataset row plus either answer text or
        :class:`~answer_engineering.GenerationResult`.

    Produces:
        A typed evaluation record consumed by
        :class:`~ae_paper_reproduction.SubrunResult`, summaries, pairwise
        reports, telemetry aggregation, and paper tables.

    Invariants:
        Judgement metadata must stay aligned with the exact answer text and
        telemetry produced for the row.

    Developer Notes:
        Keep dataset provenance, answer normalization, correctness derivation,
        telemetry attachment, and runtime timing aligned here. Reports should
        consume already evaluated rows rather than reaching back into runtime
        sessions.

    Todo:
        Make judgement policy and telemetry typing more explicit as evaluation
        moves from notebook reproduction toward a stable public API.

    See Also:
        :class:`~answer_engineering.GenerationResult`
        :class:`~ae_paper_reproduction.SubrunResult`
        :class:`~ae_paper_reproduction.Summary`

    """

    id: str
    case_type: str
    gold: str
    answer: str
    ok: bool
    ae_telemetry: RuntimeTelemetrySnapshot | None = None
    question: str | None = None
    runtime_sec: float | None = None

    def __init__(
        self,
        row: DatasetRow,
        *,
        answer: str | GenerationResult,
        ok: bool | None = None,
        ae_telemetry: RuntimeTelemetrySnapshot | None = None,
        runtime_sec: float | None = None,
    ) -> None:
        """Populate one immutable evaluation row from a dataset row and answer.

        Examples:
            ```python
            evaluated = RulesetEvaluationResult(
                task.row,
                answer="The correct option is SSNHL.",
                ok=True,
                runtime_sec=0.42,
            )
            ```

            ```python
            result = runtime.generate(
                request,
                policy=policy,
                rules=task.compiled_rules,
            )
            evaluated = RulesetEvaluationResult(task.row, answer=result)
            ```

        Use this constructor in reproduction notebooks after one question has
        been answered. It copies the stable case metadata from the dataset row,
        normalizes raw answer text or a full ``GenerationResult`` into the same
        stored shape, and attaches correctness, telemetry, and timing fields
        used by later reports.

        The constructor is intentionally convenient for notebook loops: callers
        can pass the full runtime result directly and let this object extract
        the answer text, Answer Engineering telemetry snapshot, and runtime
        seconds. When callers are replaying saved answers or building synthetic
        comparisons, they can pass plain text and provide explicit metadata
        instead.

        Args:
            row: Source dataset row for the evaluated case. The constructor
                reads the row's identifier, case type, question text, and gold
                answer from this object so reporting remains tied to the
                original dataset provenance.
            answer: Generated answer for the case. Pass a ``GenerationResult``
                when the answer came directly from
                ``GenerationRuntime.generate`` and telemetry should be preserved
                automatically; pass a string when evaluating saved, edited, or
                externally produced text.
            ok: Optional correctness override. When omitted, the repository
                gold-check logic compares the normalized answer text with the
                dataset gold answer. Provide this when a notebook deliberately
                applies custom adjudication or imports externally scored rows.
            ae_telemetry: Optional Answer Engineering runtime telemetry snapshot
                to attach when ``answer`` is plain text. When ``answer`` is a
                ``GenerationResult``, the telemetry snapshot is taken from that
                result and this argument is only relevant to custom construction
                paths.
            runtime_sec: Optional elapsed runtime in seconds to attach when
                ``answer`` is plain text. When ``answer`` is a
                ``GenerationResult``, the runtime is taken from that result so
                notebook summaries can report timing without separate
                bookkeeping.

        Notes:
            This object is a reporting row, not a runtime controller. It should
            describe what happened for one evaluated case after generation is
            already complete. For custom reproduction experiments, prefer
            creating additional ``RulesetEvaluationResult`` rows over mutating
            existing rows; immutable rows make comparisons and paper-metric
            generation easier to audit.

        Developer notes:
            This constructor is the evaluation boundary between dataset
            provenance, runtime output, and reporting. Keep derivation rules
            local and explicit so notebook code does not need to know how
            ``GenerationResult`` stores text, telemetry, or timing. Keep the
            convenient construction modes explicit because notebooks may pass
            either a full generation result or imported answer text. Do not move
            this normalization into table renderers; reports should receive
            already evaluated rows with stable provenance and optional telemetry
            attached.

        Todo:
            If future datasets use richer answer schemas, keep this constructor
            as the normalization point and document the scoring policy here
            before exposing the new fields to notebook users.

        """
        object.__setattr__(self, "id", row.id)
        object.__setattr__(self, "case_type", row.case_type)
        object.__setattr__(self, "gold", row.gold)
        object.__setattr__(self, "question", row.question)
        answer_text = (
            answer.text if isinstance(answer, GenerationResult) else answer
        )
        derived_ok = (
            gold_checks.check_gold(answer_text, self.gold)[0]
            if ok is None
            else ok
        )
        derived_telemetry = (
            answer.ae_telemetry
            if isinstance(answer, GenerationResult)
            else ae_telemetry
        )
        derived_runtime = (
            answer.runtime_sec
            if isinstance(answer, GenerationResult)
            else runtime_sec
        )
        object.__setattr__(self, "answer", answer_text)
        object.__setattr__(self, "ok", derived_ok)
        object.__setattr__(self, "ae_telemetry", derived_telemetry)
        object.__setattr__(self, "runtime_sec", derived_runtime)


@dataclass(frozen=True, slots=True, init=False)
class PairwiseComparisonReport:
    """Summarize how a candidate run differs from an anchor run.

    Purpose:
        Compute overall accuracy delta, per-case-type deltas, and transition
        counts showing where the candidate improved, degraded, or matched the
        anchor.

    Architectural role:
        Pairwise comparison object in the evaluation boundary.

    Inputs (architectural provenance):
        Constructed from two evaluated result sets and their accuracy reports.

    Outputs (downstream usage):
        Comparison metrics consumed by aggregation and summary code.

    Invariants/constraints:
        The anchor and candidate results must be aligned by case id before the
        comparison is meaningful.

    """

    anchor: AccuracyReport
    candidate: AccuracyReport
    delta_overall: float
    delta_by_case: dict[str, float]
    outcome_transitions: PairwiseOutcomeTransitions

    def __init__(
        self,
        *,
        anchor: AccuracyReport | None = None,
        candidate: AccuracyReport | None = None,
        delta_overall: float | None = None,
        delta_by_case: dict[str, float] | None = None,
        outcome_transitions: PairwiseOutcomeTransitions | None = None,
        anchor_results: Iterable[RulesetEvaluationResult] | None = None,
        anchor_report: AccuracyReport | None = None,
        candidate_results: Iterable[RulesetEvaluationResult] | None = None,
        candidate_report: AccuracyReport | None = None,
    ) -> None:
        """Compute comparison deltas or accept a precomputed comparison.

        Purpose:
            Support both live comparison construction from evaluated
            anchor/candidate runs and explicit reconstruction from
            already-computed report fields.

        Architectural role:
            Rich constructor for the pairwise comparison value object. It is the
            single place that aligns transition counts with overall and per-case
            accuracy deltas.

        Inputs (architectural provenance):
            Result/report mode receives evaluated rows and accuracy reports for
            the anchor and candidate subruns. Explicit mode receives the
            already-normalized reports, delta fields, and transition counts from
            upstream summary code.

        Outputs (downstream usage):
            Stores anchor and candidate accuracy reports, overall delta,
            per-case delta mapping, and outcome transitions consumed by
            aggregation and renderers.

        Invariants/constraints:
            Callers must provide one complete construction mode. In
            result/report mode, anchor and candidate rows must describe the same
            case-id set so transition counts and accuracy deltas refer to the
            same comparison population.

        """
        if (
            anchor_results is not None
            and anchor_report is not None
            and candidate_results is not None
            and candidate_report is not None
        ):
            transitions = PairwiseOutcomeTransitions(
                anchor_results, candidate_results
            )
            computed_delta_by_case = {
                case_type: candidate_report.accuracy_by_case().get(
                    case_type, 0.0
                )
                - anchor_report.accuracy_by_case().get(case_type, 0.0)
                for case_type in set(anchor_report.by_case)
                | set(candidate_report.by_case)
            }
            object.__setattr__(self, "anchor", anchor_report)
            object.__setattr__(self, "candidate", candidate_report)
            object.__setattr__(
                self,
                "delta_overall",
                candidate_report.accuracy - anchor_report.accuracy,
            )
            object.__setattr__(self, "delta_by_case", computed_delta_by_case)
            object.__setattr__(self, "outcome_transitions", transitions)
            return

        if (
            anchor is None
            or candidate is None
            or delta_overall is None
            or delta_by_case is None
            or outcome_transitions is None
        ):
            msg = (
                "Provide either anchor_results/anchor_report/"
                "candidate_results/candidate_report or explicit report fields."
            )
            raise TypeError(msg)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "delta_overall", delta_overall)
        object.__setattr__(self, "delta_by_case", delta_by_case)
        object.__setattr__(self, "outcome_transitions", outcome_transitions)


def pair_results_by_case_id(
    anchor_results: Iterable[RulesetEvaluationResult],
    candidate_results: Iterable[RulesetEvaluationResult],
) -> list[tuple[str, RulesetEvaluationResult, RulesetEvaluationResult]]:
    """Pair anchor and candidate results by dataset case id.

    Purpose:
        Align anchor and candidate results by dataset case id.

    Architectural role:
        Pure comparison helper inside the evaluation boundary.

    Inputs (architectural provenance):
        Consumes two evaluated result sequences drawn from anchor and candidate
        runs.

    Outputs (downstream usage):
        A paired mapping or transition counts consumed by pairwise reports.

    Invariants/constraints:
        Results are only meaningful when both sides refer to the same case id
        space.

    """
    anchor_by_id = {result.id: result for result in anchor_results}
    candidate_by_id = {result.id: result for result in candidate_results}
    if set(anchor_by_id) != set(candidate_by_id):
        msg = (
            "anchor_results and candidate_results must contain "
            "the same case ids"
        )
        raise ValueError(msg)
    return [
        (case_id, anchor_by_id[case_id], candidate_by_id[case_id])
        for case_id in sorted(anchor_by_id)
    ]


@dataclass(frozen=True, slots=True, init=False)
class PairwiseOutcomeTransitions:
    """Store outcome transition counts for one anchor/candidate comparison.

    Purpose:
        Record how many cases stayed correct, stayed incorrect, improved, or
        degraded when moving from the anchor run to the candidate run.

    Architectural role:
        Transition-summary value object used by pairwise comparison reports.

    Inputs (architectural provenance):
        Constructed from aligned anchor and candidate evaluation results.

    Outputs (downstream usage):
        Transition counters consumed by group comparison rows and report
        renderers.

    Invariants/constraints:
        All counts should refer to the same aligned comparison set.

    """

    anchor_correct_to_candidate_correct: int
    anchor_correct_to_candidate_incorrect: int
    anchor_incorrect_to_candidate_correct: int
    anchor_incorrect_to_candidate_incorrect: int

    def __init__(
        self,
        anchor_results: Iterable[RulesetEvaluationResult],
        candidate_results: Iterable[RulesetEvaluationResult],
    ) -> None:
        """Count correctness transitions between aligned runs.

        Purpose:
            Derive the four transition buckets that show which cases stayed
            correct, degraded, improved, or stayed incorrect when moving from
            anchor to candidate output.

        Architectural role:
            Constructor-level aggregation boundary for pairwise outcome
            movement. It delegates case-id alignment to
            `pair_results_by_case_id` and stores only the normalized counts.

        Inputs (architectural provenance):
            `anchor_results` and `candidate_results` come from completed subrun
            evaluations that should cover the same dataset cases.

        Outputs (downstream usage):
            Stores transition counters consumed by pairwise reports, group
            comparison rows, and summary renderers.

        Invariants/constraints:
            Both result collections must contain the same case ids. Each aligned
            pair contributes to exactly one transition bucket.

        """
        same_correct = 0
        degraded = 0
        improved = 0
        same_incorrect = 0
        for _, anchor_result, candidate_result in pair_results_by_case_id(
            anchor_results, candidate_results
        ):
            anchor_ok = bool(anchor_result.ok)
            candidate_ok = bool(candidate_result.ok)
            if anchor_ok and candidate_ok:
                same_correct += 1
            elif anchor_ok and not candidate_ok:
                degraded += 1
            elif not anchor_ok and candidate_ok:
                improved += 1
            else:
                same_incorrect += 1
        object.__setattr__(
            self, "anchor_correct_to_candidate_correct", same_correct
        )
        object.__setattr__(
            self, "anchor_correct_to_candidate_incorrect", degraded
        )
        object.__setattr__(
            self, "anchor_incorrect_to_candidate_correct", improved
        )
        object.__setattr__(
            self, "anchor_incorrect_to_candidate_incorrect", same_incorrect
        )


@dataclass(frozen=True, slots=True)
class RunOutcomeTransitions:
    """Store baseline-versus-edited outcome transitions for one run comparison.

    Purpose:
        Represent the same improvement/degradation categories as pairwise
        transitions using baseline/edited naming for reporting contexts that
        prefer those terms.

    Architectural role:
        Alternative naming view over pairwise transition counts.

    Inputs (architectural provenance):
        Constructed from aligned result comparisons in reporting code.

    Outputs (downstream usage):
        Transition counters consumed by presentation layers.

    Invariants/constraints:
        This type should remain a passive reporting record.

    """

    baseline_correct_to_edited_correct: int
    baseline_correct_to_edited_incorrect: int
    baseline_incorrect_to_edited_correct: int
    baseline_incorrect_to_edited_incorrect: int
