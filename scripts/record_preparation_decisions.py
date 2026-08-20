"""Reusable, non-mutating registry of preliminary data-preparation decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_DECISION_COLUMNS: Final[list[str]] = [
    "Decision ID",
    "Domain",
    "Title",
    "Affected fields",
    "Affected field count",
    "Status",
    "Phase",
    "Fit scope",
    "Operation",
    "Rationale",
    "Prerequisites",
    "Prerequisite count",
    "Acceptance criteria",
    "Source stages",
    "Evidence count",
    "Execution step count",
]

_EVIDENCE_COLUMNS: Final[list[str]] = [
    "Evidence ID",
    "Decision ID",
    "Source report",
    "Source item",
    "Observed value",
    "Expected or reference",
    "Interpretation",
]

_EXECUTION_COLUMNS: Final[list[str]] = [
    "Step ID",
    "Sequence",
    "Decision IDs",
    "Decision count",
    "Phase",
    "Action",
    "Blocking",
    "Status",
    "Temporal dependency",
    "Acceptance criteria",
]

_GUARDRAIL_COLUMNS: Final[list[str]] = [
    "Guardrail ID",
    "Domain",
    "Title",
    "Affected fields",
    "Affected field count",
    "Severity",
    "Status",
    "Prohibited operation",
    "Rationale",
    "Verification",
]

_SPLIT_POLICY_COLUMNS: Final[list[str]] = [
    "Policy item",
    "Value",
    "Status",
    "Interpretation",
]

_BLOCKER_COLUMNS: Final[list[str]] = [
    "Decision ID",
    "Title",
    "Domain",
    "Phase",
    "Status",
    "Fit scope",
    "Prerequisites",
    "Operation",
]

_READINESS_COLUMNS: Final[list[str]] = [
    "Readiness check",
    "Ready",
    "Interpretation",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Item",
    "Issue",
    "Details",
    "Potential impact",
]

_ALLOWED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Approved",
        "Conditional",
        "Deferred",
        "Prohibited",
        "Blocked",
    }
)

_ALLOWED_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "Cleaning",
        "Feature role",
        "Target governance",
        "Transformation",
        "Feature engineering",
        "Dataset splitting",
        "Leakage governance",
    }
)

_ALLOWED_PHASES: Final[frozenset[str]] = frozenset(
    {
        "Before split",
        "Split",
        "Train-only transformation",
        "Model selection",
        "External contract",
    }
)

_ALLOWED_FIT_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "Deterministic",
        "Train only",
        "Evaluation only",
        "None",
        "External",
    }
)

_ALLOWED_STEP_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Planned",
        "Ready",
        "Blocked",
        "Complete",
        "Deferred",
    }
)

_ALLOWED_GUARDRAIL_SEVERITIES: Final[frozenset[str]] = frozenset(
    {
        "Critical",
        "High",
        "Medium",
        "Low",
    }
)

_ALLOWED_GUARDRAIL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Active",
        "Planned",
        "Controlled",
    }
)

_ALLOWED_TEMPORAL_POLICY_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Resolved temporal split",
        "Resolved snapshot fallback",
        "Unresolved",
    }
)

_STATUS_ORDER: Final[dict[str, int]] = {
    "Blocked": 0,
    "Prohibited": 1,
    "Approved": 2,
    "Conditional": 3,
    "Deferred": 4,
}

_PHASE_ORDER: Final[dict[str, int]] = {
    "Before split": 0,
    "Split": 1,
    "Train-only transformation": 2,
    "Model selection": 3,
    "External contract": 4,
}

_SPLIT_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "stratify_by",
    "random_seed",
    "shuffle",
    "temporal_priority",
    "temporal_policy_status",
    "random_split_fallback",
    "test_holdout_untouched",
    "disjoint_partitions_required",
    "group_by_identifiers",
)

_SPLIT_LABELS: Final[dict[str, str]] = {
    "train_fraction": "Train fraction",
    "validation_fraction": "Validation fraction",
    "test_fraction": "Test fraction",
    "stratify_by": "Stratification field",
    "random_seed": "Random seed",
    "shuffle": "Shuffle random split",
    "temporal_priority": "Temporal split priority",
    "temporal_policy_status": "Temporal policy status",
    "random_split_fallback": "Random-split fallback",
    "test_holdout_untouched": "Final test holdout",
    "disjoint_partitions_required": "Disjoint partitions",
    "group_by_identifiers": "Identifier grouping",
}


class PreparationDecisionContractError(ValueError):
    """Raised when preparation-decision declarations or gates are invalid."""


@dataclass(frozen=True, slots=True)
class PreparationDecisionReport:
    """Record preparation decisions without applying any transformation."""

    available_fields: tuple[str, ...]
    decisions: pd.DataFrame
    evidence: pd.DataFrame
    execution_steps: pd.DataFrame
    guardrails: pd.DataFrame
    split_policy: dict[str, object]
    issues: pd.DataFrame

    @property
    def has_approved_decisions(self) -> bool:
        return bool(
            not self.decisions.empty
            and self.decisions["Status"].eq("Approved").any()
        )

    @property
    def has_conditional_decisions(self) -> bool:
        return bool(
            not self.decisions.empty
            and self.decisions["Status"].eq("Conditional").any()
        )

    @property
    def has_deferred_decisions(self) -> bool:
        return bool(
            not self.decisions.empty
            and self.decisions["Status"].eq("Deferred").any()
        )

    @property
    def has_prohibited_operations(self) -> bool:
        return bool(
            (
                not self.decisions.empty
                and self.decisions["Status"].eq("Prohibited").any()
            )
            or not self.guardrails.empty
        )

    @property
    def has_external_blockers(self) -> bool:
        if self.decisions.empty:
            return False
        return bool(
            (
                self.decisions["Status"].eq("Blocked")
                & (
                    self.decisions["Phase"].eq("External contract")
                    | self.decisions["Fit scope"].eq("External")
                )
            ).any()
        )

    @property
    def has_train_only_operations(self) -> bool:
        if self.decisions.empty:
            return False
        return bool(
            self.decisions["Fit scope"].eq("Train only").any()
        )

    @property
    def has_deterministic_cleaning_scope(self) -> bool:
        if self.decisions.empty:
            return False
        selected = self.decisions.loc[
            self.decisions["Domain"].eq("Cleaning")
            & self.decisions["Status"].eq("Approved")
            & self.decisions["Phase"].eq("Before split")
            & self.decisions["Fit scope"].eq("Deterministic")
        ]
        return not selected.empty

    @property
    def is_structurally_valid(self) -> bool:
        return self.issues.empty

    @property
    def is_ready_for_deterministic_preparation(self) -> bool:
        if not self.is_structurally_valid:
            return False
        if not self.has_deterministic_cleaning_scope:
            return False

        deterministic = self.decisions.loc[
            self.decisions["Status"].eq("Approved")
            & self.decisions["Fit scope"].eq("Deterministic")
            & self.decisions["Phase"].eq("Before split")
        ]
        if deterministic.empty:
            return False

        linked_evidence = set(self.evidence["Decision ID"].astype(str))
        return set(deterministic["Decision ID"].astype(str)).issubset(
            linked_evidence
        )

    @property
    def is_ready_for_split_execution(self) -> bool:
        if not self.is_structurally_valid:
            return False
        if not self.is_ready_for_deterministic_preparation:
            return False

        temporal_status = str(
            self.split_policy.get("temporal_policy_status", "")
        )
        if temporal_status == "Unresolved":
            return False

        blocked_split = self.decisions.loc[
            self.decisions["Status"].eq("Blocked")
            & self.decisions["Domain"].isin(
                {"Dataset splitting", "Leakage governance"}
            )
        ]
        return blocked_split.empty

    @property
    def is_ready_for_modeling(self) -> bool:
        if not self.is_ready_for_split_execution:
            return False
        if self.has_external_blockers:
            return False

        blocking_steps = self.execution_steps.loc[
            self.execution_steps["Blocking"]
        ]
        if blocking_steps.empty:
            return True
        return bool(blocking_steps["Status"].eq("Complete").all())

    def summary_frame(self) -> pd.DataFrame:
        status_counts = self.decisions["Status"].value_counts()
        rows = [
            {
                "Metric": "Declared fields",
                "Value": len(self.available_fields),
                "Interpretation": "Fields available to preparation decisions",
            },
            {
                "Metric": "Preparation decisions",
                "Value": len(self.decisions),
                "Interpretation": "Versioned cleaning, transformation, engineering, and split decisions",
            },
            {
                "Metric": "Approved decisions",
                "Value": int(status_counts.get("Approved", 0)),
                "Interpretation": "Decisions authorized within their declared scope",
            },
            {
                "Metric": "Conditional decisions",
                "Value": int(status_counts.get("Conditional", 0)),
                "Interpretation": "Decisions requiring a later model or contract condition",
            },
            {
                "Metric": "Deferred decisions",
                "Value": int(status_counts.get("Deferred", 0)),
                "Interpretation": "Candidates intentionally postponed to evaluation",
            },
            {
                "Metric": "Prohibited decisions",
                "Value": int(status_counts.get("Prohibited", 0)),
                "Interpretation": "Operations explicitly excluded from preparation",
            },
            {
                "Metric": "Blocked decisions",
                "Value": int(status_counts.get("Blocked", 0)),
                "Interpretation": "Decisions awaiting unresolved prerequisites",
            },
            {
                "Metric": "Evidence records",
                "Value": len(self.evidence),
                "Interpretation": "Traceable support for decisions",
            },
            {
                "Metric": "Execution steps",
                "Value": len(self.execution_steps),
                "Interpretation": "Ordered future implementation plan",
            },
            {
                "Metric": "Guardrails",
                "Value": len(self.guardrails),
                "Interpretation": "Explicit prohibitions protecting preparation integrity",
            },
            {
                "Metric": "Structurally valid",
                "Value": self.is_structurally_valid,
                "Interpretation": "Decisions, evidence, steps, guardrails, and split policy are coherent",
            },
            {
                "Metric": "Ready for deterministic preparation",
                "Value": self.is_ready_for_deterministic_preparation,
                "Interpretation": "Approved deterministic cleaning may be implemented",
            },
            {
                "Metric": "Ready for split execution",
                "Value": self.is_ready_for_split_execution,
                "Interpretation": "Temporal policy and split blockers are resolved",
            },
            {
                "Metric": "Ready for modeling",
                "Value": self.is_ready_for_modeling,
                "Interpretation": "All blocking execution and governance conditions are complete",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def decisions_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.decisions)

    def evidence_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.evidence)

    def execution_plan_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.execution_steps)

    def guardrails_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.guardrails)

    def split_policy_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        temporal_status = str(
            self.split_policy.get("temporal_policy_status", "")
        )
        for key in _SPLIT_REQUIRED_KEYS:
            value = deepcopy(self.split_policy.get(key))
            status = "Resolved"
            if key == "temporal_policy_status" and value == "Unresolved":
                status = "Unresolved"
            elif key in {"temporal_priority", "test_holdout_untouched", "disjoint_partitions_required"} and value is not True:
                status = "Invalid"
            elif key == "random_split_fallback" and not _text(value):
                status = "Missing"

            interpretation = _split_interpretation(key, value, temporal_status)
            rows.append(
                {
                    "Policy item": _SPLIT_LABELS[key],
                    "Value": value,
                    "Status": status,
                    "Interpretation": interpretation,
                }
            )
        return pd.DataFrame(rows, columns=_SPLIT_POLICY_COLUMNS)

    def blockers_frame(self) -> pd.DataFrame:
        if self.decisions.empty:
            return pd.DataFrame(columns=_BLOCKER_COLUMNS)
        selected = self.decisions.loc[
            self.decisions["Status"].eq("Blocked")
        ]
        rows = [
            {
                "Decision ID": row["Decision ID"],
                "Title": row["Title"],
                "Domain": row["Domain"],
                "Phase": row["Phase"],
                "Status": row["Status"],
                "Fit scope": row["Fit scope"],
                "Prerequisites": deepcopy(row["Prerequisites"]),
                "Operation": row["Operation"],
            }
            for _, row in selected.iterrows()
        ]
        return pd.DataFrame(rows, columns=_BLOCKER_COLUMNS)

    def readiness_frame(self) -> pd.DataFrame:
        rows = [
            {
                "Readiness check": "Structural contract",
                "Ready": self.is_structurally_valid,
                "Interpretation": (
                    "All declarations and references are valid"
                    if self.is_structurally_valid
                    else "Structural contract issues must be corrected"
                ),
            },
            {
                "Readiness check": "Deterministic preparation",
                "Ready": self.is_ready_for_deterministic_preparation,
                "Interpretation": (
                    "Approved deterministic cleaning is traceable and bounded"
                    if self.is_ready_for_deterministic_preparation
                    else "Deterministic preparation scope is incomplete"
                ),
            },
            {
                "Readiness check": "Split execution",
                "Ready": self.is_ready_for_split_execution,
                "Interpretation": (
                    "Split policy is operationally resolved"
                    if self.is_ready_for_split_execution
                    else "Temporal precedence or split blockers remain unresolved"
                ),
            },
            {
                "Readiness check": "Modeling clearance",
                "Ready": self.is_ready_for_modeling,
                "Interpretation": (
                    "All blocking preparation and governance steps are complete"
                    if self.is_ready_for_modeling
                    else "Blocking steps or external prerequisites remain open"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_READINESS_COLUMNS)

    def issues_frame(self) -> pd.DataFrame:
        return _defensive_frame(self.issues)

    def raise_if_invalid(
        self,
        *,
        require_unique_decision_ids: bool = True,
        require_unique_evidence_ids: bool = True,
        require_unique_step_ids: bool = True,
        require_unique_guardrail_ids: bool = True,
        require_known_fields: bool = True,
        require_evidence_for_decisions: bool = True,
        require_acceptance_criteria: bool = True,
        require_valid_statuses: bool = True,
        require_valid_domains: bool = True,
        require_valid_phases: bool = True,
        require_valid_fit_scopes: bool = True,
        require_valid_references: bool = True,
        require_complete_split_policy: bool = True,
    ) -> None:
        selected: set[str] = set()
        if require_unique_decision_ids:
            selected.add("Duplicate decision ID")
        if require_unique_evidence_ids:
            selected.add("Duplicate evidence ID")
        if require_unique_step_ids:
            selected.add("Duplicate step ID")
        if require_unique_guardrail_ids:
            selected.add("Duplicate guardrail ID")
        if require_known_fields:
            selected.add("Unknown affected field")
        if require_evidence_for_decisions:
            selected.add("Decision without evidence")
        if require_acceptance_criteria:
            selected.update(
                {
                    "Missing decision acceptance criteria",
                    "Missing step acceptance criteria",
                }
            )
        if require_valid_statuses:
            selected.update(
                {
                    "Invalid decision status",
                    "Invalid step status",
                    "Invalid guardrail status",
                }
            )
        if require_valid_domains:
            selected.update(
                {
                    "Invalid decision domain",
                    "Invalid guardrail domain",
                }
            )
        if require_valid_phases:
            selected.update(
                {
                    "Invalid decision phase",
                    "Invalid step phase",
                }
            )
        if require_valid_fit_scopes:
            selected.add("Invalid fit scope")
        if require_valid_references:
            selected.update(
                {
                    "Unknown decision reference",
                    "Unknown prerequisite reference",
                }
            )
        if require_complete_split_policy:
            selected.update(
                {
                    "Incomplete split policy",
                    "Invalid split proportion",
                    "Invalid split total",
                    "Unknown stratification field",
                    "Invalid random seed",
                    "Invalid temporal policy status",
                    "Missing random-split fallback",
                    "Invalid temporal priority",
                    "Invalid test holdout contract",
                    "Invalid disjoint partition contract",
                    "Unknown grouping identifier",
                }
            )

        failures = self.issues.loc[self.issues["Issue"].isin(selected)]
        if failures.empty:
            return

        details = "; ".join(str(value) for value in failures["Details"])
        raise PreparationDecisionContractError(
            "Invalid preparation-decision contract: " + details
        )

    def raise_if_split_not_ready(
        self,
        *,
        require_temporal_policy_resolved: bool = True,
        require_no_blocked_split_decisions: bool = True,
        require_stratification_contract: bool = True,
        require_disjoint_partition_contract: bool = True,
    ) -> None:
        reasons: list[str] = []

        if not self.is_structurally_valid:
            reasons.append("the structural preparation-decision contract is invalid")

        if (
            require_temporal_policy_resolved
            and self.split_policy.get("temporal_policy_status") == "Unresolved"
        ):
            reasons.append("the temporal split policy remains unresolved")

        if require_no_blocked_split_decisions:
            blocked = self.decisions.loc[
                self.decisions["Status"].eq("Blocked")
                & self.decisions["Domain"].isin(
                    {"Dataset splitting", "Leakage governance"}
                )
            ]
            if not blocked.empty:
                reasons.append("blocked split or leakage-governance decisions remain")

        if require_stratification_contract:
            stratify_by = _text(self.split_policy.get("stratify_by"))
            if not stratify_by or stratify_by not in self.available_fields:
                reasons.append("the stratification contract is incomplete")

        if (
            require_disjoint_partition_contract
            and self.split_policy.get("disjoint_partitions_required") is not True
        ):
            reasons.append("the disjoint-partition contract is not enforced")

        if reasons:
            raise PreparationDecisionContractError(
                "Preparation split is not ready: " + "; ".join(reasons)
            )


def record_preparation_decisions(
    *,
    available_fields: Sequence[object],
    decisions: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    execution_steps: Sequence[Mapping[str, object]],
    guardrails: Sequence[Mapping[str, object]],
    split_policy: Mapping[str, object],
) -> PreparationDecisionReport:
    """Normalize and validate preliminary preparation decisions."""
    fields = _unique_text_tuple(available_fields)
    decision_declarations = deepcopy(list(decisions))
    evidence_declarations = deepcopy(list(evidence))
    step_declarations = deepcopy(list(execution_steps))
    guardrail_declarations = deepcopy(list(guardrails))
    split_declaration = deepcopy(dict(split_policy))

    issues: list[dict[str, object]] = []

    decision_rows = _normalize_decisions(
        decision_declarations,
        available_fields=fields,
        issues=issues,
    )
    evidence_rows = _normalize_evidence(evidence_declarations, issues=issues)
    step_rows = _normalize_steps(step_declarations, issues=issues)
    guardrail_rows = _normalize_guardrails(
        guardrail_declarations,
        available_fields=fields,
        issues=issues,
    )

    decision_ids = {str(row["Decision ID"]) for row in decision_rows}
    _validate_references_and_coverage(
        decision_rows=decision_rows,
        evidence_rows=evidence_rows,
        step_rows=step_rows,
        decision_ids=decision_ids,
        issues=issues,
    )
    _validate_split_policy(
        split_declaration,
        available_fields=fields,
        issues=issues,
    )

    evidence_counts: dict[str, int] = {}
    for row in evidence_rows:
        decision_id = str(row["Decision ID"])
        evidence_counts[decision_id] = evidence_counts.get(decision_id, 0) + 1

    step_counts: dict[str, int] = {}
    for row in step_rows:
        for decision_id in row["Decision IDs"]:
            key = str(decision_id)
            step_counts[key] = step_counts.get(key, 0) + 1

    for row in decision_rows:
        decision_id = str(row["Decision ID"])
        row["Evidence count"] = evidence_counts.get(decision_id, 0)
        row["Execution step count"] = step_counts.get(decision_id, 0)

    decisions_frame = pd.DataFrame(decision_rows, columns=_DECISION_COLUMNS)
    if not decisions_frame.empty:
        decisions_frame["_phase"] = decisions_frame["Phase"].map(
            _PHASE_ORDER
        ).fillna(99)
        decisions_frame["_status"] = decisions_frame["Status"].map(
            _STATUS_ORDER
        ).fillna(99)
        decisions_frame = (
            decisions_frame.sort_values(["_phase", "_status", "Decision ID"])
            .drop(columns=["_phase", "_status"])
            .reset_index(drop=True)
        )

    evidence_frame = pd.DataFrame(evidence_rows, columns=_EVIDENCE_COLUMNS)
    if not evidence_frame.empty:
        evidence_frame = evidence_frame.sort_values(
            ["Decision ID", "Evidence ID"]
        ).reset_index(drop=True)

    steps_frame = pd.DataFrame(step_rows, columns=_EXECUTION_COLUMNS)
    if not steps_frame.empty:
        steps_frame = steps_frame.sort_values(
            ["Sequence", "Step ID"]
        ).reset_index(drop=True)

    guardrails_frame = pd.DataFrame(guardrail_rows, columns=_GUARDRAIL_COLUMNS)
    if not guardrails_frame.empty:
        guardrails_frame = guardrails_frame.sort_values(
            ["Severity", "Guardrail ID"],
            key=lambda series: series.map(
                {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
            ) if series.name == "Severity" else series,
        ).reset_index(drop=True)

    issues_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    if not issues_frame.empty:
        issues_frame = issues_frame.sort_values(
            ["Scope", "Item", "Issue"]
        ).reset_index(drop=True)

    return PreparationDecisionReport(
        available_fields=fields,
        decisions=decisions_frame,
        evidence=evidence_frame,
        execution_steps=steps_frame,
        guardrails=guardrails_frame,
        split_policy=split_declaration,
        issues=issues_frame,
    )



def record_static_multiclass_preparation_decisions(
    *,
    available_fields: Sequence[object],
    target: str,
    target_classes: Sequence[object],
    candidate_features: Sequence[object],
    identifiers: Sequence[object] | None,
    duplicate_report: object,
    target_report: object,
    numerical_report: object,
    feature_relationship_report: object,
    leakage_report: object,
    quality_report: object,
    exploratory_insights_report: object,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> PreparationDecisionReport:
    """Build a traceable preparation plan for a static multiclass snapshot.

    The function records decisions only. It never mutates the source data,
    splits observations, fits transformers, resamples classes, selects
    features, or trains a model. Dataset-specific source semantics remain
    explicit in the notebook through ``target``, ``target_classes``,
    ``candidate_features``, and the split parameters.
    """
    fields = _unique_text_tuple(available_fields)
    features = _unique_text_tuple(candidate_features)
    id_columns = _unique_text_tuple(identifiers or ())
    classes = tuple(deepcopy(tuple(target_classes)))
    target_name = _text(target)

    if not target_name or target_name not in set(fields):
        raise PreparationDecisionContractError(
            f"Target {target_name!r} is not available for preparation."
        )
    if not features:
        raise PreparationDecisionContractError(
            "At least one candidate feature is required."
        )
    unknown_features = tuple(value for value in features if value not in set(fields))
    if unknown_features:
        raise PreparationDecisionContractError(
            f"Candidate features are not available: {unknown_features!r}."
        )
    unknown_ids = tuple(value for value in id_columns if value not in set(fields))
    if unknown_ids:
        raise PreparationDecisionContractError(
            f"Identifier fields are not available: {unknown_ids!r}."
        )
    if target_name in set(features):
        raise PreparationDecisionContractError(
            "The target cannot be included in candidate_features."
        )
    if len(classes) < 3:
        raise PreparationDecisionContractError(
            "Static multiclass preparation requires at least three target classes."
        )

    if not bool(getattr(quality_report, "is_structurally_valid", False)):
        raise PreparationDecisionContractError(
            "The initial data-quality report must be structurally valid."
        )
    if bool(getattr(quality_report, "has_must_fix_actions", False)):
        raise PreparationDecisionContractError(
            "Deterministic must-fix data-quality actions require an explicit "
            "dataset-specific preparation decision before automatic planning."
        )
    if bool(getattr(quality_report, "has_external_blockers", False)):
        raise PreparationDecisionContractError(
            "External data-quality blockers must be resolved before preparation planning."
        )
    if not bool(
        getattr(
            exploratory_insights_report,
            "is_ready_for_preparation_decisions",
            False,
        )
    ):
        raise PreparationDecisionContractError(
            "Exploratory insights are not ready to inform preparation decisions."
        )

    if int(getattr(target_report, "class_count", 0)) != len(classes):
        raise PreparationDecisionContractError(
            "Observed multiclass target cardinality does not match the declared contract."
        )
    if bool(getattr(target_report, "has_issues", True)):
        raise PreparationDecisionContractError(
            "Target distribution contains missing, absent, or unexpected classes."
        )
    if bool(getattr(leakage_report, "has_direct_target_leakage", True)):
        raise PreparationDecisionContractError(
            "Direct target leakage must be resolved before preparation decisions are recorded."
        )

    dependency_frame = leakage_report.dependency_frame()
    if dependency_frame.empty:
        unconfirmed_dependencies: tuple[str, ...] = ()
    else:
        mask = dependency_frame["Dependency status"].eq(
            "Declared dependency not confirmed"
        )
        unconfirmed_dependencies = tuple(
            str(value)
            for value in dependency_frame.loc[mask, "Derived feature"]
        )

    redundancy_count = 0
    numerical_relationships = getattr(
        feature_relationship_report,
        "numerical_relationships",
        pd.DataFrame(),
    )
    if (
        isinstance(numerical_relationships, pd.DataFrame)
        and not numerical_relationships.empty
        and "Potential redundancy" in numerical_relationships.columns
    ):
        redundancy_count = int(
            numerical_relationships["Potential redundancy"].fillna(False).sum()
        )

    exact_duplicate_groups = int(
        getattr(duplicate_report, "exact_duplicate_group_count", 0)
    )
    exact_duplicate_rows = int(
        getattr(duplicate_report, "exact_duplicate_row_count", 0)
    )
    has_source_identifiers = bool(
        getattr(duplicate_report, "has_source_identifiers", False)
    )
    outlier_features = tuple(
        str(value)
        for value in getattr(numerical_report, "features_with_outliers", ())
    )
    imbalance_ratio = getattr(target_report, "imbalance_ratio", None)
    normalized_entropy = getattr(
        target_report, "normalized_class_entropy", None
    )

    def decision(
        decision_id: str,
        domain: str,
        title: str,
        affected_fields: Sequence[object],
        status: str,
        phase: str,
        fit_scope: str,
        operation: str,
        rationale: str,
        acceptance_criteria: str,
        source_stages: Sequence[object],
        prerequisites: Sequence[object] = (),
    ) -> dict[str, object]:
        return {
            "decision_id": decision_id,
            "domain": domain,
            "title": title,
            "affected_fields": tuple(affected_fields),
            "status": status,
            "phase": phase,
            "fit_scope": fit_scope,
            "operation": operation,
            "rationale": rationale,
            "prerequisites": tuple(prerequisites),
            "acceptance_criteria": acceptance_criteria,
            "source_stages": tuple(str(value) for value in source_stages),
        }

    decisions: list[dict[str, object]] = [
        decision(
            "PREP-001",
            "Cleaning",
            "Preserve the validated source observations",
            fields,
            "Approved",
            "Before split",
            "Deterministic",
            (
                "Create a defensive prepared copy without changing values, "
                "row count, column semantics, or target labels."
            ),
            (
                "Source-backed type, domain, and value checks found no "
                "deterministic repair requirement."
            ),
            (
                "Prepared row count and values match the validated source "
                "before any split or learned transformation."
            ),
            ("7", "8", "16"),
        ),
        decision(
            "PREP-002",
            "Cleaning",
            "Prohibit unsupported deduplication and generic outlier treatment",
            fields,
            "Prohibited",
            "Before split",
            "None",
            (
                "Do not drop exact row matches without independent source "
                "identity evidence; do not delete, clip, winsorize, or replace "
                "values solely because they are IQR outlier candidates."
            ),
            (
                "The source provides no observation identifier, while numerical "
                "extremes remain domain-valid measurements."
            ),
            (
                "No row or value is altered by equality-only deduplication or a "
                "generic IQR rule."
            ),
            ("9", "11", "16", "17"),
        ),
        decision(
            "PREP-003",
            "Target governance",
            "Preserve the nominal multiclass target contract",
            (target_name,),
            "Approved",
            "Before split",
            "None",
            (
                "Keep every declared class label readable in prepared data and "
                "exclude the target and any target derivative from predictors."
            ),
            "All declared classes are observed and no binary positive-class semantics apply.",
            (
                "Prepared y contains exactly the declared class set and X never "
                "contains the target."
            ),
            ("5", "10", "15", "16"),
        ),
        decision(
            "PREP-004",
            "Feature role",
            "Use the complete validated candidate-feature set as the baseline",
            features,
            "Approved",
            "Before split",
            "None",
            (
                "Start downstream model evaluation with all validated numerical "
                "candidate features, including documented derived measurements."
            ),
            (
                "Correlation and mathematical redundancy are exploratory evidence, "
                "not sufficient grounds for global feature deletion."
            ),
            "Baseline X contains exactly the declared candidate features.",
            ("6", "12", "15", "17"),
        ),
        decision(
            "PREP-005",
            "Dataset splitting",
            "Use a reproducible stratified snapshot split",
            (target_name,),
            "Approved",
            "Split",
            "None",
            (
                f"Partition the validated snapshot into {train_fraction:.0%} train, "
                f"{validation_fraction:.0%} validation, and {test_fraction:.0%} test "
                f"with stratification by {target_name} and random seed {random_seed}."
            ),
            (
                "The released table has no chronological evaluation field, while "
                "unequal multiclass support makes stratification important."
            ),
            (
                "Partitions are reproducible, disjoint, preserve every target class, "
                "and keep the final test holdout untouched."
            ),
            ("10", "15", "17"),
            ("PREP-001", "PREP-003"),
        ),
        decision(
            "PREP-006",
            "Transformation",
            "Make numerical scaling model-dependent and train-fitted",
            features,
            "Conditional",
            "Train-only transformation",
            "Train only",
            (
                "Preserve original numerical values and fit scaling only inside "
                "candidate pipelines whose model family requires or benefits from it."
            ),
            "All predictors are numerical, but scale sensitivity depends on model family.",
            (
                "Any scaler is fitted on training data only; validation and test "
                "are transform-only and never refit preprocessing."
            ),
            ("7", "11", "15", "17"),
            ("PREP-005",),
        ),
        decision(
            "PREP-007",
            "Feature engineering",
            "Defer class-imbalance mitigation to model selection",
            (target_name,),
            "Deferred",
            "Model selection",
            "Evaluation only",
            (
                "Use stratification as the baseline policy and compare class "
                "weighting or training-only resampling only if validation evidence "
                "shows a benefit for macro and per-class metrics."
            ),
            (
                "Class support is unequal, but frequency alone does not justify "
                "altering the released distribution."
            ),
            (
                "Any imbalance strategy is evaluated inside training/validation; "
                "validation and test prevalence remains untouched."
            ),
            ("10", "16", "17"),
            ("PREP-005",),
        ),
        decision(
            "PREP-008",
            "Feature engineering",
            "Defer redundancy reduction to leakage-safe ablation",
            features,
            "Deferred",
            "Model selection",
            "Evaluation only",
            (
                "Compare the all-feature baseline with regularized or ablated "
                "feature sets using training and validation evidence only."
            ),
            (
                f"{redundancy_count} feature pair(s) meet the declared redundancy "
                "review rule and multiple measurements are mathematically derived."
            ),
            (
                "No feature is removed globally from EDA rankings; any reduced set "
                "must match or improve validation objectives before adoption."
            ),
            ("12", "15", "17"),
            ("PREP-004", "PREP-005"),
        ),
        decision(
            "PREP-010",
            "Leakage governance",
            "Split before every learned target-aware or distribution-aware operation",
            tuple(features) + (target_name,),
            "Approved",
            "Split",
            "None",
            (
                "Perform the approved split before fitting scalers, selectors, "
                "resampling strategies, target-aware rankings, or model parameters."
            ),
            "Held-out partitions must not influence learned preparation choices.",
            (
                "All learned operations record train-only fit scope and the final "
                "test partition is used only after the model contract is frozen."
            ),
            ("15", "17"),
            ("PREP-005",),
        ),
    ]

    if unconfirmed_dependencies:
        decisions.append(
            decision(
                "PREP-009",
                "Feature engineering",
                "Keep unconfirmed derived dependencies out of automatic pruning",
                unconfirmed_dependencies,
                "Deferred",
                "Model selection",
                "Evaluation only",
                (
                    "Retain these features in the baseline and verify source formula "
                    "or tolerance assumptions before using dependency claims to prune them."
                ),
                (
                    "At least one declared mathematical dependency did not reproduce "
                    "the released values within the exploratory tolerance."
                ),
                (
                    "Each unresolved dependency is either source-verified or treated "
                    "as an ordinary candidate feature during model evaluation."
                ),
                ("15", "16", "17"),
                ("PREP-004",),
            )
        )

    evidence_specs = {
        "PREP-001": (
            "quality_report",
            "validated source quality",
            {
                "must_fix_actions": bool(getattr(quality_report, "has_must_fix_actions", False)),
                "external_blockers": bool(getattr(quality_report, "has_external_blockers", False)),
            },
            "No deterministic repair is required before copying the source.",
        ),
        "PREP-002": (
            "duplicate_report + numerical_report",
            "review-only cleaning evidence",
            {
                "source_identifiers_available": has_source_identifiers,
                "exact_duplicate_groups": exact_duplicate_groups,
                "exact_duplicate_rows": exact_duplicate_rows,
                "features_with_iqr_candidates": outlier_features,
            },
            "Equality without source identity and IQR extremeness do not prove invalid observations.",
        ),
        "PREP-003": (
            "target_report",
            "multiclass target contract",
            {"class_count": len(classes), "classes": classes},
            "All declared classes must remain present and target-only.",
        ),
        "PREP-004": (
            "feature_relationship_report + leakage_report",
            "baseline feature governance",
            {
                "candidate_feature_count": len(features),
                "redundancy_candidates": redundancy_count,
                "confirmed_derived_dependencies": int(
                    getattr(leakage_report, "confirmed_derived_dependency_count", 0)
                ),
            },
            "Redundancy requires validation, not global deletion during preparation.",
        ),
        "PREP-005": (
            "target_report",
            "class-support evidence",
            {
                "imbalance_ratio": imbalance_ratio,
                "normalized_class_entropy": normalized_entropy,
            },
            "Use a reproducible class-aware split for the static snapshot.",
        ),
        "PREP-006": (
            "numerical_report",
            "numerical predictor contract",
            {"numerical_feature_count": len(features)},
            "Scaling policy must be selected per model and fitted on training only.",
        ),
        "PREP-007": (
            "target_report",
            "unequal multiclass support",
            {"imbalance_ratio": imbalance_ratio},
            "Do not alter class prevalence without validation evidence.",
        ),
        "PREP-008": (
            "feature_relationship_report",
            "redundancy review",
            {"redundancy_candidates": redundancy_count},
            "Ablation belongs inside model selection.",
        ),
        "PREP-010": (
            "leakage_report",
            "target-isolation audit",
            {"direct_target_leakage": bool(getattr(leakage_report, "has_direct_target_leakage", False))},
            "Learned operations must never fit on held-out partitions.",
        ),
    }
    if unconfirmed_dependencies:
        evidence_specs["PREP-009"] = (
            "leakage_report + quality_report",
            "unconfirmed derived-feature provenance",
            {"features": unconfirmed_dependencies},
            "Unconfirmed formulas cannot justify automatic feature pruning.",
        )

    evidence: list[dict[str, object]] = []
    for index, item in enumerate(decisions, start=1):
        decision_id = str(item["decision_id"])
        source_report, source_item, observed_value, interpretation = evidence_specs[decision_id]
        evidence.append(
            {
                "evidence_id": f"PDE-{index:03d}",
                "decision_id": decision_id,
                "source_report": source_report,
                "source_item": source_item,
                "observed_value": observed_value,
                "expected_or_reference": "Stage 18 preparation policy",
                "interpretation": interpretation,
            }
        )

    deferred_decision_ids = [
        item["decision_id"]
        for item in decisions
        if item["status"] == "Deferred"
    ]

    execution_steps: list[dict[str, object]] = [
        {
            "step_id": "STEP-001",
            "sequence": 1,
            "decision_ids": ("PREP-001", "PREP-002", "PREP-003", "PREP-004"),
            "phase": "Before split",
            "action": "Revalidate the raw table and create an unchanged prepared projection.",
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": "Source values and row count are preserved and analytical roles are isolated.",
        },
        {
            "step_id": "STEP-002",
            "sequence": 2,
            "decision_ids": ("PREP-005", "PREP-010"),
            "phase": "Split",
            "action": "Create reproducible stratified train, validation, and test partitions.",
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": "Partitions are disjoint, class-aware, reproducible, and preserve the final test holdout.",
        },
        {
            "step_id": "STEP-003",
            "sequence": 3,
            "decision_ids": ("PREP-006",),
            "phase": "Train-only transformation",
            "action": "Fit model-dependent numerical preprocessing on training data only.",
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": "Validation and test are transform-only and no scaler is globally fitted.",
        },
        {
            "step_id": "STEP-004",
            "sequence": 4,
            "decision_ids": tuple(deferred_decision_ids),
            "phase": "Model selection",
            "action": "Evaluate imbalance and redundancy alternatives without changing the baseline handoff.",
            "blocking": False,
            "status": "Deferred",
            "temporal_dependency": False,
            "acceptance_criteria": "Alternatives are adopted only from training/validation evidence.",
        },
        {
            "step_id": "STEP-005",
            "sequence": 5,
            "decision_ids": tuple(item["decision_id"] for item in decisions),
            "phase": "Model selection",
            "action": "Freeze the selected preparation and feature contract before final test evaluation.",
            "blocking": True,
            "status": "Deferred",
            "temporal_dependency": False,
            "acceptance_criteria": "The final test set is accessed only after preprocessing, feature policy, and model-selection choices are frozen.",
        },
    ]

    def guardrail(
        guardrail_id: str,
        domain: str,
        title: str,
        affected_fields: Sequence[object],
        severity: str,
        prohibited_operation: str,
        rationale: str,
        verification: str,
    ) -> dict[str, object]:
        return {
            "guardrail_id": guardrail_id,
            "domain": domain,
            "title": title,
            "affected_fields": tuple(affected_fields),
            "severity": severity,
            "status": "Active",
            "prohibited_operation": prohibited_operation,
            "rationale": rationale,
            "verification": verification,
        }

    guardrails = (
        guardrail(
            "GRD-001", "Cleaning", "Preserve the raw evidence", fields, "Critical",
            "Modify or overwrite the acquired raw table in place.",
            "Reproducible preparation requires immutable source evidence.",
            "Raw shape, values, index, and dtypes remain unchanged.",
        ),
        guardrail(
            "GRD-002", "Cleaning", "Do not deduplicate from row equality alone", fields, "High",
            "Drop exact matches without independent observation identity evidence.",
            "The released Dry Bean table does not provide a source observation identifier.",
            "No source row is removed solely because all released values match another row.",
        ),
        guardrail(
            "GRD-003", "Cleaning", "Do not apply generic outlier cleaning", features, "High",
            "Delete, clip, winsorize, or replace values solely from an IQR flag.",
            "IQR flags are distributional evidence and the audited values remain domain-valid.",
            "The baseline prepared projection preserves every numerical value.",
        ),
        guardrail(
            "GRD-004", "Target governance", "Keep the target outside predictors", (target_name,), "Critical",
            "Include the target or a direct derivative in X or feature transformations.",
            "This would constitute direct target leakage.",
            "Predictor matrices contain exactly candidate features and never the target.",
        ),
        guardrail(
            "GRD-005", "Leakage governance", "Fit learned preprocessing on training only", features, "Critical",
            "Fit scaling, selection, resampling, or learned transformations before splitting or on held-out data.",
            "Held-out data must not influence learned preparation parameters.",
            "Every learned transformer records train-only fit scope.",
        ),
        guardrail(
            "GRD-006", "Dataset splitting", "Do not resample validation or test", (target_name,), "Critical",
            "Over- or undersample validation or test partitions.",
            "Held-out prevalence must remain representative of the released snapshot.",
            "Validation and test row sets retain their original class composition.",
        ),
        guardrail(
            "GRD-007", "Feature engineering", "Do not select features globally", features, "Critical",
            "Use full-data target associations or redundancy rankings to choose the final feature set.",
            "Global selection would bias held-out evaluation.",
            "Ablation and selection use training/validation evidence only.",
        ),
        guardrail(
            "GRD-008", "Dataset splitting", "Protect the final test holdout", (target_name,), "Critical",
            "Use final-test metrics for model, feature, resampling, or hyperparameter selection.",
            "Repeated test access converts the final holdout into validation data.",
            "Final test evaluation occurs only after the analysis contract is frozen.",
        ),
    )

    split_policy = {
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "stratify_by": target_name,
        "random_seed": random_seed,
        "shuffle": True,
        "temporal_priority": True,
        "temporal_policy_status": "Resolved snapshot fallback",
        "random_split_fallback": (
            "Approved for this source-released static classification snapshot: "
            "no chronological observation field is available in the analytical table."
        ),
        "test_holdout_untouched": True,
        "disjoint_partitions_required": True,
        "group_by_identifiers": id_columns,
    }

    report = record_preparation_decisions(
        available_fields=fields,
        decisions=decisions,
        evidence=evidence,
        execution_steps=execution_steps,
        guardrails=guardrails,
        split_policy=split_policy,
    )
    report.raise_if_invalid()
    return report


def record_static_continuous_regression_preparation_decisions(
    *,
    available_fields: Sequence[object],
    target: str,
    candidate_features: Sequence[object],
    identifiers: Sequence[object] | None,
    duplicate_report: object,
    target_report: object,
    numerical_report: object,
    feature_relationship_report: object,
    regression_structure_report: object,
    leakage_report: object,
    quality_report: object,
    exploratory_insights_report: object,
    source_type_resolutions: Mapping[str, Mapping[str, object]] | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> PreparationDecisionReport:
    """Build a traceable preparation plan for a static continuous-regression snapshot.

    The function records decisions only. It never mutates source values, splits
    observations, fits transformers, engineers nonlinear terms, selects
    features, or trains a model. Dataset-specific source-type resolutions are
    explicit inputs because source metadata can disagree with released numeric
    values without making those observations invalid.
    """
    fields = _unique_text_tuple(available_fields)
    features = _unique_text_tuple(candidate_features)
    id_columns = _unique_text_tuple(identifiers or ())
    target_name = _text(target)

    if not target_name or target_name not in set(fields):
        raise PreparationDecisionContractError(
            f"Target {target_name!r} is not available for preparation."
        )
    if not features:
        raise PreparationDecisionContractError(
            "At least one candidate feature is required."
        )
    unknown_features = tuple(value for value in features if value not in set(fields))
    if unknown_features:
        raise PreparationDecisionContractError(
            f"Candidate features are not available: {unknown_features!r}."
        )
    unknown_ids = tuple(value for value in id_columns if value not in set(fields))
    if unknown_ids:
        raise PreparationDecisionContractError(
            f"Identifier fields are not available: {unknown_ids!r}."
        )
    if target_name in set(features):
        raise PreparationDecisionContractError(
            "The target cannot be included in candidate_features."
        )

    if not bool(getattr(quality_report, "is_structurally_valid", False)):
        raise PreparationDecisionContractError(
            "The initial data-quality report must be structurally valid."
        )
    if bool(getattr(quality_report, "has_external_blockers", False)):
        raise PreparationDecisionContractError(
            "External data-quality blockers must be resolved before preparation planning."
        )
    if not bool(
        getattr(
            exploratory_insights_report,
            "is_ready_for_preparation_decisions",
            False,
        )
    ):
        raise PreparationDecisionContractError(
            "Exploratory insights are not ready to inform preparation decisions."
        )

    if bool(getattr(target_report, "has_missing_values", True)):
        raise PreparationDecisionContractError(
            "Continuous target missing values must be resolved before preparation planning."
        )
    if bool(getattr(target_report, "has_non_finite_values", True)):
        raise PreparationDecisionContractError(
            "Continuous target non-finite values must be resolved before preparation planning."
        )
    if not bool(getattr(target_report, "has_variation", False)):
        raise PreparationDecisionContractError(
            "Continuous target must contain more than one finite value."
        )
    if bool(getattr(leakage_report, "has_direct_target_leakage", True)):
        raise PreparationDecisionContractError(
            "Direct target leakage must be resolved before preparation decisions are recorded."
        )

    raw_resolutions = dict(source_type_resolutions or {})
    resolutions: dict[str, dict[str, str]] = {}
    required_resolution_keys = (
        "source_declared_type",
        "effective_analytical_type",
        "operation",
        "rationale",
    )
    for raw_field, raw_resolution in raw_resolutions.items():
        field = _text(raw_field)
        if not field or field not in set(fields):
            raise PreparationDecisionContractError(
                f"Source-type resolution field {field!r} is not available."
            )
        if not isinstance(raw_resolution, Mapping):
            raise PreparationDecisionContractError(
                f"Source-type resolution for {field!r} must be a mapping."
            )
        normalized = {
            key: _text(raw_resolution.get(key))
            for key in required_resolution_keys
        }
        missing_keys = tuple(
            key for key, value in normalized.items() if not value
        )
        if missing_keys:
            raise PreparationDecisionContractError(
                f"Source-type resolution for {field!r} is incomplete: "
                f"{missing_keys!r}."
            )
        resolutions[field] = normalized

    findings_method = getattr(quality_report, "findings_frame", None)
    if not callable(findings_method):
        raise PreparationDecisionContractError(
            "The initial data-quality report must expose findings_frame()."
        )
    findings = findings_method()
    if not isinstance(findings, pd.DataFrame):
        raise PreparationDecisionContractError(
            "quality_report.findings_frame() must return a pandas DataFrame."
        )

    must_fix = pd.DataFrame()
    if not findings.empty and "Disposition" in findings.columns:
        must_fix = findings.loc[findings["Disposition"].eq("Must fix")].copy()

    unresolved_fields: set[str] = set()
    unresolved_items: list[str] = []
    if not must_fix.empty:
        for _, row in must_fix.iterrows():
            affected = _tuple_values(row.get("Affected fields", ()))
            if not affected:
                unresolved_items.append(_text(row.get("Finding ID")) or "<unknown>")
                continue
            for field in affected:
                if field not in resolutions:
                    unresolved_fields.add(field)

    if unresolved_fields or unresolved_items:
        details: list[str] = []
        if unresolved_fields:
            details.append(
                "unresolved fields=" + repr(tuple(sorted(unresolved_fields)))
            )
        if unresolved_items:
            details.append(
                "unresolved findings=" + repr(tuple(unresolved_items))
            )
        raise PreparationDecisionContractError(
            "Deterministic must-fix data-quality actions require explicit "
            "source-type resolutions: " + "; ".join(details)
        )

    dependency_frame = leakage_report.dependency_frame()
    if dependency_frame.empty:
        unconfirmed_dependencies: tuple[str, ...] = ()
    else:
        mask = dependency_frame["Dependency status"].eq(
            "Declared dependency not confirmed"
        )
        unconfirmed_dependencies = tuple(
            str(value)
            for value in dependency_frame.loc[mask, "Derived feature"]
        )

    redundancy_count = 0
    numerical_relationships = getattr(
        feature_relationship_report,
        "numerical_relationships",
        pd.DataFrame(),
    )
    if (
        isinstance(numerical_relationships, pd.DataFrame)
        and not numerical_relationships.empty
        and "Potential redundancy" in numerical_relationships.columns
    ):
        redundancy_count = int(
            numerical_relationships["Potential redundancy"].fillna(False).sum()
        )

    exact_duplicate_groups = int(
        getattr(duplicate_report, "exact_duplicate_group_count", 0)
    )
    exact_duplicate_rows = int(
        getattr(duplicate_report, "exact_duplicate_row_count", 0)
    )
    repeated_profile_groups = int(
        getattr(duplicate_report, "repeated_profile_group_count", 0)
    )
    conflicting_profile_groups = int(
        getattr(duplicate_report, "target_conflict_group_count", 0)
    )
    has_source_identifiers = bool(
        getattr(duplicate_report, "has_source_identifiers", False)
    )
    outlier_features = tuple(
        str(value)
        for value in getattr(numerical_report, "features_with_outliers", ())
    )

    nonlinearity_frame = getattr(
        regression_structure_report,
        "nonlinearity",
        pd.DataFrame(),
    )
    interaction_frame = getattr(
        regression_structure_report,
        "interactions",
        pd.DataFrame(),
    )
    nonlinearity_count = (
        int(nonlinearity_frame["Nonlinearity signal"].fillna(False).sum())
        if isinstance(nonlinearity_frame, pd.DataFrame)
        and not nonlinearity_frame.empty
        and "Nonlinearity signal" in nonlinearity_frame.columns
        else 0
    )
    interaction_count = (
        int(interaction_frame["Interaction signal"].fillna(False).sum())
        if isinstance(interaction_frame, pd.DataFrame)
        and not interaction_frame.empty
        and "Interaction signal" in interaction_frame.columns
        else 0
    )

    def decision(
        decision_id: str,
        domain: str,
        title: str,
        affected_fields: Sequence[object],
        status: str,
        phase: str,
        fit_scope: str,
        operation: str,
        rationale: str,
        acceptance_criteria: str,
        source_stages: Sequence[object],
        prerequisites: Sequence[object] = (),
    ) -> dict[str, object]:
        return {
            "decision_id": decision_id,
            "domain": domain,
            "title": title,
            "affected_fields": tuple(affected_fields),
            "status": status,
            "phase": phase,
            "fit_scope": fit_scope,
            "operation": operation,
            "rationale": rationale,
            "prerequisites": tuple(prerequisites),
            "acceptance_criteria": acceptance_criteria,
            "source_stages": tuple(str(value) for value in source_stages),
        }

    decisions: list[dict[str, object]] = []

    if resolutions:
        resolution_fields = tuple(resolutions)
        resolution_operations = "; ".join(
            f"{field}: {resolutions[field]['operation']}"
            for field in resolution_fields
        )
        resolution_rationales = "; ".join(
            f"{field}: {resolutions[field]['rationale']}"
            for field in resolution_fields
        )
        decisions.append(
            decision(
                "PREP-001",
                "Cleaning",
                "Resolve source-type metadata conflicts without altering released values",
                resolution_fields,
                "Approved",
                "Before split",
                "Deterministic",
                resolution_operations,
                resolution_rationales,
                (
                    "Every declared source-type conflict has an explicit effective "
                    "analytical type and the released numerical values are preserved "
                    "without rounding, truncation, deletion, or imputation."
                ),
                ("7", "8", "16", "17"),
            )
        )

    preserve_prerequisites = ("PREP-001",) if resolutions else ()
    decisions.extend(
        [
            decision(
                "PREP-002",
                "Cleaning",
                "Preserve the validated source observations",
                fields,
                "Approved",
                "Before split",
                "Deterministic",
                (
                    "Create a defensive prepared copy that preserves row count and "
                    "released numerical values while applying only the explicitly "
                    "declared analytical-type interpretation."
                ),
                (
                    "No missing, non-finite, domain-invalid, or leakage-derived value "
                    "requires source-row mutation after the declared metadata resolution."
                ),
                (
                    "Prepared row count and numerical values match the validated source "
                    "before any split or learned transformation."
                ),
                ("7", "8", "15", "16"),
                preserve_prerequisites,
            ),
            decision(
                "PREP-003",
                "Cleaning",
                "Prohibit unsupported deduplication and generic outlier treatment",
                fields,
                "Prohibited",
                "Before split",
                "None",
                (
                    "Do not drop exact row matches without independent source identity "
                    "evidence; do not delete, clip, winsorize, or replace values solely "
                    "because they are IQR or target-extreme candidates."
                ),
                (
                    "The source provides no observation identifier, repeated profiles "
                    "can represent valid measurements, and audited extremes remain "
                    "within the declared physical/domain constraints."
                ),
                (
                    "No row or value is altered by equality-only deduplication or a "
                    "generic statistical-extreme rule."
                ),
                ("9", "10", "11", "16", "17"),
            ),
            decision(
                "PREP-004",
                "Target governance",
                "Preserve the continuous target on its original measurement scale",
                (target_name,),
                "Approved",
                "Before split",
                "None",
                (
                    "Keep the regression target numeric and continuous in its released "
                    "measurement scale; do not discretize it into classes or use it as "
                    "a predictor."
                ),
                (
                    "The analytical contract is continuous regression and downstream "
                    "errors must remain interpretable on the original target scale."
                ),
                (
                    "Prepared y preserves the released numeric target values and X "
                    "never contains the target or a direct target derivative."
                ),
                ("5", "10", "15", "17"),
            ),
            decision(
                "PREP-005",
                "Feature role",
                "Use the complete validated candidate-feature set as the baseline",
                features,
                "Approved",
                "Before split",
                "None",
                (
                    "Start downstream regression evaluation with all validated "
                    "candidate features in their released numerical representation."
                ),
                (
                    "Exploratory association, redundancy, curvature, and interaction "
                    "signals do not justify global feature deletion during preparation."
                ),
                "Baseline X contains exactly the declared candidate features.",
                ("6", "12", "13", "14", "15", "17"),
            ),
            decision(
                "PREP-006",
                "Dataset splitting",
                "Use a reproducible non-stratified snapshot split",
                (target_name,),
                "Approved",
                "Split",
                "None",
                (
                    f"Partition the validated snapshot into {train_fraction:.0%} train, "
                    f"{validation_fraction:.0%} validation, and {test_fraction:.0%} test "
                    f"with random seed {random_seed}, without discretizing the continuous "
                    "target solely to manufacture stratification bins."
                ),
                (
                    "The analytical table has no chronological evaluation field and the "
                    "continuous target has no natural class labels for direct stratification."
                ),
                (
                    "Partitions are reproducible and disjoint; the final test holdout "
                    "remains untouched, and per-partition target summaries are reported "
                    "descriptively without seed shopping."
                ),
                ("10", "15", "17"),
                ("PREP-002", "PREP-004"),
            ),
            decision(
                "PREP-007",
                "Transformation",
                "Make numerical scaling model-dependent and train-fitted",
                features,
                "Conditional",
                "Train-only transformation",
                "Train only",
                (
                    "Preserve released numerical values and fit scaling only inside "
                    "candidate pipelines whose regression family requires or benefits "
                    "from it."
                ),
                (
                    "All predictors are numerical, but sensitivity to feature scale "
                    "depends on the candidate regression family."
                ),
                (
                    "Any scaler is fitted on training data only; validation and test "
                    "are transform-only and never refit preprocessing."
                ),
                ("7", "11", "15", "17"),
                ("PREP-006",),
            ),
            decision(
                "PREP-008",
                "Feature engineering",
                "Defer nonlinear and interaction representation to model selection",
                features,
                "Deferred",
                "Model selection",
                "Evaluation only",
                (
                    "Compare a transparent additive baseline with candidate families "
                    "or pipeline terms capable of representing nonlinear effects and "
                    "interactions using training/validation evidence only."
                ),
                (
                    f"Exploration flagged {nonlinearity_count} nonlinearity signal(s) "
                    f"and {interaction_count} interaction signal(s), but the diagnostics "
                    "are in-sample and do not establish generalization benefit."
                ),
                (
                    "Nonlinear or interaction-aware alternatives are adopted only when "
                    "leakage-safe validation improves the declared regression objectives."
                ),
                ("14", "17"),
                ("PREP-005", "PREP-006"),
            ),
            decision(
                "PREP-009",
                "Feature engineering",
                "Carry repeated-profile ambiguity into model-evaluation sensitivity checks",
                features + (target_name,),
                "Deferred",
                "Model selection",
                "Evaluation only",
                (
                    "Preserve repeated observations in the baseline and quantify whether "
                    "repeated candidate-feature profiles materially affect validation "
                    "error or residual interpretation."
                ),
                (
                    f"Exploration found {repeated_profile_groups} repeated feature-profile "
                    f"group(s), including {conflicting_profile_groups} group(s) with "
                    "different continuous target values."
                ),
                (
                    "Any repeated-profile sensitivity analysis is reported separately "
                    "and does not rewrite the official source distribution."
                ),
                ("9", "16", "17"),
                ("PREP-003", "PREP-006"),
            ),
            decision(
                "PREP-010",
                "Leakage governance",
                "Split before every learned target-aware or distribution-aware operation",
                features + (target_name,),
                "Approved",
                "Split",
                "None",
                (
                    "Perform the approved split before fitting scalers, selectors, "
                    "target transformations, feature engineering, or model parameters."
                ),
                "Held-out partitions must not influence learned preparation choices.",
                (
                    "All learned operations record train-only fit scope and the final "
                    "test partition is used only after the modeling contract is frozen."
                ),
                ("15", "17"),
                ("PREP-006",),
            ),
        ]
    )

    if redundancy_count > 0 or unconfirmed_dependencies:
        affected = tuple(
            dict.fromkeys(
                list(features)
                + list(unconfirmed_dependencies)
            )
        )
        decisions.append(
            decision(
                "PREP-011",
                "Feature engineering",
                "Keep redundancy and unconfirmed dependencies inside leakage-safe ablation",
                affected,
                "Deferred",
                "Model selection",
                "Evaluation only",
                (
                    "Retain all affected features in the baseline and compare any "
                    "reduced representation using training/validation evidence only."
                ),
                (
                    f"{redundancy_count} redundancy review candidate(s) and "
                    f"{len(unconfirmed_dependencies)} unconfirmed dependency claim(s) "
                    "require validation rather than global pruning."
                ),
                (
                    "No feature is removed globally from EDA evidence; any reduced set "
                    "must meet the declared validation objectives before adoption."
                ),
                ("12", "15", "17"),
                ("PREP-005", "PREP-006"),
            )
        )

    evidence_specs: dict[str, tuple[str, str, object, str]] = {}

    if resolutions:
        evidence_specs["PREP-001"] = (
            "quality_report",
            "source-type resolution",
            {
                "resolved_fields": tuple(resolutions),
                "resolutions": deepcopy(resolutions),
                "must_fix_findings": int(len(must_fix)),
            },
            "Explicit metadata interpretation resolves preparation blockers without mutating released values.",
        )

    evidence_specs.update(
        {
            "PREP-002": (
                "quality_report",
                "validated source preservation",
                {
                    "must_fix_findings": int(len(must_fix)),
                    "external_blockers": bool(
                        getattr(quality_report, "has_external_blockers", False)
                    ),
                },
                "After explicit source-type resolution, no additional source-row mutation is authorized.",
            ),
            "PREP-003": (
                "duplicate_report + numerical_report + target_report",
                "review-only cleaning evidence",
                {
                    "source_identifiers_available": has_source_identifiers,
                    "exact_duplicate_groups": exact_duplicate_groups,
                    "exact_duplicate_rows": exact_duplicate_rows,
                    "features_with_iqr_candidates": outlier_features,
                    "target_extreme_count": int(getattr(target_report, "extreme_count", 0)),
                },
                "Equality and statistical extremeness do not independently prove invalid observations.",
            ),
            "PREP-004": (
                "target_report",
                "continuous target contract",
                {
                    "target": target_name,
                    "unit": getattr(target_report, "unit", None),
                    "finite_values": int(getattr(target_report, "finite_count", 0)),
                    "unique_values": int(getattr(target_report, "unique_count", 0)),
                    "observed_range": getattr(target_report, "observed_range", None),
                },
                "Regression evaluation must preserve the continuous outcome and original scale.",
            ),
            "PREP-005": (
                "feature_relationship_report + leakage_report",
                "baseline feature governance",
                {
                    "candidate_feature_count": len(features),
                    "redundancy_candidates": redundancy_count,
                    "confirmed_derived_dependencies": int(
                        getattr(leakage_report, "confirmed_derived_dependency_count", 0)
                    ),
                },
                "Exploratory structure requires validation, not global feature deletion during preparation.",
            ),
            "PREP-006": (
                "target_report",
                "continuous snapshot split contract",
                {
                    "target_unique_values": int(getattr(target_report, "unique_count", 0)),
                    "stratification": None,
                    "random_seed": random_seed,
                },
                "Use a fixed random snapshot split without manufacturing class labels from the target.",
            ),
            "PREP-007": (
                "numerical_report",
                "numerical predictor contract",
                {"numerical_feature_count": len(features)},
                "Scaling remains candidate-family dependent and train-fitted.",
            ),
            "PREP-008": (
                "regression_structure_report",
                "nonlinearity and interaction diagnostics",
                {
                    "nonlinearity_signals": nonlinearity_count,
                    "interaction_signals": interaction_count,
                },
                "Structural signals motivate later comparison but do not select a model or engineered terms.",
            ),
            "PREP-009": (
                "duplicate_report",
                "repeated-profile ambiguity",
                {
                    "repeated_profile_groups": repeated_profile_groups,
                    "target_conflict_groups": conflicting_profile_groups,
                },
                "Preserve repeated measurements and quantify their influence during validation.",
            ),
            "PREP-010": (
                "leakage_report",
                "target-isolation audit",
                {
                    "direct_target_leakage": bool(
                        getattr(leakage_report, "has_direct_target_leakage", False)
                    )
                },
                "Learned operations must never fit on held-out partitions.",
            ),
        }
    )

    if "PREP-011" in {str(item["decision_id"]) for item in decisions}:
        evidence_specs["PREP-011"] = (
            "feature_relationship_report + leakage_report",
            "redundancy and dependency review",
            {
                "redundancy_candidates": redundancy_count,
                "unconfirmed_dependencies": unconfirmed_dependencies,
            },
            "Ablation belongs inside model selection.",
        )

    evidence: list[dict[str, object]] = []
    for index, item in enumerate(decisions, start=1):
        decision_id = str(item["decision_id"])
        source_report, source_item, observed_value, interpretation = evidence_specs[
            decision_id
        ]
        evidence.append(
            {
                "evidence_id": f"PDE-{index:03d}",
                "decision_id": decision_id,
                "source_report": source_report,
                "source_item": source_item,
                "observed_value": observed_value,
                "expected_or_reference": "Stage 18 continuous-regression preparation policy",
                "interpretation": interpretation,
            }
        )

    deferred_decision_ids = tuple(
        str(item["decision_id"])
        for item in decisions
        if item["status"] == "Deferred"
    )

    before_split_ids = tuple(
        decision_id
        for decision_id in ("PREP-001", "PREP-002", "PREP-003", "PREP-004", "PREP-005")
        if decision_id in {str(item["decision_id"]) for item in decisions}
    )

    execution_steps: list[dict[str, object]] = [
        {
            "step_id": "STEP-001",
            "sequence": 1,
            "decision_ids": before_split_ids,
            "phase": "Before split",
            "action": (
                "Revalidate the raw table, apply only the declared source-type "
                "interpretation, and create a value-preserving prepared projection."
            ),
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": (
                "Source row count and numerical values are preserved and every "
                "preparation-blocking metadata conflict has an explicit resolution."
            ),
        },
        {
            "step_id": "STEP-002",
            "sequence": 2,
            "decision_ids": ("PREP-006", "PREP-010"),
            "phase": "Split",
            "action": (
                "Create reproducible non-stratified train, validation, and test "
                "partitions and record target summaries without tuning the split."
            ),
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": (
                "Partitions are disjoint, reproducible, preserve the final test "
                "holdout, and use no artificial target-class bins."
            ),
        },
        {
            "step_id": "STEP-003",
            "sequence": 3,
            "decision_ids": ("PREP-007",),
            "phase": "Train-only transformation",
            "action": "Fit model-dependent numerical preprocessing on training data only.",
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": (
                "Validation and test are transform-only and no scaler is globally fitted."
            ),
        },
        {
            "step_id": "STEP-004",
            "sequence": 4,
            "decision_ids": deferred_decision_ids,
            "phase": "Model selection",
            "action": (
                "Evaluate nonlinear structure, repeated-profile sensitivity, and any "
                "ablation candidates without changing the baseline preparation handoff."
            ),
            "blocking": False,
            "status": "Deferred",
            "temporal_dependency": False,
            "acceptance_criteria": (
                "Alternatives are adopted only from leakage-safe training/validation evidence."
            ),
        },
        {
            "step_id": "STEP-005",
            "sequence": 5,
            "decision_ids": tuple(str(item["decision_id"]) for item in decisions),
            "phase": "Model selection",
            "action": (
                "Freeze preprocessing, feature policy, target handling, and the "
                "selected regression contract before final test evaluation."
            ),
            "blocking": True,
            "status": "Deferred",
            "temporal_dependency": False,
            "acceptance_criteria": (
                "Final test data are accessed only after all model-selection choices "
                "and learned preparation choices are frozen."
            ),
        },
    ]

    def guardrail(
        guardrail_id: str,
        domain: str,
        title: str,
        affected_fields: Sequence[object],
        severity: str,
        prohibited_operation: str,
        rationale: str,
        verification: str,
    ) -> dict[str, object]:
        return {
            "guardrail_id": guardrail_id,
            "domain": domain,
            "title": title,
            "affected_fields": tuple(affected_fields),
            "severity": severity,
            "status": "Active",
            "prohibited_operation": prohibited_operation,
            "rationale": rationale,
            "verification": verification,
        }

    guardrails_list: list[dict[str, object]] = [
        guardrail(
            "GRD-001",
            "Cleaning",
            "Preserve the raw evidence",
            fields,
            "Critical",
            "Modify or overwrite the acquired raw table in place.",
            "Reproducible preparation requires immutable source evidence.",
            "Raw shape, values, index, and dtypes remain unchanged.",
        )
    ]
    if resolutions:
        guardrails_list.append(
            guardrail(
                "GRD-002",
                "Cleaning",
                "Do not coerce released decimals to satisfy source metadata",
                tuple(resolutions),
                "Critical",
                "Round, truncate, or coerce released decimal measurements to integer.",
                (
                    "The explicit source-type resolution changes analytical "
                    "interpretation, not released measurement values."
                ),
                "Prepared values remain numerically identical to the acquired source.",
            )
        )

    guardrails_list.extend(
        [
            guardrail(
                "GRD-003",
                "Cleaning",
                "Do not deduplicate from row equality alone",
                fields,
                "High",
                "Drop exact matches without independent observation identity evidence.",
                "The released table does not provide a source observation identifier.",
                "No source row is removed solely because released values match another row.",
            ),
            guardrail(
                "GRD-004",
                "Cleaning",
                "Do not apply generic outlier cleaning",
                features + (target_name,),
                "High",
                "Delete, clip, winsorize, or replace values solely from an IQR flag.",
                "IQR and target-extreme flags are descriptive, not validity rules.",
                "The baseline prepared projection preserves every domain-valid numerical value.",
            ),
            guardrail(
                "GRD-005",
                "Target governance",
                "Keep the continuous target outside predictors",
                (target_name,),
                "Critical",
                "Include the target or a direct numerical derivative in X.",
                "This would constitute direct target leakage.",
                "Predictor matrices contain exactly candidate features and never the target.",
            ),
            guardrail(
                "GRD-006",
                "Target governance",
                "Do not manufacture target classes for preparation",
                (target_name,),
                "High",
                "Discretize the continuous target into bins as a replacement prediction target.",
                "The declared problem is continuous regression on the original measurement scale.",
                "Prepared y remains numeric, continuous, and unbinned.",
            ),
            guardrail(
                "GRD-007",
                "Leakage governance",
                "Fit learned preprocessing on training only",
                features,
                "Critical",
                (
                    "Fit scaling, selection, target transformation, engineered terms, "
                    "or other learned transformations before splitting or on held-out data."
                ),
                "Held-out data must not influence learned preparation parameters.",
                "Every learned transformer records train-only fit scope.",
            ),
            guardrail(
                "GRD-008",
                "Feature engineering",
                "Do not select features globally",
                features,
                "Critical",
                (
                    "Use full-data target associations, nonlinearity diagnostics, "
                    "interaction diagnostics, or redundancy rankings to choose the final feature set."
                ),
                "Global selection would bias held-out evaluation.",
                "Ablation and selection use training/validation evidence only.",
            ),
            guardrail(
                "GRD-009",
                "Dataset splitting",
                "Protect the final test holdout",
                (target_name,),
                "Critical",
                (
                    "Use final-test metrics for model, feature, transformation, "
                    "interaction, or hyperparameter selection."
                ),
                "Repeated test access converts the final holdout into validation data.",
                "Final test evaluation occurs only after the analysis contract is frozen.",
            ),
        ]
    )

    split_policy = {
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "stratify_by": None,
        "random_seed": random_seed,
        "shuffle": True,
        "temporal_priority": True,
        "temporal_policy_status": "Resolved snapshot fallback",
        "random_split_fallback": (
            "Approved for this source-released static continuous-regression snapshot: "
            "no chronological observation field is available and no artificial target "
            "bins are introduced solely for stratification."
        ),
        "test_holdout_untouched": True,
        "disjoint_partitions_required": True,
        "group_by_identifiers": id_columns,
    }

    report = record_preparation_decisions(
        available_fields=fields,
        decisions=decisions,
        evidence=evidence,
        execution_steps=execution_steps,
        guardrails=guardrails_list,
        split_policy=split_policy,
    )
    report.raise_if_invalid()
    return report


def _normalize_decisions(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        fallback = f"decision[{position}]"
        decision_id = _text(declaration.get("decision_id")) or fallback
        if decision_id in seen:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Duplicate decision ID",
                    f"Decision ID {decision_id!r} is declared more than once",
                    "Evidence and execution steps cannot be linked deterministically",
                )
            )
        seen.add(decision_id)

        fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Decision",
                        decision_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The decision may target a non-existent variable",
                    )
                )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_STATUSES:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid decision status",
                    f"Unsupported decision status {status!r}",
                    "Readiness cannot be determined reliably",
                )
            )

        domain = _text(declaration.get("domain"))
        if domain not in _ALLOWED_DOMAINS:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid decision domain",
                    f"Unsupported decision domain {domain!r}",
                    "Scope summaries become unreliable",
                )
            )

        phase = _text(declaration.get("phase"))
        if phase not in _ALLOWED_PHASES:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid decision phase",
                    f"Unsupported decision phase {phase!r}",
                    "Execution ordering becomes ambiguous",
                )
            )

        fit_scope = _text(declaration.get("fit_scope"))
        if fit_scope not in _ALLOWED_FIT_SCOPES:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Invalid fit scope",
                    f"Unsupported fit scope {fit_scope!r}",
                    "Train/test isolation cannot be audited",
                )
            )

        acceptance = _text(declaration.get("acceptance_criteria"))
        if not acceptance:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Missing decision acceptance criteria",
                    f"Decision {decision_id!r} has no measurable acceptance criteria",
                    "Implementation cannot be verified",
                )
            )

        prerequisites = _tuple_values(declaration.get("prerequisites", ()))
        rows.append(
            {
                "Decision ID": decision_id,
                "Domain": domain,
                "Title": _text(declaration.get("title")),
                "Affected fields": fields,
                "Affected field count": len(fields),
                "Status": status,
                "Phase": phase,
                "Fit scope": fit_scope,
                "Operation": _text(declaration.get("operation")),
                "Rationale": _text(declaration.get("rationale")),
                "Prerequisites": prerequisites,
                "Prerequisite count": len(prerequisites),
                "Acceptance criteria": acceptance,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Evidence count": 0,
                "Execution step count": 0,
            }
        )

    return rows


def _normalize_evidence(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        fallback = f"evidence[{position}]"
        evidence_id = _text(declaration.get("evidence_id")) or fallback
        if evidence_id in seen:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Duplicate evidence ID",
                    f"Evidence ID {evidence_id!r} is declared more than once",
                    "Decision support cannot be traced deterministically",
                )
            )
        seen.add(evidence_id)

        rows.append(
            {
                "Evidence ID": evidence_id,
                "Decision ID": _text(declaration.get("decision_id")),
                "Source report": _text(declaration.get("source_report")),
                "Source item": _text(declaration.get("source_item")),
                "Observed value": deepcopy(declaration.get("observed_value")),
                "Expected or reference": deepcopy(
                    declaration.get("expected_or_reference")
                ),
                "Interpretation": _text(declaration.get("interpretation")),
            }
        )

    return rows


def _normalize_steps(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        fallback = f"step[{position}]"
        step_id = _text(declaration.get("step_id")) or fallback
        if step_id in seen:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Duplicate step ID",
                    f"Step ID {step_id!r} is declared more than once",
                    "Execution order cannot be traced deterministically",
                )
            )
        seen.add(step_id)

        phase = _text(declaration.get("phase"))
        if phase not in _ALLOWED_PHASES:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Invalid step phase",
                    f"Unsupported execution phase {phase!r}",
                    "Execution ordering becomes ambiguous",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_STEP_STATUSES:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Invalid step status",
                    f"Unsupported execution-step status {status!r}",
                    "Readiness cannot be determined reliably",
                )
            )

        acceptance = _text(declaration.get("acceptance_criteria"))
        if not acceptance:
            issues.append(
                _issue(
                    "Execution step",
                    step_id,
                    "Missing step acceptance criteria",
                    f"Execution step {step_id!r} has no acceptance criteria",
                    "Future implementation cannot be verified",
                )
            )

        decision_ids = _tuple_values(declaration.get("decision_ids", ()))
        rows.append(
            {
                "Step ID": step_id,
                "Sequence": _integer(declaration.get("sequence"), default=position + 1),
                "Decision IDs": decision_ids,
                "Decision count": len(decision_ids),
                "Phase": phase,
                "Action": _text(declaration.get("action")),
                "Blocking": bool(declaration.get("blocking", False)),
                "Status": status,
                "Temporal dependency": bool(
                    declaration.get("temporal_dependency", False)
                ),
                "Acceptance criteria": acceptance,
            }
        )

    return rows


def _normalize_guardrails(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        fallback = f"guardrail[{position}]"
        guardrail_id = _text(declaration.get("guardrail_id")) or fallback
        if guardrail_id in seen:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Duplicate guardrail ID",
                    f"Guardrail ID {guardrail_id!r} is declared more than once",
                    "Prohibition coverage cannot be traced deterministically",
                )
            )
        seen.add(guardrail_id)

        fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Guardrail",
                        guardrail_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The guardrail may not protect the intended variable",
                    )
                )

        domain = _text(declaration.get("domain"))
        if domain not in _ALLOWED_DOMAINS:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Invalid guardrail domain",
                    f"Unsupported guardrail domain {domain!r}",
                    "Guardrail summaries become unreliable",
                )
            )

        severity = _text(declaration.get("severity"))
        if severity not in _ALLOWED_GUARDRAIL_SEVERITIES:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Invalid guardrail severity",
                    f"Unsupported guardrail severity {severity!r}",
                    "Risk priority cannot be interpreted",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_GUARDRAIL_STATUSES:
            issues.append(
                _issue(
                    "Guardrail",
                    guardrail_id,
                    "Invalid guardrail status",
                    f"Unsupported guardrail status {status!r}",
                    "Protection state cannot be interpreted",
                )
            )

        rows.append(
            {
                "Guardrail ID": guardrail_id,
                "Domain": domain,
                "Title": _text(declaration.get("title")),
                "Affected fields": fields,
                "Affected field count": len(fields),
                "Severity": severity,
                "Status": status,
                "Prohibited operation": _text(
                    declaration.get("prohibited_operation")
                ),
                "Rationale": _text(declaration.get("rationale")),
                "Verification": _text(declaration.get("verification")),
            }
        )

    return rows


def _validate_references_and_coverage(
    *,
    decision_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    decision_ids: set[str],
    issues: list[dict[str, object]],
) -> None:
    evidence_links: set[str] = set()
    for row in evidence_rows:
        decision_id = str(row["Decision ID"])
        if decision_id not in decision_ids:
            issues.append(
                _issue(
                    "Evidence",
                    str(row["Evidence ID"]),
                    "Unknown decision reference",
                    f"Decision ID {decision_id!r} does not exist",
                    "Evidence cannot support a declared decision",
                )
            )
        else:
            evidence_links.add(decision_id)

    for row in decision_rows:
        decision_id = str(row["Decision ID"])
        if decision_id not in evidence_links:
            issues.append(
                _issue(
                    "Decision",
                    decision_id,
                    "Decision without evidence",
                    f"Decision {decision_id!r} has no linked evidence record",
                    "The decision lacks traceable support",
                )
            )

        for prerequisite in row["Prerequisites"]:
            key = str(prerequisite)
            if key not in decision_ids:
                issues.append(
                    _issue(
                        "Decision",
                        decision_id,
                        "Unknown prerequisite reference",
                        f"Prerequisite decision {key!r} does not exist",
                        "Decision dependency order cannot be validated",
                    )
                )

    for row in step_rows:
        for decision_id in row["Decision IDs"]:
            key = str(decision_id)
            if key not in decision_ids:
                issues.append(
                    _issue(
                        "Execution step",
                        str(row["Step ID"]),
                        "Unknown decision reference",
                        f"Decision ID {key!r} does not exist",
                        "Execution step cannot be traced to a decision",
                    )
                )


def _validate_split_policy(
    split_policy: Mapping[str, object],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> None:
    missing = [key for key in _SPLIT_REQUIRED_KEYS if key not in split_policy]
    if missing:
        issues.append(
            _issue(
                "Split policy",
                "policy",
                "Incomplete split policy",
                f"Missing required split-policy keys: {tuple(missing)!r}",
                "Partition behavior cannot be reproduced",
            )
        )

    fractions: list[float] = []
    for key in ("train_fraction", "validation_fraction", "test_fraction"):
        value = split_policy.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = -1.0
        if not 0.0 < numeric < 1.0:
            issues.append(
                _issue(
                    "Split policy",
                    key,
                    "Invalid split proportion",
                    f"{key} must be a number strictly between 0 and 1",
                    "A valid train/validation/test partition cannot be formed",
                )
            )
        fractions.append(numeric)

    if all(value > 0.0 for value in fractions) and abs(sum(fractions) - 1.0) > 1e-9:
        issues.append(
            _issue(
                "Split policy",
                "fractions",
                "Invalid split total",
                f"Split fractions sum to {sum(fractions):.12g}, not 1.0",
                "Rows may be omitted or assigned more than once",
            )
        )

    raw_stratify_by = split_policy.get("stratify_by")
    stratify_by = _text(raw_stratify_by)
    if stratify_by and stratify_by not in set(available_fields):
        issues.append(
            _issue(
                "Split policy",
                "stratify_by",
                "Unknown stratification field",
                f"Stratification field {stratify_by!r} is not available",
                "Requested stratification cannot be reproduced",
            )
        )

    seed = split_policy.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        issues.append(
            _issue(
                "Split policy",
                "random_seed",
                "Invalid random seed",
                "Random seed must be an integer",
                "Random fallback cannot be reproduced",
            )
        )

    temporal_status = _text(split_policy.get("temporal_policy_status"))
    if temporal_status not in _ALLOWED_TEMPORAL_POLICY_STATUSES:
        issues.append(
            _issue(
                "Split policy",
                "temporal_policy_status",
                "Invalid temporal policy status",
                f"Unsupported temporal policy status {temporal_status!r}",
                "Temporal precedence cannot be evaluated",
            )
        )

    if not _text(split_policy.get("random_split_fallback")):
        issues.append(
            _issue(
                "Split policy",
                "random_split_fallback",
                "Missing random-split fallback",
                "The snapshot fallback rule is empty",
                "Random splitting may be used without source justification",
            )
        )

    if split_policy.get("temporal_priority") is not True:
        issues.append(
            _issue(
                "Split policy",
                "temporal_priority",
                "Invalid temporal priority",
                "Temporal split must explicitly take precedence when valid timing exists",
                "Evaluation may fail to represent future inference",
            )
        )

    if split_policy.get("test_holdout_untouched") is not True:
        issues.append(
            _issue(
                "Split policy",
                "test_holdout_untouched",
                "Invalid test holdout contract",
                "The final test partition must remain untouched until final evaluation",
                "Model selection may leak into final evaluation",
            )
        )

    if split_policy.get("disjoint_partitions_required") is not True:
        issues.append(
            _issue(
                "Split policy",
                "disjoint_partitions_required",
                "Invalid disjoint partition contract",
                "Train, validation, and test partitions must be disjoint",
                "The same observation may appear in multiple partitions",
            )
        )

    identifiers = _tuple_values(split_policy.get("group_by_identifiers", ()))
    for identifier in identifiers:
        if identifier not in set(available_fields):
            issues.append(
                _issue(
                    "Split policy",
                    "group_by_identifiers",
                    "Unknown grouping identifier",
                    f"Grouping identifier {identifier!r} is not available",
                    "Entity overlap cannot be checked reliably",
                )
            )



def _defensive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame copy with independently copied object values."""
    result = frame.copy(deep=True)
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(deepcopy)
    return result

def _split_interpretation(
    key: str,
    value: object,
    temporal_status: str,
) -> str:
    interpretations = {
        "train_fraction": "Provisional share allocated to model fitting",
        "validation_fraction": "Provisional share reserved for model selection",
        "test_fraction": "Final holdout share reserved for one-time evaluation",
        "stratify_by": (
            "No stratification is applied in the random fallback"
            if not _text(value)
            else "Field used to preserve representation in the random fallback"
        ),
        "random_seed": "Seed used only for reproducible random fallback",
        "shuffle": "Random fallback shuffles rows before partitioning",
        "temporal_priority": "Chronological partitioning overrides random fallback when timing is valid",
        "temporal_policy_status": (
            "Temporal nature is unresolved and split execution remains blocked"
            if temporal_status == "Unresolved"
            else "Temporal partition strategy has an explicit resolution"
        ),
        "random_split_fallback": "Condition required before using the provisional random split",
        "test_holdout_untouched": "The final test set is excluded from all fitting and selection",
        "disjoint_partitions_required": "Partition indices and identifiers may not overlap",
        "group_by_identifiers": "Entity identifiers used to verify cross-partition isolation",
    }
    return interpretations.get(key, f"Declared value: {value!r}")


def _issue(
    scope: str,
    item: str,
    issue: str,
    details: str,
    potential_impact: str,
) -> dict[str, object]:
    return {
        "Scope": scope,
        "Item": item,
        "Issue": issue,
        "Details": details,
        "Potential impact": potential_impact,
    }


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tuple_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError:
        values = [value]
    return tuple(
        text
        for text in (_text(item) for item in values)
        if text
    )


def _unique_text_tuple(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _integer(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
