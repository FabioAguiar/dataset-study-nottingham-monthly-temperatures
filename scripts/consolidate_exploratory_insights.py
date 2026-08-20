"""Reusable, non-mutating consolidation of key exploratory insights."""

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

_INSIGHT_COLUMNS: Final[list[str]] = [
    "Insight ID",
    "Theme",
    "Title",
    "Insight type",
    "Affected fields",
    "Affected field count",
    "Relevance",
    "Status",
    "Source stages",
    "Summary",
    "Modeling implication",
    "Interpretation boundary",
    "Evidence count",
    "Hypothesis count",
]

_EVIDENCE_COLUMNS: Final[list[str]] = [
    "Evidence ID",
    "Insight ID",
    "Evidence kind",
    "Source report",
    "Source metric",
    "Observed value",
    "Comparison value",
    "Direction",
    "Interpretation",
]

_HYPOTHESIS_COLUMNS: Final[list[str]] = [
    "Hypothesis ID",
    "Linked insight IDs",
    "Linked insight count",
    "Title",
    "Hypothesis",
    "Status",
    "Confounding risks",
    "Confounding risk count",
    "Required validation",
    "Decision stage",
    "Validation action count",
]

_ACTION_COLUMNS: Final[list[str]] = [
    "Action ID",
    "Hypothesis IDs",
    "Hypothesis count",
    "Validation type",
    "Action",
    "Stage",
    "Blocking",
    "Status",
    "Acceptance criteria",
]

_LIMITATION_COLUMNS: Final[list[str]] = [
    "Limitation ID",
    "Theme",
    "Title",
    "Limitation type",
    "Affected fields",
    "Affected field count",
    "Severity",
    "Status",
    "Source stages",
    "Implication",
    "Required resolution",
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

_ALLOWED_INSIGHT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Pattern",
        "Contrast",
        "Dependency",
        "Data-quality condition",
        "Modeling limitation",
        "Governance limitation",
    }
)

_ALLOWED_RELEVANCE: Final[frozenset[str]] = frozenset(
    {
        "High",
        "Medium",
        "Low",
        "Contextual",
    }
)

_ALLOWED_INSIGHT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Observed",
        "Exploratory",
        "Controlled",
        "Unresolved",
    }
)

_ALLOWED_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "Data quality",
        "Target distribution",
        "Numerical pattern",
        "Categorical contrast",
        "Feature dependency",
        "Feature-to-target",
        "Leakage/governance",
        "Regression structure",
    }
)

_ALLOWED_HYPOTHESIS_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Unvalidated",
        "Partially validated",
        "Validated",
        "Rejected",
        "Deferred",
    }
)

_ALLOWED_DECISION_STAGES: Final[frozenset[str]] = frozenset(
    {
        "Data preparation",
        "Model selection",
        "Model evaluation",
        "External contract",
    }
)

_ALLOWED_VALIDATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Interaction test",
        "Ablation",
        "Cross-validation",
        "Calibration",
        "Confounder control",
        "Error analysis",
        "External data",
        "Temporal validation",
    }
)

_ALLOWED_ACTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Planned",
        "In progress",
        "Complete",
        "Blocked",
        "Deferred",
    }
)

_ALLOWED_LIMITATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Data sufficiency",
        "Modeling",
        "Governance",
        "Temporal",
        "Inference availability",
        "Data quality",
    }
)

_ALLOWED_LIMITATION_SEVERITIES: Final[frozenset[str]] = frozenset(
    {
        "Critical",
        "High",
        "Medium",
        "Low",
        "Contextual",
    }
)

_ALLOWED_LIMITATION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Open",
        "Controlled",
        "Accepted",
        "Resolved",
        "Unresolved",
    }
)

_RELEVANCE_ORDER: Final[dict[str, int]] = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
    "Contextual": 3,
}

_HYPOTHESIS_STATUS_ORDER: Final[dict[str, int]] = {
    "Unvalidated": 0,
    "Partially validated": 1,
    "Deferred": 2,
    "Validated": 3,
    "Rejected": 4,
}

_VALIDATION_TYPE_ORDER: Final[dict[str, int]] = {
    "Interaction test": 0,
    "Ablation": 1,
    "Confounder control": 2,
    "Cross-validation": 3,
    "Calibration": 4,
    "Error analysis": 5,
    "Temporal validation": 6,
    "External data": 7,
}

_LIMITATION_SEVERITY_ORDER: Final[dict[str, int]] = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
    "Contextual": 4,
}

_UNRESOLVED_LIMITATION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Open",
        "Unresolved",
    }
)

_GOVERNANCE_LIMITATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Governance",
        "Temporal",
        "Inference availability",
    }
)

_TEMPORAL_LIMITATION_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Temporal",
        "Inference availability",
    }
)


class ExploratoryInsightsConsolidationError(ValueError):
    """Raised when insight declarations or readiness requirements fail."""


@dataclass(frozen=True, slots=True)
class KeyExploratoryInsightsReport:
    """Consolidate exploratory evidence without recomputing analysis."""

    available_fields: tuple[str, ...]
    insights: pd.DataFrame
    evidence: pd.DataFrame
    hypotheses: pd.DataFrame
    validation_actions: pd.DataFrame
    limitations: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_high_relevance_insights(self) -> bool:
        """Return whether at least one insight has high relevance."""
        if self.insights.empty:
            return False
        return bool(self.insights["Relevance"].eq("High").any())

    @property
    def has_unvalidated_hypotheses(self) -> bool:
        """Return whether hypotheses still require validation."""
        if self.hypotheses.empty:
            return False
        return bool(
            self.hypotheses["Status"].isin(
                {"Unvalidated", "Partially validated", "Deferred"}
            ).any()
        )

    @property
    def has_confounding_risks(self) -> bool:
        """Return whether hypotheses declare possible confounders."""
        if self.hypotheses.empty:
            return False
        return bool(self.hypotheses["Confounding risk count"].gt(0).any())

    @property
    def has_structural_dependencies(self) -> bool:
        """Return whether dependency insights are documented."""
        if self.insights.empty:
            return False
        return bool(self.insights["Insight type"].eq("Dependency").any())

    @property
    def has_modeling_limitations(self) -> bool:
        """Return whether modeling limitations are documented."""
        insight_limit = (
            not self.insights.empty
            and self.insights["Insight type"].eq("Modeling limitation").any()
        )
        declared_limit = (
            not self.limitations.empty
            and self.limitations["Limitation type"].eq("Modeling").any()
        )
        return bool(insight_limit or declared_limit)

    @property
    def has_unresolved_governance_limits(self) -> bool:
        """Return whether governance, timing, or availability remains open."""
        insight_limit = False
        if not self.insights.empty:
            insight_limit = bool(
                (
                    self.insights["Insight type"].eq("Governance limitation")
                    & self.insights["Status"].eq("Unresolved")
                ).any()
            )

        declared_limit = False
        if not self.limitations.empty:
            declared_limit = bool(
                (
                    self.limitations["Limitation type"].isin(
                        _GOVERNANCE_LIMITATION_TYPES
                    )
                    & self.limitations["Status"].isin(
                        _UNRESOLVED_LIMITATION_STATUSES
                    )
                ).any()
            )

        return insight_limit or declared_limit

    @property
    def has_unresolved_temporal_limits(self) -> bool:
        """Return whether temporal or inference availability remains open."""
        if self.limitations.empty:
            return False
        return bool(
            (
                self.limitations["Limitation type"].isin(
                    _TEMPORAL_LIMITATION_TYPES
                )
                & self.limitations["Status"].isin(
                    _UNRESOLVED_LIMITATION_STATUSES
                )
            ).any()
        )

    @property
    def is_structurally_valid(self) -> bool:
        """Return whether identifiers, references, and values are valid."""
        return self.issues.empty

    @property
    def is_ready_for_preparation_decisions(self) -> bool:
        """Return whether insights can safely inform preparation decisions."""
        return self.is_structurally_valid

    @property
    def is_ready_for_modeling(self) -> bool:
        """Return whether no unresolved governance limitation blocks modeling."""
        return (
            self.is_ready_for_preparation_decisions
            and not self.has_unresolved_governance_limits
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic counts and readiness indicators."""
        rows = [
            {
                "Metric": "Declared fields",
                "Value": len(self.available_fields),
                "Interpretation": "Fields available to referenced insights",
            },
            {
                "Metric": "Consolidated insights",
                "Value": len(self.insights),
                "Interpretation": "Prioritized exploratory conclusions",
            },
            {
                "Metric": "High-relevance insights",
                "Value": int(
                    self.insights["Relevance"].eq("High").sum()
                )
                if not self.insights.empty
                else 0,
                "Interpretation": "Insights prioritized for downstream review",
            },
            {
                "Metric": "Evidence records",
                "Value": len(self.evidence),
                "Interpretation": "Traceable support from prior analytical stages",
            },
            {
                "Metric": "Exploratory hypotheses",
                "Value": len(self.hypotheses),
                "Interpretation": "Explanations that still require validation",
            },
            {
                "Metric": "Unvalidated hypotheses",
                "Value": int(self.has_unvalidated_hypotheses),
                "Interpretation": "At least one hypothesis remains unvalidated",
            },
            {
                "Metric": "Validation actions",
                "Value": len(self.validation_actions),
                "Interpretation": "Traceable future validation work",
            },
            {
                "Metric": "Declared limitations",
                "Value": len(self.limitations),
                "Interpretation": "Known interpretation and modeling constraints",
            },
            {
                "Metric": "Unresolved governance limits",
                "Value": int(self.has_unresolved_governance_limits),
                "Interpretation": "Open timing, governance, or inference constraints",
            },
            {
                "Metric": "Structurally valid",
                "Value": self.is_structurally_valid,
                "Interpretation": "Insight, evidence, and validation contracts are coherent",
            },
            {
                "Metric": "Ready for preparation decisions",
                "Value": self.is_ready_for_preparation_decisions,
                "Interpretation": "Insights can inform preliminary preparation scope",
            },
            {
                "Metric": "Ready for modeling",
                "Value": self.is_ready_for_modeling,
                "Interpretation": "No unresolved governance limitation remains",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def exploratory_overview_frame(self) -> pd.DataFrame:
        """Return a compact stage-17 overview without claiming model performance."""
        high_count = int(self.insights["Relevance"].eq("High").sum()) if not self.insights.empty else 0
        rows = [
            {"Metric": "Key exploratory insights", "Value": len(self.insights), "Interpretation": "Evidence-backed conclusions synthesized from prior stages"},
            {"Metric": "High-relevance insights", "Value": high_count, "Interpretation": "Insights prioritized for downstream validation"},
            {"Metric": "Exploratory hypotheses", "Value": len(self.hypotheses), "Interpretation": "Testable explanations deferred to later notebooks"},
            {"Metric": "Validation actions", "Value": len(self.validation_actions), "Interpretation": "Planned checks linked to exploratory hypotheses"},
            {"Metric": "Interpretation limitations", "Value": len(self.limitations), "Interpretation": "Boundaries that prevent overclaiming from EDA"},
            {"Metric": "Structural consolidation valid", "Value": self.is_structurally_valid, "Interpretation": "Insight, evidence, and reference contracts are coherent"},
            {"Metric": "Ready for preparation decisions", "Value": self.is_ready_for_preparation_decisions, "Interpretation": "Stage-17 evidence can inform preparation decisions"},
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def key_insights_frame(self) -> pd.DataFrame:
        """Return the concise insight view intended for notebooks."""
        columns = [
            "Insight ID", "Theme", "Title", "Relevance", "Status",
            "Summary", "Modeling implication", "Interpretation boundary",
        ]
        if self.insights.empty:
            return pd.DataFrame(columns=columns)
        return self.insights.loc[:, columns].copy(deep=True)

    def insights_frame(self) -> pd.DataFrame:
        """Return a defensive copy of consolidated insights."""
        return self.insights.copy(deep=True)

    def evidence_frame(self) -> pd.DataFrame:
        """Return a defensive copy of insight evidence."""
        return self.evidence.copy(deep=True)

    def hypotheses_frame(self) -> pd.DataFrame:
        """Return a defensive copy of exploratory hypotheses."""
        return self.hypotheses.copy(deep=True)

    def validation_actions_frame(self) -> pd.DataFrame:
        """Return a defensive copy of validation actions."""
        return self.validation_actions.copy(deep=True)

    def limitations_frame(self) -> pd.DataFrame:
        """Return a defensive copy of exploratory limitations."""
        return self.limitations.copy(deep=True)

    def readiness_frame(self) -> pd.DataFrame:
        """Return preparation and modeling readiness checks."""
        rows = [
            {
                "Readiness check": "Structural contract",
                "Ready": self.is_structurally_valid,
                "Interpretation": (
                    "Insight declarations and references are valid"
                    if self.is_structurally_valid
                    else "Structural issues must be corrected"
                ),
            },
            {
                "Readiness check": "Preparation decisions",
                "Ready": self.is_ready_for_preparation_decisions,
                "Interpretation": (
                    "Exploratory evidence can inform preliminary preparation"
                    if self.is_ready_for_preparation_decisions
                    else "Preparation decisions are not safely traceable"
                ),
            },
            {
                "Readiness check": "Modeling clearance",
                "Ready": self.is_ready_for_modeling,
                "Interpretation": (
                    "No unresolved governance limitation remains"
                    if self.is_ready_for_modeling
                    else "Governance or temporal limits still block modeling"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_READINESS_COLUMNS)

    def issues_frame(self) -> pd.DataFrame:
        """Return a defensive copy of structural consolidation issues."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_unique_insight_ids: bool = True,
        require_unique_evidence_ids: bool = True,
        require_unique_hypothesis_ids: bool = True,
        require_unique_action_ids: bool = True,
        require_unique_limitation_ids: bool = True,
        require_known_fields: bool = True,
        require_evidence_for_insights: bool = True,
        require_validation_for_hypotheses: bool = True,
        require_interpretation_boundaries: bool = True,
        require_valid_relevance: bool = True,
        require_valid_statuses: bool = True,
        require_valid_types: bool = True,
        require_valid_references: bool = True,
        require_acceptance_criteria: bool = True,
    ) -> None:
        """Raise when selected structural requirements are not satisfied."""
        selected: set[str] = set()
        if require_unique_insight_ids:
            selected.add("Duplicate insight ID")
        if require_unique_evidence_ids:
            selected.add("Duplicate evidence ID")
        if require_unique_hypothesis_ids:
            selected.add("Duplicate hypothesis ID")
        if require_unique_action_ids:
            selected.add("Duplicate action ID")
        if require_unique_limitation_ids:
            selected.add("Duplicate limitation ID")
        if require_known_fields:
            selected.add("Unknown affected field")
        if require_evidence_for_insights:
            selected.add("Insight without evidence")
        if require_validation_for_hypotheses:
            selected.update(
                {
                    "Hypothesis without linked insight",
                    "Hypothesis without validation action",
                    "Missing required validation",
                }
            )
        if require_interpretation_boundaries:
            selected.add("Missing interpretation boundary")
        if require_valid_relevance:
            selected.add("Invalid relevance")
        if require_valid_statuses:
            selected.update(
                {
                    "Invalid insight status",
                    "Invalid hypothesis status",
                    "Invalid validation status",
                    "Invalid limitation status",
                }
            )
        if require_valid_types:
            selected.update(
                {
                    "Invalid insight type",
                    "Invalid evidence kind",
                    "Invalid validation type",
                    "Invalid decision stage",
                    "Invalid limitation type",
                    "Invalid limitation severity",
                }
            )
        if require_valid_references:
            selected.update(
                {
                    "Unknown insight reference",
                    "Unknown hypothesis reference",
                }
            )
        if require_acceptance_criteria:
            selected.add("Missing acceptance criteria")

        failures = self.issues.loc[self.issues["Issue"].isin(selected)]
        if failures.empty:
            return

        details = "; ".join(
            str(value) for value in failures["Details"].tolist()
        )
        raise ExploratoryInsightsConsolidationError(
            "Invalid exploratory-insight consolidation: " + details
        )

    def raise_if_modeling_not_ready(
        self,
        *,
        require_no_unvalidated_critical_hypotheses: bool = False,
        require_temporal_contract_complete: bool = True,
        require_no_unresolved_governance_limits: bool = True,
    ) -> None:
        """Raise when selected downstream-modeling gates remain open."""
        reasons: list[str] = []

        if not self.is_structurally_valid:
            reasons.append("the structural insight contract is invalid")

        if (
            require_no_unvalidated_critical_hypotheses
            and self.has_unvalidated_hypotheses
            and self.has_high_relevance_insights
        ):
            reasons.append("high-relevance hypotheses remain unvalidated")

        if require_temporal_contract_complete and self.has_unresolved_temporal_limits:
            reasons.append("the temporal or inference-time contract is incomplete")

        if (
            require_no_unresolved_governance_limits
            and self.has_unresolved_governance_limits
        ):
            reasons.append("unresolved governance limitations remain")

        if reasons:
            raise ExploratoryInsightsConsolidationError(
                "Exploratory insights do not clear modeling: "
                + "; ".join(reasons)
            )


def consolidate_key_exploratory_insights(
    *,
    available_fields: Sequence[object],
    insights: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    hypotheses: Sequence[Mapping[str, object]],
    validation_actions: Sequence[Mapping[str, object]],
    limitations: Sequence[Mapping[str, object]],
) -> KeyExploratoryInsightsReport:
    """Normalize and validate exploratory insight declarations."""
    fields = _unique_text_tuple(available_fields)
    insight_declarations = deepcopy(list(insights))
    evidence_declarations = deepcopy(list(evidence))
    hypothesis_declarations = deepcopy(list(hypotheses))
    action_declarations = deepcopy(list(validation_actions))
    limitation_declarations = deepcopy(list(limitations))

    issues: list[dict[str, object]] = []

    insight_rows = _normalize_insights(
        insight_declarations,
        available_fields=fields,
        issues=issues,
    )
    evidence_rows = _normalize_evidence(
        evidence_declarations,
        issues=issues,
    )
    hypothesis_rows = _normalize_hypotheses(
        hypothesis_declarations,
        issues=issues,
    )
    action_rows = _normalize_actions(
        action_declarations,
        issues=issues,
    )
    limitation_rows = _normalize_limitations(
        limitation_declarations,
        available_fields=fields,
        issues=issues,
    )

    insight_ids = {str(row["Insight ID"]) for row in insight_rows}
    hypothesis_ids = {str(row["Hypothesis ID"]) for row in hypothesis_rows}

    _validate_references_and_coverage(
        insight_rows=insight_rows,
        evidence_rows=evidence_rows,
        hypothesis_rows=hypothesis_rows,
        action_rows=action_rows,
        insight_ids=insight_ids,
        hypothesis_ids=hypothesis_ids,
        issues=issues,
    )

    evidence_count_by_insight: dict[str, int] = {}
    for row in evidence_rows:
        insight_id = str(row["Insight ID"])
        evidence_count_by_insight[insight_id] = (
            evidence_count_by_insight.get(insight_id, 0) + 1
        )

    hypothesis_count_by_insight: dict[str, int] = {}
    for row in hypothesis_rows:
        for insight_id in row["Linked insight IDs"]:
            key = str(insight_id)
            hypothesis_count_by_insight[key] = (
                hypothesis_count_by_insight.get(key, 0) + 1
            )

    action_count_by_hypothesis: dict[str, int] = {}
    for row in action_rows:
        for hypothesis_id in row["Hypothesis IDs"]:
            key = str(hypothesis_id)
            action_count_by_hypothesis[key] = (
                action_count_by_hypothesis.get(key, 0) + 1
            )

    for row in insight_rows:
        insight_id = str(row["Insight ID"])
        row["Evidence count"] = evidence_count_by_insight.get(insight_id, 0)
        row["Hypothesis count"] = hypothesis_count_by_insight.get(insight_id, 0)

    for row in hypothesis_rows:
        hypothesis_id = str(row["Hypothesis ID"])
        row["Validation action count"] = action_count_by_hypothesis.get(
            hypothesis_id, 0
        )

    insights_frame = pd.DataFrame(insight_rows, columns=_INSIGHT_COLUMNS)
    if not insights_frame.empty:
        insights_frame["_order"] = insights_frame["Relevance"].map(
            _RELEVANCE_ORDER
        ).fillna(99)
        insights_frame = (
            insights_frame.sort_values(["_order", "Insight ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    evidence_frame = pd.DataFrame(evidence_rows, columns=_EVIDENCE_COLUMNS)
    if not evidence_frame.empty:
        evidence_frame = evidence_frame.sort_values(
            ["Insight ID", "Evidence ID"]
        ).reset_index(drop=True)

    hypotheses_frame = pd.DataFrame(hypothesis_rows, columns=_HYPOTHESIS_COLUMNS)
    if not hypotheses_frame.empty:
        hypotheses_frame["_order"] = hypotheses_frame["Status"].map(
            _HYPOTHESIS_STATUS_ORDER
        ).fillna(99)
        hypotheses_frame = (
            hypotheses_frame.sort_values(["_order", "Hypothesis ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    actions_frame = pd.DataFrame(action_rows, columns=_ACTION_COLUMNS)
    if not actions_frame.empty:
        actions_frame["_order"] = actions_frame["Validation type"].map(
            _VALIDATION_TYPE_ORDER
        ).fillna(99)
        actions_frame = (
            actions_frame.sort_values(["_order", "Action ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    limitations_frame = pd.DataFrame(
        limitation_rows, columns=_LIMITATION_COLUMNS
    )
    if not limitations_frame.empty:
        limitations_frame["_order"] = limitations_frame["Severity"].map(
            _LIMITATION_SEVERITY_ORDER
        ).fillna(99)
        limitations_frame = (
            limitations_frame.sort_values(["_order", "Limitation ID"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    issues_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    if not issues_frame.empty:
        issues_frame = issues_frame.sort_values(
            ["Scope", "Item", "Issue"]
        ).reset_index(drop=True)

    return KeyExploratoryInsightsReport(
        available_fields=fields,
        insights=insights_frame,
        evidence=evidence_frame,
        hypotheses=hypotheses_frame,
        validation_actions=actions_frame,
        limitations=limitations_frame,
        issues=issues_frame,
    )



def consolidate_key_exploratory_insights_from_reports(
    *,
    available_fields: Sequence[object],
    quality_report: object,
    target_report: object,
    numerical_report: object,
    feature_relationship_report: object,
    feature_target_report: object,
    class_profile_report: object,
    leakage_report: object,
) -> KeyExploratoryInsightsReport:
    """Build stage-17 insight contracts only from already-produced reports."""
    fields = _unique_text_tuple(available_fields)
    target = _text(getattr(target_report, "target", ""))

    insights: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    hypotheses: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    limitations: list[dict[str, object]] = []

    def add(
        insight_id: str,
        *,
        theme: str,
        title: str,
        insight_type: str,
        affected_fields: Sequence[object],
        relevance: str,
        status: str,
        stages: Sequence[object],
        summary: str,
        implication: str,
        boundary: str,
        evidence_kind: str,
        source_report: str,
        source_metric: str,
        observed: object,
        comparison: object = None,
        direction: str = "",
    ) -> None:
        insights.append({
            "insight_id": insight_id,
            "theme": theme,
            "title": title,
            "insight_type": insight_type,
            "affected_fields": tuple(affected_fields),
            "relevance": relevance,
            "status": status,
            "source_stages": tuple(stages),
            "summary": summary,
            "modeling_implication": implication,
            "interpretation_boundary": boundary,
        })
        evidence.append({
            "evidence_id": f"EVI-{insight_id.split('-')[-1]}",
            "insight_id": insight_id,
            "evidence_kind": evidence_kind,
            "source_report": source_report,
            "source_metric": source_metric,
            "observed_value": deepcopy(observed),
            "comparison_value": deepcopy(comparison),
            "direction": direction,
            "interpretation": summary,
        })

    # Quality is a synthesis of stage 16, not a new audit.
    findings = quality_report.findings_frame()
    blockers = quality_report.blockers_frame()
    non_issues = quality_report.validated_non_issues_frame()
    add(
        "INS-001",
        theme="Data quality",
        title="Structural data quality supports controlled preparation" if blockers.empty else "Data-quality conditions require targeted review",
        insight_type="Data-quality condition",
        affected_fields=fields,
        relevance="High",
        status="Observed" if blockers.empty else "Unresolved",
        stages=("7", "8", "9", "16"),
        summary=f"Stage 16 consolidated {len(findings)} findings, {len(blockers)} blockers, and {len(non_issues)} validated non-issues.",
        implication="Restrict preparation to evidence-backed actions and preserve conditions already validated as non-issues.",
        boundary="Structural data quality does not establish predictive performance.",
        evidence_kind="Data quality",
        source_report="quality_findings_report",
        source_metric="Findings, blockers, validated non-issues",
        observed={"findings": len(findings), "blockers": len(blockers), "validated_non_issues": len(non_issues)},
        comparison={"blockers": 0},
        direction="Controlled" if blockers.empty else "Review required",
    )

    # Target support.
    ratio = getattr(target_report, "imbalance_ratio", None)
    entropy = getattr(target_report, "normalized_class_entropy", None)
    unequal = ratio is not None and float(ratio) > 1.05
    add(
        "INS-002",
        theme="Target distribution",
        title="Multiclass support is unequal across target classes" if unequal else "Multiclass support is broadly even across target classes",
        insight_type="Pattern",
        affected_fields=(target,) if target else (),
        relevance="High",
        status="Observed",
        stages=("10",),
        summary=(
            f"The target contains {getattr(target_report, 'class_count', 0)} classes; "
            f"majority={tuple(getattr(target_report, 'majority_classes', ())) or 'n/a'}, "
            f"minority={tuple(getattr(target_report, 'minority_classes', ())) or 'n/a'}, "
            f"majority/minority ratio={_format_metric(ratio)}, normalized entropy={_format_metric(entropy)}."
        ),
        implication="Use stratified partitions and macro/per-class metrics; do not infer a resampling requirement from class counts alone.",
        boundary="Class frequency describes support, not class difficulty or model bias.",
        evidence_kind="Target distribution",
        source_report="target_report",
        source_metric="Class support and normalized entropy",
        observed={"imbalance_ratio": ratio, "normalized_entropy": entropy},
        comparison={"uniform_entropy": 1.0},
        direction="Unequal support" if unequal else "Broadly balanced",
    )

    # Distribution extremes only when observed.
    outlier_features = tuple(getattr(numerical_report, "features_with_outliers", ()))
    if outlier_features:
        outlier_frame = numerical_report.outlier_summary_frame()
        candidate_count = int(outlier_frame["Outlier count"].sum()) if "Outlier count" in outlier_frame else None
        add(
            "INS-003",
            theme="Numerical distributions",
            title="IQR candidates occur in numerical feature distributions",
            insight_type="Pattern",
            affected_fields=outlier_features,
            relevance="Medium",
            status="Observed",
            stages=("11", "16"),
            summary=f"{len(outlier_features)} features contain IQR review candidates (candidate flags={candidate_count if candidate_count is not None else 'n/a'}).",
            implication="Keep raw observations and evaluate robustness or transformations only inside validated pipelines.",
            boundary="IQR candidates are statistical extremes, not proven measurement errors.",
            evidence_kind="Numerical pattern",
            source_report="numerical_report",
            source_metric="IQR outlier candidates",
            observed={"features": outlier_features, "candidate_flags": candidate_count},
            comparison=0,
            direction="Review only",
        )

    # Redundancy + derived dependencies.
    review_pairs = feature_relationship_report.numerical_review_frame()
    redundant = (
        review_pairs.loc[review_pairs["Potential redundancy"]].reset_index(drop=True)
        if not review_pairs.empty and "Potential redundancy" in review_pairs
        else pd.DataFrame()
    )
    confirmed = int(getattr(leakage_report, "confirmed_derived_dependency_count", 0))
    if not redundant.empty or confirmed:
        strongest_pair: tuple[str, str] = ()
        strongest_value: object = None
        if not review_pairs.empty:
            first = review_pairs.iloc[0]
            strongest_pair = (_text(first.get("Feature A")), _text(first.get("Feature B")))
            strongest_value = first.get("Maximum absolute association")
        dep_frame = leakage_report.dependency_frame()
        derived_fields: tuple[str, ...] = ()
        if not dep_frame.empty and "Dependency status" in dep_frame:
            selected = dep_frame.loc[dep_frame["Dependency status"].eq("Confirmed from retained columns")]
            if "Derived feature" in selected:
                derived_fields = tuple(selected["Derived feature"].astype(str))
        add(
            "INS-004",
            theme="Feature dependency",
            title="Strong associations and derived measurements create structural redundancy",
            insight_type="Dependency",
            affected_fields=_unique_text_tuple((*strongest_pair, *derived_fields)),
            relevance="High",
            status="Observed",
            stages=("12", "15"),
            summary=(
                f"{len(redundant)} feature pairs meet the redundancy-review threshold and "
                f"{confirmed} derived dependencies are numerically confirmed. "
                f"Strongest reviewed pair={strongest_pair or 'n/a'} ({_format_metric(strongest_value)})."
            ),
            implication="Use the complete validated feature set as a baseline, then compare regularized and ablated variants inside validation.",
            boundary="Redundancy does not by itself justify feature removal before the split.",
            evidence_kind="Feature dependency",
            source_report="feature_relationship_report + leakage_report",
            source_metric="Redundancy-review pairs and confirmed dependencies",
            observed={"redundancy_pairs": len(redundant), "confirmed_dependencies": confirmed, "strongest_pair": strongest_pair, "strongest_association": strongest_value},
            comparison={"redundancy_threshold": getattr(feature_relationship_report, "redundancy_review_threshold", None)},
            direction="Structural redundancy",
        )
        hypotheses.append({
            "hypothesis_id": "HYP-001",
            "linked_insight_ids": ("INS-004",),
            "title": "Some redundant features may add limited incremental value",
            "hypothesis": "A reduced or regularized feature set may match the all-feature baseline without degrading multiclass performance.",
            "status": "Unvalidated",
            "confounding_risks": (),
            "required_validation": "Compare all-feature, regularized, and ablated pipelines under the same cross-validation protocol.",
            "decision_stage": "Model selection",
        })
        actions.append({
            "action_id": "VAL-001",
            "hypothesis_ids": ("HYP-001",),
            "validation_type": "Ablation",
            "action": "Compare the full feature set with dependency-aware ablation variants inside training folds.",
            "stage": "Model selection",
            "blocking": False,
            "status": "Planned",
            "acceptance_criteria": "Any removal is supported by stable held-out multiclass metrics across folds.",
        })

    # Univariate relationship with target.
    rel = feature_target_report.relationships_frame()
    if not rel.empty:
        first = rel.iloc[0]
        top_feature = _text(first.get("Feature"))
        top_association = first.get("Maximum association")
        review_count = int(rel["Review flag"].sum()) if "Review flag" in rel else 0
        add(
            "INS-005",
            theme="Feature-to-target association",
            title="Morphological features show measurable univariate multiclass association",
            insight_type="Pattern",
            affected_fields=tuple(feature_target_report.requested_features) + ((target,) if target else ()),
            relevance="High",
            status="Observed",
            stages=("13",),
            summary=f"{review_count} features meet the exploratory association threshold; strongest={top_feature or 'n/a'} ({_format_metric(top_association)}).",
            implication="Retain the validated feature set for the baseline and evaluate incremental contribution jointly during model selection.",
            boundary="Univariate association is not causality, incremental importance, or held-out predictive performance.",
            evidence_kind="Feature-to-target",
            source_report="feature_target_report",
            source_metric="Maximum univariate multiclass association",
            observed={"review_candidates": review_count, "top_feature": top_feature, "top_association": top_association},
            comparison={"review_threshold": getattr(feature_target_report, "association_review_threshold", None)},
            direction="Class-associated",
        )

    # Multivariate class overlap.
    pairs = class_profile_report.pairwise_overlap_frame()
    if not pairs.empty:
        first = pairs.iloc[0]
        pair = (_text(first.get("Class A")), _text(first.get("Class B")))
        overlap = first.get("Mean IQR overlap coefficient")
        gap = first.get("RMS robust median gap")
        pca_variance = sum(float(v) for v in getattr(class_profile_report, "pca_explained_variance_ratio", (0.0, 0.0)))
        add(
            "INS-006",
            theme="Class separation",
            title="Some class pairs retain substantial central-profile overlap",
            insight_type="Contrast",
            affected_fields=tuple(feature_target_report.requested_features) + ((target,) if target else ()),
            relevance="High",
            status="Observed",
            stages=("14",),
            summary=f"Greatest central overlap={pair[0]} vs {pair[1]} (mean IQR overlap={_format_metric(overlap)}, robust median gap={_format_metric(gap)}); PCA-2D coverage={_format_percent(pca_variance)}.",
            implication="Evaluate confusion matrices, macro metrics, and per-class recall because class-specific difficulty may differ.",
            boundary="Profile overlap and PCA proximity do not estimate classifier confusion.",
            evidence_kind="Feature-to-target",
            source_report="class_profile_report",
            source_metric="Greatest class-IQR overlap and PCA coverage",
            observed={"pair": pair, "mean_iqr_overlap": overlap, "robust_median_gap": gap, "pca_two_component_variance": pca_variance},
            direction="Overlap",
        )
        hypotheses.append({
            "hypothesis_id": "HYP-002",
            "linked_insight_ids": ("INS-006",),
            "title": "High-overlap class pairs may concentrate predictive errors",
            "hypothesis": "Pairs with greater exploratory profile overlap may show more mutual confusion after modeling.",
            "status": "Unvalidated",
            "confounding_risks": ("Model family", "Decision boundary"),
            "required_validation": "Inspect repeated-validation confusion matrices and per-class precision/recall for the highest-overlap pairs.",
            "decision_stage": "Model evaluation",
        })
        actions.append({
            "action_id": "VAL-002",
            "hypothesis_ids": ("HYP-002",),
            "validation_type": "Error analysis",
            "action": "Compare held-out model confusion with the exploratory class-pair overlap ranking from stage 14.",
            "stage": "Model evaluation",
            "blocking": False,
            "status": "Planned",
            "acceptance_criteria": "Any class-confusion claim is supported by held-out confusion matrices and per-class metrics.",
        })

    # Target leakage / governance.
    direct_leakage = bool(getattr(leakage_report, "has_direct_target_leakage", False))
    proxy_count = len(leakage_report.target_proxy_candidates_frame())
    add(
        "INS-007",
        theme="Leakage governance",
        title="Direct target leakage requires resolution" if direct_leakage else "No direct target leakage was detected in candidate features",
        insight_type="Governance limitation" if direct_leakage else "Data-quality condition",
        affected_fields=tuple(getattr(leakage_report, "candidate_features", ())) + ((target,) if target else ()),
        relevance="High",
        status="Unresolved" if direct_leakage else "Controlled",
        stages=("15", "16"),
        summary=f"Direct target leakage={direct_leakage}; target proxy candidates={proxy_count}; confirmed non-target derived dependencies={getattr(leakage_report, 'confirmed_derived_dependency_count', 0)}.",
        implication="Resolve target-derived proxies before modeling." if direct_leakage else "Keep target semantics isolated and preserve leakage-safe fitting rules.",
        boundary="Absence of direct proxies does not replace train/validation isolation for learned transformations.",
        evidence_kind="Leakage/governance",
        source_report="leakage_report",
        source_metric="Direct target leakage audit",
        observed={"direct_target_leakage": direct_leakage, "target_proxy_candidates": proxy_count},
        comparison={"direct_target_leakage": False},
        direction="Blocked" if direct_leakage else "Controlled",
    )

    limitations.append({
        "limitation_id": "LIM-001",
        "theme": "Exploratory interpretation",
        "title": "EDA associations do not establish classifier performance",
        "limitation_type": "Modeling",
        "affected_fields": tuple(feature_target_report.requested_features) + ((target,) if target else ()),
        "severity": "Contextual",
        "status": "Accepted",
        "source_stages": ("12", "13", "14", "17"),
        "implication": "Correlation, eta squared, overlap, and PCA patterns cannot be reported as out-of-sample predictive performance.",
        "required_resolution": "Validate predictive claims with leakage-safe cross-validation and held-out multiclass metrics.",
    })

    return consolidate_key_exploratory_insights(
        available_fields=fields,
        insights=insights,
        evidence=evidence,
        hypotheses=hypotheses,
        validation_actions=actions,
        limitations=limitations,
    )



def consolidate_continuous_regression_key_exploratory_insights_from_reports(
    *,
    available_fields: Sequence[object],
    quality_report: object,
    target_report: object,
    numerical_report: object,
    feature_relationship_report: object,
    feature_target_report: object,
    regression_structure_report: object,
    leakage_report: object,
) -> KeyExploratoryInsightsReport:
    """Build stage-17 continuous-regression insights from prior reports only.

    The function consolidates evidence already produced by stages 10-16. It
    does not refit models, recompute correlations, alter observations, or turn
    exploratory thresholds into preparation decisions.
    """
    fields = _unique_text_tuple(available_fields)
    target = _text(
        getattr(
            target_report,
            "target",
            getattr(feature_target_report, "target_name", ""),
        )
    )

    insights: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    hypotheses: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    limitations: list[dict[str, object]] = []

    def add(
        insight_id: str,
        *,
        theme: str,
        title: str,
        insight_type: str,
        affected_fields: Sequence[object],
        relevance: str,
        status: str,
        stages: Sequence[object],
        summary: str,
        implication: str,
        boundary: str,
        evidence_kind: str,
        source_report: str,
        source_metric: str,
        observed: object,
        comparison: object = None,
        direction: str = "",
    ) -> None:
        insights.append({
            "insight_id": insight_id,
            "theme": theme,
            "title": title,
            "insight_type": insight_type,
            "affected_fields": tuple(affected_fields),
            "relevance": relevance,
            "status": status,
            "source_stages": tuple(stages),
            "summary": summary,
            "modeling_implication": implication,
            "interpretation_boundary": boundary,
        })
        evidence.append({
            "evidence_id": f"EVI-{insight_id.split('-')[-1]}",
            "insight_id": insight_id,
            "evidence_kind": evidence_kind,
            "source_report": source_report,
            "source_metric": source_metric,
            "observed_value": deepcopy(observed),
            "comparison_value": deepcopy(comparison),
            "direction": direction,
            "interpretation": summary,
        })

    # Stage 16 is the canonical quality synthesis; stage 17 only prioritizes it.
    findings = quality_report.findings_frame()
    blockers = quality_report.blockers_frame()
    non_issues = quality_report.validated_non_issues_frame()
    add(
        "INS-001",
        theme="Data quality",
        title=(
            "Structural data quality supports controlled preparation"
            if blockers.empty
            else "Data-quality conditions require targeted review"
        ),
        insight_type="Data-quality condition",
        affected_fields=fields,
        relevance="High",
        status="Observed" if blockers.empty else "Unresolved",
        stages=("7", "8", "9", "16"),
        summary=(
            f"Stage 16 consolidated {len(findings)} findings, {len(blockers)} "
            f"blockers, and {len(non_issues)} validated non-issues."
        ),
        implication=(
            "Restrict preparation to evidence-backed actions and preserve "
            "conditions already validated as non-issues."
        ),
        boundary="Structural data quality does not establish regression performance.",
        evidence_kind="Data quality",
        source_report="quality_findings_report",
        source_metric="Findings, blockers, validated non-issues",
        observed={
            "findings": len(findings),
            "blockers": len(blockers),
            "validated_non_issues": len(non_issues),
        },
        comparison={"blockers": 0},
        direction="Controlled" if blockers.empty else "Review required",
    )

    # Continuous-target scale, spread, and descriptive extremes.
    target_range = getattr(target_report, "observed_range", None)
    extreme_count = int(getattr(target_report, "extreme_count", 0) or 0)
    extreme_share = getattr(target_report, "extreme_share", None)
    unit = _text(getattr(target_report, "unit", "")) or "target units"
    add(
        "INS-002",
        theme="Target distribution",
        title="Continuous target scale and extremes define the regression evaluation context",
        insight_type="Pattern",
        affected_fields=(target,) if target else (),
        relevance="High",
        status="Observed",
        stages=("10",),
        summary=(
            f"Observed target range={_format_metric(target_range)} {unit}; "
            f"mean={_format_metric(getattr(target_report, 'mean', None))}, "
            f"median={_format_metric(getattr(target_report, 'median', None))}, "
            f"standard deviation={_format_metric(getattr(target_report, 'standard_deviation', None))}, "
            f"1.5-IQR extreme flags={extreme_count} ({_format_percent(extreme_share)})."
        ),
        implication=(
            "Evaluate regression error on the original target scale and retain "
            "extreme-value sensitivity as a validation concern rather than an "
            "automatic cleaning rule."
        ),
        boundary=(
            "Target spread and Tukey-fence extremes describe the observed sample; "
            "they do not establish prediction difficulty or justify clipping."
        ),
        evidence_kind="Target distribution",
        source_report="target_distribution_report",
        source_metric="Range, dispersion, and 1.5-IQR target extremes",
        observed={
            "minimum": getattr(target_report, "minimum", None),
            "maximum": getattr(target_report, "maximum", None),
            "range": target_range,
            "mean": getattr(target_report, "mean", None),
            "median": getattr(target_report, "median", None),
            "standard_deviation": getattr(target_report, "standard_deviation", None),
            "extreme_count": extreme_count,
            "extreme_share": extreme_share,
        },
        direction="Continuous spread",
    )

    # Feature-level extreme-value review remains descriptive.
    outlier_features = tuple(getattr(numerical_report, "features_with_outliers", ()))
    if outlier_features:
        outlier_frame = numerical_report.outlier_summary_frame()
        candidate_count = (
            int(outlier_frame["Outlier count"].sum())
            if "Outlier count" in outlier_frame
            else None
        )
        add(
            "INS-003",
            theme="Numerical distributions",
            title="IQR candidates occur in numerical feature distributions",
            insight_type="Pattern",
            affected_fields=outlier_features,
            relevance="Medium",
            status="Observed",
            stages=("11", "16"),
            summary=(
                f"{len(outlier_features)} features contain IQR review candidates "
                f"(candidate flags={candidate_count if candidate_count is not None else 'n/a'})."
            ),
            implication=(
                "Keep raw observations and compare robustness or transformations "
                "only inside leakage-safe validation pipelines."
            ),
            boundary="IQR candidates are statistical extremes, not proven measurement errors.",
            evidence_kind="Numerical pattern",
            source_report="numerical_report",
            source_metric="IQR feature outlier candidates",
            observed={
                "features": outlier_features,
                "candidate_flags": candidate_count,
            },
            comparison=0,
            direction="Review only",
        )

    # Pairwise association and source-backed dependency evidence.
    review_pairs = feature_relationship_report.numerical_review_frame()
    redundant = (
        review_pairs.loc[review_pairs["Potential redundancy"]].reset_index(drop=True)
        if not review_pairs.empty and "Potential redundancy" in review_pairs
        else pd.DataFrame()
    )
    confirmed = int(getattr(leakage_report, "confirmed_derived_dependency_count", 0))
    if not redundant.empty or confirmed:
        strongest_pair: tuple[str, str] = ()
        strongest_value: object = None
        if not review_pairs.empty:
            first = review_pairs.iloc[0]
            strongest_pair = (
                _text(first.get("Feature A")),
                _text(first.get("Feature B")),
            )
            strongest_value = first.get("Maximum absolute association")

        dependency_frame = leakage_report.dependency_frame()
        derived_fields: tuple[str, ...] = ()
        if not dependency_frame.empty and "Dependency status" in dependency_frame:
            selected = dependency_frame.loc[
                dependency_frame["Dependency status"].eq(
                    "Confirmed from retained columns"
                )
            ]
            if "Derived feature" in selected:
                derived_fields = tuple(selected["Derived feature"].astype(str))

        add(
            "INS-004",
            theme="Feature dependency",
            title="Strong feature associations may create redundant regression information",
            insight_type="Dependency",
            affected_fields=_unique_text_tuple((*strongest_pair, *derived_fields)),
            relevance="High",
            status="Observed",
            stages=("12", "15"),
            summary=(
                f"{len(redundant)} feature pairs meet the redundancy-review threshold "
                f"and {confirmed} retained derived dependencies are numerically confirmed. "
                f"Strongest reviewed pair={strongest_pair or 'n/a'} "
                f"({_format_metric(strongest_value)})."
            ),
            implication=(
                "Use the complete validated feature set as a baseline, then compare "
                "regularized or ablated variants under the same regression validation protocol."
            ),
            boundary="Strong association does not by itself justify feature removal before splitting.",
            evidence_kind="Feature dependency",
            source_report="feature_relationship_report + leakage_report",
            source_metric="Redundancy-review pairs and confirmed dependencies",
            observed={
                "redundancy_pairs": len(redundant),
                "confirmed_dependencies": confirmed,
                "strongest_pair": strongest_pair,
                "strongest_association": strongest_value,
            },
            comparison={
                "redundancy_threshold": getattr(
                    feature_relationship_report,
                    "redundancy_review_threshold",
                    None,
                )
            },
            direction="Structural redundancy",
        )
        hypotheses.append({
            "hypothesis_id": "HYP-001",
            "linked_insight_ids": ("INS-004",),
            "title": "Some strongly associated inputs may add limited incremental value",
            "hypothesis": (
                "A regularized or dependency-aware reduced feature set may match "
                "the all-feature regression baseline without materially degrading "
                "held-out error."
            ),
            "status": "Unvalidated",
            "confounding_risks": (),
            "required_validation": (
                "Compare all-feature, regularized, and ablated pipelines under "
                "the same cross-validation protocol and regression metrics."
            ),
            "decision_stage": "Model selection",
        })
        actions.append({
            "action_id": "VAL-001",
            "hypothesis_ids": ("HYP-001",),
            "validation_type": "Ablation",
            "action": (
                "Compare the full feature set with dependency-aware ablation "
                "variants inside training folds."
            ),
            "stage": "Model selection",
            "blocking": False,
            "status": "Planned",
            "acceptance_criteria": (
                "Any removal is supported by stable held-out regression metrics "
                "across folds."
            ),
        })

    # Univariate continuous feature-to-target association.
    relationships = feature_target_report.relationships_frame()
    if not relationships.empty:
        first = relationships.iloc[0]
        top_feature = _text(first.get("Feature"))
        top_association = first.get("Maximum absolute association")
        pearson = first.get("Pearson correlation")
        spearman = first.get("Spearman correlation")
        review_count = (
            int(relationships["Review flag"].sum())
            if "Review flag" in relationships
            else 0
        )
        add(
            "INS-005",
            theme="Feature-to-target association",
            title="Numerical inputs show measurable univariate association with the continuous target",
            insight_type="Pattern",
            affected_fields=(
                tuple(getattr(feature_target_report, "requested_features", ()))
                + ((target,) if target else ())
            ),
            relevance="High",
            status="Observed",
            stages=("13",),
            summary=(
                f"{review_count} features meet the exploratory association threshold; "
                f"strongest={top_feature or 'n/a'} "
                f"(max absolute association={_format_metric(top_association)}, "
                f"Pearson={_format_metric(pearson)}, Spearman={_format_metric(spearman)})."
            ),
            implication=(
                "Retain the validated inputs for the baseline and evaluate their "
                "incremental contribution jointly during model selection."
            ),
            boundary=(
                "Univariate association is not causality, incremental importance, "
                "or held-out predictive performance."
            ),
            evidence_kind="Feature-to-target",
            source_report="feature_target_report",
            source_metric="Pearson/Spearman continuous-target association",
            observed={
                "review_candidates": review_count,
                "top_feature": top_feature,
                "top_association": top_association,
                "top_pearson": pearson,
                "top_spearman": spearman,
            },
            comparison={
                "review_threshold": getattr(
                    feature_target_report,
                    "association_review_threshold",
                    None,
                )
            },
            direction="Continuous association",
        )

    # Stage 14 structural diagnostics: curvature and pairwise interactions.
    nonlinearity = regression_structure_report.nonlinearity_frame()
    interactions = regression_structure_report.interaction_frame()
    nonlinear_signals = (
        nonlinearity.loc[nonlinearity["Nonlinearity signal"]].reset_index(drop=True)
        if not nonlinearity.empty and "Nonlinearity signal" in nonlinearity
        else pd.DataFrame()
    )
    interaction_signals = (
        interactions.loc[interactions["Interaction signal"]].reset_index(drop=True)
        if not interactions.empty and "Interaction signal" in interactions
        else pd.DataFrame()
    )

    strongest_nonlinear_feature = ""
    strongest_nonlinear_gain: object = None
    if not nonlinear_signals.empty:
        first = nonlinear_signals.iloc[0]
        strongest_nonlinear_feature = _text(first.get("Feature"))
        strongest_nonlinear_gain = first.get("Adjusted R squared gain")

    strongest_interaction_pair: tuple[str, str] = ()
    strongest_interaction_gain: object = None
    if not interaction_signals.empty:
        first = interaction_signals.iloc[0]
        strongest_interaction_pair = (
            _text(first.get("Feature A")),
            _text(first.get("Feature B")),
        )
        strongest_interaction_gain = first.get("Adjusted R squared gain")

    structural_signal = bool(len(nonlinear_signals) or len(interaction_signals))
    add(
        "INS-006",
        theme="Regression structure",
        title=(
            "Exploratory diagnostics indicate nonlinear or interaction structure"
            if structural_signal
            else "Exploratory structural diagnostics do not cross review thresholds"
        ),
        insight_type="Pattern",
        affected_fields=(
            tuple(getattr(regression_structure_report, "requested_features", ()))
            + ((target,) if target else ())
        ),
        relevance="High" if structural_signal else "Medium",
        status="Observed",
        stages=("14",),
        summary=(
            f"Nonlinearity signals={len(nonlinear_signals)}; interaction signals={len(interaction_signals)}; "
            f"strongest nonlinear feature={strongest_nonlinear_feature or 'n/a'} "
            f"(adjusted-R² gain={_format_metric(strongest_nonlinear_gain)}); "
            f"strongest interaction={strongest_interaction_pair or 'n/a'} "
            f"(adjusted-R² gain={_format_metric(strongest_interaction_gain)})."
        ),
        implication=(
            "Compare a transparent additive baseline with model families capable "
            "of representing nonlinearities and interactions under the same "
            "leakage-safe cross-validation protocol."
            if structural_signal
            else
            "Retain an additive baseline, but do not exclude flexible model families "
            "solely because these in-sample thresholds were not crossed."
        ),
        boundary=(
            "Adjusted-R² gains are in-sample structural diagnostics; they do not "
            "estimate generalization performance or establish causal interactions."
        ),
        evidence_kind="Regression structure",
        source_report="regression_structure_report",
        source_metric="Adjusted-R² gains for quadratic and interaction terms",
        observed={
            "nonlinearity_signals": len(nonlinear_signals),
            "interaction_signals": len(interaction_signals),
            "strongest_nonlinear_feature": strongest_nonlinear_feature,
            "strongest_nonlinear_gain": strongest_nonlinear_gain,
            "strongest_interaction_pair": strongest_interaction_pair,
            "strongest_interaction_gain": strongest_interaction_gain,
        },
        comparison={
            "nonlinearity_threshold": getattr(
                regression_structure_report,
                "nonlinearity_review_threshold",
                None,
            ),
            "interaction_threshold": getattr(
                regression_structure_report,
                "interaction_review_threshold",
                None,
            ),
        },
        direction="Structural signals observed" if structural_signal else "Below review thresholds",
    )

    if structural_signal:
        hypotheses.append({
            "hypothesis_id": "HYP-002",
            "linked_insight_ids": ("INS-006",),
            "title": "Flexible regression families may improve on a strictly additive linear baseline",
            "hypothesis": (
                "Model families that can represent nonlinear effects or interactions "
                "may reduce held-out regression error relative to a strictly additive "
                "linear baseline."
            ),
            "status": "Unvalidated",
            "confounding_risks": (
                "In-sample diagnostic optimism",
                "Feature scale and correlation",
            ),
            "required_validation": (
                "Compare additive linear and flexible candidate families using the "
                "same leakage-safe cross-validation folds and regression metrics."
            ),
            "decision_stage": "Model selection",
        })
        actions.append({
            "action_id": "VAL-002",
            "hypothesis_ids": ("HYP-002",),
            "validation_type": "Cross-validation",
            "action": (
                "Compare a transparent additive baseline with nonlinear and "
                "interaction-capable regression candidates."
            ),
            "stage": "Model selection",
            "blocking": False,
            "status": "Planned",
            "acceptance_criteria": (
                "Any complexity claim is supported by stable held-out regression "
                "error improvements across folds, not by in-sample adjusted-R² alone."
            ),
        })

    # Static target leakage remains a governance gate.
    direct_leakage = bool(getattr(leakage_report, "has_direct_target_leakage", False))
    proxy_count = len(leakage_report.target_proxy_candidates_frame())
    add(
        "INS-007",
        theme="Leakage governance",
        title=(
            "Direct target leakage requires resolution"
            if direct_leakage
            else "No direct numerical target leakage was detected in candidate features"
        ),
        insight_type=(
            "Governance limitation" if direct_leakage else "Data-quality condition"
        ),
        affected_fields=(
            tuple(getattr(leakage_report, "candidate_features", ()))
            + ((target,) if target else ())
        ),
        relevance="High",
        status="Unresolved" if direct_leakage else "Controlled",
        stages=("15", "16"),
        summary=(
            f"Direct target leakage={direct_leakage}; target proxy candidates={proxy_count}; "
            f"confirmed non-target derived dependencies="
            f"{getattr(leakage_report, 'confirmed_derived_dependency_count', 0)}."
        ),
        implication=(
            "Resolve target-derived numerical proxies before modeling."
            if direct_leakage
            else "Keep target semantics isolated and preserve leakage-safe fitting rules."
        ),
        boundary=(
            "Absence of static target proxies does not replace train/validation "
            "isolation for learned transformations and model fitting."
        ),
        evidence_kind="Leakage/governance",
        source_report="leakage_report",
        source_metric="Continuous target leakage audit",
        observed={
            "direct_target_leakage": direct_leakage,
            "target_proxy_candidates": proxy_count,
        },
        comparison={"direct_target_leakage": False},
        direction="Blocked" if direct_leakage else "Controlled",
    )

    limitations.append({
        "limitation_id": "LIM-001",
        "theme": "Exploratory interpretation",
        "title": "EDA associations and structural fits do not establish regression performance",
        "limitation_type": "Modeling",
        "affected_fields": (
            tuple(getattr(feature_target_report, "requested_features", ()))
            + ((target,) if target else ())
        ),
        "severity": "Contextual",
        "status": "Accepted",
        "source_stages": ("12", "13", "14", "17"),
        "implication": (
            "Correlation, adjusted-R² gains, redundancy signals, and interaction "
            "diagnostics cannot be reported as out-of-sample predictive performance."
        ),
        "required_resolution": (
            "Validate predictive claims with leakage-safe cross-validation, held-out "
            "regression metrics, and residual/error analysis."
        ),
    })

    return consolidate_key_exploratory_insights(
        available_fields=fields,
        insights=insights,
        evidence=evidence,
        hypotheses=hypotheses,
        validation_actions=actions,
        limitations=limitations,
    )

def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_insights(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        item = f"insight[{position}]"
        insight_id = _text(declaration.get("insight_id")) or item
        if insight_id in seen:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Duplicate insight ID",
                    f"Insight ID {insight_id!r} is declared more than once",
                    "Evidence and hypotheses cannot be linked deterministically",
                )
            )
        seen.add(insight_id)

        affected_fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in affected_fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Insight",
                        insight_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The insight may refer to a non-existent variable",
                    )
                )

        insight_type = _text(declaration.get("insight_type"))
        if insight_type not in _ALLOWED_INSIGHT_TYPES:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Invalid insight type",
                    f"Unsupported insight type {insight_type!r}",
                    "Theme summaries and readiness are unreliable",
                )
            )

        relevance = _text(declaration.get("relevance"))
        if relevance not in _ALLOWED_RELEVANCE:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Invalid relevance",
                    f"Unsupported relevance {relevance!r}",
                    "Priority ordering is not reliable",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_INSIGHT_STATUSES:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Invalid insight status",
                    f"Unsupported insight status {status!r}",
                    "Observed facts and unresolved hypotheses may be mixed",
                )
            )

        boundary = _text(declaration.get("interpretation_boundary"))
        if not boundary:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Missing interpretation boundary",
                    "No interpretation boundary was declared",
                    "Correlation may be presented as causation or a final decision",
                )
            )

        rows.append(
            {
                "Insight ID": insight_id,
                "Theme": _text(declaration.get("theme")),
                "Title": _text(declaration.get("title")),
                "Insight type": insight_type,
                "Affected fields": affected_fields,
                "Affected field count": len(affected_fields),
                "Relevance": relevance,
                "Status": status,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Summary": _text(declaration.get("summary")),
                "Modeling implication": _text(
                    declaration.get("modeling_implication")
                ),
                "Interpretation boundary": boundary,
                "Evidence count": 0,
                "Hypothesis count": 0,
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
        item = f"evidence[{position}]"
        evidence_id = _text(declaration.get("evidence_id")) or item
        if evidence_id in seen:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Duplicate evidence ID",
                    f"Evidence ID {evidence_id!r} is declared more than once",
                    "Insight support cannot be traced uniquely",
                )
            )
        seen.add(evidence_id)

        evidence_kind = _text(declaration.get("evidence_kind"))
        if evidence_kind not in _ALLOWED_EVIDENCE_KINDS:
            issues.append(
                _issue(
                    "Evidence",
                    evidence_id,
                    "Invalid evidence kind",
                    f"Unsupported evidence kind {evidence_kind!r}",
                    "Evidence matrices cannot be constructed consistently",
                )
            )

        rows.append(
            {
                "Evidence ID": evidence_id,
                "Insight ID": _text(declaration.get("insight_id")),
                "Evidence kind": evidence_kind,
                "Source report": _text(declaration.get("source_report")),
                "Source metric": _text(declaration.get("source_metric")),
                "Observed value": deepcopy(declaration.get("observed_value")),
                "Comparison value": deepcopy(
                    declaration.get("comparison_value")
                ),
                "Direction": _text(declaration.get("direction")),
                "Interpretation": _text(declaration.get("interpretation")),
            }
        )

    return rows


def _normalize_hypotheses(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        item = f"hypothesis[{position}]"
        hypothesis_id = _text(declaration.get("hypothesis_id")) or item
        if hypothesis_id in seen:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Duplicate hypothesis ID",
                    f"Hypothesis ID {hypothesis_id!r} is declared more than once",
                    "Validation actions cannot be linked uniquely",
                )
            )
        seen.add(hypothesis_id)

        linked_ids = _tuple_values(declaration.get("linked_insight_ids", ()))
        if not linked_ids:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Hypothesis without linked insight",
                    "No observed insight supports this hypothesis",
                    "The hypothesis is detached from exploratory evidence",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_HYPOTHESIS_STATUSES:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Invalid hypothesis status",
                    f"Unsupported hypothesis status {status!r}",
                    "Validation readiness cannot be evaluated consistently",
                )
            )

        required_validation = _text(declaration.get("required_validation"))
        if not required_validation:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Missing required validation",
                    "No required validation was declared",
                    "The hypothesis may be treated as a conclusion",
                )
            )

        decision_stage = _text(declaration.get("decision_stage"))
        if decision_stage not in _ALLOWED_DECISION_STAGES:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Invalid decision stage",
                    f"Unsupported decision stage {decision_stage!r}",
                    "Future work cannot be routed consistently",
                )
            )

        confounding_risks = _tuple_values(
            declaration.get("confounding_risks", ())
        )

        rows.append(
            {
                "Hypothesis ID": hypothesis_id,
                "Linked insight IDs": linked_ids,
                "Linked insight count": len(linked_ids),
                "Title": _text(declaration.get("title")),
                "Hypothesis": _text(declaration.get("hypothesis")),
                "Status": status,
                "Confounding risks": confounding_risks,
                "Confounding risk count": len(confounding_risks),
                "Required validation": required_validation,
                "Decision stage": decision_stage,
                "Validation action count": 0,
            }
        )

    return rows


def _normalize_actions(
    declarations: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, declaration in enumerate(declarations):
        item = f"action[{position}]"
        action_id = _text(declaration.get("action_id")) or item
        if action_id in seen:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Duplicate action ID",
                    f"Action ID {action_id!r} is declared more than once",
                    "Validation work cannot be referenced uniquely",
                )
            )
        seen.add(action_id)

        validation_type = _text(declaration.get("validation_type"))
        if validation_type not in _ALLOWED_VALIDATION_TYPES:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Invalid validation type",
                    f"Unsupported validation type {validation_type!r}",
                    "The validation plan cannot be summarized consistently",
                )
            )

        stage = _text(declaration.get("stage"))
        if stage not in _ALLOWED_DECISION_STAGES:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Invalid decision stage",
                    f"Unsupported validation stage {stage!r}",
                    "Future work cannot be routed consistently",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_ACTION_STATUSES:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Invalid validation status",
                    f"Unsupported validation status {status!r}",
                    "Validation progress cannot be evaluated consistently",
                )
            )

        acceptance = _text(declaration.get("acceptance_criteria"))
        if not acceptance:
            issues.append(
                _issue(
                    "Validation action",
                    action_id,
                    "Missing acceptance criteria",
                    "No measurable acceptance criteria were declared",
                    "The validation cannot be completed objectively",
                )
            )

        hypothesis_ids = _tuple_values(
            declaration.get("hypothesis_ids", ())
        )

        rows.append(
            {
                "Action ID": action_id,
                "Hypothesis IDs": hypothesis_ids,
                "Hypothesis count": len(hypothesis_ids),
                "Validation type": validation_type,
                "Action": _text(declaration.get("action")),
                "Stage": stage,
                "Blocking": bool(declaration.get("blocking", False)),
                "Status": status,
                "Acceptance criteria": acceptance,
            }
        )

    return rows


def _normalize_limitations(
    declarations: Sequence[Mapping[str, object]],
    *,
    available_fields: tuple[str, ...],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    available = set(available_fields)

    for position, declaration in enumerate(declarations):
        item = f"limitation[{position}]"
        limitation_id = _text(declaration.get("limitation_id")) or item
        if limitation_id in seen:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Duplicate limitation ID",
                    f"Limitation ID {limitation_id!r} is declared more than once",
                    "Readiness may count the same limitation multiple times",
                )
            )
        seen.add(limitation_id)

        affected_fields = _tuple_values(declaration.get("affected_fields", ()))
        for field in affected_fields:
            if field not in available:
                issues.append(
                    _issue(
                        "Limitation",
                        limitation_id,
                        "Unknown affected field",
                        f"Affected field {field!r} is not available",
                        "The limitation may refer to a non-existent variable",
                    )
                )

        limitation_type = _text(declaration.get("limitation_type"))
        if limitation_type not in _ALLOWED_LIMITATION_TYPES:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Invalid limitation type",
                    f"Unsupported limitation type {limitation_type!r}",
                    "Readiness gates cannot classify the limitation",
                )
            )

        severity = _text(declaration.get("severity"))
        if severity not in _ALLOWED_LIMITATION_SEVERITIES:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Invalid limitation severity",
                    f"Unsupported limitation severity {severity!r}",
                    "Limitation priority ordering is unreliable",
                )
            )

        status = _text(declaration.get("status"))
        if status not in _ALLOWED_LIMITATION_STATUSES:
            issues.append(
                _issue(
                    "Limitation",
                    limitation_id,
                    "Invalid limitation status",
                    f"Unsupported limitation status {status!r}",
                    "Modeling readiness cannot be evaluated consistently",
                )
            )

        rows.append(
            {
                "Limitation ID": limitation_id,
                "Theme": _text(declaration.get("theme")),
                "Title": _text(declaration.get("title")),
                "Limitation type": limitation_type,
                "Affected fields": affected_fields,
                "Affected field count": len(affected_fields),
                "Severity": severity,
                "Status": status,
                "Source stages": _tuple_values(
                    declaration.get("source_stages", ())
                ),
                "Implication": _text(declaration.get("implication")),
                "Required resolution": _text(
                    declaration.get("required_resolution")
                ),
            }
        )

    return rows


def _validate_references_and_coverage(
    *,
    insight_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    hypothesis_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    insight_ids: set[str],
    hypothesis_ids: set[str],
    issues: list[dict[str, object]],
) -> None:
    evidence_by_insight: dict[str, int] = {}
    for row in evidence_rows:
        insight_id = str(row["Insight ID"])
        if insight_id not in insight_ids:
            issues.append(
                _issue(
                    "Evidence",
                    str(row["Evidence ID"]),
                    "Unknown insight reference",
                    f"Insight ID {insight_id!r} is not declared",
                    "Evidence cannot be traced to a known insight",
                )
            )
        evidence_by_insight[insight_id] = evidence_by_insight.get(insight_id, 0) + 1

    for row in insight_rows:
        insight_id = str(row["Insight ID"])
        if evidence_by_insight.get(insight_id, 0) == 0:
            issues.append(
                _issue(
                    "Insight",
                    insight_id,
                    "Insight without evidence",
                    "No evidence record references this insight",
                    "The insight is not traceable to prior analysis",
                )
            )

    for row in hypothesis_rows:
        hypothesis_id = str(row["Hypothesis ID"])
        for insight_id in row["Linked insight IDs"]:
            if str(insight_id) not in insight_ids:
                issues.append(
                    _issue(
                        "Hypothesis",
                        hypothesis_id,
                        "Unknown insight reference",
                        f"Insight ID {insight_id!r} is not declared",
                        "The hypothesis is linked to unavailable evidence",
                    )
                )

    actions_by_hypothesis: dict[str, int] = {}
    for row in action_rows:
        action_id = str(row["Action ID"])
        for hypothesis_id in row["Hypothesis IDs"]:
            key = str(hypothesis_id)
            if key not in hypothesis_ids:
                issues.append(
                    _issue(
                        "Validation action",
                        action_id,
                        "Unknown hypothesis reference",
                        f"Hypothesis ID {key!r} is not declared",
                        "Validation work cannot be traced to a hypothesis",
                    )
                )
            actions_by_hypothesis[key] = actions_by_hypothesis.get(key, 0) + 1

    for row in hypothesis_rows:
        hypothesis_id = str(row["Hypothesis ID"])
        if actions_by_hypothesis.get(hypothesis_id, 0) == 0:
            issues.append(
                _issue(
                    "Hypothesis",
                    hypothesis_id,
                    "Hypothesis without validation action",
                    "No validation action references this hypothesis",
                    "The hypothesis has no executable validation plan",
                )
            )


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
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)

    result: list[str] = []
    for item in values:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _unique_text_tuple(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


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
