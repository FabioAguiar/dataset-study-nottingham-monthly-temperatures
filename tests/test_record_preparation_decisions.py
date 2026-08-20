"""Tests for the preparation-decision contract."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from scripts.record_preparation_decisions import (
    PreparationDecisionContractError,
    record_preparation_decisions,
)


AVAILABLE_FIELDS = (
    "customerID",
    "tenure",
    "TotalCharges",
    "Contract",
    "Churn",
)


def _decisions() -> list[dict[str, object]]:
    return [
        {
            "decision_id": "PREP-001",
            "domain": "Cleaning",
            "title": "Materialize TotalCharges deterministically",
            "affected_fields": ("TotalCharges", "tenure"),
            "status": "Approved",
            "phase": "Before split",
            "fit_scope": "Deterministic",
            "operation": "Materialize validated blanks and convert to numeric.",
            "rationale": "The raw field contains validated tenure-zero blanks.",
            "prerequisites": (),
            "acceptance_criteria": "All rows remain and TotalCharges is numeric.",
            "source_stages": ("8", "11", "16"),
        },
        {
            "decision_id": "PREP-002",
            "domain": "Transformation",
            "title": "Fit encoders on training data only",
            "affected_fields": ("Contract",),
            "status": "Conditional",
            "phase": "Train-only transformation",
            "fit_scope": "Train only",
            "operation": "Fit categorical encoding after the split.",
            "rationale": "Global fitting would contaminate held-out partitions.",
            "prerequisites": ("PREP-001",),
            "acceptance_criteria": "Validation and test are transform-only.",
            "source_stages": ("15", "16"),
        },
        {
            "decision_id": "PREP-003",
            "domain": "Dataset splitting",
            "title": "Resolve temporal split precedence",
            "affected_fields": ("customerID", "Churn"),
            "status": "Blocked",
            "phase": "External contract",
            "fit_scope": "External",
            "operation": "Determine whether valid observation time exists.",
            "rationale": "Temporal ordering is unresolved.",
            "prerequisites": (),
            "acceptance_criteria": "Temporal or snapshot policy is explicit.",
            "source_stages": ("15", "17"),
        },
        {
            "decision_id": "PREP-004",
            "domain": "Cleaning",
            "title": "Do not remove generic IQR outliers",
            "affected_fields": ("tenure", "TotalCharges"),
            "status": "Prohibited",
            "phase": "Before split",
            "fit_scope": "None",
            "operation": "Do not remove or clip values generically.",
            "rationale": "No IQR candidates were identified.",
            "prerequisites": (),
            "acceptance_criteria": "No row is removed by a generic IQR rule.",
            "source_stages": ("11", "16"),
        },
        {
            "decision_id": "PREP-005",
            "domain": "Feature engineering",
            "title": "Evaluate interactions later",
            "affected_fields": ("tenure", "Contract"),
            "status": "Deferred",
            "phase": "Model selection",
            "fit_scope": "Evaluation only",
            "operation": "Test interaction candidates inside validation.",
            "rationale": "Exploratory hypotheses are not base-data mutations.",
            "prerequisites": ("PREP-002",),
            "acceptance_criteria": "Interactions are compared on held-out folds.",
            "source_stages": ("17",),
        },
    ]


def _evidence() -> list[dict[str, object]]:
    return [
        {
            "evidence_id": f"PDE-{index:03d}",
            "decision_id": decision["decision_id"],
            "source_report": "quality_report",
            "source_item": str(decision["title"]),
            "observed_value": {"value": index},
            "expected_or_reference": "Declared preparation policy",
            "interpretation": "Traceable support for the decision.",
        }
        for index, decision in enumerate(_decisions(), start=1)
    ]


def _steps() -> list[dict[str, object]]:
    return [
        {
            "step_id": "STEP-001",
            "sequence": 1,
            "decision_ids": ("PREP-001", "PREP-004"),
            "phase": "Before split",
            "action": "Validate and create a deterministic prepared copy.",
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": False,
            "acceptance_criteria": "Prepared values satisfy the raw-data contract.",
        },
        {
            "step_id": "STEP-002",
            "sequence": 2,
            "decision_ids": ("PREP-003",),
            "phase": "External contract",
            "action": "Resolve temporal or snapshot split policy.",
            "blocking": True,
            "status": "Blocked",
            "temporal_dependency": True,
            "acceptance_criteria": "Temporal policy status is no longer unresolved.",
        },
        {
            "step_id": "STEP-003",
            "sequence": 3,
            "decision_ids": ("PREP-002", "PREP-005"),
            "phase": "Train-only transformation",
            "action": "Fit transformations and evaluate candidates.",
            "blocking": True,
            "status": "Planned",
            "temporal_dependency": True,
            "acceptance_criteria": "Held-out partitions are transform-only.",
        },
    ]


def _guardrails() -> list[dict[str, object]]:
    return [
        {
            "guardrail_id": "GRD-001",
            "domain": "Target governance",
            "title": "Do not use Churn as a predictor",
            "affected_fields": ("Churn",),
            "severity": "Critical",
            "status": "Active",
            "prohibited_operation": "Include Churn or a derivative in X.",
            "rationale": "This would be direct target leakage.",
            "verification": "Predictor matrices do not contain Churn.",
        },
        {
            "guardrail_id": "GRD-002",
            "domain": "Feature role",
            "title": "Do not use customerID as a predictor",
            "affected_fields": ("customerID",),
            "severity": "High",
            "status": "Controlled",
            "prohibited_operation": "Encode customerID in X.",
            "rationale": "The identifier is traceability-only.",
            "verification": "customerID is absent from predictor matrices.",
        },
    ]


def _split_policy(**overrides: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "stratify_by": "Churn",
        "random_seed": 42,
        "shuffle": True,
        "temporal_priority": True,
        "temporal_policy_status": "Unresolved",
        "random_split_fallback": (
            "Use only if the source confirms a non-temporal snapshot."
        ),
        "test_holdout_untouched": True,
        "disjoint_partitions_required": True,
        "group_by_identifiers": ("customerID",),
    }
    policy.update(overrides)
    return policy


def _report(**overrides: object):
    parameters: dict[str, object] = {
        "available_fields": AVAILABLE_FIELDS,
        "decisions": _decisions(),
        "evidence": _evidence(),
        "execution_steps": _steps(),
        "guardrails": _guardrails(),
        "split_policy": _split_policy(),
    }
    parameters.update(overrides)
    return record_preparation_decisions(**parameters)


def test_valid_contract_exposes_expected_readiness() -> None:
    report = _report()

    assert report.is_structurally_valid
    assert report.is_ready_for_deterministic_preparation
    assert not report.is_ready_for_split_execution
    assert not report.is_ready_for_modeling
    assert report.has_approved_decisions
    assert report.has_conditional_decisions
    assert report.has_deferred_decisions
    assert report.has_prohibited_operations
    assert report.has_external_blockers
    assert report.has_train_only_operations
    assert report.has_deterministic_cleaning_scope


def test_all_report_tables_are_available() -> None:
    report = _report()

    assert len(report.decisions_frame()) == 5
    assert len(report.evidence_frame()) == 5
    assert len(report.execution_plan_frame()) == 3
    assert len(report.guardrails_frame()) == 2
    assert len(report.split_policy_frame()) == 12
    assert len(report.blockers_frame()) == 1
    assert len(report.readiness_frame()) == 4


def test_summary_contains_contract_counts() -> None:
    summary = _report().summary_frame().set_index("Metric")

    assert summary.loc["Preparation decisions", "Value"] == 5
    assert summary.loc["Approved decisions", "Value"] == 1
    assert summary.loc["Blocked decisions", "Value"] == 1
    assert bool(summary.loc["Ready for deterministic preparation", "Value"])
    assert not bool(summary.loc["Ready for split execution", "Value"])


def test_decisions_are_sorted_by_phase_and_status() -> None:
    ids = list(_report().decisions_frame()["Decision ID"])

    assert ids == [
        "PREP-004",
        "PREP-001",
        "PREP-002",
        "PREP-005",
        "PREP-003",
    ]


def test_execution_steps_are_sorted_by_sequence() -> None:
    ids = list(_report().execution_plan_frame()["Step ID"])
    assert ids == ["STEP-001", "STEP-002", "STEP-003"]


def test_duplicate_decision_id_is_invalid() -> None:
    decisions = _decisions()
    decisions.append(copy.deepcopy(decisions[0]))
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="declared more than once"):
        report.raise_if_invalid()


def test_duplicate_evidence_id_is_invalid() -> None:
    evidence = _evidence()
    evidence.append(copy.deepcopy(evidence[0]))
    report = _report(evidence=evidence)

    with pytest.raises(PreparationDecisionContractError, match="Evidence ID"):
        report.raise_if_invalid()


def test_duplicate_step_id_is_invalid() -> None:
    steps = _steps()
    steps.append(copy.deepcopy(steps[0]))
    report = _report(execution_steps=steps)

    with pytest.raises(PreparationDecisionContractError, match="Step ID"):
        report.raise_if_invalid()


def test_duplicate_guardrail_id_is_invalid() -> None:
    guardrails = _guardrails()
    guardrails.append(copy.deepcopy(guardrails[0]))
    report = _report(guardrails=guardrails)

    with pytest.raises(PreparationDecisionContractError, match="Guardrail ID"):
        report.raise_if_invalid()


def test_unknown_affected_field_is_invalid() -> None:
    decisions = _decisions()
    decisions[0]["affected_fields"] = ("unknown",)
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="is not available"):
        report.raise_if_invalid()


def test_decision_without_evidence_is_invalid() -> None:
    evidence = [row for row in _evidence() if row["decision_id"] != "PREP-001"]
    report = _report(evidence=evidence)

    with pytest.raises(PreparationDecisionContractError, match="no linked evidence"):
        report.raise_if_invalid()


def test_missing_decision_acceptance_criteria_is_invalid() -> None:
    decisions = _decisions()
    decisions[0]["acceptance_criteria"] = ""
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="acceptance criteria"):
        report.raise_if_invalid()


def test_missing_step_acceptance_criteria_is_invalid() -> None:
    steps = _steps()
    steps[0]["acceptance_criteria"] = ""
    report = _report(execution_steps=steps)

    with pytest.raises(PreparationDecisionContractError, match="acceptance criteria"):
        report.raise_if_invalid()


@pytest.mark.parametrize("status", ["Pending", "Done", ""])
def test_invalid_decision_status_is_rejected(status: str) -> None:
    decisions = _decisions()
    decisions[0]["status"] = status
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="Unsupported decision status"):
        report.raise_if_invalid()


@pytest.mark.parametrize("domain", ["Preprocessing", "Analysis", ""])
def test_invalid_decision_domain_is_rejected(domain: str) -> None:
    decisions = _decisions()
    decisions[0]["domain"] = domain
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="Unsupported decision domain"):
        report.raise_if_invalid()


@pytest.mark.parametrize("phase", ["Training", "After split", ""])
def test_invalid_decision_phase_is_rejected(phase: str) -> None:
    decisions = _decisions()
    decisions[0]["phase"] = phase
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="Unsupported decision phase"):
        report.raise_if_invalid()


@pytest.mark.parametrize("scope", ["Global", "Both", ""])
def test_invalid_fit_scope_is_rejected(scope: str) -> None:
    decisions = _decisions()
    decisions[0]["fit_scope"] = scope
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="Unsupported fit scope"):
        report.raise_if_invalid()


def test_unknown_evidence_decision_reference_is_invalid() -> None:
    evidence = _evidence()
    evidence[0]["decision_id"] = "PREP-999"
    report = _report(evidence=evidence)

    with pytest.raises(PreparationDecisionContractError, match="does not exist"):
        report.raise_if_invalid()


def test_unknown_step_decision_reference_is_invalid() -> None:
    steps = _steps()
    steps[0]["decision_ids"] = ("PREP-999",)
    report = _report(execution_steps=steps)

    with pytest.raises(PreparationDecisionContractError, match="does not exist"):
        report.raise_if_invalid()


def test_unknown_prerequisite_reference_is_invalid() -> None:
    decisions = _decisions()
    decisions[1]["prerequisites"] = ("PREP-999",)
    report = _report(decisions=decisions)

    with pytest.raises(PreparationDecisionContractError, match="Prerequisite decision"):
        report.raise_if_invalid()


def test_split_fractions_must_sum_to_one() -> None:
    report = _report(split_policy=_split_policy(test_fraction=0.20))

    with pytest.raises(PreparationDecisionContractError, match="sum to"):
        report.raise_if_invalid()


@pytest.mark.parametrize("key", ["train_fraction", "validation_fraction", "test_fraction"])
def test_each_split_fraction_must_be_between_zero_and_one(key: str) -> None:
    report = _report(split_policy=_split_policy(**{key: 0.0}))

    with pytest.raises(PreparationDecisionContractError, match="strictly between"):
        report.raise_if_invalid()


def test_stratification_field_must_exist() -> None:
    report = _report(split_policy=_split_policy(stratify_by="unknown"))

    with pytest.raises(PreparationDecisionContractError, match="not available"):
        report.raise_if_invalid()


def test_split_policy_allows_explicit_no_stratification() -> None:
    report = _report(split_policy=_split_policy(stratify_by=None))

    report.raise_if_invalid()
    assert report.split_policy["stratify_by"] is None


@pytest.mark.parametrize("seed", [42.0, "42", True, None])
def test_random_seed_must_be_integer(seed: object) -> None:
    report = _report(split_policy=_split_policy(random_seed=seed))

    with pytest.raises(PreparationDecisionContractError, match="must be an integer"):
        report.raise_if_invalid()


def test_temporal_priority_is_required() -> None:
    report = _report(split_policy=_split_policy(temporal_priority=False))

    with pytest.raises(PreparationDecisionContractError, match="Temporal split must"):
        report.raise_if_invalid()


def test_random_fallback_requires_explicit_condition() -> None:
    report = _report(split_policy=_split_policy(random_split_fallback=""))

    with pytest.raises(PreparationDecisionContractError, match="fallback rule is empty"):
        report.raise_if_invalid()


def test_final_test_holdout_is_required() -> None:
    report = _report(split_policy=_split_policy(test_holdout_untouched=False))

    with pytest.raises(PreparationDecisionContractError, match="final test partition"):
        report.raise_if_invalid()


def test_disjoint_partitions_are_required() -> None:
    report = _report(split_policy=_split_policy(disjoint_partitions_required=False))

    with pytest.raises(PreparationDecisionContractError, match="must be disjoint"):
        report.raise_if_invalid()


def test_grouping_identifier_must_exist() -> None:
    report = _report(split_policy=_split_policy(group_by_identifiers=("unknown",)))

    with pytest.raises(PreparationDecisionContractError, match="Grouping identifier"):
        report.raise_if_invalid()


def test_split_gate_reports_unresolved_temporal_policy() -> None:
    with pytest.raises(PreparationDecisionContractError, match="temporal split policy"):
        _report().raise_if_split_not_ready()


def test_split_can_become_ready_after_temporal_resolution() -> None:
    decisions = _decisions()
    decisions[2]["status"] = "Approved"
    decisions[2]["phase"] = "Split"
    decisions[2]["fit_scope"] = "None"
    report = _report(
        decisions=decisions,
        split_policy=_split_policy(
            temporal_policy_status="Resolved snapshot fallback"
        ),
    )

    assert report.is_ready_for_split_execution
    report.raise_if_split_not_ready()


def test_modeling_requires_blocking_steps_to_be_complete() -> None:
    decisions = _decisions()
    decisions[2]["status"] = "Approved"
    decisions[2]["phase"] = "Split"
    decisions[2]["fit_scope"] = "None"
    steps = _steps()
    for step in steps:
        step["status"] = "Complete"
    report = _report(
        decisions=decisions,
        execution_steps=steps,
        split_policy=_split_policy(
            temporal_policy_status="Resolved snapshot fallback"
        ),
    )

    assert report.is_ready_for_modeling


def test_defensive_copies_do_not_mutate_report() -> None:
    report = _report()
    decisions = report.decisions_frame()
    decisions.loc[:, "Title"] = "changed"

    assert "changed" not in set(report.decisions_frame()["Title"])


def test_input_declarations_are_not_mutated() -> None:
    decisions = _decisions()
    evidence = _evidence()
    steps = _steps()
    guardrails = _guardrails()
    policy = _split_policy()
    before = copy.deepcopy((decisions, evidence, steps, guardrails, policy))

    record_preparation_decisions(
        available_fields=AVAILABLE_FIELDS,
        decisions=decisions,
        evidence=evidence,
        execution_steps=steps,
        guardrails=guardrails,
        split_policy=policy,
    )

    assert (decisions, evidence, steps, guardrails, policy) == before


def test_nested_output_values_are_defensive() -> None:
    report = _report()
    evidence = report.evidence_frame()
    evidence.at[0, "Observed value"]["value"] = 999

    assert report.evidence_frame().at[0, "Observed value"] == {"value": 1}


def test_results_are_deterministic() -> None:
    first = _report()
    second = _report()

    pd.testing.assert_frame_equal(first.decisions_frame(), second.decisions_frame())
    pd.testing.assert_frame_equal(first.evidence_frame(), second.evidence_frame())
    pd.testing.assert_frame_equal(
        first.execution_plan_frame(), second.execution_plan_frame()
    )
    pd.testing.assert_frame_equal(first.guardrails_frame(), second.guardrails_frame())
    pd.testing.assert_frame_equal(first.split_policy_frame(), second.split_policy_frame())


# ---------------------------------------------------------------------------
# Static multiclass preparation planning
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from scripts.record_preparation_decisions import (
    record_static_continuous_regression_preparation_decisions,
    record_static_multiclass_preparation_decisions,
)


def _static_multiclass_reports(*, unconfirmed_dependency: bool = True):
    dependency_rows = [
        {
            "Derived feature": "ShapeFactor1",
            "Dependency status": "Confirmed from retained columns",
        }
    ]
    if unconfirmed_dependency:
        dependency_rows.append(
            {
                "Derived feature": "ShapeFactor2",
                "Dependency status": "Declared dependency not confirmed",
            }
        )

    leakage = SimpleNamespace(
        has_direct_target_leakage=False,
        confirmed_derived_dependency_count=1,
        dependency_frame=lambda: pd.DataFrame(dependency_rows),
    )
    duplicate = SimpleNamespace(
        has_source_identifiers=False,
        exact_duplicate_group_count=68,
        exact_duplicate_row_count=136,
    )
    target = SimpleNamespace(
        class_count=7,
        has_issues=False,
        imbalance_ratio=6.7931,
        normalized_class_entropy=0.9427,
    )
    numerical = SimpleNamespace(
        features_with_outliers=("Area", "Perimeter"),
    )
    relationships = SimpleNamespace(
        numerical_relationships=pd.DataFrame(
            {
                "Potential redundancy": [True, True, False],
            }
        ),
    )
    quality = SimpleNamespace(
        is_structurally_valid=True,
        has_must_fix_actions=False,
        has_external_blockers=False,
    )
    insights = SimpleNamespace(
        is_ready_for_preparation_decisions=True,
    )
    return {
        "duplicate_report": duplicate,
        "target_report": target,
        "numerical_report": numerical,
        "feature_relationship_report": relationships,
        "leakage_report": leakage,
        "quality_report": quality,
        "exploratory_insights_report": insights,
    }


def _static_multiclass_report(**overrides: object):
    reports = _static_multiclass_reports()
    reports.update(overrides)
    return record_static_multiclass_preparation_decisions(
        available_fields=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2", "Class"),
        target="Class",
        target_classes=("A", "B", "C", "D", "E", "F", "G"),
        candidate_features=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2"),
        identifiers=(),
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        **reports,
    )


def test_static_multiclass_plan_is_structurally_valid_and_split_ready() -> None:
    report = _static_multiclass_report()

    assert report.is_structurally_valid
    assert report.is_ready_for_deterministic_preparation
    assert report.is_ready_for_split_execution
    assert not report.is_ready_for_modeling


def test_static_multiclass_plan_preserves_snapshot_split_contract() -> None:
    report = _static_multiclass_report()
    policy = report.split_policy

    assert policy["train_fraction"] == pytest.approx(0.70)
    assert policy["validation_fraction"] == pytest.approx(0.15)
    assert policy["test_fraction"] == pytest.approx(0.15)
    assert policy["stratify_by"] == "Class"
    assert policy["temporal_policy_status"] == "Resolved snapshot fallback"
    assert policy["group_by_identifiers"] == ()


def test_static_multiclass_plan_defers_unconfirmed_dependency_pruning() -> None:
    report = _static_multiclass_report()
    decisions = report.decisions_frame().set_index("Decision ID")

    assert decisions.loc["PREP-009", "Status"] == "Deferred"
    assert decisions.loc["PREP-009", "Affected fields"] == ("ShapeFactor2",)
    assert decisions.loc["PREP-004", "Status"] == "Approved"


def test_static_multiclass_plan_omits_dependency_decision_when_fully_confirmed() -> None:
    reports = _static_multiclass_reports(unconfirmed_dependency=False)
    report = record_static_multiclass_preparation_decisions(
        available_fields=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2", "Class"),
        target="Class",
        target_classes=("A", "B", "C", "D", "E", "F", "G"),
        candidate_features=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2"),
        identifiers=(),
        **reports,
    )

    assert "PREP-009" not in set(report.decisions_frame()["Decision ID"])


def test_static_multiclass_plan_rejects_must_fix_quality_actions() -> None:
    reports = _static_multiclass_reports()
    reports["quality_report"] = SimpleNamespace(
        is_structurally_valid=True,
        has_must_fix_actions=True,
        has_external_blockers=False,
    )

    with pytest.raises(PreparationDecisionContractError, match="must-fix"):
        record_static_multiclass_preparation_decisions(
            available_fields=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2", "Class"),
            target="Class",
            target_classes=("A", "B", "C", "D", "E", "F", "G"),
            candidate_features=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2"),
            identifiers=(),
            **reports,
        )


def test_static_multiclass_plan_rejects_direct_target_leakage() -> None:
    reports = _static_multiclass_reports()
    reports["leakage_report"] = SimpleNamespace(
        has_direct_target_leakage=True,
        confirmed_derived_dependency_count=0,
        dependency_frame=lambda: pd.DataFrame(),
    )

    with pytest.raises(PreparationDecisionContractError, match="Direct target leakage"):
        record_static_multiclass_preparation_decisions(
            available_fields=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2", "Class"),
            target="Class",
            target_classes=("A", "B", "C", "D", "E", "F", "G"),
            candidate_features=("Area", "Perimeter", "ShapeFactor1", "ShapeFactor2"),
            identifiers=(),
            **reports,
        )


# ---------------------------------------------------------------------------
# Static continuous-regression preparation planning
# ---------------------------------------------------------------------------


def _static_continuous_reports(*, with_must_fix: bool = True):
    dependency_frame = pd.DataFrame(
        columns=["Derived feature", "Dependency status"]
    )
    leakage = SimpleNamespace(
        has_direct_target_leakage=False,
        confirmed_derived_dependency_count=0,
        dependency_frame=lambda: dependency_frame.copy(deep=True),
    )
    duplicate = SimpleNamespace(
        has_source_identifiers=False,
        exact_duplicate_group_count=11,
        exact_duplicate_row_count=36,
        repeated_profile_group_count=19,
        target_conflict_group_count=9,
    )
    target = SimpleNamespace(
        has_missing_values=False,
        has_non_finite_values=False,
        has_variation=True,
        finite_count=1030,
        unique_count=845,
        observed_range=80.27,
        extreme_count=4,
        unit="MPa",
    )
    numerical = SimpleNamespace(
        features_with_outliers=("Water", "Superplasticizer", "Age"),
    )
    relationships = SimpleNamespace(
        numerical_relationships=pd.DataFrame(
            {"Potential redundancy": [False, False]}
        ),
    )
    regression_structure = SimpleNamespace(
        nonlinearity=pd.DataFrame(
            {"Nonlinearity signal": [True, True, True, False]}
        ),
        interactions=pd.DataFrame(
            {"Interaction signal": [True, True, True, True, False]}
        ),
    )
    findings = pd.DataFrame(
        [
            {
                "Finding ID": "DQ-001",
                "Disposition": "Must fix" if with_must_fix else "Monitor",
                "Affected fields": ("Blast Furnace Slag",),
            },
            {
                "Finding ID": "DQ-002",
                "Disposition": "Must fix" if with_must_fix else "Monitor",
                "Affected fields": ("Blast Furnace Slag",),
            },
        ]
    )
    quality = SimpleNamespace(
        is_structurally_valid=True,
        has_must_fix_actions=with_must_fix,
        has_external_blockers=False,
        findings_frame=lambda: findings.copy(deep=True),
    )
    insights = SimpleNamespace(
        is_ready_for_preparation_decisions=True,
    )
    return {
        "duplicate_report": duplicate,
        "target_report": target,
        "numerical_report": numerical,
        "feature_relationship_report": relationships,
        "regression_structure_report": regression_structure,
        "leakage_report": leakage,
        "quality_report": quality,
        "exploratory_insights_report": insights,
    }


def _slag_resolution():
    return {
        "Blast Furnace Slag": {
            "source_declared_type": "Integer",
            "effective_analytical_type": "Continuous numeric",
            "operation": (
                "Preserve released decimal values without rounding or "
                "integer coercion."
            ),
            "rationale": (
                "Released values contain valid decimal measurements that "
                "conflict with the source metadata annotation."
            ),
        }
    }


def _static_continuous_report(**overrides: object):
    reports = _static_continuous_reports()
    reports.update(overrides)
    return record_static_continuous_regression_preparation_decisions(
        available_fields=(
            "Cement",
            "Blast Furnace Slag",
            "Water",
            "Age",
            "Concrete compressive strength",
        ),
        target="Concrete compressive strength",
        candidate_features=(
            "Cement",
            "Blast Furnace Slag",
            "Water",
            "Age",
        ),
        identifiers=(),
        source_type_resolutions=_slag_resolution(),
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
        **reports,
    )


def test_static_continuous_plan_is_structurally_valid_and_split_ready() -> None:
    report = _static_continuous_report()

    assert report.is_structurally_valid
    assert report.is_ready_for_deterministic_preparation
    assert report.is_ready_for_split_execution
    assert not report.is_ready_for_modeling

    report.raise_if_split_not_ready(require_stratification_contract=False)


def test_static_continuous_plan_uses_non_stratified_snapshot_split() -> None:
    report = _static_continuous_report()
    policy = report.split_policy

    assert policy["train_fraction"] == pytest.approx(0.70)
    assert policy["validation_fraction"] == pytest.approx(0.15)
    assert policy["test_fraction"] == pytest.approx(0.15)
    assert policy["stratify_by"] is None
    assert policy["random_seed"] == 42
    assert policy["temporal_policy_status"] == "Resolved snapshot fallback"
    assert policy["group_by_identifiers"] == ()

    split_frame = report.split_policy_frame().set_index("Policy item")
    assert split_frame.loc["Stratification field", "Status"] == "Resolved"
    assert "No stratification" in split_frame.loc[
        "Stratification field", "Interpretation"
    ]


def test_static_continuous_plan_resolves_source_type_without_value_coercion() -> None:
    report = _static_continuous_report()
    decisions = report.decisions_frame().set_index("Decision ID")
    guardrails = report.guardrails_frame().set_index("Guardrail ID")

    assert decisions.loc["PREP-001", "Status"] == "Approved"
    assert decisions.loc["PREP-001", "Affected fields"] == (
        "Blast Furnace Slag",
    )
    assert "preserv" in decisions.loc["PREP-001", "Operation"].lower()
    assert "round" in guardrails.loc["GRD-002", "Prohibited operation"].lower()


def test_static_continuous_plan_preserves_original_target_semantics() -> None:
    report = _static_continuous_report()
    decisions = report.decisions_frame().set_index("Decision ID")

    assert decisions.loc["PREP-004", "Status"] == "Approved"
    assert decisions.loc["PREP-004", "Affected fields"] == (
        "Concrete compressive strength",
    )
    assert "continuous" in decisions.loc["PREP-004", "Operation"].lower()
    assert "discretize" in decisions.loc["PREP-004", "Operation"].lower()


def test_static_continuous_plan_defers_structure_and_repeat_sensitivity() -> None:
    report = _static_continuous_report()
    decisions = report.decisions_frame().set_index("Decision ID")

    assert decisions.loc["PREP-008", "Status"] == "Deferred"
    assert decisions.loc["PREP-009", "Status"] == "Deferred"
    assert "3 nonlinearity" in decisions.loc["PREP-008", "Rationale"]
    assert "4 interaction" in decisions.loc["PREP-008", "Rationale"]
    assert "19 repeated feature-profile" in decisions.loc["PREP-009", "Rationale"]
    assert "9 group(s)" in decisions.loc["PREP-009", "Rationale"]


def test_static_continuous_plan_rejects_unresolved_must_fix_quality_field() -> None:
    reports = _static_continuous_reports()

    with pytest.raises(
        PreparationDecisionContractError,
        match="explicit source-type resolutions",
    ):
        record_static_continuous_regression_preparation_decisions(
            available_fields=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
                "Concrete compressive strength",
            ),
            target="Concrete compressive strength",
            candidate_features=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
            ),
            identifiers=(),
            source_type_resolutions={},
            **reports,
        )


def test_static_continuous_plan_rejects_unknown_resolution_field() -> None:
    reports = _static_continuous_reports()
    bad_resolution = _slag_resolution()
    bad_resolution["Unknown"] = bad_resolution.pop("Blast Furnace Slag")

    with pytest.raises(
        PreparationDecisionContractError,
        match="is not available",
    ):
        record_static_continuous_regression_preparation_decisions(
            available_fields=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
                "Concrete compressive strength",
            ),
            target="Concrete compressive strength",
            candidate_features=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
            ),
            identifiers=(),
            source_type_resolutions=bad_resolution,
            **reports,
        )


def test_static_continuous_plan_rejects_direct_target_leakage() -> None:
    reports = _static_continuous_reports()
    reports["leakage_report"] = SimpleNamespace(
        has_direct_target_leakage=True,
        confirmed_derived_dependency_count=0,
        dependency_frame=lambda: pd.DataFrame(),
    )

    with pytest.raises(
        PreparationDecisionContractError,
        match="Direct target leakage",
    ):
        record_static_continuous_regression_preparation_decisions(
            available_fields=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
                "Concrete compressive strength",
            ),
            target="Concrete compressive strength",
            candidate_features=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
            ),
            identifiers=(),
            source_type_resolutions=_slag_resolution(),
            **reports,
        )


def test_static_continuous_plan_rejects_constant_target() -> None:
    reports = _static_continuous_reports()
    reports["target_report"] = SimpleNamespace(
        has_missing_values=False,
        has_non_finite_values=False,
        has_variation=False,
    )

    with pytest.raises(
        PreparationDecisionContractError,
        match="more than one finite value",
    ):
        record_static_continuous_regression_preparation_decisions(
            available_fields=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
                "Concrete compressive strength",
            ),
            target="Concrete compressive strength",
            candidate_features=(
                "Cement",
                "Blast Furnace Slag",
                "Water",
                "Age",
            ),
            identifiers=(),
            source_type_resolutions=_slag_resolution(),
            **reports,
        )
