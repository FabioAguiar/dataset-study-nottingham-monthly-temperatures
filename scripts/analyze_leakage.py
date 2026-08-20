"""Reusable, non-mutating audit of potential data leakage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd


_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_CONTEXT_COLUMNS: Final[list[str]] = [
    "Context field",
    "Value",
    "Resolved",
    "Interpretation",
]

_FIELD_AUDIT_COLUMNS: Final[list[str]] = [
    "Field",
    "Role",
    "Present",
    "Uses target",
    "Identifier risk",
    "Available at inference",
    "Available before target",
    "Direct target leakage",
    "Temporal risk",
    "Inference availability issue",
    "Model disposition",
    "Risk level",
    "Reason",
]

_TRANSFORMATION_AUDIT_COLUMNS: Final[list[str]] = [
    "Transformation",
    "Sources",
    "Uses target",
    "Learned from data",
    "Fit scope",
    "Inference reproducible",
    "Aggregation cutoff respected",
    "Target-derived",
    "Global-fit risk",
    "Temporal risk",
    "Invalid source count",
    "Model disposition",
    "Risk level",
    "Reason",
]

_PROXY_COLUMNS: Final[list[str]] = [
    "Field",
    "Detection method",
    "Valid paired rows",
    "Exact match rate",
    "Normalized match rate",
    "Deterministic mapping",
    "Bidirectional mapping",
    "Candidate severity",
    "Interpretation",
]

_PIPELINE_POLICY_COLUMNS: Final[list[str]] = [
    "Rule",
    "Compliant",
    "Interpretation",
]

_RISK_REGISTER_COLUMNS: Final[list[str]] = [
    "Risk ID",
    "Scope",
    "Item",
    "Risk type",
    "Severity",
    "Status",
    "Required action",
]

_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Item",
    "Issue",
    "Details",
    "Potential impact",
]

_ALLOWED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "Prohibited",
        "Exclude",
        "Exploratory only",
        "Train-only fit",
        "Conditionally safe",
        "Safe deterministic",
        "Safe",
        "Unresolved",
    }
)

_ALLOWED_FIT_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "none",
        "global",
        "training_only",
        "fold_only",
        "domain_frozen",
        "exploratory",
    }
)

_REQUIRED_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "prediction_unit",
    "scoring_population",
    "observation_time_definition",
    "target_event_definition",
    "prediction_horizon",
    "feature_snapshot_cutoff",
    "source_snapshot_precedes_target",
)

_DEFAULT_PIPELINE_RULES: Final[dict[str, bool]] = {
    "split_before_learning_transformations": False,
    "fit_imputation_on_training_only": False,
    "fit_scaling_on_training_only": False,
    "fit_encoding_on_training_only": False,
    "fit_rare_category_rules_on_training_only": False,
    "fit_outlier_rules_on_training_only": False,
    "fit_feature_selection_on_training_only": False,
    "apply_resampling_to_training_only": False,
    "keep_test_set_untouched_until_final_evaluation": False,
}


class DataLeakageAnalysisError(ValueError):
    """Raised when leakage-audit structure or modeling safety is invalid."""


@dataclass(frozen=True, slots=True)
class DataLeakageReport:
    """Summarize leakage risks without mutating data or declarations."""

    target: str
    identifiers: tuple[str, ...]
    candidate_features: tuple[str, ...]
    row_count: int
    target_present: bool
    missing_identifiers: tuple[str, ...]
    missing_candidate_features: tuple[str, ...]
    undeclared_fields: tuple[str, ...]
    missing_context_fields: tuple[str, ...]
    field_audit: pd.DataFrame
    transformation_audit: pd.DataFrame
    target_proxy_candidates: pd.DataFrame
    pipeline_policy: pd.DataFrame
    risk_register: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_direct_target_leakage(self) -> bool:
        """Return whether a model input directly uses or reproduces target."""
        candidate_rows = self.field_audit.loc[
            self.field_audit["Field"].isin(self.candidate_features)
        ]
        declared = (
            False
            if candidate_rows.empty
            else bool(candidate_rows["Direct target leakage"].any())
        )
        return declared or not self.target_proxy_candidates.empty

    @property
    def has_identifier_risks(self) -> bool:
        """Return whether identifier fields are present in the audit."""
        if self.field_audit.empty:
            return False
        return bool(self.field_audit["Identifier risk"].any())

    @property
    def has_target_proxy_candidates(self) -> bool:
        """Return whether simple direct target proxies were detected."""
        return not self.target_proxy_candidates.empty

    @property
    def has_temporal_risks(self) -> bool:
        """Return whether fields or transformations have temporal risk."""
        field_risk = (
            False
            if self.field_audit.empty
            else bool(self.field_audit["Temporal risk"].any())
        )
        transformation_risk = (
            False
            if self.transformation_audit.empty
            else bool(self.transformation_audit["Temporal risk"].any())
        )
        return field_risk or transformation_risk

    @property
    def has_inference_availability_issues(self) -> bool:
        """Return whether candidate inputs are unavailable or unresolved."""
        if self.field_audit.empty:
            return False
        candidate_rows = self.field_audit.loc[
            self.field_audit["Field"].isin(self.candidate_features)
        ]
        return bool(candidate_rows["Inference availability issue"].any())

    @property
    def has_target_derived_transformations(self) -> bool:
        """Return whether any transformation uses the target."""
        if self.transformation_audit.empty:
            return False
        return bool(self.transformation_audit["Target-derived"].any())

    @property
    def has_global_fit_risks(self) -> bool:
        """Return whether a learned transformation is globally fitted."""
        if self.transformation_audit.empty:
            return False
        return bool(self.transformation_audit["Global-fit risk"].any())

    @property
    def has_unresolved_temporal_contract(self) -> bool:
        """Return whether required inference-time context is unresolved."""
        return bool(self.missing_context_fields)

    @property
    def has_prohibited_model_inputs(self) -> bool:
        """Return whether prohibited fields remain among candidate inputs."""
        if self.field_audit.empty:
            return False
        candidate_rows = self.field_audit.loc[
            self.field_audit["Field"].isin(self.candidate_features)
        ]
        return bool(
            candidate_rows["Model disposition"].eq("Prohibited").any()
        )

    @property
    def is_structurally_valid(self) -> bool:
        """Return whether declarations and references are structurally valid."""
        return self.issues.empty

    @property
    def is_modeling_ready(self) -> bool:
        """Return whether no recorded condition blocks safe model fitting."""
        pipeline_ready = (
            not self.pipeline_policy.empty
            and bool(self.pipeline_policy["Compliant"].all())
        )
        unsafe_transformations = (
            False
            if self.transformation_audit.empty
            else bool(
                self.transformation_audit["Model disposition"].isin(
                    {
                        "Prohibited",
                        "Unresolved",
                    }
                ).any()
            )
        )
        return bool(
            self.is_structurally_valid
            and not self.has_unresolved_temporal_contract
            and not self.has_prohibited_model_inputs
            and not self.has_target_proxy_candidates
            and not self.has_inference_availability_issues
            and not self.has_global_fit_risks
            and not unsafe_transformations
            and pipeline_ready
        )

    def summary_frame(self) -> pd.DataFrame:
        """Return deterministic top-level audit metrics."""
        rows = [
            {
                "Metric": "Rows audited",
                "Value": self.row_count,
                "Interpretation": "Observations supplied to the audit",
            },
            {
                "Metric": "Candidate features",
                "Value": len(self.candidate_features),
                "Interpretation": "Declared model-input candidates",
            },
            {
                "Metric": "Identifier fields",
                "Value": len(self.identifiers),
                "Interpretation": "Fields excluded for identity risk",
            },
            {
                "Metric": "Target proxy candidates",
                "Value": len(self.target_proxy_candidates),
                "Interpretation": "Simple direct proxy patterns detected",
            },
            {
                "Metric": "Target-derived transformations",
                "Value": int(
                    self.transformation_audit["Target-derived"].sum()
                )
                if not self.transformation_audit.empty
                else 0,
                "Interpretation": "Transformations prohibited as raw inputs",
            },
            {
                "Metric": "Global-fit risks",
                "Value": int(
                    self.transformation_audit["Global-fit risk"].sum()
                )
                if not self.transformation_audit.empty
                else 0,
                "Interpretation": "Learned operations not isolated to training",
            },
            {
                "Metric": "Unresolved temporal context fields",
                "Value": len(self.missing_context_fields),
                "Interpretation": "Required timing declarations still missing",
            },
            {
                "Metric": "Structurally valid",
                "Value": self.is_structurally_valid,
                "Interpretation": "Declarations and references are coherent",
            },
            {
                "Metric": "Modeling ready",
                "Value": self.is_modeling_ready,
                "Interpretation": "All leakage-safety gates are satisfied",
            },
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def context_frame(self) -> pd.DataFrame:
        """Return a defensive copy of inference-context evidence."""
        rows = self.risk_register.attrs.get("context_rows", [])
        return pd.DataFrame(rows, columns=_CONTEXT_COLUMNS).copy(deep=True)

    def field_audit_frame(self) -> pd.DataFrame:
        """Return a defensive field-audit copy."""
        return self.field_audit.copy(deep=True)

    def transformation_audit_frame(self) -> pd.DataFrame:
        """Return a defensive transformation-audit copy."""
        return self.transformation_audit.copy(deep=True)

    def target_proxy_candidates_frame(self) -> pd.DataFrame:
        """Return a defensive direct-proxy-candidate copy."""
        return self.target_proxy_candidates.copy(deep=True)

    def pipeline_policy_frame(self) -> pd.DataFrame:
        """Return a defensive pipeline-policy copy."""
        return self.pipeline_policy.copy(deep=True)

    def risk_register_frame(self) -> pd.DataFrame:
        """Return a defensive risk-register copy."""
        return self.risk_register.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return a defensive structural-issues copy."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(
        self,
        *,
        require_target_present: bool = True,
        require_identifiers_present: bool = True,
        require_candidate_features_present: bool = True,
        require_all_fields_declared: bool = True,
        require_unique_declarations: bool = True,
        require_valid_dispositions: bool = True,
        require_valid_transformation_sources: bool = True,
    ) -> None:
        """Raise when selected structural requirements are not satisfied."""
        failures: list[str] = []

        if require_target_present and not self.target_present:
            failures.append(f"target field is missing: {self.target}")
        if require_identifiers_present and self.missing_identifiers:
            failures.append(
                "identifier fields are missing: "
                f"{list(self.missing_identifiers)}"
            )
        if require_candidate_features_present and self.missing_candidate_features:
            failures.append(
                "candidate features are missing: "
                f"{list(self.missing_candidate_features)}"
            )
        if require_all_fields_declared and self.undeclared_fields:
            failures.append(
                "fields have no leakage declaration: "
                f"{list(self.undeclared_fields)}"
            )

        if not self.issues.empty:
            issue_names = set(self.issues["Issue"])
            if require_unique_declarations and issue_names.intersection(
                {
                    "Duplicate identifier declaration",
                    "Duplicate candidate declaration",
                    "Overlapping field roles",
                }
            ):
                failures.append("field declarations are not unique")
            if require_valid_dispositions and "Invalid disposition" in issue_names:
                failures.append("one or more dispositions are invalid")
            if (
                require_valid_transformation_sources
                and "Unknown transformation source" in issue_names
            ):
                failures.append("one or more transformation sources are invalid")

        if failures:
            raise DataLeakageAnalysisError(
                "Leakage audit is structurally invalid: " + "; ".join(failures)
            )

    def raise_if_modeling_unsafe(
        self,
        *,
        require_temporal_contract_complete: bool = True,
        require_no_prohibited_model_inputs: bool = True,
        require_no_unresolved_inference_fields: bool = True,
        require_train_only_fit_policies: bool = True,
        require_no_target_proxy_candidates: bool = True,
    ) -> None:
        """Raise when selected leakage-safety gates are not satisfied."""
        failures: list[str] = []

        if (
            require_temporal_contract_complete
            and self.has_unresolved_temporal_contract
        ):
            failures.append(
                "temporal contract is incomplete: "
                f"{list(self.missing_context_fields)}"
            )
        if (
            require_no_prohibited_model_inputs
            and self.has_prohibited_model_inputs
        ):
            failures.append("prohibited fields remain among candidate inputs")
        if (
            require_no_unresolved_inference_fields
            and self.has_inference_availability_issues
        ):
            failures.append("candidate inference availability is unresolved")
        if require_train_only_fit_policies and (
            self.pipeline_policy.empty
            or not bool(self.pipeline_policy["Compliant"].all())
            or self.has_global_fit_risks
        ):
            failures.append("train-only fitting policies are not fully satisfied")
        if (
            require_no_target_proxy_candidates
            and self.has_target_proxy_candidates
        ):
            failures.append("target proxy candidates require resolution")

        if failures:
            raise DataLeakageAnalysisError(
                "Leakage audit does not clear modeling: " + "; ".join(failures)
            )


def analyze_data_leakage(
    dataframe: pd.DataFrame,
    *,
    target: str,
    identifiers: Sequence[str],
    candidate_features: Sequence[str],
    field_declarations: Mapping[str, Mapping[str, object]],
    transformation_declarations: Mapping[str, Mapping[str, object]],
    inference_context: Mapping[str, object],
    pipeline_rules: Mapping[str, bool] | None = None,
    detect_exact_target_proxies: bool = True,
    detect_deterministic_target_proxies: bool = True,
) -> DataLeakageReport:
    """Audit raw fields, transformations, context, and model-pipeline rules."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(target, str) or not target:
        raise DataLeakageAnalysisError("target must be a non-empty field name")

    source = dataframe.copy(deep=True)
    identifier_fields = tuple(str(value) for value in identifiers)
    candidate_fields = tuple(str(value) for value in candidate_features)
    declarations = {
        str(name): dict(declaration)
        for name, declaration in field_declarations.items()
    }
    transformations = {
        str(name): dict(declaration)
        for name, declaration in transformation_declarations.items()
    }
    context = dict(inference_context)
    policies = dict(_DEFAULT_PIPELINE_RULES)
    if pipeline_rules is not None:
        policies.update({str(name): bool(value) for name, value in pipeline_rules.items()})

    issues: list[dict[str, object]] = []

    duplicate_identifiers = _duplicates(identifier_fields)
    duplicate_candidates = _duplicates(candidate_fields)
    overlap = tuple(
        field for field in candidate_fields if field in set(identifier_fields)
    )

    for field in duplicate_identifiers:
        issues.append(
            _issue(
                "Fields",
                field,
                "Duplicate identifier declaration",
                "Identifier appears more than once.",
                "Role assignment becomes ambiguous.",
            )
        )
    for field in duplicate_candidates:
        issues.append(
            _issue(
                "Fields",
                field,
                "Duplicate candidate declaration",
                "Candidate feature appears more than once.",
                "Audit counts and downstream selection become ambiguous.",
            )
        )
    for field in overlap:
        issues.append(
            _issue(
                "Fields",
                field,
                "Overlapping field roles",
                "Field is both identifier and candidate feature.",
                "Identifier may leak identity into model inputs.",
            )
        )

    target_present = target in source.columns
    missing_identifiers = tuple(
        field for field in _unique(identifier_fields) if field not in source.columns
    )
    missing_candidates = tuple(
        field for field in _unique(candidate_fields) if field not in source.columns
    )

    if not target_present:
        issues.append(
            _issue(
                "Fields",
                target,
                "Missing target",
                "Declared target is absent from the DataFrame.",
                "Leakage proxies cannot be evaluated.",
            )
        )
    for field in missing_identifiers:
        issues.append(
            _issue(
                "Fields",
                field,
                "Missing identifier",
                "Declared identifier is absent from the DataFrame.",
                "Identity-risk policy cannot be verified.",
            )
        )
    for field in missing_candidates:
        issues.append(
            _issue(
                "Fields",
                field,
                "Missing candidate feature",
                "Declared candidate feature is absent from the DataFrame.",
                "The intended model input contract is incomplete.",
            )
        )

    required_declared_fields = _unique(
        (target, *identifier_fields, *candidate_fields)
    )
    undeclared_fields = tuple(
        field for field in required_declared_fields if field not in declarations
    )
    for field in undeclared_fields:
        issues.append(
            _issue(
                "Fields",
                field,
                "Missing field declaration",
                "No leakage declaration was provided.",
                "Availability and disposition cannot be audited.",
            )
        )

    field_rows: list[dict[str, object]] = []
    for field in required_declared_fields:
        declaration = declarations.get(field, {})
        role = str(
            declaration.get(
                "role",
                _inferred_role(field, target, identifier_fields, candidate_fields),
            )
        )
        disposition = str(declaration.get("model_disposition", "Unresolved"))
        if disposition not in _ALLOWED_DISPOSITIONS:
            issues.append(
                _issue(
                    "Fields",
                    field,
                    "Invalid disposition",
                    f"Unsupported disposition: {disposition!r}.",
                    "The model-input policy is not machine-checkable.",
                )
            )

        uses_target = bool(declaration.get("uses_target", field == target))
        available_at_inference = _optional_bool(
            declaration.get("available_at_inference")
        )
        available_before_target = _optional_bool(
            declaration.get("available_before_target")
        )
        identifier_risk = role.casefold() == "identifier" or field in identifier_fields
        is_candidate = field in candidate_fields
        direct_target = bool(field == target or (is_candidate and uses_target))
        temporal_risk = bool(
            is_candidate and available_before_target is not True
        )
        inference_issue = bool(
            is_candidate and available_at_inference is not True
        )
        risk_level = _field_risk_level(
            direct_target=direct_target,
            identifier_risk=identifier_risk,
            temporal_value=available_before_target,
            inference_value=available_at_inference,
            disposition=disposition,
        )

        field_rows.append(
            {
                "Field": field,
                "Role": role,
                "Present": field in source.columns,
                "Uses target": uses_target,
                "Identifier risk": identifier_risk,
                "Available at inference": available_at_inference,
                "Available before target": available_before_target,
                "Direct target leakage": direct_target,
                "Temporal risk": temporal_risk,
                "Inference availability issue": inference_issue,
                "Model disposition": disposition,
                "Risk level": risk_level,
                "Reason": str(declaration.get("reason", "Not declared.")),
            }
        )

    known_sources = set(source.columns)
    transformation_rows: list[dict[str, object]] = []
    for name, declaration in transformations.items():
        sources = tuple(str(value) for value in declaration.get("sources", ()))
        invalid_sources = tuple(value for value in sources if value not in known_sources)
        for invalid_source in invalid_sources:
            issues.append(
                _issue(
                    "Transformations",
                    name,
                    "Unknown transformation source",
                    f"Source field does not exist: {invalid_source}.",
                    "The transformation cannot be reproduced safely.",
                )
            )

        disposition = str(declaration.get("model_disposition", "Unresolved"))
        if disposition not in _ALLOWED_DISPOSITIONS:
            issues.append(
                _issue(
                    "Transformations",
                    name,
                    "Invalid disposition",
                    f"Unsupported disposition: {disposition!r}.",
                    "The transformation policy is not machine-checkable.",
                )
            )

        fit_scope = str(declaration.get("fit_scope", "none"))
        if fit_scope not in _ALLOWED_FIT_SCOPES:
            issues.append(
                _issue(
                    "Transformations",
                    name,
                    "Invalid fit scope",
                    f"Unsupported fit scope: {fit_scope!r}.",
                    "Train/test isolation cannot be verified.",
                )
            )

        uses_target = bool(declaration.get("uses_target", False))
        learned = bool(declaration.get("learned_from_data", False))
        inference_reproducible = _optional_bool(
            declaration.get("inference_reproducible")
        )
        cutoff_respected = _optional_bool(
            declaration.get("aggregation_cutoff_respected")
        )
        global_fit_risk = bool(learned and fit_scope == "global")
        temporal_risk = bool(cutoff_respected is not True and any(
            token in name.casefold()
            for token in ("total", "aggregate", "cumulative", "rolling")
        ))
        target_derived = uses_target or target in sources
        risk_level = _transformation_risk_level(
            target_derived=target_derived,
            global_fit_risk=global_fit_risk,
            temporal_risk=temporal_risk,
            disposition=disposition,
        )

        transformation_rows.append(
            {
                "Transformation": name,
                "Sources": sources,
                "Uses target": uses_target,
                "Learned from data": learned,
                "Fit scope": fit_scope,
                "Inference reproducible": inference_reproducible,
                "Aggregation cutoff respected": cutoff_respected,
                "Target-derived": target_derived,
                "Global-fit risk": global_fit_risk,
                "Temporal risk": temporal_risk,
                "Invalid source count": len(invalid_sources),
                "Model disposition": disposition,
                "Risk level": risk_level,
                "Reason": str(declaration.get("reason", "Not declared.")),
            }
        )

    missing_context_fields = tuple(
        name
        for name in _REQUIRED_CONTEXT_FIELDS
        if not _context_value_resolved(context.get(name))
    )
    context_rows = [
        {
            "Context field": name,
            "Value": context.get(name),
            "Resolved": name not in missing_context_fields,
            "Interpretation": _context_interpretation(
                name,
                name not in missing_context_fields,
            ),
        }
        for name in _REQUIRED_CONTEXT_FIELDS
    ]

    field_audit = pd.DataFrame(field_rows, columns=_FIELD_AUDIT_COLUMNS)
    transformation_audit = pd.DataFrame(
        transformation_rows,
        columns=_TRANSFORMATION_AUDIT_COLUMNS,
    )

    proxy_rows: list[dict[str, object]] = []
    if target_present and (detect_exact_target_proxies or detect_deterministic_target_proxies):
        target_values = source[target].copy(deep=True)
        auditable_fields = _unique((*identifier_fields, *candidate_fields))
        role_by_field = {
            str(row["Field"]): str(row["Role"])
            for row in field_rows
        }
        for field in auditable_fields:
            if field not in source.columns:
                continue
            candidate = _detect_target_proxy(
                source[field],
                target_values,
                field=field,
                role=role_by_field.get(field, "Candidate feature"),
                detect_exact=detect_exact_target_proxies,
                detect_deterministic=detect_deterministic_target_proxies,
            )
            if candidate is not None:
                proxy_rows.append(candidate)

    proxy_candidates = pd.DataFrame(proxy_rows, columns=_PROXY_COLUMNS)
    pipeline_policy = pd.DataFrame(
        [
            {
                "Rule": rule,
                "Compliant": bool(compliant),
                "Interpretation": _pipeline_rule_interpretation(
                    rule,
                    bool(compliant),
                ),
            }
            for rule, compliant in policies.items()
        ],
        columns=_PIPELINE_POLICY_COLUMNS,
    )

    risk_rows = _build_risk_register(
        field_audit=field_audit,
        transformation_audit=transformation_audit,
        proxy_candidates=proxy_candidates,
        pipeline_policy=pipeline_policy,
        missing_context_fields=missing_context_fields,
    )
    risk_register = pd.DataFrame(risk_rows, columns=_RISK_REGISTER_COLUMNS)
    risk_register.attrs["context_rows"] = context_rows

    return DataLeakageReport(
        target=target,
        identifiers=_unique(identifier_fields),
        candidate_features=_unique(candidate_fields),
        row_count=len(source),
        target_present=target_present,
        missing_identifiers=missing_identifiers,
        missing_candidate_features=missing_candidates,
        undeclared_fields=undeclared_fields,
        missing_context_fields=missing_context_fields,
        field_audit=field_audit,
        transformation_audit=transformation_audit,
        target_proxy_candidates=proxy_candidates,
        pipeline_policy=pipeline_policy,
        risk_register=risk_register,
        issues=pd.DataFrame(issues, columns=_ISSUE_COLUMNS),
    )


def _detect_target_proxy(
    feature: pd.Series,
    target: pd.Series,
    *,
    field: str,
    role: str,
    detect_exact: bool,
    detect_deterministic: bool,
) -> dict[str, object] | None:
    paired = pd.DataFrame(
        {
            "feature": feature.copy(deep=True),
            "target": target.copy(deep=True),
        }
    ).dropna()
    if paired.empty:
        return None

    exact_rate = float(paired["feature"].eq(paired["target"]).mean())
    normalized_feature = paired["feature"].map(_normalize_value)
    normalized_target = paired["target"].map(_normalize_value)
    normalized_rate = float(normalized_feature.eq(normalized_target).mean())

    method: str | None = None
    deterministic = False
    bidirectional = False
    severity = "Medium"

    if detect_exact and exact_rate == 1.0:
        method = "Exact target copy"
        deterministic = True
        bidirectional = True
        severity = "Critical"
    elif detect_exact and normalized_rate == 1.0:
        method = "Normalized target copy"
        deterministic = True
        bidirectional = True
        severity = "Critical"
    elif detect_deterministic:
        feature_cardinality = int(normalized_feature.nunique(dropna=True))
        target_cardinality = int(normalized_target.nunique(dropna=True))
        unique_ratio = feature_cardinality / len(paired)
        grouped_forward = paired.assign(
            normalized_feature=normalized_feature,
            normalized_target=normalized_target,
        ).groupby("normalized_feature", dropna=False)["normalized_target"].nunique()
        grouped_reverse = paired.assign(
            normalized_feature=normalized_feature,
            normalized_target=normalized_target,
        ).groupby("normalized_target", dropna=False)["normalized_feature"].nunique()
        deterministic = bool(not grouped_forward.empty and grouped_forward.max() == 1)
        bidirectional = bool(
            deterministic
            and not grouped_reverse.empty
            and grouped_reverse.max() == 1
        )

        role_is_identifier = role.casefold() == "identifier"
        low_cardinality = feature_cardinality <= max(10, target_cardinality * 5)
        repeated_values = unique_ratio <= 0.5

        if deterministic and bidirectional and feature_cardinality == target_cardinality:
            values = set(normalized_feature.unique())
            if values.issubset({0, 1, True, False, "0", "1", "true", "false"}):
                method = "Binary target encoding"
            else:
                method = "Bidirectional target mapping"
            severity = "High"
        elif (
            deterministic
            and low_cardinality
            and repeated_values
            and not role_is_identifier
        ):
            method = "Deterministic target mapping"
            severity = "High"

    if method is None:
        return None

    return {
        "Field": field,
        "Detection method": method,
        "Valid paired rows": len(paired),
        "Exact match rate": exact_rate,
        "Normalized match rate": normalized_rate,
        "Deterministic mapping": deterministic,
        "Bidirectional mapping": bidirectional,
        "Candidate severity": severity,
        "Interpretation": (
            "The field reproduces or deterministically encodes the target; "
            "it must not be used as a predictor without resolving provenance."
        ),
    }


def _build_risk_register(
    *,
    field_audit: pd.DataFrame,
    transformation_audit: pd.DataFrame,
    proxy_candidates: pd.DataFrame,
    pipeline_policy: pd.DataFrame,
    missing_context_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        scope: str,
        item: str,
        risk_type: str,
        severity: str,
        status: str,
        action: str,
    ) -> None:
        rows.append(
            {
                "Risk ID": f"LKG-{len(rows) + 1:03d}",
                "Scope": scope,
                "Item": item,
                "Risk type": risk_type,
                "Severity": severity,
                "Status": status,
                "Required action": action,
            }
        )

    for _, row in field_audit.iterrows():
        if bool(row["Direct target leakage"]):
            add(
                "Field",
                str(row["Field"]),
                "Direct target leakage",
                "Critical",
                "Open",
                "Keep the field outside every predictor matrix.",
            )
        elif bool(row["Identifier risk"]):
            add(
                "Field",
                str(row["Field"]),
                "Identifier memorization risk",
                "High",
                "Controlled" if row["Model disposition"] == "Exclude" else "Open",
                "Exclude the identifier from model inputs.",
            )
        elif bool(row["Temporal risk"]):
            add(
                "Field",
                str(row["Field"]),
                "Temporal availability unresolved",
                "Medium",
                "Open",
                "Confirm the field snapshot precedes the target event.",
            )

    for _, row in proxy_candidates.iterrows():
        add(
            "Field",
            str(row["Field"]),
            "Target proxy candidate",
            str(row["Candidate severity"]),
            "Open",
            "Exclude or prove that the detected mapping is not target-derived.",
        )

    for _, row in transformation_audit.iterrows():
        if bool(row["Target-derived"]):
            add(
                "Transformation",
                str(row["Transformation"]),
                "Target-derived transformation",
                "Critical",
                "Controlled"
                if row["Model disposition"] == "Exploratory only"
                else "Open",
                "Keep the result out of raw model inputs.",
            )
        if bool(row["Global-fit risk"]):
            add(
                "Transformation",
                str(row["Transformation"]),
                "Global-fit contamination",
                "High",
                "Open",
                "Fit the operation using training data only.",
            )
        if bool(row["Temporal risk"]):
            add(
                "Transformation",
                str(row["Transformation"]),
                "Aggregation cutoff unresolved",
                "High",
                "Open",
                "Prove that aggregation stops at the feature snapshot cutoff.",
            )

    if missing_context_fields:
        add(
            "Context",
            "Inference-time contract",
            "Temporal contract incomplete",
            "High",
            "Open",
            "Resolve: " + ", ".join(missing_context_fields) + ".",
        )

    for _, row in pipeline_policy.loc[~pipeline_policy["Compliant"]].iterrows():
        add(
            "Pipeline",
            str(row["Rule"]),
            "Train/test isolation policy unresolved",
            "High",
            "Open",
            "Adopt and enforce the declared train-only policy.",
        )

    return rows


def _field_risk_level(
    *,
    direct_target: bool,
    identifier_risk: bool,
    temporal_value: bool | None,
    inference_value: bool | None,
    disposition: str,
) -> str:
    if direct_target or disposition == "Prohibited":
        return "Critical"
    if identifier_risk or temporal_value is False or inference_value is False:
        return "High"
    if temporal_value is None or inference_value is None or disposition == "Unresolved":
        return "Medium"
    return "Low"


def _transformation_risk_level(
    *,
    target_derived: bool,
    global_fit_risk: bool,
    temporal_risk: bool,
    disposition: str,
) -> str:
    if target_derived or disposition == "Prohibited":
        return "Critical"
    if global_fit_risk or temporal_risk:
        return "High"
    if disposition in {"Unresolved", "Conditionally safe", "Train-only fit"}:
        return "Medium"
    return "Low"


def _context_value_resolved(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() not in {
            "unknown",
            "unresolved",
            "pending",
        }
    return True


def _context_interpretation(name: str, resolved: bool) -> str:
    if resolved:
        return "Declared for the current inference contract"
    return f"{name.replace('_', ' ')} must be resolved before modeling"


def _pipeline_rule_interpretation(rule: str, compliant: bool) -> str:
    readable = rule.replace("_", " ")
    if compliant:
        return f"Compliant: {readable}"
    return f"Unresolved or non-compliant: {readable}"


def _inferred_role(
    field: str,
    target: str,
    identifiers: tuple[str, ...],
    candidate_features: tuple[str, ...],
) -> str:
    if field == target:
        return "target"
    if field in identifiers:
        return "identifier"
    if field in candidate_features:
        return "candidate feature"
    return "other"


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise DataLeakageAnalysisError(
        f"Expected boolean or None, received {value!r}."
    )


def _normalize_value(value: object) -> object:
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _issue(
    scope: str,
    item: str,
    issue: str,
    details: str,
    impact: str,
) -> dict[str, object]:
    return {
        "Scope": scope,
        "Item": item,
        "Issue": issue,
        "Details": details,
        "Potential impact": impact,
    }


# ---------------------------------------------------------------------------
# Static classification leakage and derived-feature dependency audit
# ---------------------------------------------------------------------------

_STATIC_SUMMARY_COLUMNS: Final[list[str]] = [
    "Metric",
    "Value",
    "Interpretation",
]

_DERIVED_DEPENDENCY_COLUMNS: Final[list[str]] = [
    "Derived feature",
    "Source features",
    "Formula",
    "Auditable from retained columns",
    "Valid rows",
    "Match rate",
    "Maximum absolute error",
    "Maximum relative error",
    "Target-derived",
    "Dependency status",
    "Interpretation",
]

_STATIC_ISSUE_COLUMNS: Final[list[str]] = [
    "Scope",
    "Item",
    "Issue",
    "Details",
]


@dataclass(frozen=True, slots=True)
class DerivedFeatureDependencySpec:
    """Declare one source-independent mathematical feature dependency."""

    feature: str
    sources: tuple[str, ...]
    formula: str
    operation: str | None
    note: str = ""


@dataclass(frozen=True, slots=True)
class StaticClassificationLeakageReport:
    """Audit direct target leakage and deterministic feature dependencies."""

    target: str
    candidate_features: tuple[str, ...]
    identifiers: tuple[str, ...]
    row_count: int
    target_present: bool
    missing_candidate_features: tuple[str, ...]
    missing_identifiers: tuple[str, ...]
    target_proxy_candidates: pd.DataFrame
    dependency_audit: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_target_proxy_candidates(self) -> bool:
        """Return whether a candidate directly or deterministically encodes target."""
        return not self.target_proxy_candidates.empty

    @property
    def has_target_derived_dependencies(self) -> bool:
        """Return whether a declared derived feature depends on the target."""
        if self.dependency_audit.empty:
            return False
        return bool(self.dependency_audit["Target-derived"].any())

    @property
    def confirmed_derived_dependency_count(self) -> int:
        """Return the number of numerically confirmed retained dependencies."""
        if self.dependency_audit.empty:
            return 0
        return int(
            self.dependency_audit["Dependency status"]
            .eq("Confirmed from retained columns")
            .sum()
        )

    @property
    def external_dependency_count(self) -> int:
        """Return dependencies documented but not reconstructible from table."""
        if self.dependency_audit.empty:
            return 0
        return int(
            self.dependency_audit["Dependency status"]
            .eq("Documented; source column not retained")
            .sum()
        )

    @property
    def has_direct_target_leakage(self) -> bool:
        """Return whether static leakage evidence requires resolution."""
        return bool(
            self.has_target_proxy_candidates
            or self.has_target_derived_dependencies
            or self.target in self.candidate_features
        )

    @property
    def is_structurally_valid(self) -> bool:
        """Return whether required fields and declarations are coherent."""
        return self.issues.empty

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact leakage/dependency summary for notebooks."""
        rows = [
            {
                "Metric": "Rows audited",
                "Value": self.row_count,
                "Interpretation": "Observations included in the static audit",
            },
            {
                "Metric": "Candidate features",
                "Value": len(self.candidate_features),
                "Interpretation": "Predictor candidates reviewed for target leakage",
            },
            {
                "Metric": "Source identifiers",
                "Value": len(self.identifiers),
                "Interpretation": "Identifier fields kept outside predictor semantics",
            },
            {
                "Metric": "Target proxy candidates",
                "Value": len(self.target_proxy_candidates),
                "Interpretation": "Simple deterministic target encodings detected",
            },
            {
                "Metric": "Confirmed derived dependencies",
                "Value": self.confirmed_derived_dependency_count,
                "Interpretation": "Features numerically reproduced from retained inputs",
            },
            {
                "Metric": "External-source dependencies",
                "Value": self.external_dependency_count,
                "Interpretation": "Documented derived features requiring an omitted primitive",
            },
            {
                "Metric": "Target-derived dependencies",
                "Value": int(self.has_target_derived_dependencies),
                "Interpretation": "Derived features whose declared formula uses the target",
            },
            {
                "Metric": "Direct target leakage detected",
                "Value": self.has_direct_target_leakage,
                "Interpretation": (
                    "No direct target leakage evidence found"
                    if not self.has_direct_target_leakage
                    else "Leakage evidence requires resolution before modeling"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_STATIC_SUMMARY_COLUMNS)

    def dependency_frame(self) -> pd.DataFrame:
        """Return the derived-feature dependency audit."""
        return self.dependency_audit.copy(deep=True)

    def target_proxy_candidates_frame(self) -> pd.DataFrame:
        """Return direct target-proxy evidence."""
        return self.target_proxy_candidates.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return structural audit issues."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(self) -> None:
        """Raise when required fields or dependency declarations are invalid."""
        if self.issues.empty:
            return
        details = "; ".join(
            f"{row['Item']}: {row['Issue']}"
            for _, row in self.issues.iterrows()
        )
        raise DataLeakageAnalysisError(
            "Static classification leakage audit is structurally invalid: "
            + details
        )


def ratio_dependency(
    feature: str,
    *,
    numerator: str,
    denominator: str,
    note: str = "",
) -> DerivedFeatureDependencySpec:
    """Declare feature = numerator / denominator."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=(numerator, denominator),
        formula=f"{numerator} / {denominator}",
        operation="ratio",
        note=note,
    )


def ellipse_eccentricity_dependency(
    feature: str,
    *,
    major_axis: str,
    minor_axis: str,
    note: str = "",
) -> DerivedFeatureDependencySpec:
    """Declare ellipse eccentricity from major and minor axes."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=(major_axis, minor_axis),
        formula=f"sqrt(1 - ({minor_axis} / {major_axis})^2)",
        operation="ellipse_eccentricity",
        note=note,
    )


def equivalent_circle_diameter_dependency(
    feature: str,
    *,
    area: str,
    note: str = "",
) -> DerivedFeatureDependencySpec:
    """Declare the diameter of a circle with the supplied area."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=(area,),
        formula=f"sqrt(4 * {area} / pi)",
        operation="equivalent_circle_diameter",
        note=note,
    )


def roundness_dependency(
    feature: str,
    *,
    area: str,
    perimeter: str,
    note: str = "",
) -> DerivedFeatureDependencySpec:
    """Declare circularity/roundness = 4*pi*area/perimeter^2."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=(area, perimeter),
        formula=f"4 * pi * {area} / {perimeter}^2",
        operation="roundness",
        note=note,
    )


def shape_factor_3_dependency(
    feature: str,
    *,
    area: str,
    major_axis: str,
    note: str = "",
) -> DerivedFeatureDependencySpec:
    """Declare SF3 = 4*area/(pi*major_axis^2)."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=(area, major_axis),
        formula=f"4 * {area} / (pi * {major_axis}^2)",
        operation="shape_factor_3",
        note=note,
    )


def shape_factor_4_dependency(
    feature: str,
    *,
    area: str,
    major_axis: str,
    minor_axis: str,
    note: str = "",
) -> DerivedFeatureDependencySpec:
    """Declare SF4 = 4*area/(pi*major_axis*minor_axis)."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=(area, major_axis, minor_axis),
        formula=f"4 * {area} / (pi * {major_axis} * {minor_axis})",
        operation="shape_factor_4",
        note=note,
    )


def external_dependency(
    feature: str,
    *,
    sources: Sequence[str],
    formula: str,
    note: str,
) -> DerivedFeatureDependencySpec:
    """Declare a documented dependency that cannot be reconstructed here."""
    return DerivedFeatureDependencySpec(
        feature=feature,
        sources=tuple(str(value) for value in sources),
        formula=formula,
        operation=None,
        note=note,
    )


def analyze_static_classification_leakage(
    dataframe: pd.DataFrame,
    *,
    target: str,
    candidate_features: Sequence[str],
    identifiers: Sequence[str] | None = None,
    derived_dependencies: Sequence[DerivedFeatureDependencySpec] = (),
    relative_tolerance: float = 1e-4,
    absolute_tolerance: float = 1e-9,
    confirmation_rate: float = 0.99,
) -> StaticClassificationLeakageReport:
    """Audit static classification leakage without temporal-event assumptions."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(target, str) or not target:
        raise DataLeakageAnalysisError("target must be a non-empty field name")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise DataLeakageAnalysisError("dependency tolerances must be non-negative")
    if not 0 <= confirmation_rate <= 1:
        raise DataLeakageAnalysisError("confirmation_rate must be between 0 and 1")

    source = dataframe.copy(deep=True)
    candidate_fields = _unique(tuple(str(value) for value in candidate_features))
    identifier_fields = _unique(
        tuple(str(value) for value in (identifiers or ()))
    )
    specs = tuple(derived_dependencies)
    issues: list[dict[str, object]] = []

    target_present = target in source.columns
    missing_candidates = tuple(
        field for field in candidate_fields if field not in source.columns
    )
    missing_identifiers = tuple(
        field for field in identifier_fields if field not in source.columns
    )

    if not target_present:
        issues.append(
            _static_issue("Fields", target, "Missing target", "Target column is absent.")
        )
    for field in missing_candidates:
        issues.append(
            _static_issue(
                "Fields", field, "Missing candidate feature", "Candidate column is absent."
            )
        )
    for field in missing_identifiers:
        issues.append(
            _static_issue(
                "Fields", field, "Missing identifier", "Declared identifier is absent."
            )
        )
    if target in candidate_fields:
        issues.append(
            _static_issue(
                "Fields",
                target,
                "Target included as candidate feature",
                "The response variable must remain outside every predictor matrix.",
            )
        )
    for field in candidate_fields:
        if field in set(identifier_fields):
            issues.append(
                _static_issue(
                    "Fields",
                    field,
                    "Identifier included as candidate feature",
                    "Identifier semantics must remain outside predictor inputs.",
                )
            )

    seen_derived: set[str] = set()
    dependency_rows: list[dict[str, object]] = []
    for spec in specs:
        if not isinstance(spec, DerivedFeatureDependencySpec):
            raise TypeError(
                "derived_dependencies must contain DerivedFeatureDependencySpec values"
            )
        if spec.feature in seen_derived:
            issues.append(
                _static_issue(
                    "Dependency",
                    spec.feature,
                    "Duplicate dependency declaration",
                    "Each derived feature may be declared only once.",
                )
            )
        seen_derived.add(spec.feature)

        target_derived = target in spec.sources
        feature_present = spec.feature in source.columns
        missing_sources = tuple(
            name for name in spec.sources if name not in source.columns
        )

        if not feature_present:
            issues.append(
                _static_issue(
                    "Dependency",
                    spec.feature,
                    "Derived feature missing",
                    "Declared derived feature is absent from the dataframe.",
                )
            )

        if spec.operation is not None and missing_sources:
            issues.append(
                _static_issue(
                    "Dependency",
                    spec.feature,
                    "Dependency source missing",
                    f"Cannot evaluate retained dependency; missing {list(missing_sources)}.",
                )
            )

        auditable = bool(
            spec.operation is not None
            and feature_present
            and not missing_sources
        )
        valid_rows = 0
        match_rate: float | None = None
        max_abs_error: float | None = None
        max_rel_error: float | None = None

        if auditable:
            expected = _evaluate_derived_dependency(source, spec)
            observed = pd.to_numeric(source[spec.feature], errors="coerce")
            expected = pd.to_numeric(expected, errors="coerce")
            valid_mask = observed.notna() & expected.notna()
            valid_rows = int(valid_mask.sum())

            if valid_rows:
                observed_valid = observed.loc[valid_mask].astype(float)
                expected_valid = expected.loc[valid_mask].astype(float)
                abs_error = (observed_valid - expected_valid).abs()
                tolerance = absolute_tolerance + relative_tolerance * expected_valid.abs()
                matches = abs_error <= tolerance
                match_rate = float(matches.mean())
                max_abs_error = float(abs_error.max())
                denominator = expected_valid.abs().where(
                    expected_valid.abs() > absolute_tolerance
                )
                relative_error = (abs_error / denominator).dropna()
                max_rel_error = (
                    float(relative_error.max())
                    if not relative_error.empty
                    else 0.0
                )
            else:
                match_rate = 0.0

        if target_derived:
            status = "Target-derived leakage"
            interpretation = (
                "The declared feature dependency uses the target and must not be a predictor."
            )
        elif spec.operation is None:
            status = "Documented; source column not retained"
            interpretation = spec.note or (
                "The feature is derived, but at least one primitive source is not retained."
            )
        elif match_rate is not None and match_rate >= confirmation_rate:
            status = "Confirmed from retained columns"
            interpretation = (
                "The released feature is numerically reproducible from other retained features; "
                "treat it as structural redundancy, not target leakage."
            )
        else:
            status = "Declared dependency not confirmed"
            interpretation = (
                "The declared mathematical relationship is not reproduced at the configured "
                "tolerance and should be reviewed before relying on it."
            )

        dependency_rows.append(
            {
                "Derived feature": spec.feature,
                "Source features": spec.sources,
                "Formula": spec.formula,
                "Auditable from retained columns": auditable,
                "Valid rows": valid_rows,
                "Match rate": match_rate,
                "Maximum absolute error": max_abs_error,
                "Maximum relative error": max_rel_error,
                "Target-derived": target_derived,
                "Dependency status": status,
                "Interpretation": interpretation,
            }
        )

    proxy_rows: list[dict[str, object]] = []
    if target_present:
        target_values = source[target].copy(deep=True)
        for field in candidate_fields:
            if field not in source.columns:
                continue
            candidate = _detect_target_proxy(
                source[field],
                target_values,
                field=field,
                role="candidate feature",
                detect_exact=True,
                detect_deterministic=True,
            )
            if candidate is not None:
                proxy_rows.append(candidate)

    return StaticClassificationLeakageReport(
        target=target,
        candidate_features=candidate_fields,
        identifiers=identifier_fields,
        row_count=len(source),
        target_present=target_present,
        missing_candidate_features=missing_candidates,
        missing_identifiers=missing_identifiers,
        target_proxy_candidates=pd.DataFrame(proxy_rows, columns=_PROXY_COLUMNS),
        dependency_audit=pd.DataFrame(
            dependency_rows,
            columns=_DERIVED_DEPENDENCY_COLUMNS,
        ),
        issues=pd.DataFrame(issues, columns=_STATIC_ISSUE_COLUMNS),
    )


def _evaluate_derived_dependency(
    dataframe: pd.DataFrame,
    spec: DerivedFeatureDependencySpec,
) -> pd.Series:
    """Evaluate a supported derived-feature relationship without eval()."""
    values = {
        name: pd.to_numeric(dataframe[name], errors="coerce")
        for name in spec.sources
    }

    if spec.operation == "ratio":
        numerator, denominator = spec.sources
        return values[numerator] / values[denominator]
    if spec.operation == "ellipse_eccentricity":
        major_axis, minor_axis = spec.sources
        inside = 1.0 - (values[minor_axis] / values[major_axis]).pow(2)
        return inside.clip(lower=0.0).pow(0.5)
    if spec.operation == "equivalent_circle_diameter":
        (area,) = spec.sources
        return (4.0 * values[area] / math.pi).pow(0.5)
    if spec.operation == "roundness":
        area, perimeter = spec.sources
        return 4.0 * math.pi * values[area] / values[perimeter].pow(2)
    if spec.operation == "shape_factor_3":
        area, major_axis = spec.sources
        return 4.0 * values[area] / (math.pi * values[major_axis].pow(2))
    if spec.operation == "shape_factor_4":
        area, major_axis, minor_axis = spec.sources
        return 4.0 * values[area] / (
            math.pi * values[major_axis] * values[minor_axis]
        )

    raise DataLeakageAnalysisError(
        f"Unsupported derived dependency operation: {spec.operation!r}."
    )


def _static_issue(
    scope: str,
    item: str,
    issue: str,
    details: str,
) -> dict[str, object]:
    return {
        "Scope": scope,
        "Item": item,
        "Issue": issue,
        "Details": details,
    }


# ---------------------------------------------------------------------------
# Static continuous-regression leakage and derived-feature dependency audit
# ---------------------------------------------------------------------------

_CONTINUOUS_PROXY_COLUMNS: Final[list[str]] = [
    "Field",
    "Detection method",
    "Valid paired rows",
    "Slope",
    "Intercept",
    "Match rate",
    "Maximum absolute error",
    "Maximum relative error",
    "Candidate severity",
    "Interpretation",
]


@dataclass(frozen=True, slots=True)
class StaticContinuousRegressionLeakageReport:
    """Audit continuous-target leakage and deterministic feature dependencies."""

    target: str
    candidate_features: tuple[str, ...]
    identifiers: tuple[str, ...]
    row_count: int
    target_present: bool
    missing_candidate_features: tuple[str, ...]
    missing_identifiers: tuple[str, ...]
    target_proxy_candidates: pd.DataFrame
    dependency_audit: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_target_proxy_candidates(self) -> bool:
        """Return whether a candidate numerically reproduces the target."""
        return not self.target_proxy_candidates.empty

    @property
    def has_target_derived_dependencies(self) -> bool:
        """Return whether a declared derived feature depends on the target."""
        if self.dependency_audit.empty:
            return False
        return bool(self.dependency_audit["Target-derived"].any())

    @property
    def confirmed_derived_dependency_count(self) -> int:
        """Return numerically confirmed retained dependencies."""
        if self.dependency_audit.empty:
            return 0
        return int(
            self.dependency_audit["Dependency status"]
            .eq("Confirmed from retained columns")
            .sum()
        )

    @property
    def external_dependency_count(self) -> int:
        """Return documented dependencies requiring an omitted primitive."""
        if self.dependency_audit.empty:
            return 0
        return int(
            self.dependency_audit["Dependency status"]
            .eq("Documented; source column not retained")
            .sum()
        )

    @property
    def has_direct_target_leakage(self) -> bool:
        """Return whether static leakage evidence requires resolution."""
        return bool(
            self.has_target_proxy_candidates
            or self.has_target_derived_dependencies
            or self.target in self.candidate_features
        )

    @property
    def is_structurally_valid(self) -> bool:
        """Return whether required fields and declarations are coherent."""
        return self.issues.empty

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact continuous-regression leakage summary."""
        rows = [
            {
                "Metric": "Rows audited",
                "Value": self.row_count,
                "Interpretation": "Observations included in the static audit",
            },
            {
                "Metric": "Candidate features",
                "Value": len(self.candidate_features),
                "Interpretation": "Predictor candidates reviewed for target leakage",
            },
            {
                "Metric": "Source identifiers",
                "Value": len(self.identifiers),
                "Interpretation": "Identifier fields kept outside predictor semantics",
            },
            {
                "Metric": "Target proxy candidates",
                "Value": len(self.target_proxy_candidates),
                "Interpretation": (
                    "Exact or affine numerical target encodings detected"
                ),
            },
            {
                "Metric": "Declared derived dependencies",
                "Value": len(self.dependency_audit),
                "Interpretation": (
                    "Source-backed mathematical feature dependencies audited"
                ),
            },
            {
                "Metric": "Confirmed derived dependencies",
                "Value": self.confirmed_derived_dependency_count,
                "Interpretation": (
                    "Features numerically reproduced from retained inputs"
                ),
            },
            {
                "Metric": "External-source dependencies",
                "Value": self.external_dependency_count,
                "Interpretation": (
                    "Documented derived features requiring an omitted primitive"
                ),
            },
            {
                "Metric": "Target-derived dependencies",
                "Value": int(self.has_target_derived_dependencies),
                "Interpretation": (
                    "Derived features whose declared formula uses the target"
                ),
            },
            {
                "Metric": "Direct target leakage detected",
                "Value": self.has_direct_target_leakage,
                "Interpretation": (
                    "No direct target leakage evidence found"
                    if not self.has_direct_target_leakage
                    else "Leakage evidence requires resolution before modeling"
                ),
            },
        ]
        return pd.DataFrame(rows, columns=_STATIC_SUMMARY_COLUMNS)

    def dependency_frame(self) -> pd.DataFrame:
        """Return the derived-feature dependency audit."""
        return self.dependency_audit.copy(deep=True)

    def target_proxy_candidates_frame(self) -> pd.DataFrame:
        """Return numerical target-proxy evidence."""
        return self.target_proxy_candidates.copy(deep=True)

    def issues_frame(self) -> pd.DataFrame:
        """Return structural audit issues."""
        return self.issues.copy(deep=True)

    def raise_if_invalid(self) -> None:
        """Raise when required fields or dependency declarations are invalid."""
        if self.issues.empty:
            return
        details = "; ".join(
            f"{row['Item']}: {row['Issue']}"
            for _, row in self.issues.iterrows()
        )
        raise DataLeakageAnalysisError(
            "Static continuous-regression leakage audit is structurally invalid: "
            + details
        )


def analyze_static_continuous_regression_leakage(
    dataframe: pd.DataFrame,
    *,
    target: str,
    candidate_features: Sequence[str],
    identifiers: Sequence[str] | None = None,
    derived_dependencies: Sequence[DerivedFeatureDependencySpec] = (),
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
    confirmation_rate: float = 0.99,
) -> StaticContinuousRegressionLeakageReport:
    """Audit static continuous-regression leakage without fitting a model."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(target, str) or not target:
        raise DataLeakageAnalysisError("target must be a non-empty field name")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise DataLeakageAnalysisError("dependency tolerances must be non-negative")
    if not 0 <= confirmation_rate <= 1:
        raise DataLeakageAnalysisError("confirmation_rate must be between 0 and 1")

    source = dataframe.copy(deep=True)
    candidate_fields = _unique(tuple(str(value) for value in candidate_features))
    identifier_fields = _unique(
        tuple(str(value) for value in (identifiers or ()))
    )
    specs = tuple(derived_dependencies)
    issues: list[dict[str, object]] = []

    target_present = target in source.columns
    missing_candidates = tuple(
        field for field in candidate_fields if field not in source.columns
    )
    missing_identifiers = tuple(
        field for field in identifier_fields if field not in source.columns
    )

    if not target_present:
        issues.append(
            _static_issue("Fields", target, "Missing target", "Target column is absent.")
        )
    for field in missing_candidates:
        issues.append(
            _static_issue(
                "Fields",
                field,
                "Missing candidate feature",
                "Candidate column is absent.",
            )
        )
    for field in missing_identifiers:
        issues.append(
            _static_issue(
                "Fields",
                field,
                "Missing identifier",
                "Declared identifier is absent.",
            )
        )
    if target in candidate_fields:
        issues.append(
            _static_issue(
                "Fields",
                target,
                "Target included as candidate feature",
                "The response variable must remain outside every predictor matrix.",
            )
        )
    for field in candidate_fields:
        if field in set(identifier_fields):
            issues.append(
                _static_issue(
                    "Fields",
                    field,
                    "Identifier included as candidate feature",
                    "Identifier semantics must remain outside predictor inputs.",
                )
            )

    dependency_rows = _audit_static_derived_dependencies(
        source,
        target=target,
        specs=specs,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        confirmation_rate=confirmation_rate,
        issues=issues,
    )

    proxy_rows: list[dict[str, object]] = []
    if target_present:
        target_values = source[target].copy(deep=True)
        numeric_target = pd.to_numeric(target_values, errors="coerce")
        invalid_target_count = int(
            target_values.notna().sum() - numeric_target.notna().sum()
        )
        if invalid_target_count:
            issues.append(
                _static_issue(
                    "Target",
                    target,
                    "Non-numeric target values",
                    (
                        f"{invalid_target_count} non-missing target values cannot "
                        "be interpreted as numbers."
                    ),
                )
            )
        elif numeric_target.dropna().nunique() < 2:
            issues.append(
                _static_issue(
                    "Target",
                    target,
                    "Insufficient target variation",
                    "Continuous-regression leakage audit requires target variation.",
                )
            )
        else:
            for field in candidate_fields:
                if field not in source.columns or field == target:
                    continue
                candidate = _detect_continuous_target_proxy(
                    source[field],
                    target_values,
                    field=field,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    confirmation_rate=confirmation_rate,
                )
                if candidate is not None:
                    proxy_rows.append(candidate)

    return StaticContinuousRegressionLeakageReport(
        target=target,
        candidate_features=candidate_fields,
        identifiers=identifier_fields,
        row_count=len(source),
        target_present=target_present,
        missing_candidate_features=missing_candidates,
        missing_identifiers=missing_identifiers,
        target_proxy_candidates=pd.DataFrame(
            proxy_rows,
            columns=_CONTINUOUS_PROXY_COLUMNS,
        ),
        dependency_audit=pd.DataFrame(
            dependency_rows,
            columns=_DERIVED_DEPENDENCY_COLUMNS,
        ),
        issues=pd.DataFrame(issues, columns=_STATIC_ISSUE_COLUMNS),
    )


def _audit_static_derived_dependencies(
    dataframe: pd.DataFrame,
    *,
    target: str,
    specs: Sequence[DerivedFeatureDependencySpec],
    relative_tolerance: float,
    absolute_tolerance: float,
    confirmation_rate: float,
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Audit declared deterministic feature dependencies."""
    seen_derived: set[str] = set()
    dependency_rows: list[dict[str, object]] = []

    for spec in specs:
        if not isinstance(spec, DerivedFeatureDependencySpec):
            raise TypeError(
                "derived_dependencies must contain DerivedFeatureDependencySpec values"
            )
        if spec.feature in seen_derived:
            issues.append(
                _static_issue(
                    "Dependency",
                    spec.feature,
                    "Duplicate dependency declaration",
                    "Each derived feature may be declared only once.",
                )
            )
        seen_derived.add(spec.feature)

        target_derived = target in spec.sources
        feature_present = spec.feature in dataframe.columns
        missing_sources = tuple(
            name for name in spec.sources if name not in dataframe.columns
        )

        if not feature_present:
            issues.append(
                _static_issue(
                    "Dependency",
                    spec.feature,
                    "Derived feature missing",
                    "Declared derived feature is absent from the dataframe.",
                )
            )

        if spec.operation is not None and missing_sources:
            issues.append(
                _static_issue(
                    "Dependency",
                    spec.feature,
                    "Dependency source missing",
                    f"Cannot evaluate retained dependency; missing {list(missing_sources)}.",
                )
            )

        auditable = bool(
            spec.operation is not None
            and feature_present
            and not missing_sources
        )
        valid_rows = 0
        match_rate: float | None = None
        max_abs_error: float | None = None
        max_rel_error: float | None = None

        if auditable:
            expected = _evaluate_derived_dependency(dataframe, spec)
            observed = pd.to_numeric(dataframe[spec.feature], errors="coerce")
            expected = pd.to_numeric(expected, errors="coerce")
            valid_mask = observed.notna() & expected.notna()
            valid_rows = int(valid_mask.sum())

            if valid_rows:
                observed_valid = observed.loc[valid_mask].astype(float)
                expected_valid = expected.loc[valid_mask].astype(float)
                abs_error = (observed_valid - expected_valid).abs()
                tolerance = (
                    absolute_tolerance
                    + relative_tolerance * expected_valid.abs()
                )
                matches = abs_error <= tolerance
                match_rate = float(matches.mean())
                max_abs_error = float(abs_error.max())
                denominator = expected_valid.abs().where(
                    expected_valid.abs() > absolute_tolerance
                )
                relative_error = (abs_error / denominator).dropna()
                max_rel_error = (
                    float(relative_error.max())
                    if not relative_error.empty
                    else 0.0
                )
            else:
                match_rate = 0.0

        if target_derived:
            status = "Target-derived leakage"
            interpretation = (
                "The declared feature dependency uses the target and must not "
                "be a predictor."
            )
        elif spec.operation is None:
            status = "Documented; source column not retained"
            interpretation = spec.note or (
                "The feature is derived, but at least one primitive source is "
                "not retained."
            )
        elif match_rate is not None and match_rate >= confirmation_rate:
            status = "Confirmed from retained columns"
            interpretation = (
                "The released feature is numerically reproducible from other "
                "retained features; treat it as structural redundancy, not "
                "target leakage."
            )
        else:
            status = "Declared dependency not confirmed"
            interpretation = (
                "The declared mathematical relationship is not reproduced at "
                "the configured tolerance and should be reviewed before relying "
                "on it."
            )

        dependency_rows.append(
            {
                "Derived feature": spec.feature,
                "Source features": spec.sources,
                "Formula": spec.formula,
                "Auditable from retained columns": auditable,
                "Valid rows": valid_rows,
                "Match rate": match_rate,
                "Maximum absolute error": max_abs_error,
                "Maximum relative error": max_rel_error,
                "Target-derived": target_derived,
                "Dependency status": status,
                "Interpretation": interpretation,
            }
        )

    return dependency_rows


def _detect_continuous_target_proxy(
    feature: pd.Series,
    target: pd.Series,
    *,
    field: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    confirmation_rate: float,
) -> dict[str, object] | None:
    """Detect exact or affine numerical reproductions of a continuous target."""
    paired = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature.copy(deep=True), errors="coerce"),
            "target": pd.to_numeric(target.copy(deep=True), errors="coerce"),
        }
    ).dropna()
    if len(paired) < 3:
        return None

    finite_mask = paired.apply(
        lambda column: column.map(
            lambda value: math.isfinite(float(value))
        )
    ).all(axis=1)
    paired = paired.loc[finite_mask].astype(float)
    if len(paired) < 3 or paired["target"].nunique() < 3:
        return None

    target_values = paired["target"]
    feature_values = paired["feature"]

    def evidence(
        predicted: pd.Series,
    ) -> tuple[float, float, float]:
        abs_error = (feature_values - predicted).abs()
        tolerance = (
            absolute_tolerance + relative_tolerance * predicted.abs()
        )
        match_rate = float((abs_error <= tolerance).mean())
        max_abs_error = float(abs_error.max())
        denominator = predicted.abs().where(
            predicted.abs() > absolute_tolerance
        )
        relative_error = (abs_error / denominator).dropna()
        max_rel_error = (
            float(relative_error.max())
            if not relative_error.empty
            else 0.0
        )
        return match_rate, max_abs_error, max_rel_error

    exact_rate, exact_abs, exact_rel = evidence(target_values)
    if exact_rate >= confirmation_rate:
        return {
            "Field": field,
            "Detection method": "Exact numeric target copy",
            "Valid paired rows": len(paired),
            "Slope": 1.0,
            "Intercept": 0.0,
            "Match rate": exact_rate,
            "Maximum absolute error": exact_abs,
            "Maximum relative error": exact_rel,
            "Candidate severity": "Critical",
            "Interpretation": (
                "The field numerically reproduces the target and must not be "
                "used as a predictor without resolving provenance."
            ),
        }

    centered_target = target_values - float(target_values.mean())
    target_ss = float(centered_target.pow(2).sum())
    if target_ss <= 0:
        return None

    centered_feature = feature_values - float(feature_values.mean())
    slope = float((centered_target * centered_feature).sum() / target_ss)
    intercept = float(feature_values.mean() - slope * target_values.mean())
    predicted = target_values * slope + intercept

    match_rate, max_abs_error, max_rel_error = evidence(predicted)
    if match_rate < confirmation_rate:
        return None

    residual_ss = float((feature_values - predicted).pow(2).sum())
    feature_ss = float(centered_feature.pow(2).sum())
    r_squared = (
        1.0
        if feature_ss == 0.0 and residual_ss == 0.0
        else (
            1.0 - residual_ss / feature_ss
            if feature_ss > 0.0
            else 0.0
        )
    )
    if r_squared < confirmation_rate:
        return None

    return {
        "Field": field,
        "Detection method": "Affine numeric target transform",
        "Valid paired rows": len(paired),
        "Slope": slope,
        "Intercept": intercept,
        "Match rate": match_rate,
        "Maximum absolute error": max_abs_error,
        "Maximum relative error": max_rel_error,
        "Candidate severity": "Critical",
        "Interpretation": (
            "The field is an effectively affine numerical transform of the "
            "target. Treat it as a target-proxy candidate until provenance "
            "proves otherwise."
        ),
    }
