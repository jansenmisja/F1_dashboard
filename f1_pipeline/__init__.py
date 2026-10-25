"""
F1 Fastest Lap pipeline package.
"""

from .config import (
    CONFIG,
    PipelineConfig,
    PRIMARY_KEY,
    TRAINING_YEAR_MIN,
    TRAINING_YEAR_MAX,
    OPERATIONAL_BASELINE_COLUMN,
    OPERATIONAL_PREDICTION_COLUMN,
    PREDICTION_COLUMNS,
    PREDICTION_OFFSET_COLUMNS,
    ACTUAL_COLUMN,
)

__all__ = [
    "CONFIG",
    "PipelineConfig",
    "PRIMARY_KEY",
    "TRAINING_YEAR_MIN",
    "TRAINING_YEAR_MAX",
    "OPERATIONAL_BASELINE_COLUMN",
    "OPERATIONAL_PREDICTION_COLUMN",
    "PREDICTION_COLUMNS",
    "PREDICTION_OFFSET_COLUMNS",
    "ACTUAL_COLUMN",
]


from .schemas import (
    ContractResult,
    get_target_mask,
    validate_event_uniqueness,
    require_one_event,
    assert_training_year_guard,
    validate_training_contract,
    validate_test_contract,
    validate_overview_contract,
    validate_operational_dataset_contracts,
)


from .io import (
    CommitResult,
    read_csv_checked,
    round_float_columns,
    backup_csv,
    safe_commit_csv,
    safe_commit_many,
    io_preflight,
)


from .lifecycle import (
    WORKFLOW_STATUS_ORDER,
    STATUS_INDEX,
    TransitionResult,
    EventState,
    validate_status_transition,
    expected_next_status,
    get_event_state,
    transition_event_status,
)


from .registry import (
    REGISTRY_REQUIRED_COLUMNS,
    RegistryState,
    build_race_event_id,
    validate_registry,
    bootstrap_authoritative_registry,
    regenerate_eligible_mirror,
    sync_event_registry,
    get_registry_state,
)


from .readiness import (
    GateResult,
    ReadinessReport,
    ACCEPTED_RECORDLAP_STATUSES,
    validate_qualifying,
    validate_event_timing,
    validate_race_weather,
    validate_recordlap_readiness,
    validate_identity_layout,
    validate_feature_snapshot,
    validate_race_context_plausibility,
    validate_leakage_guard,
    build_prediction_readiness_report,
    prediction_readiness_ok,
)


from .readiness import (
    PreparePredictionReadyResult,
    prepare_prediction_ready,
)


from .prediction import (
    PredictionDependencyState,
    dependency_preflight,
    assert_prediction_dependencies,
    run_final_prediction,
    validate_final_prediction_artifacts,
    promote_prediction_to_predicted,
)
