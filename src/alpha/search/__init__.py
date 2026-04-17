"""Search framework: orchestrator, context, pipeline, evolution."""

from .checkpoint import CheckpointManager, SearchCheckpoint
from .context import (
    FactorCatalog,
    FactorCatalogEntry,
    SearchContext,
    SearchStrategy,
    StatefulStrategy,
    StrategySnapshot,
    build_individual,
)
from .enumerator import FormulaEnumerator
from .evolution import BreedingSpec, EvalResult, FitnessEngine, FitnessPolicy, Individual, SearchResult
from .orchestrator import SearchOrchestrator
from .pipeline import ArchiveEntry, Lineage, PipelineRecord, RoundRecord, StageKind, StageRecord
