"""Search framework: orchestrator, context, pipeline, evolution."""

from .context import FactorCatalog, FactorCatalogEntry, SearchContext, SearchStrategy, build_individual
from .context import StatefulStrategy, StrategySnapshot
from .evolution import BreedingSpec, EvalResult, FitnessEngine, FitnessPolicy, Individual, SearchResult
from .orchestrator import SearchOrchestrator
from .pipeline import ArchiveEntry, Lineage, PipelineRecord, RoundRecord, StageKind, StageRecord
from .enumerator import FormulaEnumerator
from .checkpoint import CheckpointManager, SearchCheckpoint
