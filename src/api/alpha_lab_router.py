"""
Alpha Lab API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.alpha_lab import AlphaLabService
from src.config.settings import get_alpha_lab_config, get_bitget_config


router = APIRouter(prefix="/alpha-lab", tags=["alpha-lab"])
service = AlphaLabService()


class CompileRequest(BaseModel):
    formula: str


class EvaluateRequest(BaseModel):
    formula: str
    fields: dict[str, list[list[float]]]
    liquidity_mask: list[list[bool]] | None = None
    session_mask: list[list[bool]] | None = None


class SeedPopulationRequest(BaseModel):
    seeds: list[str] = Field(default_factory=list)
    population_size: int = 8


class BreedRequest(BaseModel):
    formulas: list[str] = Field(default_factory=list)
    offspring_count: int = 4


@router.get("/operators")
async def list_operators():
    return {
        "operators": service.list_operators(),
        "defaults": {
            "alpha_lab": get_alpha_lab_config().model_dump(),
            "bitget": get_bitget_config().model_dump(),
        },
    }


@router.post("/validate")
async def validate_formula(request: CompileRequest):
    return service.validate_formula(request.formula)


@router.post("/compile")
async def compile_formula(request: CompileRequest):
    try:
        return service.compile_formula(request.formula)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluate")
async def evaluate_formula(request: EvaluateRequest):
    try:
        return service.evaluate_formula(
            formula=request.formula,
            fields=request.fields,
            liquidity_mask=request.liquidity_mask,
            session_mask=request.session_mask,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/population/seed")
async def seed_population(request: SeedPopulationRequest):
    return {
        "population": service.seed_population(request.seeds, request.population_size),
    }


@router.post("/population/breed")
async def breed_population(request: BreedRequest):
    return {
        "offspring": service.breed_population(request.formulas, request.offspring_count),
    }


@router.get("/bitget/config")
async def get_bitget_defaults():
    return get_bitget_config().model_dump()
