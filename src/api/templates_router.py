"""
Templates Router - API endpoints for strategy templates.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.strategies.templates import (
    TemplateCategory,
    TemplateRegistry,
)

router = APIRouter(prefix="/templates", tags=["templates"])


class TemplateResponse(BaseModel):
    """Template response model"""

    name: str
    label: str
    category: str
    description: str
    parameters: List[Dict[str, Any]]
    indicators: List[str]
    entry_rules: List[str]
    exit_rules: List[str]


class TemplateListResponse(BaseModel):
    """Template list response"""

    templates: List[TemplateResponse]
    count: int


@router.get("", response_model=TemplateListResponse)
async def list_templates(category: Optional[str] = None):
    """
    List all available strategy templates.

    Args:
        category: Optional filter by category
    """
    if category:
        try:
            cat = TemplateCategory(category)
            templates = TemplateRegistry.list_by_category(cat)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category: {category}. Valid: {[c.value for c in TemplateCategory]}",
            )
    else:
        templates = TemplateRegistry.list_all()

    return {
        "templates": [t.to_dict() for t in templates],
        "count": len(templates),
    }


@router.get("/categories")
async def list_categories():
    """List all template categories"""
    return {
        "categories": [{"name": c.name, "value": c.value} for c in TemplateCategory]
    }


@router.get("/{template_name}", response_model=TemplateResponse)
async def get_template(template_name: str):
    """Get a specific template by name"""
    template = TemplateRegistry.get(template_name)
    if not template:
        raise HTTPException(
            status_code=404, detail=f"Template not found: {template_name}"
        )
    return template.to_dict()


@router.get("/{template_name}/defaults")
async def get_template_defaults(template_name: str):
    """Get default parameter values for a template"""
    template = TemplateRegistry.get(template_name)
    if not template:
        raise HTTPException(
            status_code=404, detail=f"Template not found: {template_name}"
        )
    return {
        "template": template_name,
        "defaults": template.get_default_params(),
    }


class ValidateParamsRequest(BaseModel):
    """Request to validate parameters"""

    params: Dict[str, Any]


@router.post("/{template_name}/validate")
async def validate_params(template_name: str, request: ValidateParamsRequest):
    """Validate parameters for a template"""
    template = TemplateRegistry.get(template_name)
    if not template:
        raise HTTPException(
            status_code=404, detail=f"Template not found: {template_name}"
        )

    errors = []
    validated = {}

    for param in template.parameters:
        value = request.params.get(param.name)

        if value is None:
            validated[param.name] = param.default
        elif not param.validate(value):
            errors.append(
                {
                    "param": param.name,
                    "value": value,
                    "expected": param.param_type,
                    "min": param.min_value,
                    "max": param.max_value,
                    "options": param.options,
                }
            )
        else:
            validated[param.name] = value

    return {
        "valid": len(errors) == 0,
        "validated_params": validated if not errors else None,
        "errors": errors,
    }
