"""
Strategy Template System - Templates for creating new strategies quickly.
Provides base templates, template registry, and strategy generation.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from src.core.base import Strategy
from src.strategies.registry import StrategyRegistry
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TemplateCategory(Enum):
    """Strategy template categories"""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    FACTOR = "factor"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "ml"
    CUSTOM = "custom"


@dataclass
class TemplateParameter:
    """Template parameter definition"""

    name: str
    param_type: str  # int, float, str, bool, list
    default: Any
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.param_type,
            "default": self.default,
            "description": self.description,
            "min": self.min_value,
            "max": self.max_value,
            "options": self.options,
        }

    def validate(self, value: Any) -> bool:
        """Validate parameter value"""
        if self.param_type == "int":
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.param_type == "float":
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.param_type == "str":
            if not isinstance(value, str):
                return False
            if self.options and value not in self.options:
                return False
        elif self.param_type == "bool":
            if not isinstance(value, bool):
                return False
        elif self.param_type == "list":
            if not isinstance(value, list):
                return False
        return True


@dataclass
class StrategyTemplate:
    """Strategy template definition"""

    name: str
    label: str
    category: TemplateCategory
    description: str
    parameters: List[TemplateParameter] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category.value,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "indicators": self.indicators,
            "entry_rules": self.entry_rules,
            "exit_rules": self.exit_rules,
        }

    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values"""
        return {p.name: p.default for p in self.parameters}


class TemplateRegistry:
    """Registry for strategy templates"""

    _templates: Dict[str, StrategyTemplate] = {}

    @classmethod
    def register(cls, template: StrategyTemplate) -> None:
        """Register a template"""
        cls._templates[template.name] = template
        logger.info(f"Registered template: {template.name}")

    @classmethod
    def get(cls, name: str) -> Optional[StrategyTemplate]:
        """Get a template by name"""
        return cls._templates.get(name)

    @classmethod
    def list_all(cls) -> List[StrategyTemplate]:
        """List all templates"""
        return list(cls._templates.values())

    @classmethod
    def list_by_category(cls, category: TemplateCategory) -> List[StrategyTemplate]:
        """List templates by category"""
        return [t for t in cls._templates.values() if t.category == category]

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export all templates as dict"""
        return {name: template.to_dict() for name, template in cls._templates.items()}


# ============= Built-in Templates =============

# Momentum Template
MOMENTUM_TEMPLATE = StrategyTemplate(
    name="momentum",
    label="Momentum Strategy",
    category=TemplateCategory.MOMENTUM,
    description="Buy assets with strong recent performance, sell weak performers",
    parameters=[
        TemplateParameter(
            name="lookback_period",
            param_type="int",
            default=20,
            description="Number of days to calculate momentum",
            min_value=5,
            max_value=252,
        ),
        TemplateParameter(
            name="top_n",
            param_type="int",
            default=10,
            description="Number of top assets to hold",
            min_value=1,
            max_value=50,
        ),
        TemplateParameter(
            name="rebalance_freq",
            param_type="str",
            default="monthly",
            description="Rebalancing frequency",
            options=["daily", "weekly", "monthly"],
        ),
        TemplateParameter(
            name="min_momentum",
            param_type="float",
            default=0.0,
            description="Minimum momentum threshold",
            min_value=-1.0,
            max_value=1.0,
        ),
    ],
    indicators=["momentum", "returns"],
    entry_rules=[
        "Asset momentum > min_momentum",
        "Asset in top_n by momentum ranking",
    ],
    exit_rules=[
        "Asset drops out of top_n",
        "Momentum < min_momentum",
    ],
)

# Mean Reversion Template
MEAN_REVERSION_TEMPLATE = StrategyTemplate(
    name="mean_reversion",
    label="Mean Reversion Strategy",
    category=TemplateCategory.MEAN_REVERSION,
    description="Buy oversold assets, sell overbought assets",
    parameters=[
        TemplateParameter(
            name="lookback_period",
            param_type="int",
            default=20,
            description="Period for calculating mean",
            min_value=5,
            max_value=100,
        ),
        TemplateParameter(
            name="z_entry",
            param_type="float",
            default=-2.0,
            description="Z-score threshold for entry (negative = buy)",
            min_value=-4.0,
            max_value=0.0,
        ),
        TemplateParameter(
            name="z_exit",
            param_type="float",
            default=0.0,
            description="Z-score threshold for exit",
            min_value=-2.0,
            max_value=2.0,
        ),
        TemplateParameter(
            name="max_holding_days",
            param_type="int",
            default=10,
            description="Maximum holding period",
            min_value=1,
            max_value=60,
        ),
    ],
    indicators=["sma", "std", "zscore"],
    entry_rules=[
        "Z-score < z_entry (oversold)",
    ],
    exit_rules=[
        "Z-score > z_exit (mean reversion)",
        "Holding days > max_holding_days",
    ],
)

# Trend Following Template
TREND_FOLLOWING_TEMPLATE = StrategyTemplate(
    name="trend_following",
    label="Trend Following Strategy",
    category=TemplateCategory.TREND_FOLLOWING,
    description="Follow trends using moving average crossovers",
    parameters=[
        TemplateParameter(
            name="fast_period",
            param_type="int",
            default=10,
            description="Fast moving average period",
            min_value=2,
            max_value=50,
        ),
        TemplateParameter(
            name="slow_period",
            param_type="int",
            default=30,
            description="Slow moving average period",
            min_value=10,
            max_value=200,
        ),
        TemplateParameter(
            name="ma_type",
            param_type="str",
            default="ema",
            description="Moving average type",
            options=["sma", "ema", "wma"],
        ),
        TemplateParameter(
            name="atr_multiplier",
            param_type="float",
            default=2.0,
            description="ATR multiplier for stop loss",
            min_value=0.5,
            max_value=5.0,
        ),
    ],
    indicators=["sma", "ema", "atr"],
    entry_rules=[
        "Fast MA crosses above Slow MA (buy)",
        "Fast MA crosses below Slow MA (sell)",
    ],
    exit_rules=[
        "Opposite crossover signal",
        "Price drops below stop loss (entry - ATR * multiplier)",
    ],
)

# Factor Strategy Template
FACTOR_TEMPLATE = StrategyTemplate(
    name="multi_factor",
    label="Multi-Factor Strategy",
    category=TemplateCategory.FACTOR,
    description="Combine multiple factors for stock selection",
    parameters=[
        TemplateParameter(
            name="momentum_weight",
            param_type="float",
            default=0.3,
            description="Weight for momentum factor",
            min_value=0.0,
            max_value=1.0,
        ),
        TemplateParameter(
            name="value_weight",
            param_type="float",
            default=0.3,
            description="Weight for value factor",
            min_value=0.0,
            max_value=1.0,
        ),
        TemplateParameter(
            name="quality_weight",
            param_type="float",
            default=0.2,
            description="Weight for quality factor",
            min_value=0.0,
            max_value=1.0,
        ),
        TemplateParameter(
            name="volatility_weight",
            param_type="float",
            default=0.2,
            description="Weight for low volatility factor",
            min_value=0.0,
            max_value=1.0,
        ),
        TemplateParameter(
            name="top_n",
            param_type="int",
            default=20,
            description="Number of stocks to hold",
            min_value=5,
            max_value=100,
        ),
    ],
    indicators=["momentum", "pe_ratio", "roe", "volatility"],
    entry_rules=[
        "Stock in top_n by composite factor score",
    ],
    exit_rules=[
        "Stock drops out of top_n",
        "Monthly rebalance",
    ],
)

# Register built-in templates
TemplateRegistry.register(MOMENTUM_TEMPLATE)
TemplateRegistry.register(MEAN_REVERSION_TEMPLATE)
TemplateRegistry.register(TREND_FOLLOWING_TEMPLATE)
TemplateRegistry.register(FACTOR_TEMPLATE)


# ============= Template-Based Strategy Generator =============


class TemplateBasedStrategy(Strategy):
    """
    Base class for strategies generated from templates.
    Provides common functionality for template-based strategies.
    """

    template: StrategyTemplate

    def __init__(self, db_client, session_id: Optional[str] = None, **params):
        super().__init__(session_id)
        self.db_client = db_client
        self.params = {**self.template.get_default_params(), **params}

        # Validate parameters
        for p in self.template.parameters:
            value = self.params.get(p.name)
            if value is not None and not p.validate(value):
                raise ValueError(f"Invalid value for {p.name}: {value}")

        self._initialize()

    def _initialize(self):
        """Initialize strategy-specific state. Override in subclass."""
        pass

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema from template"""
        return {
            p.name: {
                "type": p.param_type,
                "default": p.default,
                "description": p.description,
                "min": p.min_value,
                "max": p.max_value,
                "options": p.options,
            }
            for p in cls.template.parameters
        }


def create_strategy_from_template(
    template_name: str,
    strategy_name: str,
    on_bar_impl: str,
) -> Type[Strategy]:
    """
    Create a strategy class from a template.

    Args:
        template_name: Name of the template
        strategy_name: Name for the new strategy
        on_bar_impl: Python code for on_bar method

    Returns:
        New strategy class
    """
    template = TemplateRegistry.get(template_name)
    if not template:
        raise ValueError(f"Template not found: {template_name}")

    # Create new class
    class GeneratedStrategy(TemplateBasedStrategy):
        pass

    GeneratedStrategy.template = template
    GeneratedStrategy.__name__ = strategy_name

    # Compile on_bar implementation
    # WARNING: This uses exec - only use with trusted code
    namespace = {}
    exec(
        f"""
def on_bar(self, bars):
    {on_bar_impl}
""",
        namespace,
    )

    GeneratedStrategy.on_bar = namespace["on_bar"]

    # Register the strategy
    StrategyRegistry.register(strategy_name)(GeneratedStrategy)

    return GeneratedStrategy


# ============= Template Export/Import =============


def export_templates(filepath: str) -> None:
    """Export all templates to JSON file"""
    data = TemplateRegistry.to_dict()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Exported {len(data)} templates to {filepath}")


def activate_templates() -> int:
    """将所有已注册模板自动转换为可用的 Strategy 子类并注册到 StrategyRegistry。

    每个模板生成一个 TemplateBasedStrategy 子类，on_bar 使用安全的声明式逻辑。
    """
    from src.strategies.registry import StrategyRegistry

    count = 0
    for template in TemplateRegistry.list_all():
        class_name = f"Template_{template.name}"

        # 跳过已注册的
        if StrategyRegistry.get_strategy_class(f"tpl_{template.name}") is not None:
            continue

        # 动态创建子类 (不使用 exec，只设置类属性)
        cls = type(class_name, (TemplateBasedStrategy,), {
            "template": template,
            "__doc__": f"Auto-generated from template: {template.label}",
        })

        StrategyRegistry.register(f"tpl_{template.name}")(cls)
        count += 1
        logger.info("Template activated: tpl_{} ({})", template.name, template.label)

    return count


def import_templates(filepath: str) -> int:
    """Import templates from JSON file"""
    with open(filepath, "r") as f:
        data = json.load(f)

    count = 0
    for name, template_data in data.items():
        try:
            params = [
                TemplateParameter(**p) for p in template_data.get("parameters", [])
            ]
            template = StrategyTemplate(
                name=template_data["name"],
                label=template_data["label"],
                category=TemplateCategory(template_data["category"]),
                description=template_data["description"],
                parameters=params,
                indicators=template_data.get("indicators", []),
                entry_rules=template_data.get("entry_rules", []),
                exit_rules=template_data.get("exit_rules", []),
            )
            TemplateRegistry.register(template)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to import template {name}: {e}")

    logger.info(f"Imported {count} templates from {filepath}")
    return count
