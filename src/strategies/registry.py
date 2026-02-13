"""
Strategy registry for automatic strategy discovery and registration.
Eliminates hardcoded strategy mapping in server.py.
"""

from typing import Dict, Type, Optional, Any
from src.core.base import Strategy


class StrategyRegistry:
    """Registry for automatically discovering and registering strategies"""

    _strategies: Dict[str, Type[Strategy]] = {}
    _labels: Dict[str, str] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, label: str = None, description: str = None):
        """Decorator to register a strategy class"""

        def decorator(strategy_cls: Type[Strategy]):
            cls._strategies[name] = strategy_cls
            cls._labels[name] = label or name
            cls._descriptions[name] = description or ""
            return strategy_cls

        return decorator

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[Strategy]]:
        """Get strategy class by name"""
        return cls._strategies.get(name)

    @classmethod
    def create_strategy(
        cls, name: str, db_client, session_id: str = None, **kwargs
    ) -> Optional[Strategy]:
        """Create a strategy instance by name"""
        strategy_cls = cls.get_strategy_class(name)
        if strategy_cls is None:
            return None
        return strategy_cls(db_client, session_id=session_id, **kwargs)

    @classmethod
    def get_parameters(cls, name: str) -> Dict[str, Any]:
        """Get parameters for a strategy"""
        strategy_cls = cls.get_strategy_class(name)
        if strategy_cls is None:
            return {}
        if hasattr(strategy_cls, "get_parameters"):
            return strategy_cls.get_parameters()
        return {}

    @classmethod
    def list_strategies(cls) -> list:
        """List all registered strategies"""
        return [
            {
                "name": name,
                "label": cls._labels.get(name, name),
                "description": cls._descriptions.get(name, ""),
                "params": cls.get_parameters(name),
            }
            for name in cls._strategies
        ]

    @classmethod
    def register_all(cls):
        """Import all strategy modules to trigger registration"""
        from src.strategies import jsg_strategy
        from src.strategies import rotation_strategy
        from src.strategies import multi_factor_strategy

        # Import forces decorator execution
        _ = jsg_strategy, rotation_strategy, multi_factor_strategy


# Auto-register all strategies on module import
StrategyRegistry.register_all()


# Convenience function
def get_strategy(
    name: str, db_client, session_id: str = None, **kwargs
) -> Optional[Strategy]:
    """Get a strategy instance by name"""
    return StrategyRegistry.create_strategy(
        name, db_client, session_id=session_id, **kwargs
    )
