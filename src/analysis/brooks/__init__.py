"""Brooks price-action analysis — L1 features, L2 market structure, L3 patterns.

Layered design:
  L1 features.py     — stateless / per-bar numeric features
  L2 structure.py    — swing points, always-in, channels, legs (stateful dataclass)
  L3 patterns/       — H2/L2 FSM + Double TB + Wedge + Final Flag + Measured Move
  L3 aggregator.py   — SignalAggregator with confluence
"""
