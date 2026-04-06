# Quent Documentation

## Overview

Quent is an AI-powered quantitative trading platform for crypto and A-share markets. It combines LLM-driven alpha factor discovery with a full backtesting and live trading stack.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, services, data flow, tech stack |
| [API Reference](api.md) | REST endpoints, WebSocket protocol, request/response models |
| [Alpha Search](alpha-search.md) | Factor discovery: DSL, VM, search strategies, LLM integration |
| [Backtesting](backtesting.md) | Trading engine, brokers, risk management, strategies |
| [Data Pipeline](data-pipeline.md) | Market data sources, processors, ClickHouse storage |
| [Frontend](frontend.md) | React UI architecture, components, state management |
| [Development](development.md) | Local setup, testing, Docker deployment, configuration |

## Quick Links

- **Project root**: [CLAUDE.md](../CLAUDE.md) - AI-agent-friendly project overview
- **Alpha research papers**: [docs/pdf/](pdf/) - Referenced research papers
- **Factor storage**: [jointdata/README.md](../jointdata/README.md) - ClickHouse factor storage module
