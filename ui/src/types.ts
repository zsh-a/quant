export interface Trade {
    timestamp: string;
    symbol: string;
    name: string;
    type: string;
    price: number;
    quantity: number;
    amount: number;
    commission?: number;
}

export interface Position {
    qty: number;
    name: string;
    price: number;
    value: number;
    avg_cost?: number;
    unrealized_pnl?: number;
    pnl_pct?: number;
}

export interface EquityPoint {
    timestamp: string;
    total_equity: number;
    daily_pnl?: number;
    daily_return?: number;
    cash: number;
    positions: Record<string, Position>;
}

export interface SessionSummary {
    id: string;
    strategy: string;
    symbol: string;
    status: string;
    mode: string;
    progress: number;
    start_date: string;
    end_date?: string;
    params?: Record<string, unknown>;
}

export interface BenchmarkData {
    timestamp: string;
    value: number;
}

export interface StrategyParam {
    type: string;
    default: any;
    description: string;
    min?: number;
    max?: number;
    options?: string[];
}

export interface StrategyMeta {
    name: string;
    label: string;
    params: Record<string, StrategyParam>;
}
