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
    source?: string;
    job_id?: string;
    run_id?: string;
    last_processed_at?: string;
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

export interface SimulationJob {
    job_id: string;
    name: string;
    strategy_name: string;
    symbol: string;
    mode: string;
    start_date: string;
    end_date?: string;
    params: Record<string, unknown>;
    enabled: boolean;
    status: string;
    schedule: string;
    last_processed_at?: string;
    last_update_at?: string;
    latest_session_id?: string;
    latest_run_id?: string;
    snapshot?: Record<string, unknown> | null;
    error?: string | null;
    created_at: string;
    updated_at: string;
}

export interface SimulationRun {
    run_id: string;
    job_id: string;
    session_id?: string;
    update_run_id?: string;
    trigger_source: string;
    start_date: string;
    end_date?: string;
    status: string;
    progress: number;
    bars_processed: number;
    steps_recorded: number;
    summary?: Record<string, unknown>;
    error?: string | null;
    started_at?: string;
    completed_at?: string;
    created_at: string;
}

export interface SimulationStep {
    id: number;
    run_id: string;
    session_id?: string;
    step_index: number;
    timestamp?: string;
    event_type: string;
    payload: Record<string, any>;
    created_at: string;
}

export interface DataUpdateRun {
    update_run_id: string;
    trigger_source: string;
    status: string;
    has_new_data: boolean;
    details?: Record<string, any>;
    error?: string | null;
    started_at?: string;
    completed_at?: string;
    created_at: string;
}
