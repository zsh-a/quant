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

export interface TelegramNotificationConfig {
    enabled?: boolean;
    chat_id?: string;
}

export interface SimulationJobNotification {
    telegram?: TelegramNotificationConfig;
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
    notification?: SimulationJobNotification;
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
    last_heartbeat_at?: string;
    completed_at?: string;
    created_at: string;
}

export interface MarketTableSummary {
    table: string;
    status: string;
    row_count?: number | null;
    distinct_count?: number | null;
    distinct_label?: string | null;
    earliest_date?: string | null;
    latest_date?: string | null;
    error?: string | null;
}

export interface MarketUpdateStepDefinition {
    key: string;
    label: string;
    description: string;
    default_selected: boolean;
}

export interface MarketUpdateCapabilities {
    steps: MarketUpdateStepDefinition[];
    default_selected_steps: string[];
    share_start_date_default: string;
    reference_symbol: string;
}

export interface MarketDbOverview {
    reference_symbol: string;
    latest_market_date?: string | null;
    data_lag_days?: number | null;
    stock_coverage: {
        tracked_stock_codes: number;
        tracked_etf_codes: number;
    };
    tables: Record<string, MarketTableSummary>;
    last_update_run?: DataUpdateRun | null;
    running_update?: DataUpdateRun | null;
    generated_at: string;
}

export interface AlphaLabOperator {
    name: string;
    category?: string;
    arity?: number;
    description?: string;
    output_type?: string;
}

export interface AlphaLabValidationReport {
    ok: boolean;
    normalized_formula?: string;
    errors?: string[];
    warnings?: string[];
}

export interface AlphaLabDatasetSummary {
    provider: string;
    interval: string;
    symbols: string[];
    shape: [number, number];
    timestamps?: string[];
}

export interface AlphaLabRunSummary {
    run_id: string;
    saved_at?: string;
    path?: string;
    dataset?: Record<string, unknown>;
    top_results?: number;
}

export interface AlphaLabZooEntry {
    formula: string;
    expr_hash?: string;
    fitness?: number;
    metrics?: Record<string, number>;
    lineage?: Record<string, unknown>;
    note?: string | null;
    tags?: string[];
    source?: string;
    saved_at?: string;
    run_id?: string;
    path?: string;
    validation?: AlphaLabValidationReport;
}

export interface AlphaLabRunDetail {
    run_id: string;
    dataset?: AlphaLabDatasetSummary;
    validation?: Record<string, unknown>;
    timing?: Record<string, unknown>;
    generations?: Array<Record<string, unknown>>;
    top_results?: AlphaLabZooEntry[];
    lineage?: Array<Record<string, unknown>>;
    evaluations?: Record<string, unknown>;
}

export interface AlphaLabEvaluationSummary {
    dataset?: AlphaLabDatasetSummary;
    backend?: string;
    device?: string;
    expr_hash?: string;
    normalized_formula?: string;
    metrics: Record<string, number>;
    alpha_tail?: number[][];
    weights_tail?: number[][];
    equity_tail?: number[];
}

export interface AlphaLabWorkspaceDefaults {
    alpha_lab: Record<string, unknown>;
    bitget: Record<string, unknown>;
    crypto_market: Record<string, unknown>;
    providers: string[];
    intervals: string[];
    sample_formulas: string[];
}

export interface AlphaLabWorkspace {
    operators: AlphaLabOperator[];
    defaults: AlphaLabWorkspaceDefaults;
    runs: AlphaLabRunSummary[];
    zoo: AlphaLabZooEntry[];
}
