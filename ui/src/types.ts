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

/* ======================================================================== */
/*  Alpha Lab types                                                         */
/* ======================================================================== */

export interface AlphaLabOperator {
    name: string;
    category?: string;
    min_args?: number;
    max_args?: number;
    output_kind?: string;
}

export interface AlphaLabValidationReport {
    ok: boolean;
    normalized_formula?: string;
    errors?: string[];
    warnings?: string[];
}

export interface AlphaLabDatasetSummary {
    interval: string;
    symbols: string[];
    shape: [number, number];
}

export interface AlphaLabRunSummary {
    run_id: string;
    saved_at?: string;
    dataset?: { interval?: string; symbols?: string[]; shape?: [number, number] };
    top_results?: number;
    search_stats?: { total_evaluations?: number; total_rejected?: number; archive_size?: number };
    timing_seconds?: number;
    best_fitness?: number;
    best_sharpe?: number;
    best_ic?: number;
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
}

export interface AlphaLabRunDetail {
    run_id: string;
    saved_at?: string;
    dataset?: AlphaLabDatasetSummary;
    timing?: Record<string, number>;
    top_results?: Array<AlphaLabZooEntry & { split_metrics?: { train?: Record<string, number>; valid?: Record<string, number>; test?: Record<string, number> } }>;
    lineage?: Array<Record<string, unknown>>;
    search_stats?: { total_evaluations?: number; total_rejected?: number; archive_size?: number };
    validation?: Record<string, unknown>;
    pipeline?: AlphaPipelineRecord;
    rounds?: Array<Record<string, unknown>>;
}

export interface AlphaLabSeriesPoint {
    i: number;
    v: number;
}

export interface AlphaLabEvaluationSummary {
    dataset?: AlphaLabDatasetSummary;
    backend?: string;
    device?: string;
    expr_hash?: string;
    normalized_formula?: string;
    metrics: Record<string, number>;
    equity_series?: AlphaLabSeriesPoint[];
    drawdown_series?: AlphaLabSeriesPoint[];
    turnover_series?: AlphaLabSeriesPoint[];
}

export interface AlphaLabEngineInfo {
    backend: string;
    device: string;
    triton: boolean;
}

export interface AlphaLabSearchJob {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    created_at?: string;
    error?: string;
    run_id?: string;
    top_results?: AlphaLabZooEntry[];
    top_count?: number;
    search_stats?: Record<string, number>;
    timing?: Record<string, number>;
    pipeline?: AlphaPipelineRecord;
}

/* ── Pipeline types ─────────────────────────────────────────────────── */

export interface AlphaStageRecord {
    kind: 'generate' | 'quick_screen' | 'evaluate' | 'fitness' | 'archive';
    strategy: string;
    round: number;
    input: number;
    output: number;
    duration_ms: number;
    best_fitness?: number;
    metadata?: Record<string, unknown>;
}

export interface AlphaArchiveEntry {
    formula: string;
    expr_hash: string;
    fitness: number;
    rank_ic: number;
    sharpe: number;
    turnover: number;
    origin: string;
}

export interface AlphaRoundRecord {
    round: number;
    strategies: string[];
    stages: AlphaStageRecord[];
    archive_snapshot: AlphaArchiveEntry[];
    archive_size: number;
    population_size: number;
    best_fitness: number;
    duration_ms: number;
}

export interface AlphaPipelineRecord {
    job_id: string;
    rounds: AlphaRoundRecord[];
    total_evaluations: number;
    total_rejected: number;
}

export type SearchSSEEvent =
    | { type: 'stage'; data: AlphaStageRecord }
    | { type: 'round'; data: AlphaRoundRecord }
    | { type: 'complete'; data: { status: string; error?: string } };

export interface AlphaLabCombineResult {
    metrics?: Record<string, number>;
    combination?: {
        method: string;
        factor_count: number;
        selected_factors?: Array<{ formula: string; rank_ic: number; fitness: number }>;
        timing?: Record<string, number>;
    };
    equity_series?: AlphaLabSeriesPoint[];
    drawdown_series?: AlphaLabSeriesPoint[];
}

export interface AlphaLabWorkspaceDefaults {
    alpha_lab: Record<string, unknown>;
    bitget: Record<string, unknown>;
    crypto_market: Record<string, unknown>;
    intervals: string[];
    sample_formulas: string[];
}

export interface AlphaLabTrainingSnapshot {
    round: number;
    step: number;
    loss: number;
    avg_reward: number;
    best_reward: number;
    valid_ratio: number;
    unique: number;
    best_formula: string;
}

export interface AlphaLabTrainingHistory {
    history: AlphaLabTrainingSnapshot[];
    plot_available: boolean;
}

export interface AlphaLabWorkspace {
    operators: AlphaLabOperator[];
    defaults: AlphaLabWorkspaceDefaults;
    runs: AlphaLabRunSummary[];
    zoo: AlphaLabZooEntry[];
    engine?: AlphaLabEngineInfo;
    strategy_modes?: string[];
}

/* ── Strategy State Management ──────────────────────────────────────── */

export interface StrategyInfo {
    name: string;
    stateful: boolean;
    stats?: Record<string, unknown>;
}

export interface StrategyStateResponse {
    strategies: StrategyInfo[];
    strategy_memory?: {
        themes: Record<string, { count: number; avg_fitness: number; success_rate: number; best_fitness: number }>;
        operators: Record<string, { count: number; avg_fitness: number }>;
    } | null;
}

export interface CheckpointEntry {
    job_id: string;
    round_idx: number;
    timestamp: number;
    path: string;
    strategies: string[] | Array<{ name: string; format: string; metadata: Record<string, unknown> }>;
    archive_count?: number;
    archive_formulas?: Array<Record<string, unknown>>;
    context_state?: Record<string, unknown>;
}

export interface FactorCatalogEntry {
    formula: string;
    expr_hash: string;
    strategy: string;
    round_idx: number;
    rank_ic: number;
    sharpe: number;
    turnover: number;
    fitness: number;
    evaluated: boolean;
}

export interface FactorCatalogResponse {
    entries: FactorCatalogEntry[];
    stats: Record<string, { total_generated: number; total_evaluated: number; best_fitness: number; avg_ic: number }>;
    total: number;
}
