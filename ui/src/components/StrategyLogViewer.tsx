import React, { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { apiFetch } from '../utils/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LogViewerProps {
    sessionId: string | null;
}

interface LogEntry {
    timestamp: string;
    level: string;
    source: string;
    message: string;
    extra?: Record<string, unknown>;
}

interface RowData {
    entries: LogEntry[];
    searchTerm: string;
    searchRegex: RegExp | null;
    activeMatchIndex: number;
    matchMap: Map<number, number[]>;  // rowIndex -> [charStart, ...]
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ROW_HEIGHT = 24;
const REFRESH_INTERVAL = 2000;

const LEVEL_COLORS: Record<string, string> = {
    DEBUG: '#8b949e',
    INFO: '#58a6ff',
    WARNING: '#d29922',
    ERROR: '#f85149',
};

const LEVEL_BG: Record<string, string> = {
    WARNING: 'rgba(210,153,34,0.06)',
    ERROR: 'rgba(248,81,73,0.08)',
};

const SOURCE_COLORS: Record<string, string> = {
    strategy: '#7ee787',
    broker: '#d2a8ff',
    engine: '#79c0ff',
    system: '#8b949e',
};

// ---------------------------------------------------------------------------
// Highlight helper – wraps matched substrings in <mark>
// ---------------------------------------------------------------------------

function highlightText(text: string, regex: RegExp | null): React.ReactNode {
    if (!regex) return text;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    // Reset lastIndex for global regex
    regex.lastIndex = 0;
    let safety = 0;
    while ((match = regex.exec(text)) !== null && safety++ < 200) {
        if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
        }
        parts.push(
            <mark key={match.index} style={{
                background: 'rgba(227,173,48,0.35)',
                color: '#e3e3e3',
                borderRadius: 2,
                padding: '0 1px',
            }}>
                {match[0]}
            </mark>,
        );
        lastIndex = match.index + match[0].length;
        if (match[0].length === 0) break; // Prevent infinite loop on zero-length matches
    }
    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }
    return parts.length > 0 ? <>{parts}</> : text;
}

// ---------------------------------------------------------------------------
// Log row (memoized)
// ---------------------------------------------------------------------------

const LogRow = memo<{ index: number; style: React.CSSProperties; data: RowData }>(
    ({ index, style, data }) => {
        const entry = data.entries[index];
        const level = (entry.level || 'INFO').toUpperCase();
        const source = entry.source || 'system';
        const levelColor = LEVEL_COLORS[level] || '#c9d1d9';
        const sourceColor = SOURCE_COLORS[source] || '#8b949e';
        const bgColor = LEVEL_BG[level] || 'transparent';
        const regex = data.searchRegex;

        const extraParts = entry.extra && Object.keys(entry.extra).length > 0
            ? Object.entries(entry.extra).map(([k, v]) => `${k}=${String(v)}`)
            : null;

        return (
            <div
                style={{
                    ...style,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0,
                    padding: '0 12px',
                    borderBottom: '1px solid #21262d',
                    background: bgColor,
                    whiteSpace: 'nowrap',
                    fontFamily: 'IBM Plex Mono, JetBrains Mono, Consolas, Monaco, monospace',
                    fontSize: 12,
                    lineHeight: `${ROW_HEIGHT}px`,
                    color: '#c9d1d9',
                    cursor: 'default',
                    userSelect: 'text',
                }}
                className="log-row"
            >
                {/* line number */}
                <span style={{ color: '#484f58', minWidth: 44, textAlign: 'right', marginRight: 12, flexShrink: 0 }}>
                    {index + 1}
                </span>

                {/* timestamp */}
                <span style={{ color: '#6e7681', marginRight: 8, flexShrink: 0 }}>
                    {highlightText(entry.timestamp.slice(11, 23) || entry.timestamp, regex)}
                </span>

                {/* level badge */}
                <span style={{
                    color: levelColor,
                    fontWeight: 600,
                    minWidth: 56,
                    marginRight: 6,
                    flexShrink: 0,
                }}>
                    {level}
                </span>

                {/* source badge */}
                <span style={{
                    color: sourceColor,
                    background: `${sourceColor}15`,
                    border: `1px solid ${sourceColor}30`,
                    borderRadius: 4,
                    padding: '0 5px',
                    fontSize: 11,
                    marginRight: 8,
                    flexShrink: 0,
                }}>
                    {source}
                </span>

                {/* message */}
                <span style={{ flexShrink: 0 }}>
                    {highlightText(entry.message, regex)}
                </span>

                {/* extra fields */}
                {extraParts && (
                    <span style={{ color: '#6e7681', marginLeft: 8, flexShrink: 0 }}>
                        {extraParts.map((part, i) => (
                            <span key={i}>
                                {i > 0 && <span style={{ margin: '0 4px', color: '#30363d' }}>·</span>}
                                {highlightText(part, regex)}
                            </span>
                        ))}
                    </span>
                )}
            </div>
        );
    },
);
LogRow.displayName = 'LogRow';

// ---------------------------------------------------------------------------
// Search bar
// ---------------------------------------------------------------------------

interface SearchBarProps {
    value: string;
    onChange: (v: string) => void;
    matchCount: number;
    activeMatch: number;
    onPrev: () => void;
    onNext: () => void;
    useRegex: boolean;
    onToggleRegex: () => void;
    caseSensitive: boolean;
    onToggleCase: () => void;
    onClose: () => void;
}

const SearchBar: React.FC<SearchBarProps> = ({
    value, onChange, matchCount, activeMatch,
    onPrev, onNext, useRegex, onToggleRegex,
    caseSensitive, onToggleCase, onClose,
}) => (
    <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 12px',
        background: '#161b22',
        borderBottom: '1px solid #30363d',
        fontSize: 13,
    }}>
        <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Search logs..."
            autoFocus
            style={{
                flex: 1,
                minWidth: 180,
                padding: '4px 8px',
                borderRadius: 6,
                border: '1px solid #30363d',
                background: '#0d1117',
                color: '#c9d1d9',
                fontSize: 13,
                fontFamily: 'IBM Plex Mono, JetBrains Mono, Consolas, monospace',
                outline: 'none',
            }}
            onKeyDown={(e) => {
                if (e.key === 'Enter') {
                    e.shiftKey ? onPrev() : onNext();
                }
                if (e.key === 'Escape') onClose();
            }}
        />
        <button onClick={onToggleRegex} title="Regular expression" style={{
            padding: '2px 6px',
            borderRadius: 4,
            border: `1px solid ${useRegex ? '#58a6ff' : '#30363d'}`,
            background: useRegex ? 'rgba(88,166,255,0.15)' : 'transparent',
            color: useRegex ? '#58a6ff' : '#8b949e',
            cursor: 'pointer',
            fontSize: 12,
            fontFamily: 'monospace',
            fontWeight: 600,
        }}>.*</button>
        <button onClick={onToggleCase} title="Case sensitive" style={{
            padding: '2px 6px',
            borderRadius: 4,
            border: `1px solid ${caseSensitive ? '#58a6ff' : '#30363d'}`,
            background: caseSensitive ? 'rgba(88,166,255,0.15)' : 'transparent',
            color: caseSensitive ? '#58a6ff' : '#8b949e',
            cursor: 'pointer',
            fontSize: 12,
            fontWeight: 600,
        }}>Aa</button>
        <span style={{ color: '#8b949e', fontSize: 12, minWidth: 60, textAlign: 'center' }}>
            {matchCount > 0 ? `${activeMatch + 1}/${matchCount}` : value ? 'No matches' : ''}
        </span>
        <button onClick={onPrev} disabled={matchCount === 0} style={navBtnStyle}>↑</button>
        <button onClick={onNext} disabled={matchCount === 0} style={navBtnStyle}>↓</button>
        <button onClick={onClose} style={{ ...navBtnStyle, color: '#8b949e' }}>✕</button>
    </div>
);

const navBtnStyle: React.CSSProperties = {
    padding: '2px 8px',
    borderRadius: 4,
    border: '1px solid #30363d',
    background: 'transparent',
    color: '#c9d1d9',
    cursor: 'pointer',
    fontSize: 13,
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const StrategyLogViewer: React.FC<LogViewerProps> = ({ sessionId }) => {
    const [entries, setEntries] = useState<LogEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [levelFilter, setLevelFilter] = useState('');
    const [sourceFilter, setSourceFilter] = useState('');
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [autoFollow, setAutoFollow] = useState(true);

    // Search state
    const [showSearch, setShowSearch] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [useRegex, setUseRegex] = useState(false);
    const [caseSensitive, setCaseSensitive] = useState(false);
    const [activeMatchIndex, setActiveMatchIndex] = useState(0);

    const listRef = useRef<List>(null);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const prevEntryCountRef = useRef(0);

    // ---- Fetch logs ----
    const fetchLogs = useCallback(async () => {
        if (!sessionId) return;
        try {
            setLoading(true);
            const params = new URLSearchParams({ limit: '2000', format: 'json' });
            if (levelFilter) params.append('level', levelFilter);
            if (sourceFilter) params.append('source', sourceFilter);
            const resp = await apiFetch(`/logs/${sessionId}?${params}`);
            if (!resp.ok) { setError(`Fetch failed: ${resp.statusText}`); return; }
            const data = await resp.json();
            setEntries(data.logs || []);
            setError(null);
        } catch (err) {
            setError(`Error: ${err}`);
        } finally {
            setLoading(false);
        }
    }, [sessionId, levelFilter, sourceFilter]);

    // ---- Polling ----
    useEffect(() => {
        fetchLogs();
        if (intervalRef.current) clearInterval(intervalRef.current);
        if (autoRefresh && sessionId) {
            intervalRef.current = setInterval(fetchLogs, REFRESH_INTERVAL);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [fetchLogs, autoRefresh, sessionId]);

    // ---- Auto-follow: scroll to bottom when new entries arrive ----
    useEffect(() => {
        if (autoFollow && entries.length > prevEntryCountRef.current && listRef.current) {
            listRef.current.scrollToItem(entries.length - 1, 'end');
        }
        prevEntryCountRef.current = entries.length;
    }, [entries.length, autoFollow]);

    // ---- Keyboard shortcut: Ctrl+F / Cmd+F ----
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                // Only capture if our container is focused/hovered
                if (containerRef.current?.contains(document.activeElement) || containerRef.current?.matches(':hover')) {
                    e.preventDefault();
                    setShowSearch(true);
                }
            }
            if (e.key === 'Escape' && showSearch) {
                setShowSearch(false);
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [showSearch]);

    // ---- Build search regex ----
    const searchRegex = useMemo(() => {
        if (!searchTerm) return null;
        try {
            const flags = caseSensitive ? 'g' : 'gi';
            const pattern = useRegex ? searchTerm : searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            return new RegExp(pattern, flags);
        } catch {
            return null;
        }
    }, [searchTerm, useRegex, caseSensitive]);

    // ---- Compute matches: which rows match ----
    const matchedRowIndices = useMemo(() => {
        if (!searchRegex) return [];
        const result: number[] = [];
        for (let i = 0; i < entries.length; i++) {
            const e = entries[i];
            const text = `${e.timestamp} ${e.level} ${e.source} ${e.message} ${
                e.extra ? Object.entries(e.extra).map(([k, v]) => `${k}=${v}`).join(' ') : ''
            }`;
            searchRegex.lastIndex = 0;
            if (searchRegex.test(text)) {
                result.push(i);
            }
        }
        return result;
    }, [entries, searchRegex]);

    // ---- Clamp active match index ----
    useEffect(() => {
        if (activeMatchIndex >= matchedRowIndices.length) {
            setActiveMatchIndex(Math.max(0, matchedRowIndices.length - 1));
        }
    }, [matchedRowIndices.length, activeMatchIndex]);

    // ---- Navigate matches ----
    const goToMatch = useCallback((dir: 1 | -1) => {
        if (matchedRowIndices.length === 0) return;
        const next = (activeMatchIndex + dir + matchedRowIndices.length) % matchedRowIndices.length;
        setActiveMatchIndex(next);
        listRef.current?.scrollToItem(matchedRowIndices[next], 'center');
    }, [activeMatchIndex, matchedRowIndices]);

    // ---- Clear logs ----
    const handleClear = async () => {
        if (!sessionId) return;
        try {
            await apiFetch(`/logs/${sessionId}`, { method: 'DELETE' });
            setEntries([]);
        } catch (err) {
            console.error('Failed to clear logs:', err);
        }
    };

    // ---- Row data (passed to memoized row component) ----
    const rowData = useMemo<RowData>(() => ({
        entries,
        searchTerm,
        searchRegex,
        activeMatchIndex,
        matchMap: new Map(),
    }), [entries, searchTerm, searchRegex, activeMatchIndex]);

    // ---- Level/source counts for filter badges ----
    const levelCounts = useMemo(() => {
        const counts: Record<string, number> = {};
        for (const e of entries) {
            const l = (e.level || 'INFO').toUpperCase();
            counts[l] = (counts[l] || 0) + 1;
        }
        return counts;
    }, [entries]);

    return (
        <SectionCard
            title="Strategy Logs"
            description="Filter session logs by level, source, and keywords."
            action={
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {loading && <span style={{ color: '#58a6ff', fontSize: 12 }}>Loading...</span>}
                    <span style={{ color: 'var(--color-text-dim)', fontSize: '0.85rem' }}>
                        {entries.length > 0 ? `${entries.length} entries` : ''} · Session: {sessionId?.slice(0, 8) || 'Not selected'}
                    </span>
                </div>
            }
        >
            {/* ---- Toolbar ---- */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                {/* Level filter pills */}
                <div style={{ display: 'flex', gap: 4 }}>
                    {['', 'INFO', 'WARNING', 'ERROR', 'DEBUG'].map((lv) => {
                        const active = levelFilter === lv;
                        const label = lv || 'All';
                        const count = lv ? (levelCounts[lv] || 0) : entries.length;
                        const color = lv ? LEVEL_COLORS[lv] : '#c9d1d9';
                        return (
                            <button
                                key={lv}
                                onClick={() => setLevelFilter(lv)}
                                style={{
                                    padding: '3px 10px',
                                    borderRadius: 6,
                                    border: `1px solid ${active ? `${color}60` : 'var(--border)'}`,
                                    background: active ? `${color}18` : 'transparent',
                                    color: active ? color : 'var(--color-text-dim)',
                                    fontSize: 12,
                                    fontWeight: active ? 600 : 400,
                                    cursor: 'pointer',
                                    transition: 'all 0.15s',
                                }}
                            >
                                {label}{count > 0 ? ` ${count}` : ''}
                            </button>
                        );
                    })}
                </div>

                <div style={{ width: 1, height: 20, background: 'var(--border)' }} />

                {/* Source filter */}
                <select
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    style={{
                        padding: '3px 8px',
                        borderRadius: 6,
                        border: '1px solid var(--border)',
                        background: 'var(--input-bg)',
                        color: 'var(--color-text)',
                        fontSize: 12,
                        cursor: 'pointer',
                    }}
                >
                    <option value="">All Sources</option>
                    <option value="strategy">Strategy</option>
                    <option value="broker">Broker</option>
                    <option value="engine">Engine</option>
                </select>

                <div style={{ flex: 1 }} />

                {/* Right-side controls */}
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--color-text-dim)', cursor: 'pointer' }}>
                    <input type="checkbox" checked={autoFollow} onChange={(e) => setAutoFollow(e.target.checked)} style={{ width: 'auto' }} />
                    Follow
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--color-text-dim)', cursor: 'pointer' }}>
                    <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} style={{ width: 'auto' }} />
                    Poll
                </label>
                <Button onClick={() => setShowSearch((v) => !v)} variant="ghost" size="sm" title="Ctrl+F">Search</Button>
                <Button onClick={fetchLogs} variant="outline" size="sm">Refresh</Button>
                <Button onClick={handleClear} variant="danger" size="sm">Clear</Button>
            </div>

            {/* ---- Search bar ---- */}
            {showSearch && (
                <SearchBar
                    value={searchTerm}
                    onChange={(v) => { setSearchTerm(v); setActiveMatchIndex(0); }}
                    matchCount={matchedRowIndices.length}
                    activeMatch={activeMatchIndex}
                    onPrev={() => goToMatch(-1)}
                    onNext={() => goToMatch(1)}
                    useRegex={useRegex}
                    onToggleRegex={() => setUseRegex((v) => !v)}
                    caseSensitive={caseSensitive}
                    onToggleCase={() => setCaseSensitive((v) => !v)}
                    onClose={() => { setShowSearch(false); setSearchTerm(''); }}
                />
            )}

            {/* ---- Log area ---- */}
            <div
                ref={containerRef}
                tabIndex={-1}
                style={{
                    height: 'calc(100vh - 360px)',
                    minHeight: 400,
                    borderRadius: 8,
                    border: '1px solid var(--border)',
                    background: '#0d1117',
                    overflow: 'hidden',
                }}
            >
                {error ? (
                    <div style={{ padding: 24, color: '#f85149' }}>{error}</div>
                ) : entries.length === 0 ? (
                    <div style={{ padding: 24, color: '#8b949e', fontSize: 13 }}>
                        {sessionId ? 'No logs for this session yet.' : 'No session selected.'}
                    </div>
                ) : (
                    <AutoSizer>
                        {({ height, width }) => (
                            <List
                                ref={listRef}
                                height={height}
                                width={width}
                                itemCount={entries.length}
                                itemSize={ROW_HEIGHT}
                                itemData={rowData}
                                overscanCount={30}
                                style={{ overflowX: 'auto', overflowY: 'auto' }}
                            >
                                {LogRow}
                            </List>
                        )}
                    </AutoSizer>
                )}
            </div>

            {/* ---- Row hover style ---- */}
            <style>{`
                .log-row:hover {
                    background: #161b22 !important;
                }
            `}</style>
        </SectionCard>
    );
};

export default StrategyLogViewer;
