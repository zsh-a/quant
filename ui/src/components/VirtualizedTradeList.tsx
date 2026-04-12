import React, { memo, useCallback, useMemo } from 'react';
import { FixedSizeList as List, areEqual } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { Trade } from '../types';
import { formatMoney } from '../utils/format';

interface VirtualizedTradeListProps {
    trades: Trade[];
    onTradeClick?: (trade: Trade) => void;
    height?: number;
    showDate?: boolean;
}

interface RowData {
    trades: Trade[];
    onTradeClick?: (trade: Trade) => void;
    showDate: boolean;
}

// Memoized row component to prevent unnecessary re-renders
const TradeRow = memo<{
    index: number;
    style: React.CSSProperties;
    data: RowData;
}>(({ index, style, data }) => {
    const trade = data.trades[index];
    const isBuy = trade.type === 'buy';

    const handleClick = useCallback(() => {
        data.onTradeClick?.(trade);
    }, [data, trade]);

    const displayDate = data.showDate 
        ? trade.timestamp.split(' ')[0] 
        : trade.timestamp.split(' ')[1] || trade.timestamp;

    return (
        <div
            style={{
                ...style,
                display: 'grid',
                gridTemplateColumns: data.showDate 
                    ? '100px 1fr 70px 90px 100px 80px' 
                    : '80px 1fr 70px 90px 100px 80px',
                gap: '8px',
                padding: '0 16px',
                borderBottom: '1px solid rgba(255,255,255,0.05)',
                cursor: data.onTradeClick ? 'pointer' : 'default',
                alignItems: 'center',
                fontSize: '13px',
            }}
            onClick={handleClick}
            onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
            }}
        >
            <div style={{ color: 'var(--color-text-dim)', fontSize: '11px' }}>
                {displayDate}
            </div>
            <div>
                <div style={{ fontWeight: 600, color: 'var(--color-text)' }}>{trade.symbol}</div>
                <div style={{ fontSize: '11px', color: 'var(--color-text-dim)' }}>{trade.name}</div>
            </div>
            <div>
                <span style={{
                    fontSize: '10px',
                    fontWeight: 600,
                    padding: '3px 8px',
                    borderRadius: '6px',
                    textTransform: 'uppercase',
                    background: isBuy ? 'color-mix(in srgb, var(--color-success) 15%, transparent)' : 'color-mix(in srgb, var(--color-danger) 15%, transparent)',
                    color: isBuy ? 'var(--color-success)' : 'var(--color-danger)',
                }}>
                    {trade.type}
                </span>
            </div>
            <div style={{ textAlign: 'right', color: 'var(--color-text)' }}>
                {formatMoney(trade.price)}
            </div>
            <div style={{ textAlign: 'right', color: 'var(--color-text)', fontWeight: 500 }}>
                {formatMoney(trade.amount, { decimals: 0 })}
            </div>
            <div style={{ textAlign: 'right', color: 'var(--color-text-dim)', fontSize: '12px' }}>
                {trade.commission ? formatMoney(trade.commission, { decimals: 1 }) : '-'}
            </div>
        </div>
    );
}, areEqual);

TradeRow.displayName = 'TradeRow';

// Header component
const TradeHeader: React.FC<{ showDate: boolean }> = memo(({ showDate }) => (
    <div style={{
        display: 'grid',
        gridTemplateColumns: showDate 
            ? '100px 1fr 70px 90px 100px 80px' 
            : '80px 1fr 70px 90px 100px 80px',
        gap: '8px',
        padding: '12px 16px',
        background: 'rgba(255,255,255,0.03)',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
        fontSize: '11px',
        fontWeight: 600,
        color: 'var(--color-text-dim)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
    }}>
        <div>{showDate ? 'Date' : 'Time'}</div>
        <div>Symbol</div>
        <div>Type</div>
        <div style={{ textAlign: 'right' }}>Price</div>
        <div style={{ textAlign: 'right' }}>Amount</div>
        <div style={{ textAlign: 'right' }}>Comm</div>
    </div>
));

TradeHeader.displayName = 'TradeHeader';

export const VirtualizedTradeList: React.FC<VirtualizedTradeListProps> = memo(({
    trades,
    onTradeClick,
    height,
    showDate = true
}) => {
    // Memoize item data to prevent re-renders
    const itemData = useMemo<RowData>(() => ({
        trades,
        onTradeClick,
        showDate
    }), [trades, onTradeClick, showDate]);

    if (trades.length === 0) {
        return (
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: height || 200,
                color: 'var(--color-text-dim)',
                fontSize: '14px',
            }}>
                No trades
            </div>
        );
    }

    const ITEM_HEIGHT = 52;
    const HEADER_HEIGHT = 44;

    // If height is provided, use fixed height mode
    if (height) {
        return (
            <div style={{ height: height, display: 'flex', flexDirection: 'column' }}>
                <TradeHeader showDate={showDate} />
                <List
                    height={height - HEADER_HEIGHT}
                    itemCount={trades.length}
                    itemSize={ITEM_HEIGHT}
                    width="100%"
                    itemData={itemData}
                    overscanCount={5}
                >
                    {TradeRow}
                </List>
            </div>
        );
    }

    // Auto-sizing mode
    return (
        <div style={{ height: '100%', minHeight: 200, display: 'flex', flexDirection: 'column' }}>
            <TradeHeader showDate={showDate} />
            <div style={{ flex: 1 }}>
                <AutoSizer>
                    {({ height: autoHeight, width }: { height: number; width: number }) => (
                        <List
                            height={autoHeight || 300}
                            itemCount={trades.length}
                            itemSize={ITEM_HEIGHT}
                            width={width || '100%'}
                            itemData={itemData}
                            overscanCount={5}
                        >
                            {TradeRow}
                        </List>
                    )}
                </AutoSizer>
            </div>
        </div>
    );
});

VirtualizedTradeList.displayName = 'VirtualizedTradeList';

export default VirtualizedTradeList;
