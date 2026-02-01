import React from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';

interface Trade {
    timestamp: string;
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
    commission: number;
    pnl?: number;
}

interface VirtualizedTradeListProps {
    trades: Trade[];
    onTradeClick?: (trade: Trade) => void;
}

const TradeRow: React.FC<{
    index: number;
    style: React.CSSProperties;
    data: {
        trades: Trade[];
        onTradeClick?: (trade: Trade) => void;
    };
}> = ({ index, style, data }) => {
    const trade = data.trades[index];
    const isProfitable = (trade.pnl || 0) > 0;

    return (
        <div
            style={style}
            className={`trade-row ${trade.side}`}
            onClick={() => data.onTradeClick?.(trade)}
        >
            <div className="trade-time">
                {new Date(trade.timestamp).toLocaleString()}
            </div>
            <div className="trade-symbol">{trade.symbol}</div>
            <div className={`trade-side ${trade.side}`}>
                {trade.side.toUpperCase()}
            </div>
            <div className="trade-quantity">{trade.quantity}</div>
            <div className="trade-price">${trade.price.toFixed(2)}</div>
            <div className="trade-commission">${trade.commission.toFixed(2)}</div>
            {trade.pnl !== undefined && (
                <div className={`trade-pnl ${isProfitable ? 'profit' : 'loss'}`}>
                    {isProfitable ? '+' : ''}${trade.pnl.toFixed(2)}
                </div>
            )}
        </div>
    );
};

export const VirtualizedTradeList: React.FC<VirtualizedTradeListProps> = ({
    trades,
    onTradeClick
}) => {
    if (trades.length === 0) {
        return (
            <div className="empty-trades">
                <p>No trades yet</p>
            </div>
        );
    }

    return (
        <div className="virtualized-trade-list">
            <div className="trade-header">
                <div>Time</div>
                <div>Symbol</div>
                <div>Side</div>
                <div>Quantity</div>
                <div>Price</div>
                <div>Commission</div>
                <div>P&L</div>
            </div>

            <AutoSizer>
                {({ height, width }: { height: number; width: number }) => (
                    <List
                        height={height || 400}
                        itemCount={trades.length}
                        itemSize={50}
                        width={width || 800}
                        itemData={{ trades, onTradeClick }}
                    >
                        {TradeRow}
                    </List>
                )}
            </AutoSizer>

            <style>{`
        .virtualized-trade-list {
          height: 100%;
          display: flex;
          flex-direction: column;
          background: #1a1a1a;
          border-radius: 8px;
          overflow: hidden;
        }

        .trade-header {
          display: grid;
          grid-template-columns: 180px 100px 80px 100px 100px 100px 120px;
          gap: 12px;
          padding: 12px 16px;
          background: #2a2a2a;
          border-bottom: 1px solid #333;
          font-size: 12px;
          font-weight: 600;
          color: #888;
        }

        .trade-row {
          display: grid;
          grid-template-columns: 180px 100px 80px 100px 100px 100px 120px;
          gap: 12px;
          padding: 12px 16px;
          border-bottom: 1px solid #222;
          cursor: pointer;
          transition: background 0.2s;
          align-items: center;
        }

        .trade-row:hover {
          background: #252525;
        }

        .trade-time {
          font-size: 12px;
          color: #888;
        }

        .trade-symbol {
          font-weight: 600;
          color: #fff;
        }

        .trade-side {
          font-size: 11px;
          font-weight: 600;
          padding: 4px 8px;
          border-radius: 4px;
          text-align: center;
        }

        .trade-side.buy {
          background: rgba(46, 204, 113, 0.2);
          color: #2ecc71;
        }

        .trade-side.sell {
          background: rgba(231, 76, 60, 0.2);
          color: #e74c3c;
        }

        .trade-quantity,
        .trade-price,
        .trade-commission {
          font-size: 13px;
          color: #ccc;
        }

        .trade-pnl {
          font-weight: 600;
          font-size: 14px;
        }

        .trade-pnl.profit {
          color: #2ecc71;
        }

        .trade-pnl.loss {
          color: #e74c3c;
        }

        .empty-trades {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 200px;
          color: #666;
        }
      `}</style>
        </div>
    );
};
