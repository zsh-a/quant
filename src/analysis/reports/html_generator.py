"""
HTML Report Generator - Generate HTML reports with interactive charts.
"""

from typing import Dict, List, Optional
from datetime import datetime
import os
from loguru import logger

from src.analysis.attribution import ReturnAttribution, RiskAttribution


class HTMLReportGenerator:
    """Generate interactive HTML reports"""

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            from src.config.paths import REPORTS_DIR
            output_dir = str(REPORTS_DIR)
        self.output_dir = output_dir

    def generate(
        self,
        session_id: str,
        strategy_name: str,
        equity_history: List[Dict],
        trades: List[Dict],
        positions: Dict,
        params: Optional[Dict] = None
    ) -> str:
        """Generate HTML report"""
        logger.info(f"Generating HTML report for {session_id}")

        # Run attribution
        attr = ReturnAttribution(trades, equity_history)
        attribution = attr.analyze()

        risk_attr = RiskAttribution(equity_history, positions)
        risk = risk_attr.analyze()

        # Calculate stats
        if equity_history:
            initial = equity_history[0].get('total_equity', 0)
            final = equity_history[-1].get('total_equity', 0)
            total_return = (final - initial) / initial if initial > 0 else 0
            start_date = equity_history[0].get('date', '')
            end_date = equity_history[-1].get('date', '')
        else:
            initial = final = total_return = 0
            start_date = end_date = ''

        # Build HTML
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>回测报告 - {strategy_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 16px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: 600; }}
        .stat-label {{ font-size: 12px; color: #8b949e; }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ color: #8b949e; }}
        .chart-container {{ height: 300px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 回测报告: {strategy_name}</h1>
        <p>会话: {session_id} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div class="card">
            <h3>执行摘要</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value {'positive' if total_return >= 0 else 'negative'}">{total_return:.2%}</div>
                    <div class="stat-label">总收益</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{risk.get('sharpe_ratio', 0):.2f}</div>
                    <div class="stat-label">夏普比率</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value negative">{risk.get('max_drawdown', 0):.2%}</div>
                    <div class="stat-label">最大回撤</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{attribution.win_rate:.1%}</div>
                    <div class="stat-label">胜率</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">¥{initial:,.0f}</div>
                    <div class="stat-label">初始资金</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">¥{final:,.0f}</div>
                    <div class="stat-label">最终权益</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(trades)}</div>
                    <div class="stat-label">交易次数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{attribution.profit_factor:.2f}</div>
                    <div class="stat-label">盈亏比</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>权益曲线</h3>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>资产归因</h3>
            <table>
                <tr><th>资产</th><th>盈亏</th></tr>
                {''.join(f"<tr><td>{s}</td><td class='{'positive' if p>=0 else 'negative'}'>¥{p:+,.0f}</td></tr>" 
                         for s, p in sorted(attribution.by_asset.items(), key=lambda x: x[1], reverse=True)[:10])}
            </table>
        </div>

        <div class="card">
            <h3>行业归因</h3>
            <table>
                <tr><th>行业</th><th>盈亏</th></tr>
                {''.join(f"<tr><td>{s}</td><td class='{'positive' if p>=0 else 'negative'}'>¥{p:+,.0f}</td></tr>" 
                         for s, p in sorted(attribution.by_sector.items(), key=lambda x: x[1], reverse=True))}
            </table>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {[e.get('date', '') for e in equity_history]},
                datasets: [{{
                    label: '权益',
                    data: {[e.get('total_equity', 0) for e in equity_history]},
                    borderColor: '#58a6ff',
                    fill: false,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
    </script>
</body>
</html>'''

        filename = f"report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved: {filepath}")
        return filepath
