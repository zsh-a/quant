"""
Report Generator - Generate backtest reports in Markdown format.
"""

from typing import Dict, List, Optional
from datetime import datetime
import os
from loguru import logger

from src.analysis.attribution import ReturnAttribution, RiskAttribution


class ReportGenerator:
    """
    Backtest report generator.
    
    Generates professional reports in Markdown format.
    """
    
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
        params: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate Markdown report.
        
        Returns:
            Path to generated report
        """
        logger.info(f"Generating report for session {session_id}")
        
        # Run attribution analysis
        return_attr = ReturnAttribution(trades, equity_history)
        attribution = return_attr.analyze()
        
        risk_attr = RiskAttribution(equity_history, positions)
        risk_metrics = risk_attr.analyze()
        
        # Build report content
        report = self._build_report(
            session_id=session_id,
            strategy_name=strategy_name,
            equity_history=equity_history,
            trades=trades,
            attribution=attribution,
            risk_metrics=risk_metrics,
            params=params or {},
            metadata=metadata or {}
        )
        
        # Save report
        filename = f"report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved: {filepath}")
        return filepath
    
    def _build_report(
        self,
        session_id: str,
        strategy_name: str,
        equity_history: List[Dict],
        trades: List[Dict],
        attribution,
        risk_metrics: Dict,
        params: Dict,
        metadata: Dict
    ) -> str:
        """Build report content"""
        
        # Calculate summary stats
        if equity_history:
            initial = equity_history[0].get('total_equity', 0)
            final = equity_history[-1].get('total_equity', 0)
            total_return = (final - initial) / initial if initial > 0 else 0
            start_date = equity_history[0].get('date', '')
            end_date = equity_history[-1].get('date', '')
        else:
            initial = final = 0
            total_return = 0
            start_date = end_date = ''
        
        lines = []
        
        # Header
        lines.append(f"# 回测报告: {strategy_name}")
        lines.append("")
        lines.append(f"**会话ID**: `{session_id}`  ")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Executive Summary
        lines.append("## 📊 执行摘要")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 回测期间 | {start_date} 至 {end_date} |")
        lines.append(f"| 初始资金 | ¥{initial:,.0f} |")
        lines.append(f"| 最终资金 | ¥{final:,.0f} |")
        lines.append(f"| **总收益率** | **{total_return:.2%}** |")
        lines.append(f"| 夏普比率 | {risk_metrics.get('sharpe_ratio', 0):.2f} |")
        lines.append(f"| 最大回撤 | {risk_metrics.get('max_drawdown', 0):.2%} |")
        lines.append(f"| 交易次数 | {len(trades)} |")
        lines.append("")
        
        # Strategy Parameters
        if params:
            lines.append("## ⚙️ 策略参数")
            lines.append("")
            lines.append("| 参数 | 值 |")
            lines.append("|------|-----|")
            for k, v in params.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        
        # Performance Metrics
        lines.append("## 📈 绩效指标")
        lines.append("")
        lines.append("### 收益指标")
        lines.append("")
        lines.append(f"- **总收益**: {total_return:.2%}")
        lines.append(f"- **年化收益**: {self._annualize_return(total_return, len(equity_history)):.2%}")
        lines.append(f"- **胜率**: {attribution.win_rate:.1%}")
        lines.append(f"- **盈亏比**: {attribution.profit_factor:.2f}")
        lines.append("")
        
        lines.append("### 风险指标")
        lines.append("")
        lines.append(f"- **年化波动率**: {risk_metrics.get('volatility', 0):.2%}")
        lines.append(f"- **最大回撤**: {risk_metrics.get('max_drawdown', 0):.2%}")
        lines.append(f"- **VaR(95%)**: {risk_metrics.get('var_95', 0):.2%}")
        lines.append(f"- **CVaR(95%)**: {risk_metrics.get('cvar_95', 0):.2%}")
        lines.append("")
        
        # Trade Analysis
        lines.append("## 📋 交易分析")
        lines.append("")
        lines.append(f"- **总交易次数**: {len(trades)}")
        lines.append(f"- **平均盈利**: {attribution.avg_win:.2%}")
        lines.append(f"- **平均亏损**: {attribution.avg_loss:.2%}")
        lines.append("")
        
        # Attribution by Asset
        if attribution.by_asset:
            lines.append("### 资产归因")
            lines.append("")
            lines.append("| 资产 | 盈亏 |")
            lines.append("|------|------|")
            sorted_assets = sorted(attribution.by_asset.items(), key=lambda x: x[1], reverse=True)
            for symbol, pnl in sorted_assets[:10]:
                pnl_str = f"¥{pnl:+,.0f}" if pnl != 0 else "¥0"
                lines.append(f"| {symbol} | {pnl_str} |")
            lines.append("")
        
        # Attribution by Sector
        if attribution.by_sector:
            lines.append("### 行业归因")
            lines.append("")
            lines.append("| 行业 | 盈亏 |")
            lines.append("|------|------|")
            sorted_sectors = sorted(attribution.by_sector.items(), key=lambda x: x[1], reverse=True)
            for sector, pnl in sorted_sectors:
                pnl_str = f"¥{pnl:+,.0f}" if pnl != 0 else "¥0"
                lines.append(f"| {sector} | {pnl_str} |")
            lines.append("")
        
        # Monthly Returns
        if attribution.by_period:
            lines.append("### 月度收益")
            lines.append("")
            lines.append("| 月份 | 收益率 |")
            lines.append("|------|--------|")
            for month, ret in sorted(attribution.by_period.items()):
                color = "🟢" if ret >= 0 else "🔴"
                lines.append(f"| {month} | {color} {ret:+.2%} |")
            lines.append("")
        
        # Recent Trades
        if trades:
            lines.append("## 📝 最近交易")
            lines.append("")
            lines.append("| 时间 | 代码 | 方向 | 数量 | 价格 |")
            lines.append("|------|------|------|------|------|")
            for trade in trades[-20:]:
                ts = trade.get('timestamp', '')[:16]
                symbol = trade.get('symbol', '')
                side = trade.get('type', trade.get('side', ''))
                qty = trade.get('quantity', 0)
                price = trade.get('price', 0)
                lines.append(f"| {ts} | {symbol} | {side} | {qty} | ¥{price:.2f} |")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*本报告由Quant Trading Platform自动生成*")
        
        return "\n".join(lines)
    
    def _annualize_return(self, total_return: float, n_days: int) -> float:
        """Annualize return"""
        if n_days <= 0:
            return 0
        years = n_days / 252
        if years <= 0:
            return total_return
        return (1 + total_return) ** (1 / years) - 1


class ReportExporter:
    """Export reports to different formats"""
    
    @staticmethod
    def to_dict(report_path: str) -> Dict:
        """Parse Markdown report to dict"""
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple parsing - extract key metrics
        result = {
            'content': content,
            'path': report_path
        }
        
        # Extract summary metrics from table
        lines = content.split('\n')
        for line in lines:
            if '初始资金' in line:
                result['initial_capital'] = line.split('|')[-2].strip()
            elif '最终资金' in line:
                result['final_capital'] = line.split('|')[-2].strip()
            elif '总收益率' in line:
                result['total_return'] = line.split('|')[-2].strip()
        
        return result
