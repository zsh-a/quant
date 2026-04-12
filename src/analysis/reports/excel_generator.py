"""
Excel Report Generator - Generate Excel reports with xlsxwriter.
"""

from typing import Dict, List, Optional
from datetime import datetime
import os
from loguru import logger

try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False
    logger.warning("xlsxwriter not installed, Excel export disabled")

from src.analysis.attribution import ReturnAttribution, RiskAttribution


class ExcelReportGenerator:
    """Generate Excel reports with multiple sheets"""

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
    ) -> Optional[str]:
        """Generate Excel report"""
        if not HAS_XLSXWRITER:
            logger.error("xlsxwriter not installed")
            return None

        logger.info(f"Generating Excel report for {session_id}")

        # Run attribution
        attr = ReturnAttribution(trades, equity_history)
        attribution = attr.analyze()

        risk_attr = RiskAttribution(equity_history, positions)
        risk = risk_attr.analyze()

        # Create workbook
        filename = f"report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        workbook = xlsxwriter.Workbook(filepath)
        
        # Formats
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white'})
        money_fmt = workbook.add_format({'num_format': '¥#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        positive_fmt = workbook.add_format({'font_color': 'green', 'num_format': '¥#,##0.00'})
        negative_fmt = workbook.add_format({'font_color': 'red', 'num_format': '¥#,##0.00'})

        # 1. Summary Sheet
        self._write_summary_sheet(
            workbook, header_fmt, money_fmt, pct_fmt,
            session_id, strategy_name, equity_history, trades, attribution, risk, params
        )

        # 2. Equity History Sheet
        self._write_equity_sheet(workbook, header_fmt, money_fmt, date_fmt, equity_history)

        # 3. Trades Sheet
        self._write_trades_sheet(workbook, header_fmt, money_fmt, date_fmt, trades)

        # 4. Attribution Sheet
        self._write_attribution_sheet(
            workbook, header_fmt, money_fmt, pct_fmt, positive_fmt, negative_fmt, attribution
        )

        workbook.close()
        logger.info(f"Excel report saved: {filepath}")
        return filepath

    def _write_summary_sheet(
        self, workbook, header_fmt, money_fmt, pct_fmt,
        session_id, strategy_name, equity_history, trades, attribution, risk, params
    ):
        """Write summary sheet"""
        sheet = workbook.add_worksheet("摘要")
        sheet.set_column('A:A', 20)
        sheet.set_column('B:B', 25)

        row = 0
        sheet.write(row, 0, "回测报告", header_fmt)
        sheet.write(row, 1, strategy_name, header_fmt)
        row += 2

        # Basic info
        info = [
            ("会话ID", session_id),
            ("生成时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ]
        
        if equity_history:
            initial = equity_history[0].get('total_equity', 0)
            final = equity_history[-1].get('total_equity', 0)
            total_return = (final - initial) / initial if initial > 0 else 0
            
            info.extend([
                ("回测期间", f"{equity_history[0].get('date', '')} 至 {equity_history[-1].get('date', '')}"),
                ("初始资金", initial),
                ("最终权益", final),
                ("总收益率", total_return),
                ("夏普比率", risk.get('sharpe_ratio', 0)),
                ("最大回撤", risk.get('max_drawdown', 0)),
                ("交易次数", len(trades)),
                ("胜率", attribution.win_rate),
                ("盈亏比", attribution.profit_factor),
            ])

        for label, value in info:
            sheet.write(row, 0, label)
            if isinstance(value, float):
                if 'rate' in label.lower() or '回撤' in label or '胜率' in label:
                    sheet.write(row, 1, value, pct_fmt)
                else:
                    sheet.write(row, 1, value, money_fmt)
            else:
                sheet.write(row, 1, value)
            row += 1

        # Strategy params
        if params:
            row += 2
            sheet.write(row, 0, "策略参数", header_fmt)
            sheet.write(row, 1, "", header_fmt)
            row += 1
            for k, v in params.items():
                sheet.write(row, 0, k)
                sheet.write(row, 1, str(v))
                row += 1

    def _write_equity_sheet(self, workbook, header_fmt, money_fmt, date_fmt, equity_history):
        """Write equity history sheet"""
        sheet = workbook.add_worksheet("权益曲线")
        sheet.set_column('A:A', 15)
        sheet.set_column('B:B', 18)
        sheet.set_column('C:C', 15)

        headers = ["日期", "总权益", "日收益率"]
        for col, h in enumerate(headers):
            sheet.write(0, col, h, header_fmt)

        for row, eq in enumerate(equity_history, 1):
            sheet.write(row, 0, eq.get('date', ''))
            sheet.write(row, 1, eq.get('total_equity', 0), money_fmt)
            
            if row > 1:
                prev = equity_history[row-2].get('total_equity', 0)
                curr = eq.get('total_equity', 0)
                if prev > 0:
                    ret = (curr - prev) / prev
                    sheet.write(row, 2, ret)

        # Add chart
        if len(equity_history) > 1:
            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'name': '权益',
                'categories': f"='权益曲线'!$A$2:$A${len(equity_history)+1}",
                'values': f"='权益曲线'!$B$2:$B${len(equity_history)+1}",
            })
            chart.set_title({'name': '权益曲线'})
            chart.set_size({'width': 600, 'height': 300})
            sheet.insert_chart('E2', chart)

    def _write_trades_sheet(self, workbook, header_fmt, money_fmt, date_fmt, trades):
        """Write trades sheet"""
        sheet = workbook.add_worksheet("交易记录")
        sheet.set_column('A:A', 20)
        sheet.set_column('B:E', 12)

        headers = ["时间", "代码", "方向", "数量", "价格"]
        for col, h in enumerate(headers):
            sheet.write(0, col, h, header_fmt)

        for row, trade in enumerate(trades, 1):
            sheet.write(row, 0, trade.get('timestamp', '')[:19])
            sheet.write(row, 1, trade.get('symbol', ''))
            sheet.write(row, 2, trade.get('type', trade.get('side', '')))
            sheet.write(row, 3, trade.get('quantity', 0))
            sheet.write(row, 4, trade.get('price', 0), money_fmt)

    def _write_attribution_sheet(
        self, workbook, header_fmt, money_fmt, pct_fmt, positive_fmt, negative_fmt, attribution
    ):
        """Write attribution sheet"""
        sheet = workbook.add_worksheet("归因分析")
        sheet.set_column('A:A', 15)
        sheet.set_column('B:B', 15)
        sheet.set_column('D:D', 15)
        sheet.set_column('E:E', 15)

        # By Asset
        sheet.write(0, 0, "资产归因", header_fmt)
        sheet.write(0, 1, "盈亏", header_fmt)
        
        row = 1
        for symbol, pnl in sorted(attribution.by_asset.items(), key=lambda x: x[1], reverse=True):
            sheet.write(row, 0, symbol)
            fmt = positive_fmt if pnl >= 0 else negative_fmt
            sheet.write(row, 1, pnl, fmt)
            row += 1

        # By Sector
        sheet.write(0, 3, "行业归因", header_fmt)
        sheet.write(0, 4, "盈亏", header_fmt)
        
        row = 1
        for sector, pnl in sorted(attribution.by_sector.items(), key=lambda x: x[1], reverse=True):
            sheet.write(row, 3, sector)
            fmt = positive_fmt if pnl >= 0 else negative_fmt
            sheet.write(row, 4, pnl, fmt)
            row += 1

        # Monthly returns
        if attribution.by_period:
            start_row = max(len(attribution.by_asset), len(attribution.by_sector)) + 3
            sheet.write(start_row, 0, "月度收益", header_fmt)
            sheet.write(start_row, 1, "收益率", header_fmt)
            
            row = start_row + 1
            for month, ret in sorted(attribution.by_period.items()):
                sheet.write(row, 0, month)
                sheet.write(row, 1, ret, pct_fmt)
                row += 1
