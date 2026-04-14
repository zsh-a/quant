"""
Analysis and Reports API Router - API endpoints for attribution analysis and report generation.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from loguru import logger

from src.analysis.attribution import ReturnAttribution, RiskAttribution
from src.analysis.reports.generator import ReportGenerator


router = APIRouter(prefix="/analysis", tags=["analysis"])

# Import session storage (from main server)
try:
    from session_db import SessionDB
    session_db = SessionDB()
except Exception:
    session_db = None


@router.get("/attribution/{session_id}")
async def get_attribution(session_id: str):
    """Get return attribution analysis"""
    if not session_db:
        raise HTTPException(500, "Session database not available")
    
    try:
        # Get session data
        equity_history = session_db.get_equity_history(session_id)
        trades = session_db.get_trades(session_id)
        
        if not equity_history:
            raise HTTPException(404, "Session not found or no data")
        
        # Run attribution
        attr = ReturnAttribution(trades, equity_history)
        result = attr.analyze()
        
        return {
            'session_id': session_id,
            'total_return': result.total_return,
            'by_asset': result.by_asset,
            'by_sector': result.by_sector,
            'by_period': result.by_period,
            'win_rate': result.win_rate,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'profit_factor': result.profit_factor
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Attribution error: {e}")
        raise HTTPException(500, str(e))


@router.get("/risk/{session_id}")
async def get_risk_analysis(session_id: str):
    """Get risk attribution analysis"""
    if not session_db:
        raise HTTPException(500, "Session database not available")
    
    try:
        equity_history = session_db.get_equity_history(session_id)
        session = session_db.get_session(session_id)
        
        if not equity_history:
            raise HTTPException(404, "Session not found or no data")
        
        positions = session.get('positions', {}) if session else {}
        
        risk_attr = RiskAttribution(equity_history, positions)
        result = risk_attr.analyze()
        
        return {
            'session_id': session_id,
            'volatility': result['volatility'],
            'max_drawdown': result['max_drawdown'],
            'var_95': result['var_95'],
            'cvar_95': result['cvar_95'],
            'sharpe_ratio': result['sharpe_ratio']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(500, str(e))


@router.get("/report/{session_id}")
async def generate_report(
    session_id: str,
    format: str = "markdown"
):
    """Generate backtest report"""
    if not session_db:
        raise HTTPException(500, "Session database not available")
    
    try:
        # Get session data
        session = session_db.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        
        equity_history = session_db.get_equity_history(session_id)
        trades = session_db.get_trades(session_id)
        positions = session.get('positions', {})
        
        # Generate report
        generator = ReportGenerator()
        report_path = generator.generate(
            session_id=session_id,
            strategy_name=session.get('strategy', 'Unknown'),
            equity_history=equity_history,
            trades=trades,
            positions=positions,
            params=session.get('params', {}),
            metadata={
                'start_date': session.get('start_date'),
                'end_date': session.get('end_date'),
                'status': session.get('status')
            }
        )
        
        if format == "markdown":
            return FileResponse(
                report_path,
                media_type='text/markdown',
                filename=f"report_{session_id}.md"
            )
        else:
            # Return content directly
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {'content': content, 'path': report_path}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(500, str(e))


@router.get("/report/{session_id}/pdf")
async def generate_pdf_report(session_id: str):
    """Generate and download PDF backtest report."""
    if not session_db:
        raise HTTPException(500, "Session database not available")
    try:
        session = session_db.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")

        equity_history = session_db.get_equity_history(session_id)
        trades = session_db.get_trades(session_id)
        positions = session.get("positions", {})

        from src.analysis.reports.html_generator import HTMLReportGenerator
        gen = HTMLReportGenerator()
        pdf_path = gen.generate_pdf(
            session_id=session_id,
            strategy_name=session.get("strategy", "Unknown"),
            equity_history=equity_history,
            trades=trades,
            positions=positions,
            params=session.get("params", {}),
        )
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"report_{session_id}.pdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF report generation error: {e}")
        raise HTTPException(500, str(e))


@router.get("/summary/{session_id}")
async def get_session_summary(session_id: str):
    """Get quick session summary"""
    if not session_db:
        raise HTTPException(500, "Session database not available")
    
    try:
        session = session_db.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        
        equity_history = session_db.get_equity_history(session_id)
        trades = session_db.get_trades(session_id)
        
        if not equity_history:
            return {
                'session_id': session_id,
                'status': session.get('status', 'unknown'),
                'total_return': 0,
                'n_trades': 0
            }
        
        initial = equity_history[0].get('total_equity', 1)
        final = equity_history[-1].get('total_equity', 1)
        total_return = (final - initial) / initial if initial > 0 else 0
        
        return {
            'session_id': session_id,
            'strategy': session.get('strategy'),
            'status': session.get('status'),
            'start_date': equity_history[0].get('date', ''),
            'end_date': equity_history[-1].get('date', ''),
            'initial_capital': initial,
            'final_equity': final,
            'total_return': total_return,
            'n_trades': len(trades)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(500, str(e))
