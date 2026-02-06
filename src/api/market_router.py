"""
Market Data API Router - Endpoints for market-wide analysis and indicators.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import talib as ta
from loguru import logger
from datetime import datetime, timedelta

from db import DB

router = APIRouter(prefix="/market", tags=["market"])

@router.get("/industry_breadth")
async def get_industry_breadth(
    start_date: str,
    end_date: Optional[str] = None,
    index_code: str = "000985"
):
    """
    Calculate industry market breadth over time.
    Logic follows JSG strategy: % of stocks in industry with close > MA20.
    """
    db_client = DB()
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # 1. Get the timeline of trading days
        # Use a common index to get trading days
        ref_df = db_client.get_kline("sh.000001", start_date, end_date)
        if ref_df.empty:
            return {"dates": [], "industries": [], "data": []}
        
        trading_days = ref_df.index.strftime("%Y-%m-%d").tolist()
        
        # 2. Get all stocks in the index for the period
        # For simplicity and performance, we take stocks as of the end date
        # and assume they are representative enough for the breadth calculation.
        stocks = db_client.get_index_stocks(index_code, end_date)
        if not stocks:
            raise HTTPException(404, f"No stocks found for index {index_code}")

        # 3. Fetch prices for MA calculation (need some lookback for the first date)
        lookback_start = (pd.to_datetime(start_date) - timedelta(days=40)).strftime("%Y-%m-%d")
        
        # This might be a large query, we'll fetch only what we need
        # Set count high enough to cover the range (e.g., 1000 bars)
        price_df = db_client.get_price(stocks, end_date, ["close"], 1000, start_date=lookback_start)
        if price_df.empty:
            return {"dates": [], "industries": [], "data": []}

        # 4. Calculate MA20 and Bias
        # Ensure we have a multi-index [code, date] or sort properly
        price_df = price_df.sort_index(level=['code', 'date'])
        
        # Calculate MA20 per stock
        price_df['ma20'] = price_df.groupby(level='code')['close'].transform(
            lambda x: ta.MA(x, timeperiod=20)
        )
        
        # Filter back to requested range
        price_df = price_df[price_df.index.get_level_values('date') >= pd.to_datetime(start_date)]
        price_df['bias'] = price_df['close'] > price_df['ma20']
        
        # 5. Map stocks to industries
        # Use the end_date for industry classification
        industry_mapping = db_client.get_stock_industry_sw(stocks, end_date)
        # industry_mapping index is code, columns include industry_name
        
        # Join industry info to price_df
        # We reset index to join on 'code'
        price_df = price_df.reset_index()
        price_df = price_df.merge(industry_mapping[['industry_name']], left_on='code', right_index=True)
        
        # 6. Group by Date and Industry
        # Calculate mean bias (breadth)
        breadth_df = price_df.groupby(['date', 'industry_name'])['bias'].mean().unstack(level=-1)
        breadth_df = (breadth_df * 100).round(1)
        breadth_df = breadth_df.fillna(0) # Fill gaps
        
        # 7. Format for Heatmap
        # ECharts heatmap usually wants [x_index, y_index, value]
        dates = breadth_df.index.strftime("%Y-%m-%d").tolist()
        industries = breadth_df.columns.tolist()
        
        heatmap_data = []
        for i, date in enumerate(dates):
            for j, industry in enumerate(industries):
                val = breadth_df.iloc[i, j]
                heatmap_data.append([i, j, float(val)])
        
        return {
            "dates": dates,
            "industries": industries,
            "data": heatmap_data
        }
        
    except Exception as e:
        logger.exception(f"Error calculating industry breadth: {e}")
        raise HTTPException(500, str(e))

@router.get("/industry_amount")
async def get_industry_amount(
    start_date: str,
    end_date: Optional[str] = None,
    index_code: str = "000985"
):
    """
    Calculate industry trading amount share over time.
    Shows the percentage of total market liquidity captured by each sector.
    """
    db_client = DB()
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        ref_df = db_client.get_kline("sh.000001", start_date, end_date)
        if ref_df.empty:
            return {"dates": [], "industries": [], "data": []}
        
        stocks = db_client.get_index_stocks(index_code, end_date)
        if not stocks:
            raise HTTPException(404, f"No stocks found for index {index_code}")

        # Fetch trading amount
        price_df = db_client.get_price(stocks, end_date, ["amount"], 1000, start_date=start_date)
        if price_df.empty:
            return {"dates": [], "industries": [], "data": []}

        # Map stocks to industries
        industry_mapping = db_client.get_stock_industry_sw(stocks, end_date)
        
        price_df = price_df.reset_index()
        price_df = price_df.merge(industry_mapping[['industry_name']], left_on='code', right_index=True)
        
        # Calculate daily total amount for normalization (Amount Share)
        daily_total = price_df.groupby('date')['amount'].sum()
        
        # Group by Date and Industry
        industry_amount = price_df.groupby(['date', 'industry_name'])['amount'].sum().unstack(level=-1)
        
        # Convert to Share (%)
        industry_share = industry_amount.div(daily_total, axis=0) * 100
        industry_share = industry_share.round(2).fillna(0)
        
        dates = industry_share.index.strftime("%Y-%m-%d").tolist()
        industries = industry_share.columns.tolist()
        
        heatmap_data = []
        for i, date in enumerate(dates):
            for j, industry in enumerate(industries):
                val = industry_share.iloc[i, j]
                heatmap_data.append([i, j, float(val)])
        
        return {
            "dates": dates,
            "industries": industries,
            "data": heatmap_data
        }
        
    except Exception as e:
        logger.exception(f"Error calculating industry amount: {e}")
        raise HTTPException(500, str(e))
