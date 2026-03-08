import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { API_BASE } from '../utils/api';

interface HeatmapData {
  dates: string[];
  industries: string[];
  data: [number, number, number][];
}

type MetricType = 'breadth' | 'amount';

const BLACK_LISTED_INDUSTRIES = ['银行', '煤炭', '有色金属', '钢铁', '银行I', '煤炭I', '有色金属I', '钢铁I'];

export const IndustryHeatmap: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [metric, setMetric] = useState<MetricType>('breadth');
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [startDate, setStartDate] = useState(() => {
    const d = new Date();
    d.setMonth(d.getMonth() - 3);
    return d.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().split('T')[0]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const endpoint = metric === 'breadth' ? 'industry_breadth' : 'industry_amount';
      const resp = await fetch(`${API_BASE}/market/${endpoint}?start_date=${startDate}&end_date=${endDate}`);
      if (resp.ok) {
        const data = await resp.json();
        setHeatmapData(data);
      }
    } catch (err) {
      console.error("Failed to fetch industry data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [metric]); // Refetch when metric changes

  const getOption = () => {
    if (!heatmapData) return {};

    const industriesWithHighlight = heatmapData.industries.map(name => {
      const isBlacklisted = BLACK_LISTED_INDUSTRIES.includes(name);
      if (isBlacklisted) {
        return {
          value: name,
          textStyle: {
            color: '#fbbf24', // Gold/Amber highlight
            fontWeight: 'bold',
            fontSize: 12
          }
        };
      }
      return name;
    });

    const isAmount = metric === 'amount';
    // Calculate a reasonable max for Amount Share mode, or fixed 100 for breadth
    const dataValues = heatmapData.data.map(d => d[2]);
    const dynamicMax = isAmount ? Math.max(...dataValues, 10) : 100;

    return {
      backgroundColor: 'transparent',
      tooltip: {
        position: 'top',
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        textStyle: { color: '#f8fafc' },
        formatter: (params: any) => {
          const [dateIdx, industryIdx, value] = params.data;
          const industryName = heatmapData.industries[industryIdx];
          const isBlack = BLACK_LISTED_INDUSTRIES.includes(industryName);
          return `
            <div style="font-weight: bold; margin-bottom: 4px;">${heatmapData.dates[dateIdx]}</div>
            <div style="color: ${isBlack ? '#fbbf24' : '#94a3b8'};">
              ${isBlack ? '⚠️ ' : ''}${industryName}
            </div>
            <div style="font-size: 16px; margin-top: 4px;">
              ${isAmount ? 'Liquidity Share' : 'Breadth'}: <span style="color: ${isAmount ? '#3b82f6' : (value > 50 ? '#10b981' : '#ef4444')}">${value}%</span>
            </div>
          `;
        }
      },
      grid: {
        height: '85%',
        top: '5%',
        left: '120px', 
        right: '30px',
        bottom: '10%',
        containLabel: false
      },
      xAxis: {
        type: 'category',
        data: heatmapData.dates,
        splitArea: { show: false },
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, rotate: 45 },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'category',
        data: industriesWithHighlight,
        splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'transparent'] } },
        axisLine: { show: false },
        axisLabel: { 
          color: '#f8fafc', 
          fontSize: 11,
          formatter: (value: string) => {
            return BLACK_LISTED_INDUSTRIES.includes(value) ? `⚠️ ${value}` : value;
          }
        },
        axisTick: { show: false }
      },
      visualMap: {
        min: 0,
        max: dynamicMax,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0',
        itemWidth: 15,
        itemHeight: 300,
        inRange: {
          color: isAmount ? 
            [
              '#1e293b', // 0% - Neutral (matches background)
              '#312e81', // 20%
              '#4338ca', // 40%
              '#7c3aed', // 60%
              '#c026d3', // 80%
              '#fbbf24'  // 100% - Hot (Amber/Gold)
            ] : 
            ['#ef4444', '#92400e', '#1e293b', '#065f46', '#10b981'] // Breadth colors (Red-Green)
        },
        textStyle: { color: '#94a3b8' }
      },
      series: [{
        name: isAmount ? 'Liquidity' : 'Breadth',
        type: 'heatmap',
        data: heatmapData.data,
        label: { show: false },
        itemStyle: {
          borderColor: 'rgba(0, 0, 0, 0.3)',
          borderWidth: 1
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
            borderColor: '#fff',
            borderWidth: 1
          }
        }
      }]
    };
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Market Structure"
        title={`Sector Heatmap (${metric === 'breadth' ? 'Market Breadth' : 'Liquidity Share'})`}
        description={
          metric === 'breadth'
            ? 'Percentage of stocks in sector above 20-day Moving Average'
            : 'Percentage of total market liquidity captured by each sector'
        }
      />

      <SectionCard
        title="Heatmap Controls"
        description="Switch metrics and adjust the time window before refreshing the market map."
        action={<Button onClick={fetchData} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh'}</Button>}
      >
        <div className="flex flex-wrap gap-2">
          <Button variant={metric === 'breadth' ? 'default' : 'outline'} size="sm" onClick={() => setMetric('breadth')}>
            Breadth
          </Button>
          <Button variant={metric === 'amount' ? 'default' : 'outline'} size="sm" onClick={() => setMetric('amount')}>
            Liquidity
          </Button>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <Input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
          <Input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
        </div>
      </SectionCard>

      <SectionCard title="Sector Map" description="Blacklisted industries are highlighted to make defensive or crowded groups easier to spot.">
        <div style={{ minHeight: '950px' }}>
          {heatmapData ? (
            <ReactECharts 
              option={getOption()} 
              style={{ height: '950px', width: '100%' }}
              theme="dark"
            />
          ) : (
            <div style={{ height: '950px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {loading ? 'Loading data...' : 'No data available'}
            </div>
          )}
        </div>
      </SectionCard>
    </div>
  );
};

export default IndustryHeatmap;
