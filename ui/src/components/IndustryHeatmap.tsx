import React, { Suspense, lazy, useState, useEffect } from 'react';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { apiFetch } from '../utils/api';

const HeatmapChart = lazy(() => import('./charts/HeatmapChart'));

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
      const resp = await apiFetch(`/market/${endpoint}?start_date=${startDate}&end_date=${endDate}`);
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
            color: '#CFA844', // Burnished gold highlight
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
        backgroundColor: 'rgba(12, 15, 24, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.08)',
        borderWidth: 1,
        textStyle: { color: '#E2E5EB' },
        formatter: (params: any) => {
          const [dateIdx, industryIdx, value] = params.data;
          const industryName = heatmapData.industries[industryIdx];
          const isBlack = BLACK_LISTED_INDUSTRIES.includes(industryName);
          return `
            <div style="font-weight: bold; margin-bottom: 4px;">${heatmapData.dates[dateIdx]}</div>
            <div style="color: ${isBlack ? '#CFA844' : '#7A828F'};">
              ${isBlack ? '⚠️ ' : ''}${industryName}
            </div>
            <div style="font-size: 16px; margin-top: 4px;">
              ${isAmount ? 'Liquidity Share' : 'Breadth'}: <span style="color: ${isAmount ? '#4DA8D4' : (value > 50 ? '#46A488' : '#CF5A55')}">${value}%</span>
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
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.07)' } },
        axisLabel: { color: '#7A828F', fontSize: 10, rotate: 45 },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'category',
        data: industriesWithHighlight,
        splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'transparent'] } },
        axisLine: { show: false },
        axisLabel: {
          color: '#E2E5EB',
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
              '#1C222F', // 0% - Neutral (matches secondary surface)
              '#2D2A6B', // 20%
              '#3E36A8', // 40%
              '#6B54C0', // 60%
              '#A040B0', // 80%
              '#CFA844'  // 100% - Hot (Burnished gold)
            ] :
            ['#CF5A55', '#8A4520', '#1C222F', '#1A5E44', '#46A488'] // Breadth: warm red → teal
        },
        textStyle: { color: '#7A828F' }
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
            <Suspense fallback={<div style={{ height: '950px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading chart...</div>}>
              <HeatmapChart 
                option={getOption()} 
                style={{ height: '950px', width: '100%' }}
                theme="dark"
              />
            </Suspense>
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
