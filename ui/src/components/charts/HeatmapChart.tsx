import React from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { HeatmapChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, VisualMapComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

interface HeatmapChartProps {
  option: any;
  style?: React.CSSProperties;
  theme?: string;
}

const HeatmapChartComponent: React.FC<HeatmapChartProps> = ({ option, style, theme }) => (
  <ReactEChartsCore echarts={echarts} option={option} style={style} theme={theme} />
);

export default HeatmapChartComponent;
