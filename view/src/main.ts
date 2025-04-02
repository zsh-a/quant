import './style.css'
import typescriptLogo from './typescript.svg'
import viteLogo from '/vite.svg'
import { setupCounter } from './counter.ts'
import { AreaSeries, BarSeries, BaselineSeries, createChart, CandlestickSeries, LineSeries } from 'lightweight-charts';
const darkTheme = {
  chart: {
    layout: {
      background: {
        type: 'solid',
        color: '#2B2B43',
      },
      lineColor: '#2B2B43',
      textColor: '#D9D9D9',
    },
    watermark: {
      color: 'rgba(0, 0, 0, 0)',
    },
    crosshair: {
      color: '#758696',
    },
    grid: {
      vertLines: {
        color: '#2B2B43',
      },
      horzLines: {
        color: '#363C4E',
      },
    },
  },
  series: {
    topColor: 'rgba(32, 226, 47, 0.56)',
    bottomColor: 'rgba(32, 226, 47, 0.04)',
    lineColor: 'rgba(32, 226, 47, 1)',
  },
};

const lightTheme = {
  chart: {
    layout: {
      background: {
        type: 'solid',
        color: '#FFFFFF',
      },
      lineColor: '#2B2B43',
      textColor: '#191919',
    },
    watermark: {
      color: 'rgba(0, 0, 0, 0)',
    },
    grid: {
      vertLines: {
        visible: false,
      },
      horzLines: {
        color: '#f0f3fa',
      },
    },
  },
  series: {
    topColor: 'rgba(33, 150, 243, 0.56)',
    bottomColor: 'rgba(33, 150, 243, 0.04)',
    lineColor: 'rgba(33, 150, 243, 1)',
  },
};

var themesData = {
  Dark: darkTheme,
  Light: lightTheme,
};


let code = 'sz.300100'
const apiUrl: string = `http://localhost:8000/stocks/${code}`;


async function fetchData() {
  try {
    // 使用 await 等待 fetch 请求完成
    const response = await fetch(apiUrl);
    // 检查响应状态是否正常
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    // 使用 await 等待将响应数据解析为 JSON 格式
    const data = await response.json();
    console.log(data);
    return data;
  } catch (error) {
    console.error('请求出错：', error);
  }
}


const kline = await fetchData();



const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };


const chart = createChart(document.querySelector<HTMLDivElement>("#stock")!);

const candlestickSeries = chart.addSeries(CandlestickSeries, {
  upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
});
candlestickSeries.setData(kline['kline']);

const colors = ["#eeeff0", '#ffd000', "#ef39b2", "#0acb5a", "#199af4"]
let i = 0;
for (let ma in kline['ma']) {
  const lineSeries = chart.addSeries(LineSeries, { color: colors[i++] });
  lineSeries.setData(kline['ma'][ma]);
}


// 创建独立窗格显示ATR
const atrPane = chart.addSeries(LineSeries, { color: '#2962FF' });
atrPane.setData(kline['ATR']);

// const lineSeries = chart.addSeries(LineSeries, { color: '#2962FF' });

// const data = [{ value: 0, time: 1642425322 }, { value: 8, time: 1642511722 }, { value: 10, time: 1642598122 }, { value: 20, time: 1642684522 }, { value: 3, time: 1642770922 }, { value: 43, time: 1642857322 }, { value: 41, time: 1642943722 }, { value: 43, time: 1643030122 }, { value: 56, time: 1643116522 }, { value: 46, time: 1643202922 }];

// lineSeries.setData(data);

chart.timeScale().fitContent();


const theme = 'Dark';

chart.applyOptions(themesData[theme].chart);
// areaSeries.applyOptions(themesData[theme].series);

chart.timeScale().fitContent();



