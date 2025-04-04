import React from 'react';
import ReactECharts from 'echarts-for-react';

export default ({ isBting, btres }) => {


    // if (btRes === undefined) {
    //     return (
    //         <div>
    //             <h1>Backtesting</h1>
    //         </div>
    //     );
    // }

    if (isBting) {
        return <>
            <h1>Backtesting...</h1>
        </>
    }
    // console.log(btres)

    const options = {
        grid: { top: 8, right: 8, bottom: 24, left: 36 },
        xAxis: {
            data: btres['revenue'].x,
        },
        yAxis: {
            type: 'value',
        },
        series: [
            {
                data: btres['revenue']?.y,
                type: 'line',
                smooth: false,
            },
            {
                data: btres['market_value'],
                type: 'line',
                smooth: false,
            },
        ],
        tooltip: {
            trigger: 'axis',
        },
    };



    return <ReactECharts option={options} style={{ width: 1920, height: 1080, minWidth: 800, minHeight: 400 }} />;
};
