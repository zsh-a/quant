import React from 'react';
import { Spinner, Text, Table } from "@chakra-ui/react"
import {
  ButtonGroup,
  Heading,
  IconButton,
  Pagination,
  Stack,
} from "@chakra-ui/react"
import {
  Stat,
  StatGroup,
} from '@chakra-ui/react'
import { LuChevronLeft, LuChevronRight } from "react-icons/lu"


import ReactECharts from 'echarts-for-react';

export default ({ isBting, btres }) => {

  if (isBting) {
    return (
      <>
        <Text>Loading...</Text>
        <Spinner size="sm" />
      </>
    )
  }

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
        name: 'Strategy Return'
      },
      {
        data: btres['market_value']?.y || btres['market_value'],
        type: 'line',
        smooth: false,
        name: 'Market Value'
      },
    ],
    tooltip: {
      trigger: 'axis',
    },
  };

  const trading_history = btres["order_stats"]['order_history']

  const trading_history_ui = (
    <Stack width="full" gap="5">
      <Heading size="xl">orders</Heading>
      <Table.Root size="sm" variant="outline" striped>
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader>symbol</Table.ColumnHeader>
            <Table.ColumnHeader>return</Table.ColumnHeader>
            <Table.ColumnHeader>open time</Table.ColumnHeader>
            <Table.ColumnHeader>close time</Table.ColumnHeader>
            {/* <Table.ColumnHeader textAlign="end">Price</Table.ColumnHeader> */}
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {trading_history.map((order, index) => (
            <Table.Row key={index}>
              <Table.Cell>{order.symbol}</Table.Cell>
              <Table.Cell>{order.order_return}</Table.Cell>
              <Table.Cell>{order.open_time}</Table.Cell>
              <Table.Cell>{order.close_time}</Table.Cell>
              {/* <Table.Cell textAlign="end">{item.price}</Table.Cell> */}
            </Table.Row>
          ))}
        </Table.Body>
      </Table.Root>

    </Stack>
  )
  return (
    <>
      <Heading size="md" mb={4}>Portfolio Metrics</Heading>
      <StatGroup mb={6}>
        <Stat.Root>
          <Stat.Label>Strategy Return</Stat.Label>
          <Stat.ValueText>{btres['strategy_return']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Annualized Return</Stat.Label>
          <Stat.ValueText>{btres['annualized_return']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Max Drawdown</Stat.Label>
          <Stat.ValueText>{btres['max_drawdown']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Sharpe Ratio</Stat.Label>
          <Stat.ValueText>{btres['sharpe_ratio']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Volatility</Stat.Label>
          <Stat.ValueText>{btres['volatility']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Calmar Ratio</Stat.Label>
          <Stat.ValueText>{btres['calmar_ratio']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Buy & Hold</Stat.Label>
          <Stat.ValueText>{btres['market_return']}%</Stat.ValueText>
        </Stat.Root>
      </StatGroup>

      <Heading size="md" mb={4}>Trade Metrics</Heading>
      <StatGroup mb={6}>
        <Stat.Root>
          <Stat.Label>Total Trades</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['total_trades']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Win Rate</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['win_rate']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Profit Factor</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['profit_factor']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Avg Profit</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['avg_profit']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Avg Loss</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['avg_loss']}</Stat.ValueText>
        </Stat.Root>
        <Stat.Root>
          <Stat.Label>Max Win</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['max_win']}</Stat.ValueText>
        </Stat.Root>
         <Stat.Root>
          <Stat.Label>Max Loss</Stat.Label>
          <Stat.ValueText>{btres['order_stats']?.['max_loss']}</Stat.ValueText>
        </Stat.Root>
      </StatGroup>

      <ReactECharts option={options} style={{ width: 1920, height: 1080, minWidth: 800, minHeight: 400 }} />
      {trading_history_ui}
    </>
  )
};
