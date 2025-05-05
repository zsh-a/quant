import * as React from 'react'
import { useEffect, useState, useRef } from 'react'

import { Provider } from "./components/ui/provider"
import { defaultSystem } from "@chakra-ui/react"
import {
  ButtonGroup,
  Flex,
  Heading,
  IconButton,
  Pagination,
  Stack,
} from "@chakra-ui/react"
import Chart from './Chart'
import Menu from './Menu'
import BTStat from './BTStat'

export default function App() {
  // 2. Wrap ChakraProvider at the root of your app
  const [codes, setCodes] = useState(['sz.002883']);
  const [name, setName] = useState('');

  const [btres, setBtres] = useState({ 'revenue': { 'x': [], 'y': [] } });
  const [isBting, setisBting] = useState(true);


  const inputRef = useRef('');

  // 添加useEffect钩子，在组件首次加载时根据默认code获取name
  // useEffect(() => {
  //   const url = `http://localhost:8000/meta/${code}`;
  //   fetch(url)
  //     .then(response => response.json())
  //     .then(data => setName(data.name));
  // }, []); // 空依赖数组确保只在组件首次加载时执行一次

  useEffect(() => {
    async function backtest() {
      const url = `http://localhost:8000/backtest/sh.000300/20230101/20260101`;
      const resp = await fetch(url);
      return resp.json();
    }

    (async () => {
      const result = await backtest();
      const code_list = result['buy_sell_points'].map((point) => point['symbol']);
      setCodes(code_list);
      setBtres(result);
      setisBting(false)
    })();
  }, []);

  function handleClick() {
    setisBting(true)
    // const url = `http://localhost:8000/meta/${inputRef.current.value}`;
    // fetch(url)
    //   .then(response => response.json())
    //   .then(data => setName(data.name));
  }


  const charts = codes.map(code => (
    <Flex key={code} width="100%" gap="5" marginBottom="20px" direction="column">
      <Menu code={code} name={name} btres={btres} onclick={handleClick} inputRef={inputRef} />
      {/* <Chart code={code} btres={btres} /> */}
    </Flex>
  ));

  return (
    <Provider>
      {/* <Menu code={code} name={name} btres={btres} onclick={handleClick} inputRef={inputRef} />
    <Chart code={code} btres={btres} /> */}

      {/* {charts} */}

      <BTStat isBting={isBting} btres={btres} />
    </Provider>
  )
}