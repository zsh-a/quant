import * as React from 'react'
import { useEffect, useState, useRef } from 'react'

import { Provider } from "./components/ui/provider"
import { defaultSystem } from "@chakra-ui/react"
import Chart from './Chart'
import Menu from './Menu'
import BTStat from './BTStat'

export default function App() {
  // 2. Wrap ChakraProvider at the root of your app
  const [code, setCode] = useState('sz.002883');
  const [name, setName] = useState('');

  const [btres, setBtres] = useState({ 'revenue': { 'x': [], 'y': [] } });
  const [isBting, setisBting] = useState(true);


  const inputRef = useRef('');

  // 添加useEffect钩子，在组件首次加载时根据默认code获取name
  useEffect(() => {
    const url = `http://localhost:8000/meta/${code}`;
    fetch(url)
      .then(response => response.json())
      .then(data => setName(data.name));
  }, []); // 空依赖数组确保只在组件首次加载时执行一次

  function handleClick() {
    setisBting(true)
    setCode(inputRef.current.value)

    const url = `http://localhost:8000/meta/${inputRef.current.value}`;
    fetch(url)
      .then(response => response.json())
      .then(data => setName(data.name));
  }

  function handlbacktest(data) {
    setBtres(data);
    setisBting(false)
  }

  return (
    <Provider>
      <Menu code={code} name={name} btres={btres} onclick={handleClick} inputRef={inputRef} />
      <Chart code={code} handlbacktest={handlbacktest} />
      {/* <BTStat isBting={isBting} btres={btres} /> */}
    </Provider>
  )
}