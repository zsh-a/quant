import * as React from 'react'
import { useEffect, useState, useRef } from 'react'

// 1. import `ChakraProvider` component
import { ChakraProvider } from '@chakra-ui/react'
import Chart from './Chart'
import Menu from './Menu'

export default function App() {
  // 2. Wrap ChakraProvider at the root of your app
  const [code, setCode] = useState('sh.000001');
  const [name, setName] = useState('');

  const [btres, setBtres] = useState({});

  const inputRef = useRef('');

  function handleClick() {
    setCode(inputRef.current.value)
    const url = `http://localhost:8000/meta/${inputRef.current.value}`;
    fetch(url)
      .then(response => response.json())
      .then(data => setName(data.name));
  }

  function handlbacktest(data) {
    setBtres(data);
  }

  return (
    <ChakraProvider>
      <Menu name={name} btres={btres} onclick={handleClick} inputRef={inputRef}/>
      <Chart code={code} handlbacktest={handlbacktest} />
    </ChakraProvider>
  )
}