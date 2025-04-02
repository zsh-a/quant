import { useEffect, useState } from 'react'
import { Button, ButtonGroup } from '@chakra-ui/react'
import { Input } from '@chakra-ui/react'
import { Flex, Spacer } from '@chakra-ui/react'

import {
    Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
    StatArrow,
    StatGroup,
} from '@chakra-ui/react'
import { Divider, Stack, Text } from '@chakra-ui/react'

export default ({ name, btres, onclick, inputRef }) => {

    return (
        <>
            <Stack direction='row' h='100px' p={4}>
                <Text>{inputRef.current.value}</Text>
                <Text>{name}</Text>
            </Stack>


            <StatGroup>
                <Stat>
                    <StatLabel>Strategy Return</StatLabel>
                    <StatNumber>{btres['strategy_return']}%</StatNumber>
                    {/* <StatHelpText>Feb 12 - Feb 28</StatHelpText> */}
                </Stat>

                <Stat>
                    <StatLabel>Max Drawdown</StatLabel>
                    <StatNumber>{btres['max_drawdown']}</StatNumber>
                    {/* <StatHelpText>
                        <StatArrow type='decrease' />
                        9.05%
                    </StatHelpText> */}
                </Stat>
                <Stat>
                    <StatLabel>Sharpe Ratio</StatLabel>
                    <StatNumber>{btres['sharpe_ratio']}</StatNumber>

                </Stat>
                <Stat>
                    <StatLabel>Win Ratio</StatLabel>
                    <StatNumber>{btres['order_stats']?.['win']} - {btres['order_stats']?.['loss']}</StatNumber>

                </Stat>
                <Stat>
                    <StatLabel>Buy & Hold</StatLabel>
                    <StatNumber>{btres['market_return']}</StatNumber>

                </Stat>
            </StatGroup>
            <Input placeholder='code' ref={inputRef} />
            <Button colorScheme='blue' onClick={onclick}>backtest</Button>

        </>
    )

}
