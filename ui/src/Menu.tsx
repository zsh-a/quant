import { useEffect, useState } from 'react'
import { Button, ButtonGroup } from '@chakra-ui/react'
import { Input } from '@chakra-ui/react'

import {
    Stat,
    StatGroup,
} from '@chakra-ui/react'
import { Divider, Stack, Text } from '@chakra-ui/react'

export default ({ code, name, btres, onclick, inputRef }) => {
    return (
        <>
            <StatGroup>
                <Stat.Root>
                    <Stat.Label>code</Stat.Label>
                    <Stat.ValueText>{code}</Stat.ValueText>
                </Stat.Root>
                <Stat.Root>
                    <Stat.Label>name</Stat.Label>
                    <Stat.ValueText>{name}</Stat.ValueText>
                </Stat.Root>
            </StatGroup>


            <StatGroup>

                <Stat.Root>
                    <Stat.Label>Strategy Return</Stat.Label>
                    <Stat.ValueText>{btres['strategy_return']}%</Stat.ValueText>
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
                    <Stat.Label>Win Ratio</Stat.Label>
                    <Stat.ValueText>{btres['order_stats']?.['win']} - {btres['order_stats']?.['loss']}</Stat.ValueText>
                </Stat.Root>
                <Stat.Root>
                    <Stat.Label>Buy & Hold</Stat.Label>
                    <Stat.ValueText>{btres['market_return']}%</Stat.ValueText>
                </Stat.Root>
            </StatGroup>
            <Input placeholder='code' ref={inputRef} />
            <Button colorScheme='blue' onClick={onclick}>backtest</Button>

        </>
    )

}
