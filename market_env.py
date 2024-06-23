from typing import Any, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from data_source import DataSource
import matplotlib.pyplot as plt
from loguru import logger
from utils import log2percent


class Order:
    def __init__(self, order_id, symbol, order_type, quantity, price=None):
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type  # 'buy' or 'sell'
        self.quantity = quantity
        self.price = price  # Limit price for limit orders
        self.status = "open"  # 'open', 'filled', 'cancelled'
        self.filled_quantity = 0
        self.timestamp = None  # Time when the order was created


class OrderPolicy:
    def __init__(self, account) -> None:
        pass

    def buy_policy(self, order):
        pass

    def sell_policy(self, order):
        pass

    def step(self, obs):
        pass


class Account:
    def __init__(self, init_capital=10000) -> None:
        self.capital = init_capital
        self.trading_cost_bps = 1e-4
        self.available = 0
        self.min_action = 100

        self.actions = [0]

        self.costs = [0]
        self.positions = [0]
        self.capitals = [init_capital]
        self.tot_values = [init_capital]

    def step(self):
        self.actions.append(0)
        self.costs.append(self.costs[-1])
        self.positions.append(self.positions[-1])
        self.capitals.append(self.capitals[-1])
        self.tot_values.append(self.tot_values[-1])
        self.available = self.positions[-1]

    def result(self, risk_free_rate):
        strategy_return = (
            np.array(self.tot_values) - self.tot_values[0]
        ) / self.tot_values[0]
        print(risk_free_rate,np.mean(strategy_return))
        return {
            "strategy_return": round(strategy_return[-1] * 100, 2) ,
            "max_drawdown": round((min(self.tot_values) / self.tot_values[0] - 1) * 100, 2),
            "max_profit": round((max(self.tot_values) / self.tot_values[0] - 1) * 100, 2),
            "sharpe_ratio": (np.mean(strategy_return) - risk_free_rate)
            / np.std(strategy_return),
        }


class OrderManager:
    def __init__(self, account, order_policy):
        self.orders = []
        self.order_id_counter = 1

        self.order_plolicy = order_policy
        self.account = account

        self.buy_sell_points = []
        self.timestamp = None

        self.obs = None

    def step(self, obs):
        self.obs = obs
        self.order_plolicy.step(obs)

    def create_order(self, symbol, order_type, quantity, price=None):
        order = Order(self.order_id_counter, symbol, order_type, quantity, price)
        for od in self.orders:
            if od.status == "tracking":
                od.status = "cancelled"
                logger.info(
                    f"cancel order | datetime : {self.get_current_timestamp()} | order_id : {od.order_id}  | order_type : {od.order_type}"
                )

        self.orders.append(order)
        logger.info(
            f"create order | datetime : {self.get_current_timestamp()} | order_id : {order.order_id}  | order_type : {order_type}"
        )
        self.order_id_counter += 1
        return order

    def match_orders(self, market_data):
        """
        Match open orders with the latest market data.
        """
        self.timestamp = market_data.name

        for order in self.orders:
            if order.status == "open" or order.status == "tracking":
                if order.order_type == "buy":
                    ok, exec_price = self.order_plolicy.buy_policy(order)
                    if ok:
                        self.execute_order(order, exec_price)
                elif order.order_type == "sell" or order.order_type == "stop":
                    ok, exec_price = self.order_plolicy.sell_policy(order)
                    if ok:
                        self.execute_order(order, exec_price)

    def execute_order(self, order, execution_price):
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.execution_price = execution_price
        order.timestamp = self.get_current_timestamp()
        self.buy_sell_points.append((order.timestamp, order.order_type))
        logger.info(
            f"complete order | datetime : {self.get_current_timestamp()} | order_id : {order.order_id}  | order_type : {order.order_type} | price : {order.execution_price} | quantity : {order.quantity}"
        )
        self.order_plolicy.order_callback(order, self)
        self.update_account(order)

    def update_account(self, order):
        """
        Update account balance and positions based on the filled order.
        """
        # Implement account and position update logic
        amount = order.execution_price * order.quantity
        cost = amount * self.account.trading_cost_bps
        if order.order_type == "buy":
            self.account.capital = self.account.capital - amount - cost
        else:
            self.account.capital = self.account.capital + amount - cost

        self.account.capitals[-1] = self.account.capital
        self.account.actions[-1] = (
            order.quantity if order.order_type == "buy" else -order.quantity
        )

        self.account.positions[-1] += self.account.actions[-1]
        self.account.tot_values[-1] = (
            self.account.capital + self.account.positions[-1] * self.obs["close"]
        )
        self.account.available = 0

    def get_current_timestamp(self):
        """
        Return the current timestamp in the desired format.
        """
        return self.timestamp

    def get_order_history(self):
        return [order for order in self.orders if order.status == "filled"]


INF = 1e9


class MarketEnv(gym.Env):
    def __init__(
        self,
        num_step,
        code="510880",
        start_date="20100531",
        end_date="20110901",
        work_dir="",
        initial_capital=100000,
        max_stake=10000,
        account=Account(),
        order_policy=None,
    ) -> None:
        super().__init__()

        self.data_source = DataSource(
            code=code,
            trading_days=num_step,
            start_date=start_date,
            end_date=end_date,
            work_dir=work_dir,
        )

        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32)
        self.observation_space = self.observation_space = gym.spaces.Box(
            low=np.array(self.data_source.min_values),
            high=np.array(self.data_source.max_values),
        )

        self.capital = initial_capital
        self.max_stake = max_stake
        self.num_step = num_step
        self.tot_values = self.capital
        self.min_action = 100

        self.tot_values = np.zeros(self.num_step + 1)
        self.tot_values[0] = self.capital

        self.capitals = np.zeros(self.num_step + 1)
        self.capitals[0] = self.capital
        self.trading_cost_bps = 1e-4
        # state
        self.cur_step = 1
        self.actions = np.zeros(self.num_step + 1)
        self.positions = np.zeros(self.num_step + 1)
        self.navs = np.ones(self.num_step + 1)
        self.trades = np.zeros(self.num_step + 1)
        self.costs = np.zeros(self.num_step + 1)
        self.strategy_returns = np.zeros(self.num_step + 1)
        self.market_returns = np.zeros(self.num_step + 1)

        self.account = account

        self.order_manager = OrderManager(account, order_policy)

    def exec_order(self, obs):
        self.order_manager.match_orders(obs)

    def step(self, action: Any) -> Tuple[Any | SupportsFloat | bool | dict[str, Any]]:
        # assert self.action_space.contains(action)
        obs, done, ori_obs = self.data_source.step()
        self.order_manager.step(ori_obs)
        self.account.step()
        self.exec_order(ori_obs)

        self.market_returns[self.cur_step] = ori_obs["returns"]

        # print(self.account.tot_values,self.cur_step)
        reward = (
            self.account.tot_values[self.cur_step]
            - self.account.tot_values[self.cur_step - 1]
        )
        info = {
            "reward": reward,
            "costs": self.costs[self.cur_step],
            "ori_obs": ori_obs,
        }
        self.cur_step += 1
        return (obs.values, self.account.positions[-1]), reward, done, info

    def result(self):
        account_res = self.account.result(self.market_returns[self.cur_step - 1])

        return account_res | {
            "market_return": log2percent(np.exp(sum(self.market_returns)))
        }

    def plot(self):
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(15, 20))
        ax1.plot(self.account.positions[: self.cur_step], label="Position")
        ax1.set_ylabel("Position/stake")

        ax2.plot(self.account.tot_values[: self.cur_step], label="Value")
        ax2.set_ylabel("Value/CNY")

        ax3.plot(
            (
                np.array(self.account.tot_values[: self.cur_step])
                - self.account.tot_values[0]
            )
            / self.account.tot_values[0]
            * 100,
            label="Strategy Return",
        )
        ax3.set_ylabel("Strategy Returns/%")
        ax4.plot(self.account.capitals[: self.cur_step], label="Capital")
        ax4.set_ylabel("Capital/CNY")

        ax5.plot(
            log2percent(np.exp(np.cumsum(self.market_returns))), label="Market Return"
        )
        ax5.set_ylabel("Market Return/%")

        ax1.legend(loc="best")
        # ax1.set_title("Market vs Strategy Returns")``
        # ax1.set_ylabel("Returns/%")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)
        ax5.grid(True)

        self.data_source.plot(self.order_manager.buy_sell_points)
        plt.show()
        pass
        # plt.savefig('trade_result.png', dpi=300, bbox_inches='tight')

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Any, dict]:
        self.cur_step = 1
        self.actions.fill(0)
        self.navs.fill(1)
        self.strategy_returns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)
        self.data_source.reset()
        obs, done, ori_obs = self.data_source.step()
        info = {"ori_obs": ori_obs}
        return (obs.values, 0), info


if __name__ == "__main__":
    env = MarketEnv(220, start_date="20230401", end_date="20240401")
    env.reset()
    print(env.step(1))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(-0.1))

    env.plot()
