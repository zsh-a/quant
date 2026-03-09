import pandas as pd
from src.alpha_mining.mcts import AlphaMiningMCTS
from src.alpha_mining.llm_agent import LLMAgent
from src.alpha_mining.evaluator import AlphaEvaluator
from src.alpha_mining.persistence import AlphaZooPersistence
from src.market_data.db import DB
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

class AlphaMiningPipeline:
    def __init__(self, start_date: str, end_date: str, index_code: str = "000852"):
        self.db = DB()
        self.start_date = start_date
        self.end_date = end_date
        self.index_code = index_code
        self.persistence = AlphaZooPersistence()
        
    def load_data(self) -> pd.DataFrame:
        """Load CSI 1000 component data with extended history"""
        logger.info(f"Fetching components for index {self.index_code}...")
        stocks = self.db.get_index_stocks(self.index_code, self.start_date)
        if not stocks:
            stocks = self.db.get_index_stocks(f"sh.{self.index_code}", self.start_date)

        if not stocks:
            raise ValueError(f"Could not find any component stocks for index {self.index_code}")

        logger.info(f"Loading prices for {len(stocks)} stocks...")
        # Increased from 500 to 1500 (approx 6 years) to reduce overfitting
        df = self.db.get_price(stocks, self.end_date,
                              fields=['open', 'high', 'low', 'close', 'volume', 'amount'],
                              count=1500, start_date=self.start_date)

        if df.empty:
            raise ValueError("Data fetching returned empty DataFrame")

        df = df.reset_index()
        df.rename(columns={'code': 'symbol'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded {len(df['date'].unique())} trading days")
        return df

    def run(self, iterations: int = 5, zoo_threshold: float = 0.02):  # 进一步降低到0.02
        # 1. Prepare Data
        data = self.load_data()

        # 2. Setup Components
        evaluator = AlphaEvaluator(data, train_ratio=0.6, val_ratio=0.2)
        llm = LLMAgent()
        mcts = AlphaMiningMCTS(evaluator, llm, zoo_threshold=zoo_threshold)

        # 3. Seed Alphas - Try multiple starting points
        seed_alphas = [
            "CSRank(-Ts_Returns(close, 5))",  # Reversal
            "CSRank(Correlation(close, volume, 10))",  # Volume-price
            "CSRank(Delta(vwap - close, 3))",  # VWAP deviation
            "CSRank(Ts_Rank(volume, 20))",  # Volume momentum
        ]

        # Select best seed
        logger.info("Testing seed alphas to find best starting point...")
        best_seed = seed_alphas[0]
        best_ic = -999

        for seed in seed_alphas:
            metrics = evaluator.evaluate(seed, mode='train')
            ic = abs(metrics.get('rank_ic', 0))
            logger.info(f"  Seed: {seed[:50]}... → IC: {metrics.get('rank_ic', 0):.4f}")
            if ic > best_ic:
                best_ic = ic
                best_seed = seed

        logger.info(f"Selected best seed with IC={best_ic:.4f}: {best_seed}")
        seed_alpha = best_seed

        # 4. Run Search
        logger.info(f"Starting optimized MCTS search on index {self.index_code}...")
        logger.info(f"Zoo threshold: {zoo_threshold}, Iterations: {iterations}")
        mcts.run(seed_alpha, iterations=iterations)

        # 5. Test set evaluation for final alphas
        logger.info("\n" + "="*60)
        logger.info("TESTING TOP ALPHAS ON HOLD-OUT TEST SET")
        logger.info("="*60)

        for node in mcts.alpha_zoo:
            test_metrics = evaluator.evaluate(node.formula, mode='test')
            node.metrics['test_rank_ic'] = test_metrics.get('rank_ic', 0)
            node.metrics['test_ic_ir'] = test_metrics.get('ic_ir', 0)

        # 6. Persist Results
        self.persistence.save_zoo(mcts.alpha_zoo, task_name=f"CSI1000_{self.start_date}")
        self.persistence.export_to_csv()

        # 7. Report with train/val/test breakdown
        top_alphas = sorted(mcts.alpha_zoo, key=lambda x: abs(x.metrics.get('test_rank_ic', 0)), reverse=True)
        print("\n" + "="*70)
        print(f"MINING COMPLETE - Top Discovered Factors (sorted by test IC)")
        print("="*70)
        print(f"{'#':<3} {'Train IC':<10} {'Val IC':<10} {'Test IC':<10} {'IR':<8} {'Decay':<8}")
        print("-"*70)
        for i, node in enumerate(top_alphas[:10]):
            m = node.metrics
            print(f"{i+1:<3} {m.get('rank_ic',0):>9.4f} {m.get('val_rank_ic',0):>9.4f} "
                  f"{m.get('test_rank_ic',0):>9.4f} {m.get('ic_ir',0):>7.2f} {m.get('ic_decay',0):>7.4f}")
            if i < 3:  # Show formula for top 3
                print(f"    Formula: {node.formula[:80]}")
        print(f"\nTotal factors discovered: {len(mcts.alpha_zoo)}")
        print(f"Alphas saved to: {self.persistence.storage_dir}")
        print("="*70)

        return mcts.alpha_zoo

    def load_best_alphas(self, limit: int = 10):
        """Helper to load previously mined alphas"""
        return self.persistence.load_all()[:limit]

if __name__ == "__main__":
    pipeline = AlphaMiningPipeline(
        start_date='2019-01-01',
        end_date='2025-01-01'
    )
    pipeline.run(iterations=10)
