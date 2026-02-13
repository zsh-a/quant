import numpy as np
from typing import List, Optional
from loguru import logger
from src.alpha_mining.node import AlphaNode
from src.alpha_mining.llm_agent import LLMAgent
from src.alpha_mining.evaluator import AlphaEvaluator

class AlphaMiningMCTS:
    def __init__(self,
                 evaluator: AlphaEvaluator,
                 llm_agent: LLMAgent,
                 c_puct: float = 1.0,
                 max_iterations: int = 10,
                 zoo_threshold: float = 0.05):  # Increased from 0.02
        self.evaluator = evaluator
        self.llm = llm_agent
        self.c_puct = c_puct
        self.max_iterations = max_iterations
        self.zoo_threshold = zoo_threshold
        self.root: Optional[AlphaNode] = None
        self.alpha_zoo: List[AlphaNode] = []
        self.factor_values_cache = {}  # Cache for correlation-based deduplication

    def run(self, initial_formula: str, iterations: int = None):
        iters = iterations or self.max_iterations

        # Evaluate on training set only
        metrics = self.evaluator.evaluate(initial_formula, mode='train')
        score = self._calculate_score(metrics)

        self.root = AlphaNode(initial_formula, c_puct=self.c_puct)
        self.root.metrics = metrics
        self.root.update(score)

        for i in range(iters):
            logger.info(f"MCTS Iteration {i+1}/{iters}")
            logger.info(f"  [Step 1/4] Starting selection phase...")
            leaf = self._select(self.root)
            logger.info(f"  [Step 1/4] Selection complete. Selected leaf: {leaf.formula[:50]}...")
            
            logger.info(f"  [Step 2/4] Starting expansion phase...")
            child = self._expand(leaf)
            logger.info(f"  [Step 2/4] Expansion complete.")

            if child:
                logger.info(f"  [Step 3/4] Starting backpropagation...")
                self._backpropagate(child, child.value)
                logger.info(f"  [Step 3/4] Backpropagation complete.")
                
                # Increased threshold and added validation check
                train_ic = child.metrics.get('rank_ic', 0)
                if abs(train_ic) > self.zoo_threshold:
                    logger.info(f"  [Step 4/4] Train IC {train_ic:.4f} exceeds threshold, validating on validation set...")
                    # Validate on validation set before adding to zoo
                    val_metrics = self.evaluator.evaluate(child.formula, mode='val')
                    val_ic = val_metrics.get('rank_ic', 0)
                    logger.info(f"  [Step 4/4] Validation complete. Val IC: {val_ic:.4f}")

                    # Check if performance holds on validation set (放宽验证集要求)
                    if abs(val_ic) > self.zoo_threshold * 0.4:  # 从0.7降低到0.4，允许更大衰减
                        child.metrics['val_rank_ic'] = val_ic
                        logger.info(f"  [Step 4/4] Adding to zoo (deduplication check)...")
                        self._add_to_zoo(child)
                        logger.info(f"Added to zoo: Train IC={train_ic:.4f}, Val IC={val_ic:.4f}")
                    else:
                        logger.warning(f"Overfitted: Train IC={train_ic:.4f}, Val IC={val_ic:.4f} (below threshold {self.zoo_threshold * 0.4:.4f})")
                else:
                    logger.info(f"  [Step 4/4] Train IC {train_ic:.4f} below threshold, skipping zoo.")
            else:
                logger.info(f"  [Step 3/4] No valid child created, skipping backpropagation.")
            
            logger.info(f"MCTS Iteration {i+1}/{iters} completed. Zoo size: {len(self.alpha_zoo)}")

    def _select(self, node: AlphaNode) -> AlphaNode:
        current = node
        while current.children:
            # Only select children that aren't "failed" (score > -0.5)
            valid_children = [c for c in current.children if c.value > -0.5]
            if not valid_children:
                break
            current = max(valid_children, key=lambda c: c.get_uct_score(self.c_puct))
        return current

    def _expand(self, node: AlphaNode) -> Optional[AlphaNode]:
        # 1. Decide dimension
        logger.info(f"    Analyzing node metrics: IR={node.metrics.get('ic_ir', 0):.4f}, RankIC={node.metrics.get('rank_ic', 0):.4f}")
        ir = node.metrics.get('ic_ir', 0)
        rank_ic = node.metrics.get('rank_ic', 0)
        
        # 增加探索多样性，避免总是选择Stability
        import random
        if random.random() < 0.3:  # 30%概率随机探索
            dimension = random.choice([
                "Stability (use smoothing or longer windows)",
                "Effectiveness (use volume-price interaction or non-linear operators)",
                "Novelty (explore new alpha space with different operators)"
            ])
            logger.info(f"    Random exploration triggered!")
        elif ir < 0.4:
            dimension = "Stability (IR is low, use smoothing or longer windows)"
        elif abs(rank_ic) < 0.05:
            dimension = "Effectiveness (RankIC is low, use volume-price interaction or non-linear operators)"
        else:
            dimension = "Novelty (High performance but needs variation to explore new alpha space)"
        
        logger.info(f"    Dimension selected: {dimension}")
        logger.info(f"    Calling LLM for refinement suggestion...")
        suggestion = self.llm.get_refinement_suggestion(node.formula, dimension, node.metrics)
        logger.info(f"    LLM suggestion received: {suggestion[:100]}...")
        
        # 2. Expansion loop with Self-Correction
        max_retries = 3
        error_msg = None
        
        for attempt in range(max_retries):
            logger.info(f"    Refinement attempt {attempt+1}/{max_retries}...")
            logger.info(f"    Calling LLM to refine formula...")
            new_formula = self.llm.refine_alpha(node.formula, suggestion, error_msg)
            logger.info(f"    LLM returned formula: {new_formula[:80] if new_formula else 'None'}...")
            
            if not new_formula or new_formula == node.formula:
                logger.info(f"    Formula unchanged or empty, skipping evaluation.")
                continue

            # Evaluate on training set
            logger.info(f"    Evaluating formula on training set...")
            metrics = self.evaluator.evaluate(new_formula, mode='train')
            logger.info(f"    Evaluation complete. RankIC: {metrics.get('rank_ic', 0):.4f}, IR: {metrics.get('ic_ir', 0):.4f}")

            if 'error' in metrics:
                error_msg = metrics['error']
                logger.warning(f"Attempt {attempt+1} failed: {error_msg}. Retrying...")
                continue

            # Success!
            child = AlphaNode(new_formula, parent=node, c_puct=self.c_puct)
            child.metrics = metrics
            child.value = self._calculate_score(metrics)
            node.add_child(child)
            logger.info(f"Expanded: {new_formula[:60]}... Score: {child.value:.4f}")
            return child
            
        # If all retries failed, add a "Dead End" child to prevent re-selecting this path
        logger.warning(f"    All {max_retries} expansion attempts failed. Adding dead end node.")
        dead_child = AlphaNode(f"FAILED_{node.formula[:10]}", parent=node)
        dead_child.value = -1.0 # Penalty
        node.add_child(dead_child)
        return None

    def _backpropagate(self, node: AlphaNode, reward: float):
        current = node
        while current:
            current.update(reward)
            current = current.parent

    def _calculate_score(self, metrics: dict) -> float:
        """
        Calculate score preserving sign information
        Formula: rank_ic + 0.1 * ic_ir - 0.05 * ic_decay
        """
        if 'error' in metrics:
            return -1.0

        rank_ic = metrics.get('rank_ic', 0)
        ic_ir = metrics.get('ic_ir', 0)
        ic_decay = metrics.get('ic_decay', 0)

        # Keep sign of rank_ic (don't use abs), penalize IC decay
        score = rank_ic + 0.1 * ic_ir - 0.05 * max(ic_decay, 0)

        return score

    def _add_to_zoo(self, node: AlphaNode):
        """Add node to zoo with correlation-based deduplication"""
        logger.info(f"    Checking for duplicates in zoo (current size: {len(self.alpha_zoo)})...")
        # Quick string-based check
        if any(node.formula[:40] == existing.formula[:40] for existing in self.alpha_zoo):
            logger.info(f"    Duplicate found (string match), skipping")
            return

        # Correlation-based deduplication
        try:
            logger.info(f"    Evaluating factor values for correlation check...")
            # Evaluate factor values on full dataset
            factor_values = eval(node.formula, {"__builtins__": {}}, self.evaluator.context)
            factor_flat = factor_values.values.flatten()
            factor_flat = factor_flat[~np.isnan(factor_flat)]
            logger.info(f"    Factor values computed. Valid values: {len(factor_flat)}")

            for idx, existing in enumerate(self.alpha_zoo):
                logger.info(f"    Comparing with zoo entry {idx+1}/{len(self.alpha_zoo)}...")
                if existing.formula in self.factor_values_cache:
                    existing_flat = self.factor_values_cache[existing.formula]
                else:
                    logger.info(f"    Computing factor values for existing formula...")
                    existing_values = eval(existing.formula, {"__builtins__": {}}, self.evaluator.context)
                    existing_flat = existing_values.values.flatten()
                    existing_flat = existing_flat[~np.isnan(existing_flat)]
                    self.factor_values_cache[existing.formula] = existing_flat

                # Check correlation
                min_len = min(len(factor_flat), len(existing_flat))
                if min_len > 100:
                    corr = np.corrcoef(factor_flat[:min_len], existing_flat[:min_len])[0, 1]
                    if abs(corr) > 0.95:
                        logger.info(f"    Factor too similar (corr={corr:.3f}), skipping")
                        return

            # Add to cache and zoo
            logger.info(f"    No duplicates found, adding to zoo...")
            self.factor_values_cache[node.formula] = factor_flat
            self.alpha_zoo.append(node)
            logger.info(f"    Successfully added to zoo. New size: {len(self.alpha_zoo)}")

        except Exception as e:
            logger.warning(f"Deduplication check failed: {e}, adding anyway")
            self.alpha_zoo.append(node)
