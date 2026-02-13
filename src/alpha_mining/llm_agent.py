from typing import List, Dict, Optional
import re
import json
import os
from openai import OpenAI
from loguru import logger

# Strict Prompting to minimize hallucinations

PROMPT_PORTRAIT_GENERATION = """
Task Description:
Design a high-performance alpha factor for CSI 1000 index.

MANDATORY Operator List (ONLY USE THESE):
- Unary: Abs(x), Log(x), Sign(x), Sqrt(x)
- Time-series: Ts_Mean(x, d), Ts_Std(x, d), Ts_Max(x, d), Ts_Min(x, d), Ts_Rank(x, d), Ts_Zscore(x, d), Ts_EMA(x, d), Ts_DecayLinear(x, d), Ts_Winsorize(x, d, n_std)
- Diff: Delta(x, d), Delay(x, d), Ts_Returns(x, d)
- Binary: Correlation(x, y, d), Covariance(x, y, d), Max(x, y), Min(x, y)
- Logic: Where(condition, x, y)  <-- Use this instead of If
- Cross-sectional: CSRank(x), Scale(x), Power(x, p), Sigmoid(x)
- Available fields: open, high, low, close, volume, amount, vwap

CRITICAL RULES:
1. Syntax MUST be valid Python with BALANCED parentheses.
2. Use '&' for AND, '|' for OR inside Where condition (e.g., Where((close > open) & (volume > 100), 1, 0)).
3. DO NOT use '&&' or 'AND' or 'If' or any undefined functions.
4. Ensure all Ts_* operators have a window 'd' parameter.
5. Always wrap the final result in CSRank().
6. Keep formula complexity reasonable (max 3-4 nested levels).
7. CSRank() takes EXACTLY 1 argument, not 2.

Output JSON: {"name": "...", "description": "...", "formula": "..."}
"""

PROMPT_REFINE_ALPHA = """
Improve this alpha: {formula}
Feedback: {suggestion}
{error_feedback}

MANDATORY SIGNATURES (ONLY USE THESE):
- Ts_Mean(x, d), Ts_Std(x, d), Ts_Rank(x, d), Ts_Zscore(x, d), Ts_EMA(x, d), Ts_DecayLinear(x, d), Ts_Winsorize(x, d, n_std)
- Delta(x, d), Delay(x, d), Correlation(x, y, d), Covariance(x, y, d)
- CSRank(x) - takes EXACTLY 1 argument
- Where(condition, x, y), Max(x, y), Min(x, y)
- Power(x, p), Sigmoid(x), Abs(x), Log(x), Sign(x), Sqrt(x)

CRITICAL SYNTAX RULES:
1. BALANCE all parentheses - count opening '(' and closing ')' must match.
2. Use '&' instead of 'AND' or '&&'.
3. Use '|' instead of 'OR' or '||'.
4. No 'If' statements, use Where().
5. CSRank(x) takes 1 argument, not 2 or more.
6. Keep complexity under control - MAX 3 nested levels, window size 5-60 days.
7. Only use functions listed above - no undefined functions.
8. AVOID excessive smoothing - don't combine Ts_DecayLinear + Ts_Mean together.
9. Prefer simple formulas over complex ones for better generalization.

Provide ONLY the improved formula string (no explanation, no markdown).
"""

class LLMAgent:
    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        api_key: str = None
    ):
        self.model_name = model_name or os.getenv("ALPHA_MINING_MODEL", "deepseek-ai/DeepSeek-V3.2")
        base_url = base_url or os.getenv("ALPHA_MINING_BASE_URL", "https://api-inference.modelscope.cn/v1")
        api_key = api_key or os.getenv("ALPHA_MINING_API_KEY")

        if not api_key:
            raise ValueError("API key not found. Please set ALPHA_MINING_API_KEY in .env file")

        # Configure client with explicit timeouts
        import httpx
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(
                connect=10.0,   # 连接超时10秒
                read=60.0,      # 读取超时60秒
                write=10.0,     # 写入超时10秒
                pool=5.0        # 连接池超时5秒
            ),
            max_retries=2       # 最多重试2次
        )
        
    def generate_alpha(self, forbidden_structures: List[str] = []) -> Dict[str, str]:
        response = self._call_llm(PROMPT_PORTRAIT_GENERATION)
        return self._parse_json_response(response)

    def refine_alpha(self, formula: str, suggestion: str, error_msg: str = None) -> str:
        error_feedback = f"\nERROR IN PREVIOUS FORMULA: {error_msg}\nPlease fix the syntax or undefined name." if error_msg else ""
        prompt = PROMPT_REFINE_ALPHA.format(
            formula=formula,
            suggestion=suggestion,
            error_feedback=error_feedback
        )
        response = self._call_llm(prompt)
        # Robust cleaning
        res = response.strip().split('\n')[0].replace('`', '').replace('formula=', '')
        return res

    def get_refinement_suggestion(self, formula: str, dimension: str, metrics: dict) -> str:
        prompt = f"Alpha: {formula}\nMetrics: RankIC={metrics.get('rank_ic',0):.4f}, IR={metrics.get('ic_ir',0):.4f}\nImprove {dimension}. Give 1 short logic tip."
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        try:
            logger.info(f"      [LLM] Sending request to {self.model_name}...")
            logger.debug(f"      [LLM] Prompt length: {len(prompt)} chars")

            import time
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,  # Minimum randomness
                stream=False,     # 明确禁用流式响应
                timeout=60        # API调用级别超时
            )

            elapsed = time.time() - start_time
            result = response.choices[0].message.content

            logger.info(f"      [LLM] Response received in {elapsed:.2f}s, length: {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"      [LLM] Call failed: {type(e).__name__}: {e}")
            return ""

    def _parse_json_response(self, response: str) -> Dict[str, str]:
        try:
            clean_res = re.sub(r'```json\s*|\s*```', '', response).strip()
            start = clean_res.find('{')
            end = clean_res.rfind('}') + 1
            return json.loads(clean_res[start:end])
        except:
            return {"name": "error", "description": "error", "formula": "CSRank(close)"}