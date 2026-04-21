# Trader's Equation — applied notes

The taxonomy section `trader_equation` defines the formula and the canonical
probability buckets. This section adds the *reasoning rules* the LLM must
apply when picking a bucket for a live setup.

## Formula recap

```
expectancy = P(win) * reward − P(loss) * risk
P(loss) = 1 − P(win)
```

Working in units of `R` (R = risk = entry − stop in absolute price), reward
is expressed as `reward_R = reward / R`. The acceptance rule used everywhere
in this codebase is:

```
accept iff (P(win) * reward_R − (1 − P(win))) >= 1.0
```

The "≥ 1R" threshold accounts for slippage, commission, and the empirical
fact that probability estimates are biased optimistically.

## Choosing a probability bucket

Pick the **lowest** bucket consistent with the regime/pattern combination.
When in doubt, drop one bucket — over-estimating P(win) is the single
biggest source of negative expectancy.

| Bucket | When to use |
|--------|-------------|
| 0.40 | Counter-trend in a strong trend; speculative scalps; H4/L4 continuation |
| 0.55 | Default for Hn/Ln in a normal trend, breakout from a trading range |
| 0.65 | Strong setups: micro-channel first pullback, BP, MTR, double top/bottom at HTF level |
| 0.75 | High-conviction confluence: H2 in strong bull + 20EMA bounce + measured-move target intact |
| 0.85 | Rare 'A+' setups — only when **multiple** independent edges align: HTF level + LTF MTR + climactic prior bar + reversal-bar quality |

## Adjustments

- **Regime mismatch** (e.g. taking an L2 short in a strong bull trend):
  drop **two** buckets, not one. Brooks calls these "always-wrong" trades.
- **Late in the trend** (leg_age > 20 bars or 4th+ pullback): drop one bucket.
- **Climactic bar within last 3 bars**: drop one bucket for continuation,
  raise one bucket for fade-the-climax setups.
- **Tight stop available** (`R < 0.5 ATR`): probability unaffected, but the
  reward needed for ≥1R expectancy is halved — these are the most attractive
  entries.

## Worked decision examples

The taxonomy `trader_equation.worked_examples` block shows two canonical
trades. Use them as anchors when grading an unfamiliar setup:

- `h2_in_strong_bull`: P=0.65, R=8t, reward=16t → expectancy ≈ 0.69R
  (close to threshold; take only with confluence).
- `counter_trend_l2_in_strong_bull`: P=0.30, R=8t, reward=8t → expectancy
  ≈ −0.40R (skip).

When in doubt, write the equation out explicitly in the trade-log comment
and let the LLM/operator audit the probability assignment.
