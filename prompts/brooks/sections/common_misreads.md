# Common Misreads — checklist before entering

The taxonomy ships a `common_misreads` block per pattern. This file
collects the cross-cutting failure modes the LLM must screen for before
emitting a signal.

## 1. Confusing a trading range with a trend

A leg inside a broad TR looks like a trend in isolation. The fix:

- Compute `range_height_atr = (range_high − range_low) / atr14`. If the
  current swing is fully inside a 2+ ATR range with `always_in == neutral`,
  the regime is `broad_trading_range`, **not** a trend.
- In a TR, every Hn/Ln upgrade should be downgraded one bucket and the
  default action is to **fade the extremes**, not chase pullbacks.

## 2. Mis-counting Hn / Ln

`Hn` and `Ln` are counted from the start of the **current** leg, reset by:

- a flip of `always_in`,
- a confirmed swing in the opposite direction beyond the prior leg's extreme,
- or a clear MTR.

If the prior swing is unconfirmed, **wait**. Counting H3 when it's actually
a fresh H1 in a new leg leads to "late entry" expectancy errors.

## 3. Treating doji bars as signal bars

Brooks signal bars must be **trend bars** in the entry direction (`is_trend
== true`, `close_position` matches direction). A doji signal bar:

- doubles the probability of being run-stop within the next 2 bars,
- should drop the bucket by one,
- and is invalid in tight ranges (the next bar is as likely to be doji too).

## 4. Counter-trend H4/L4 entries treated as continuation

H4/L4 are **reversal-biased** in Brooks practice. If your detector says
"H4 long in a strong bull", the more likely interpretation is "L1 short
forming". Default to skipping continuation, and consider the opposite-side
setup once a reversal bar prints.

## 5. Misreading micro-channels

Inside a micro-channel:

- **Do not fade.** Counter-trend trades inside a fresh micro-channel are
  the highest-loss action category in Brooks' literature.
- The first pullback after the micro-channel breaks is the highest-quality
  entry; wait for it.

## 6. Ignoring `always_in` hysteresis

`always_in` does **not** flip on every breakout bar. The hysteresis rule
in `taxonomy.yaml -> always_in_rules` requires a confirmed close beyond
the N-bar extreme **plus** a trend bar in the breakout direction. LLM
output that toggles `always_in` every other bar is wrong.

## 7. Climactic bar interpretation

A climactic bar is **not** itself a continuation entry — it is exhaustion.
Continuation after a climax requires either:

- a `breakout_pullback` whose pullback does not close back inside the
  pre-climax range, or
- a `final_flag` setup (which then *reverses*, not continues).

## 8. Measured-move target abuse

`measured_move` is a **target-side** concept; it never appears as an
`entry`. If a candidate signal lists MM in the entry field, the LLM has
hallucinated. Use MM only to size the take-profit.

## 9. Ignoring regime when grading

The same Hn pattern has very different probabilities in different regimes:

- H2 in `strong_bull_trend`: bucket 0.65–0.75
- H2 in `tight_trading_range`: bucket 0.45 (effectively a fade-buy near low)
- H2 in `weak_bear_trend`: skip — it's an L1 short instead

Always emit the matched regime alongside the pattern so the grader can
audit the bucket choice.

## 10. Stop placement off by one tick

Stops are 1 tick **beyond** the signal-bar extreme, not at it. Off-by-one
stops are run intra-bar by spread/normal noise. Use:

- long entry: `signal_bar.high + 1 tick`, stop `signal_bar.low − 1 tick`.
- short entry: `signal_bar.low − 1 tick`, stop `signal_bar.high + 1 tick`.
