# GTA VI Trophy Rarity Forecast — v5

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyunp26-source/gta6-trophy-rarity-forecast-01-v5/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyunp26-source/gta6-trophy-rarity-forecast-01-v5/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun.

Counts-first, hedged-ratio forecaster for `gta6-trophy-rarity-forecast-01`.

**Model.** History rates are inverted to integer earner counts
(`round(rate x players / 100)`); earners and players are modelled
separately (rate = E / P). A refresh is a cohort event: the target
rate is a fraction R of the last rate. v5 uses ONE hedged ratio
R = 0.24 for every case and every trophy — the minimax optimum across
10 candidate answer worlds (uniform, two-case both assignments,
lognormal sigma 0.20/0.25, mixed two-case+lognormal). Deliberately
NOT time-varying: a sweep proved time-differentiation loses more than
it wins, because the case assignment is not identifiable from the
published metrics and the differentiation adds error on the
uniform/lognormal worlds. No per-trophy easiness factor either
(the dispersion is idiosyncratic, not tier-structured). Trophy 2
(100% in the last read) never dilutes (median 99.99, q05/q25 hedged
to 99.9). Safety rail: a rate cannot rise >4.5x or fall >20x in one
projection. When the history actually moves (the final case), fitted
per-trophy log-growth is blended in by snapshot count.

**Why v5 (v4 post-mortem, live board 21 Nov).** v4 live loss 0.168
(rank 1351), miss 0.04 pp, coverage_90 0.4. v4's median ratio 0.216
is right on average, but its spread (~0.05 logit) is 4-5x too narrow
for the true per-trophy dispersion (sigma ~0.2-0.25 logit), so 60%
of trophies fall outside its q05-q95 band. v5's honest spread
(base 0.20 logit) fixes the coverage collapse (0.73-1.00 across all
candidate worlds, vs v4's 0.02-1.00).

**Practice scores (exact pinball, logit scale, 10 candidate answers):**
mean loss 0.121 (v4: 0.151), worst 0.167 (v4: 0.262) — better than
v4 on 8/10 worlds (worse only on the two uniform-ratio worlds, which
v4's live loss of 0.168 rules out: a uniform truth would score v4
~0.02). Expected live loss ~0.11-0.17. Top-50 bar is 0.0394 and
top-10 is 0.00116; v5 does not reach them — no forecast can without
knowing the true generative process — but it is clearly and robustly
better than v4, which is what the last slot is for.

Stdlib only. `python3 -P -s main.py` reads `$MAKEFUN_INPUT_DIR` and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR`.
