# Experiments

## Question

Does a world model that models the other agent separately plan better than one network that treats it as part of the environment, and how does the answer depend on whether the other agent ignores us, reacts to us, or switches behaviour without warning?

## Closest work

| Paper | What it reports | Read |
|---|---|---|
| [Dreaming of Others, 2605.31361](https://arxiv.org/abs/2605.31361) | A world model that imagines other agents. | Title and snippet only. |
| [MA-WAM, 2609.31281](https://arxiv.org/abs/2609.31281) | A multi-agent world-action model. | Title and snippet only. |
| [Multiplayer RAE, 2607.05352](https://arxiv.org/abs/2607.05352) | A world model for several players. | Title and snippet only. |

## Why the papers don't answer it

Multi-agent world models change the architecture, the data and the training together, so the effect of modelling the other agent separately can't be read from them. No search found a study that holds inputs and data fixed and varies only that split, across partners that do and don't react.

**The difference, in one sentence:** a joint and a factored model with the same inputs and data, planning against four partners whose rules are known, with the true simulator and a partner-blind simulator as ceiling and floor.

## Boring explanations to rule out

1. The factored model wins because it has more weights. At k = 3 and the `5m` width, factored has 172,918 weights and joint 146,294 (18% fewer); every learned row stores `weights`, and `--match-size` widens joint to match (question 5 in QUESTIONS.md). A wider joint model is one `--h` away and should be run if C1 passes.
2. It wins because it sees more history. Both use the same k in every claim.
3. It wins only on easy episodes. Every comparison is paired on the same episodes and sampling seed.
4. The partner never matters. K1 requires the partner-blind floor to lose at least 0.3 on reacting partners and to tie on the still one.

## Kill test: written before any comparison ran

Nothing below has run, and the time of a `5m` row is not measured.

```bash
python3 check.py --slow
python3 beat.py --selftest
python3 jobs.py 5m | sh
python3 kill.py --tier 5m
```

| Test | Failure number | What failure means |
|---|---|---|
| K0: the true simulator plans | success under 0.8 on any partner | The planner is the limit, so models can't be compared. |
| K1: the partner matters only when it moves | blind within 0.3 of the oracle on chase or switch, or more than 0.05 apart on still | The task doesn't test modelling the partner. |
| K2: learning works | joint or factored more than 0.1 below the oracle on still | The learned models are too weak to compare. |
| K3: room to gain | joint at k = 3 within 0.1 of the oracle on both reacting partners | Nothing for a better model to win. |

Failing K0 or K3 at `5m` and again at `1h` kills the question for these partners.

## Claims and predictions: written before any comparison ran

The rules are in `beat.py`. Each prediction is what we expect, so a result against it is still a result.

| Claim | Rule | Prediction |
|---|---|---|
| C1: reacting partners | factored minus joint at k = 3 on chase and on switch, paired interval | Factored wins on both, by under 0.15. |
| C2: a partner that ignores us | the same on patrol | Ties. |
| C3: hidden mode | factored at k = 3 minus k = 1 on switch | History helps. |

## Debug log

- 2026-11-06: built. `check.py` passes in about 1.5 s, `--slow` in about 3 s. Only 7-second rows have run.
- 2026-11-07: learned models switched from MLPs on positions to LeWM on 16 x 16 pictures. `check.py` passes in about 7 s, `--slow` in about 50 s.
- The position probe trained alongside the encoder read our position to within 0.09 on real frames; a probe fitted afterwards on the frozen encoder read it to within 0.023. The probe is now fitted afterwards.
- Model weights were created before the seed was set, so two identical runs differed. `make()` now seeds.
- At the `7s` size, 0 of 6 episodes reached the goal for both learned models on chase and switch. On still, joint 0.33, factored 0.00, oracle 1.00. After 300 training steps the predictor misses about 65% of the latent's one-step change, and our dot's position after 8 imagined steps is off by 0.16. K2 will fail until training gets better.
- The first table placed start, goal and partner at random. Planning that ignored the partner reached 1.00 on chase, because the partner was rarely in the way. Start and goal are now on opposite sides with the partner in between. On 60 tune episodes at the `1m` planner size, the oracle then reached 1.00, 1.00, 0.98 and 0.97 on still, patrol, chase and switch, and the blind floor 1.00, 0.70, 0.20 and 0.45.
- A `block` partner that walks between us and the goal was dropped: see the README warning.
- 2026-11-07: added the 16 questions in QUESTIONS.md, each one flag on `run.py`, plus `compare.py`. Rows now store `variant`; beat.py and kill.py read base rows only. Observations grew from 4 to 6 numbers (the optional third dot). The check "a factored model learns the patrol partner" was replaced: at 300 steps its error was 0.097 on one data seed and 0.137 on another, against 0.130 for "the partner stays still", so it passed or failed by luck. The new check asks the trained encoder to place both dots to within 0.08 (0.034 and 0.020).
