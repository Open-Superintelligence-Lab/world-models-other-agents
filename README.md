# Environment for researching whether a world model should treat another agent as an agent

> [!IMPORTANT]
> We are using this environment to write a paper for ICML 2027. It was built in [AI Research School](https://www.skool.com/become-ai-researcher-2669). The code is under the MIT licence, so you can use the code and ideas in your own work.
>
> **This is a setup, and it has no results yet.** The questions to answer, each one flag away, are in [QUESTIONS.md](QUESTIONS.md). Bug checks pass on known answers, and every partner and model has run end to end at the 7-second size. At that size the learned world models are undertrained: against the partner that never moves, `joint` reached the goal in 2 of 6 episodes and the true simulator in 6 of 6. No kill-test or claim row has run.

## Ask whether the other agent needs its own model

A **world model** guesses what happens next after an action, and a planner uses it to try plans in imagination. When another agent shares the space (a person, another robot, a car), the usual world model treats it as part of the environment: one network predicts everything that moves.

The other choice is to **model the other agent as an agent**: a separate part that predicts what it will do from what it sees, while our own body is predicted from our action. This is closer to how the world works, because the other agent can't see our next action before it moves. Papers that build multi-agent world models report gains, but they change many things at once.

| Paper | What it does | What it left open |
|---|---|---|
| [Dreaming of Others, 2605.31361](https://arxiv.org/abs/2605.31361) | A world model that imagines other agents' behaviour. | Title and search snippet only. When the separate agent model matters and when one network is enough. |
| [MA-WAM, 2609.31281](https://arxiv.org/abs/2609.31281) | A multi-agent world-action model. | Title and search snippet only. |
| [Multiplayer RAE, 2607.05352](https://arxiv.org/abs/2607.05352) | A world model for several players. | Title and search snippet only. |

> **The question: does a world model that models the other agent separately plan better than one network that treats it as part of the environment, and how does the answer depend on whether the other agent ignores us, reacts to us, or switches behaviour without warning?**

This came from polls in AI Research School and on RedNote and Douyin, where world models got the most votes and one member asked for embodied multi-agent world models.

---

## One table, one partner, four world models

| Part | What is here |
|---|---|
| **World** | `world.py`: our dot starts near one side of a table and must reach a goal near the opposite side. A partner dot starts in the middle. Touching it ends the episode as a failure. Our speed is 0.04 a step and the partner's 0.03. |
| **Partners** | `still` never moves (the control). `patrol` circles the centre and ignores us. `chase` walks at us. `switch` alternates between patrol and chase every 6 to 14 steps, and the switch time is hidden. |
| **Pictures** | `world.render`: each step is drawn as a 16 x 16 picture, our dot in red and the partner in blue. The learned models see only these pictures. |
| **Models** | `model.py`: both learned models are [LeWM](https://arxiv.org/abs/2603.19312) (LeWorldModel): a conv encoder from a picture to a 32-number latent, and a predictor from the last k latents and our action to the next latent, trained on prediction error plus 0.1 x SIGReg (copied from the authors' code). `joint` predicts the whole next latent with one network. `factored` splits the latent in two halves: one half is predicted from the last k latents and our action, the other from the last k latents only, so it can't see our current action. Nothing tells the model which half holds the partner. Both see the same pictures and data. `blind` is the true simulator with the partner frozen, the floor. `oracle` is the true simulator with the hidden mode, the ceiling. |
| **Probe** | The planner's cost needs both positions, so a small network reads the 4 positions from a latent. It is fitted after training on the frozen encoder, and both learned models get the same one. LeWM plans towards a goal picture instead; that can't work here, because a goal picture would also fix where the partner is. |
| **Planning** | `run.py`: the cross-entropy method over our next H actions, replanning every step. Cost: distance to the goal plus a penalty whenever the imagined dots come close to touching. |
| **Scores** | Per episode: `success` (goal reached before touching) and `touched`. For learned models also `other_err` and `me_err`, the distance between imagined and true positions after H steps of recorded actions, and `still_err`, the partner error of assuming it stays put. |
| **Questions** | [QUESTIONS.md](QUESTIONS.md): 15 questions that remove, add or change one part of the model or its data, each one flag on `run.py`. `compare.py` prints each variant next to the base model. |
| **Rules** | `kill.py` is the kill test and `beat.py` holds the three claims, both written before any comparison ran. `python3 beat.py --selftest` runs the rules on made-up rows. |

---

## Run the checks, then the kill test

```bash
pip install -r requirements.txt
python3 check.py
```

`check.py` takes about 7 seconds on one CPU thread, and `--slow` adds a 7-second-tier row for every partner, model and question flag through `run.py` (about 2 minutes in all). Its known answers: splits are different episodes; the still partner never moves; the patrol partner stays on its circle; the chaser closes in; the switching partner flips exactly on its period; the simulator's rollout is the true step sequence; with a still partner, planning that ignores it gives exactly the same outcomes as the truth; with a chaser, ignoring it drops success from 0.96 to 0.21; the factored model's second latent half has zero gradient from our current action and the joint model's doesn't; planning from pictures equals planning from pre-encoded latents; after 200 training steps the latent holds both dots (a probe places them to within 0.034 and 0.020); the instant partner chases our post-move position; the optional third dot moves, stays on the table and changes nothing else; a size-matched joint model is within 2% of factored's weights.

At the 7-second size, on 6 episodes each, no learned model reached the goal against chase or switch. Against the still partner `joint` reached 0.33, `factored` 0.00 and `oracle` 1.00. The first job is to make the learned models plan well at small sizes.

The kill test's commands and failure numbers are in [EXPERIMENTS.md](EXPERIMENTS.md). To run the whole method with Claude Code, use the [auto-research skill](.claude/skills/auto-research/SKILL.md).

| Tier | Data trajectories x steps | Training steps | Planning episodes per split | Plan horizon, samples | Purpose |
|---|---|---|---|---|---|
| `7s` | 48 x 24 | 150 | 6 | 6, 32 | Checks only, and its scores mean nothing. A learned-model row takes 4 to 7 seconds. |
| `1m` | 300 x 40 | 1500 | 48 | 8, 64 | Debugging. |
| `5m` | 1000 x 60 | 5000 | 150 | 10, 128 | The kill test. |
| `1h` | 4000 x 80 | 30000 | 500 | 12, 256 | The claims. |

Only `7s` rows have been timed. One LeWM training step takes about 20 ms on one Mac CPU thread, so a `1h` row needs a GPU.

---

## Know what this test can and can't judge

**It can judge,** once the kill test passes, whether splitting a small world model into "our body" and "the other agent" helps planning against partners that ignore us, react to us, or switch modes, at equal inputs and equal data, on a 2D table.

**It can't judge** large pixel world models (ours is a small conv encoder on 16 x 16 pictures, not LeWM's ViT), many agents, partners that learn during the episode, cooperation, or partners that model us back. The partners follow simple fixed rules so the truth is known.

> [!WARNING]
> - **No kill-test or claim row has run.** The rules have only read 7-second rows and made-up rows.
> - **The factored model's structure matches every partner here**, because no partner sees our current action. A partner that reacts within the same step would break that structure, and none is included.
> - **A partner that walks to a point between us and the goal was dropped.** With its speed at 0.03, even the true simulator reached 0.00. At 0.015 the true simulator reached 0.60 and planning that ignored the partner reached 1.00, because the short planning horizon can't see the way around. The planner, not the model, was the limit.
> - **`k` is the same for both learned models in every claim.** A factored win at k = 3 over joint at k = 1 would only show that history helps.

---

## Is this new?

| Work | Separate agent model | Controls inputs and data | Partner types compared | Known-answer check |
|---|---|---|---|---|
| [Dreaming of Others](https://arxiv.org/abs/2605.31361) | yes | not checked | not checked | not checked |
| [MA-WAM](https://arxiv.org/abs/2609.31281) | not checked | not checked | not checked | not checked |
| **This repo** | factored against joint | yes, same windows and data | ignores us, chases us, hidden switch | still partner and true simulator |

This table comes from literature searches on 2026-11-06, from titles and search snippets. None of these papers was read in full, and "not checked" means the search result didn't say. It doesn't settle that the question is new.
