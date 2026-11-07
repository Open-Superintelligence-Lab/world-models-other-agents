# Questions

Each question changes one part of the world model or its training data, and compares the result with the base model. Every change is one flag on `run.py`, and `compare.py` prints the answer.

## How to answer one question

1. Run the base rows once. They are shared by every question.

```bash
for m in joint factored; do for p in still patrol chase switch; do
  python3 run.py --tier 1m --model $m --partner $p --k 3
done; done
```

2. Run the question's command from the list below. Each one runs the same rows with one flag added.

3. Print the answer.

```bash
python3 compare.py --tier 1m
```

Each row of the output is one partner, model and setting. `success` is the share of episodes that reached the goal without touching the partner. `vs base` is the change against the base model, with a 95% interval over episodes. `me_err` and `other_err` say how far the imagined position of our dot and the partner is from the truth after a few steps.

**Tiers.** `7s` rows take 4 to 7 seconds on a laptop and check that the code runs. Their scores mean nothing. Use `1m` to debug a question, and `5m` or `1h` on a GPU for numbers that go in the paper. At `7s`, the base model doesn't reach the goal yet, even against the partner that never moves. So until the base model plans well at the tier you use, a "no difference" answer doesn't count.

**Example.** Question 1 at `5m` might print `chase factored 3 sigreg=0.0 0.41 -0.35 [-0.45, -0.25]`. That would mean that, against the chaser, removing SIGReg dropped success from 0.76 to 0.41, and the drop is real because the whole interval is below 0.

## Remove a part

**1. Remove SIGReg, or change its weight.** SIGReg is the term that keeps the latent spread out, so the encoder can't map every picture to the same numbers. Does planning fail without it, and does the model lose the partner before it loses our own dot? Compare `me_err` and `other_err`. Then try weights 0.01 and 1 against the base 0.1: does a strong weight blur small details, such as which way the partner is heading?

```bash
for w in 0 0.01 1; do for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --sigreg $w; done; done; done
```

**2. Remove our action from the predictor.** The predictor then sees only the past pictures. How much worse does planning get? This shows how much the model uses our move at all. Expect `me_err` to rise a lot. If `other_err` rises too, the model was using our move to predict the partner.

```bash
for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --no-action; done; done
```

**3. Remove history.** The model sees only the current picture (`--k 1`) instead of the last three. A single picture can't show which way the switching partner is going or which mode it is in, so expect the switch partner to suffer most.

```bash
for m in joint factored; do for p in patrol chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 1; done; done
```

**4. Remove colour.** Every dot is drawn white, so colour no longer says which dot is ours. Can the model still tell them apart from how each one moves after our action?

```bash
for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --gray; done; done
```

**5. Remove the split, but keep the size.** The factored model has more weights than the joint one: 12% more at the `1m` width and 18% more at `5m`. This widens the joint model to the same weight count. If factored still wins, the split helps; if the gap closes, the extra weights did it.

```bash
for p in chase switch; do python3 run.py --tier 1m --model joint --partner $p --k 3 --match-size; done
```

## Add a part

**6. Add a separate encoder for each dot.** One encoder sees only the red channel (our dot) and gives the first latent half. The other sees only the blue channel (the partner) and gives the second half. This is "object-centric" with the objects given for free. Does it beat one encoder that has to sort the dots out itself?

```bash
for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --per-dot; done; done
```

**7. Tell the model which half is the partner.** An extra training loss makes the first latent half hold our position and the second half the partner's. Does that beat letting the model find the split alone? Add `--halves` to the base and this run to see where each dot ended up.

```bash
for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --tell 1; done; done
```

**8. Add a third dot that is harmless.** A green dot drifts along a looping path, ignores everyone, and can be touched safely. Does the model waste capacity on it, and does the factored model handle it better?

```bash
for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --distractor; done; done
```

**9. Add a partner that sees our move in the same step.** The `instant` partner chases where we will be after this step's move. The factored model's partner half can't see our current move, so it can't predict this partner exactly. Does factored then lose to joint? This tests where the split stops helping. For this question the comparison is between models (factored against joint), not against a base row.

```bash
for m in joint factored oracle blind; do python3 run.py --tier 1m --model $m --partner instant --k 3; done
```

## Change the data

**10. Train on one partner, or on all four mixed.** By default each model trains only on the partner it plans against. With `--train-partners all`, the training data has an equal share from each partner, with the same total amount. Does a model that has seen every kind of partner do as well against each one?

```bash
for m in joint factored; do for p in still patrol chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --train-partners all; done; done
```

**11. Train on goal-seeking moves only, or random moves only.** By default half the training trajectories head for the goal and half move randomly. A chaser reacts to where we go, so does the model need both kinds to learn it?

```bash
for d in seek random; do for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --data $d; done; done; done
```

**12. Halve and double the training data.** Does joint need more data than factored to learn the chaser? If the gap between them shrinks with more data, the split mainly helps when data is scarce.

```bash
for s in 0.5 2; do for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --data-scale $s; done; done; done
```

## What does this part do

**13. Which half ends up holding the partner?** `--halves` fits a small network from each latent half alone to each dot's position and reports the error. A low error means that half holds that dot; about 0.25 is guessing. In the factored model, does the partner end up in the half that can't see our action (half B)? In the joint model, the halves have no meaning, so it is the control.

```bash
for m in joint factored; do for p in patrol chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --halves --out halves.jsonl; done; done
python3 compare.py --tier 1m --results halves.jsonl
```

**14. What does the latent size do?** Try 8, 16 and 64 numbers against the base 32. Does factored only win when the latent is small and the model has to choose what to keep?

```bash
for d in 8 16 64; do for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --latent $d; done; done; done
```

**15. What does the picture size do?** Try 32 pixels against the base 16. A dot is about 2 pixels wide at 16. Does a sharper picture make the positions, and so the plans, more accurate?

```bash
for m in joint factored; do for p in chase switch; do python3 run.py --tier 1m --model $m --partner $p --k 3 --img 32; done; done
```

## Where to change the code

| To change | File and function |
|---|---|
| The model's parts and options | `model.py`, `LeWM.__init__`, `LeWM.delta` and `LeWM.embed` |
| The training loss | `model.py`, `losses` and `train` |
| The partners and the third dot | `world.py`, `partner_target`, `step` and `init` |
| The pictures | `world.py`, `render` |
| The training data | `world.py`, `collect`, and `run.py`, `training_data` |
| A new flag | `run.py`, the "questions" argument group and `DEFAULTS`, then `row` |

After any change, run `python3 check.py` (7 seconds). It checks every known answer, including that factored's partner half can't see our action. `python3 check.py --slow` also runs one `7s` row for every flag (about 2 minutes).
