# Working in this repo

Follow `.claude/skills/auto-research/SKILL.md`. Run `python3 check.py` after every code change, and debug every part in runs of about 7 seconds before anything longer. After any change to `world.py`, `model.py` or the planner and scores in `run.py`, move the old `results.jsonl` aside and rerun every baseline. Write predictions in `EXPERIMENTS.md` and commit them before any comparison runs. Use `--final` once, at the end.
