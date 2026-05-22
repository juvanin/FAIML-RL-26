# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FAIML-RL-26 course project (Politecnico di Torino, AY 2025-2026).
Two-part reinforcement learning project:

- **Part 1 — COMPLETE**: Custom REINFORCE and Actor-Critic (5 variants) on Hopper-v4.
  All scripts in `part1/` are implemented and verified.
- **Part 2 — IN PROGRESS**: Sim-to-real transfer via domain randomization on PandaPush-v3.
  Train on a "source" env (cube mass = 1kg), evaluate on a "target" env (cube mass = 5kg).
  Domain randomization over cube mass should produce a policy that transfers without
  seeing the target during training.

---

## Part 2 Structure

### Phase A — Tasks 4 & 5: SB3 Baselines

**Task 4:** Compare PPO and SAC on PandaPush-v3 via stable-baselines3.
Justify which performs better in the report.

**Task 5:** Train the best algorithm on source and target separately.
Evaluate all three baseline configs over 50 episodes each:
- `src→src`: train on source, eval on source (reference)
- `src→tgt`: train on source, eval on target (**lower bound**)
- `tgt→tgt`: train on target, eval on target (**upper bound**)

`src→tgt` MUST be lower than `tgt→tgt`. If it isn't, the env setup is wrong.

### Phase B — Task 6: Domain Randomization

**UDR (Uniform Domain Randomization):** Randomize cube mass uniformly over a range
at each episode reset. The range IS the hyperparameter — try 3-5 different ranges.

**ADR (Automatic Domain Randomization, arXiv:1910.07113):** Adaptively widen/narrow
the mass range based on agent performance. Read the paper before implementing.

Evaluate both over 50 episodes each:
- `udr→src`, `udr→tgt`
- `adr→src`, `adr→tgt`

### Full evaluation matrix (7 configs, 50 episodes each):

| Config     | Train env        | Eval env | Role             |
|------------|------------------|----------|------------------|
| `src→src`  | source, no DR    | source   | Reference        |
| `src→tgt`  | source, no DR    | target   | Lower bound      |
| `tgt→tgt`  | target, no DR    | target   | Upper bound      |
| `udr→src`  | source + UDR     | source   | UDR sanity check |
| `udr→tgt`  | source + UDR     | target   | Main UDR result  |
| `adr→src`  | source + ADR     | source   | ADR sanity check |
| `adr→tgt`  | source + ADR     | target   | Main ADR result  |

---

## Active Constraints (DO NOT VIOLATE)

1. **Algorithms**: Use PPO and SAC from stable-baselines3. The skeleton `train_sb3.py`
   has a DDPG import — this is incorrect per the assignment. Replace with PPO/SAC.
   **PENDING**: Confirm with professor whether DDPG is also acceptable.

2. **Mass values**: Source cube mass = 1.0 kg, target cube mass = 5.0 kg. The 4 kg
   shift is the sim-to-real gap being simulated.

3. **Observation space**: PandaPush-v3 uses dict obs with keys `observation`,
   `achieved_goal`, `desired_goal`. SB3 requires `MultiInputPolicy` for dict obs —
   do NOT use `MlpPolicy`.

4. **Reward signal**: `reward_type="dense"` must stay hardcoded. Sparse rewards make
   PandaPush nearly unlearnable.

5. **UDR distribution**: Must NOT be centered on the target mass (5 kg) — that
   is equivalent to cheating. Design UDR without "knowing" the target. Typical choices:
   symmetric around source (e.g., Uniform[0.5, 1.5]) or asymmetric upward
   (e.g., Uniform[0.5, 3.0]).

6. **Wrapper attributes**: `rand_wrapper.py` already references `self.mass_min`,
   `self.mass_max`, and `self.last_sample_type` in its print statement. These must be
   set during `_sample_mass()` for both UDR and ADR modes.

7. **Physics modification**: Mass is set via:
   ```python
   sim.physics_client.changeDynamics(
       bodyUniqueId=sim._bodies_idx["object"],
       linkIndex=-1,
       mass=sampled_mass
   )
   ```
   The sim handle is at `self.env.unwrapped.task.sim`.

8. **Evaluation protocol**: 50 episodes per config, report mean ± std return AND
   success rate. Use SB3's `evaluate_policy` helper from
   `stable_baselines3.common.evaluation`.

9. **Part 1 patterns do NOT apply**: Part 2 uses SB3 — no manual log_prob, no
   manual loss, no hand-rolled optimizers. SB3 handles all of this internally.

---

## Files to Implement

### Modify existing skeletons:

- **`part2/train_sb3.py`**: Replace DDPG import with PPO/SAC. Add `--algorithm` CLI
  arg. Wrap env with `RandomizationWrapper` when `--sampling-strategy` is `udr` or
  `adr`. Instantiate model with `MultiInputPolicy`. Train with SB3 callbacks
  (checkpoint, eval). Save model as `.zip`.

- **`part2/eval_sb3.py`**: Currently has TODOs — needs full implementation.
  Load model from `.zip`, run `evaluate_policy` for 50 episodes, report
  mean ± std return and success rate. Support `--env-type source|target`.

- **`part2/rand_wrapper.py`**: Implement `_sample_mass()` for `udr` and `adr` modes.
  - UDR: `np.random.uniform(mass_min, mass_max)`. Set `self.last_sample_type = 'interior'`.
  - ADR: Sample from boundary or interior based on ADR algorithm (see below).
    Set `self.last_sample_type = 'boundary_low' | 'boundary_high' | 'interior'`.

### New files to create:

- **`part2/plot_results.py`**: Bar charts of return and success rate across all 7
  configs. Load from a structured results directory.
- **`part2/run_sweep.sh`** or Colab notebook: Orchestrate the full 7-config sweep.

---

## ADR Algorithm Sketch (arXiv:1910.07113)

Read the paper first. The core mechanism for a single randomized parameter (mass):

1. Maintain two circular performance buffers: `buf_low` (for boundary_low samples)
   and `buf_high` (for boundary_high samples). Buffer size is a hyperparameter (~100).
2. At each `reset()`, with probability `p_boundary` sample from a boundary
   (`mass_min` or `mass_max`), else sample uniformly from `(mass_min, mass_max)`.
3. After each episode, append the success/return to the appropriate buffer based on
   `self.last_sample_type`.
4. When a buffer is full, check performance against thresholds `t_low` and `t_high`:
   - If `buf_low` mean > `t_high`: widen by decreasing `mass_min` (harder lower end)
   - If `buf_low` mean < `t_low`: narrow by increasing `mass_min` (easier lower end)
   - If `buf_high` mean > `t_high`: widen by increasing `mass_max`
   - If `buf_high` mean < `t_low`: narrow by decreasing `mass_max`
5. Clip `mass_min` and `mass_max` to safe physical bounds (e.g., [0.1, 10.0]).

`last_sample_type` is how the wrapper communicates to the ADR buffer which buffer
to update after the episode ends.

---

## Resolved Decisions (DO NOT RE-LITIGATE)

- Part 1 is complete. Do not modify anything under `part1/`.
- Part 2 uses third-party SB3 — we are NOT writing PPO/SAC/DDPG from scratch.
  Hand-rolled implementation patterns from Part 1 do NOT apply here.
- Sim-to-real / sim-to-sim framing is settled. The source/target distinction is a
  domain shift; domain randomization is the standard remedy.
- Do not center UDR on target mass (5 kg) — this is cheating.
- Do not write PPO or SAC from scratch.
- Do not use `MlpPolicy` for PandaPush-v3.

---

## Environment Setup & Running Scripts

```bash
source venv/bin/activate
pip install -r requirements.txt

# Install panda-gym
cd part2/panda-gym && pip install -e .
```

```bash
# Verify PandaPush environment
python part2/test_random_policy.py

# Train (examples)
python part2/train_sb3.py --algorithm ppo --sampling-strategy none --env-type source --timesteps 500000
python part2/train_sb3.py --algorithm sac --sampling-strategy udr  --env-type source --timesteps 500000

# Evaluate
python part2/eval_sb3.py --model-path results/ppo_source_none.zip --episodes 50 --env-type target
```

---

## Environments Reference

| Env | State | Action | Done condition |
|---|---|---|---|
| Hopper-v4 | 11-dim flat (torso height, pitch, joint angles + velocities) | 3-dim torques ∈ [−1,1] | height < 0.7m, \|pitch\| > 0.2rad, 1000 steps |
| PandaPush-v3 | dict: `observation` (25-dim), `achieved_goal` (3-dim), `desired_goal` (3-dim) | 4-dim end-effector delta | success or max steps |

Hopper reward: `x_velocity + 1.0 − 0.001 × ‖action‖²`
PandaPush reward (dense): negative distance between cube and goal

---

## Reference Materials
- any reading from 'references' is useful. More detail below:
- `references/FAIML_RL_2026.md` — full assignment text. **Authoritative.**
- `references/lilianweng-github-io-posts-2019-05-05-domain-randomization.md` — DR overview
- AutoDR paper: arXiv:1910.07113 — **REQUIRED before ADR implementation**
- SB3 callbacks: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
- SB3 evaluation: https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
- PPO reference: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- SAC reference: https://spinningup.openai.com/en/latest/algorithms/sac.html

If anything conflicts between these references and general knowledge, **the references win.**
