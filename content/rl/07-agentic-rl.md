# 7. Agentic and multi-turn RL

## Core idea

In agentic RL, one episode has many turns: the model writes, a tool answers, the model writes again.
The trainer must train only the model's tokens, give credit to the right turn, and keep the GPUs busy while some episodes take much longer than others.

## The example

The [shared toy task](01-rl-basics.md) ("Copy: the cat sat") becomes a 2-turn episode: the model answers, calls a tool `check` that says which token is wrong, and answers again.

| Item | Value |
|---|---|
| Policy | Tabular: each position starts at [0.099, 0.242, 0.659] |
| Tool | `check`: returns 4 tokens of feedback on the answer |
| Call token | Assume the model writes `<check>` with probability 0.9 (log-prob −0.105) |
| Tool output | Assume `result: token 1 wrong`, with assumed model log-probs [−1.6, −0.9, −2.3, −1.2] |
| Final reward | 1 if the turn-2 answer is "the cat sat", else 0 |

Assume this group of 4 episodes. The partial score is the fraction of correct tokens.

| Episode | Turn 1 | Partial | Turn 2 (after `check`) | Partial | Final reward |
|---|---|---|---|---|---|
| E1 | sat cat sat | 0.667 | the cat sat | 1.000 | 1 |
| E2 | the sat sat | 0.667 | the cat sat | 1.000 | 1 |
| E3 | sat cat sat | 0.667 | sat cat sat | 0.667 | 0 |
| E4 | sat sat sat | 0.333 | sat cat sat | 0.667 | 0 |

The toy policy does not read the feedback, so turn 2 uses the turn-1 logit tables.

## 1. Multi-turn episodes and the loss mask

### Setup

An **episode** (or **trajectory**) is the full token sequence after the prompt: model turns and environment turns. The [Agents lesson](../engineer/05-agents.md#1-agent-fundamentals) explains the agent loop. Here we train it. Episode E1 has 11 tokens after the prompt.

<figure class="diagram" data-src="agentic-rl/loss-mask.svg">
<figcaption>One 2-turn episode. Only the tokens that the model wrote get a gradient.</figcaption>
</figure>

### Steps

1. **Label each token with its author.** The model wrote `sat cat sat`, `<check>` and `the cat sat`: 7 tokens. The tool wrote `result: token 1 wrong`: 4 tokens.

    **Why:** the policy gradient uses $\nabla \log \pi_\theta(y_t)$ for an action $y_t$ that the policy chose. The tool output is not an action. It is part of the state.

2. **Build the loss mask.** The **loss mask** $m_t$ is 1 for a model token and 0 for an environment token:

    $$m = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]$$

    7 of 11 tokens are trained (0.636). The call `<check>` has mask 1: the model chose to call the tool.

    **Why:** the model must learn *when* to call a tool and what to do with the result, not how to write the result.

3. **Calculate the masked loss.** Assume the episode advantage is $A = 1$ (part 2 shows that this is the group advantage of E1). The loss is the token mean over the masked tokens:

    $$L = -\frac{A}{\sum_t m_t} \sum_t m_t \log \pi_\theta(y_t)$$

    The 7 model log-probs add up to −6.508. Thus $L = 6.508 / 7 = 0.930$. Each model token gets the weight 1/7 = 0.143.

    **Why:** with $A > 0$, a gradient step on $L$ makes the 7 chosen tokens more probable.

4. **Compare with the unmasked loss.** Without the mask, the 4 tool tokens (sum −6.0) join: $L = 12.508 / 11 = 1.137$. Each model token gets only 1/11 = 0.091, and 4/11 = 0.364 of the gradient goes to tokens the model did not choose.

    **Why:** the model learns to predict the tool output, so it can start to write fake tool results. With $A < 0$, the loss also pushes down tool text that says nothing about the model's choices.

<div class="key-point">

In a multi-turn episode, train only the tokens that the model wrote, the tool calls included. The environment tokens are context: they go into the forward pass, but their mask is 0.

</div>

### From toy to real

- Real tool outputs (files, web pages, test logs) are often longer than the model turns.
- The verl multi-turn documentation adds new assistant tokens with `loss_mask += [1] * len(token_ids)`, so that only assistant tokens are in the loss mask.
- TRL's `GRPOTrainer` runs tool calls during rollouts (`tools=`). Its documentation says the length metrics then count only non-tool tokens.
- Keep the token ids of the rollout. If you decode and tokenize the text again, the ids can change, and the trainer scores tokens that were never sampled.

## 2. Credit assignment over turns

### Setup

E1 and E2 fix their mistake after `check`. E3 ignores the feedback. E4 fixes one of two wrong tokens. **Credit assignment** asks which tokens caused the reward.

Each episode has 7 model tokens: 4 in turn 1 (with `<check>`) and 3 in turn 2. We use the group advantage $(r - \text{mean}) / \text{std}$ of the [GRPO lesson](04-grpo-family.md).

### Steps

1. **Trajectory-level advantage.** The final rewards are [1, 1, 0, 0]. The group advantages are $A = [1, 1, -1, -1]$. Each of the 7 model tokens of an episode gets its episode's advantage.

    **Why:** it is the simple default: one reward for each episode. But E4 gets −1 on its good fix in turn 2, and E1 gets +1 on its wrong first answer.

2. **Per-turn rewards.** Assume the reward of a turn is its gain of partial score: $r_1$ = score after turn 1, $r_2$ = score after turn 2 − $r_1$. The rewards are E1 [0.667, 0.333], E2 [0.667, 0.333], E3 [0.667, 0], E4 [0.333, 0.333]. Each sum is the final partial score.

    **Why:** it says which turn made progress: a [dense reward](02-rewards.md) for each turn.

3. **Discounted return of each turn.** Assume the discount $\gamma = 0.9$:

    $$G_t = r_t + \gamma\, G_{t+1}$$

    For E1: $G_2 = 0.333$ and $G_1 = 0.667 + 0.9 \times 0.333 = 0.967$. All returns: E1 [0.967, 0.333], E2 [0.967, 0.333], E3 [0.667, 0], E4 [0.633, 0.333].

    **Why:** a good turn 1 makes turn 2 easier, so turn 1 gets part of the later credit. A smaller $\gamma$ gives it less.

4. **Turn-level advantages.** Normalize the returns of one turn across the group:

    | Episode | Turn 1 | Turn 2 |
    |---|---|---|
    | E1 | 0.997 | 0.577 |
    | E2 | 0.997 | 0.577 |
    | E3 | −0.892 | −1.732 |
    | E4 | −1.102 | 0.577 |

    The 7 token advantages of E4 are now [−1.102, −1.102, −1.102, −1.102, 0.577, 0.577, 0.577], not 7 times −1.

    **Why:** E4's fix now gets a positive signal, and E3's "ignore the feedback" turn gets the most negative one (−1.732).

5. **Check the effect of $\gamma$.** With $\gamma = 0$, the turn-1 advantages are [0.577, 0.577, 0.577, −1.732]: the same turn-1 score gives the same credit. With $\gamma = 1$, they are [1, 1, −1, −1], as at the trajectory level.

    **Why:** $\gamma$ moves the signal between "only this turn" and "the full episode". It changes which turn the model learns from.

<div class="key-point">

One reward at the end gives good and bad turns the same advantage. Per-turn rewards with a discount give each turn its own advantage, but you need a reward for each turn and a choice of $\gamma$.

</div>

### From toy to real

- Most real tasks give only an outcome reward (the tests pass). Per-turn credit then needs a [process reward](02-rewards.md) or a [value model](03-ppo.md).
- **GiGPO** (Group-in-Group Policy Optimization, Feng et al., 2025) adds step-level groups to the episode-level group. It finds states that occur again in several trajectories (**anchor states**) and compares the actions taken from each one, with no extra model or rollouts. It reports gains over GRPO on ALFWorld and WebShop.
- In our toy, E1 and E3 reach the same state before turn 2 (same answer, same feedback). E1 fixed the answer and E3 did not: a step-level group compares exactly these two actions.

## 3. Synchronous vs asynchronous rollouts

### Setup

Generation takes most of the time of an RL step ([Training libraries](../scientist/10-training-libraries.md#2-preference-tuning-and-reinforcement-learning-rl-libraries)). Multi-turn episodes vary a lot in length.

Assume a group of 8 rollouts with log-normal lengths: median 250 tokens, spread (standard deviation of the log-length) 0.6. We take the 8 quantiles at (i + 0.5) / 8, so no random seed is needed. The lengths are [100, 147, 186, 227, 275, 335, 426, 628] tokens.

Assume each rollout has its own slot in the batch, and each slot decodes one token at each step.

<figure class="plot">
<img src="/figures/agentic-rl/sync-timeline.svg" alt="Timeline of 8 rollouts in one synchronous batch. Lengths go from 100 to 628 decode steps. All slots wait until step 628. 54% of slot time is idle.">
<figcaption>In a synchronous batch, every slot waits for the longest rollout.</figcaption>
</figure>

### Steps

1. **Calculate the idle time of a synchronous batch.** In **synchronous** RL, the trainer waits for all rollouts of the batch, updates, then generates the next batch. The batch lasts 628 steps: 8 × 628 = 5,024 slot-steps, of which 2,324 generate a token. The idle fraction is $1 - 290.5 / 628 = 0.537$.

    **Why:** the longest rollout (a **straggler**) decides the time of the batch. The slots of the short rollouts do nothing.

2. **Make the group larger.** The more rollouts in one batch, the longer the longest one. With spread 0.6, the idle fraction goes from 0.277 (2 rollouts) to 0.537 (8) and 0.758 (128). With spread 0.9 it reaches 0.864.

    **Why:** large groups give better advantage estimates, but in a synchronous system they also waste more time.

3. **Remove the wait with asynchronous rollouts.** In **asynchronous** (or **pipelined**) RL, a slot starts a new rollout as soon as one ends, and generation continues during the update. The slots are always busy, so generation is $628 / 290.5 = 2.162$ times faster.

    **Why:** generation and training no longer wait for each other.

4. **Count the staleness.** 8 rollouts (one batch, one update) finish every 290.5 steps on average. The longest rollout lasts 628 steps, so the policy changes $\lfloor 628 / 290.5 \rfloor = 2$ times before the trainer uses it. Its data is **stale** by $k = 2$ versions: it is **off-policy** (sampled by an older policy).

    **Why:** the policy gradient is correct only for samples of the current policy.

5. **Correct with an importance weight.** Assume each update adds the sparse GRPO gradient of the track group, $g = [0.289, 0, -0.289]$, to the position-1 logits. The **behaviour policy** gave `the` the probability 0.099. The **importance weight** is

    $$w = \frac{\pi_{\text{now}}(y)}{\pi_{\text{behav}}(y)}$$

    | Updates since sampling (k) | Current P(`the`) | Weight w | Truncated at 2 |
    |---|---|---|---|
    | 0 | 0.099 | 1.000 | 1.000 |
    | 1 | 0.152 | 1.538 | 1.538 |
    | 2 | 0.223 | 2.261 | 2.000 |
    | 4 | 0.410 | 4.159 | 2.000 |

    **Why:** the weight makes the old sample count as a sample of the current policy. But it grows fast with $k$, and one large weight makes the gradient noisy. **Truncation** ($\min(w, C)$, assume $C = 2$) limits the variance and accepts a small bias.

<figure class="plot">
<img src="/figures/agentic-rl/idle-vs-group.svg" alt="Idle fraction of a synchronous batch against the number of rollouts, for spreads 0.3, 0.6 and 0.9. With spread 0.6: 0.277 at 2 rollouts, 0.537 at 8, 0.758 at 128.">
<figcaption>Larger groups and a wider spread of lengths make a synchronous batch wait longer.</figcaption>
</figure>

<div class="llm-widget" data-widget="async-rollouts"></div>

<div class="key-point">

Asynchronous rollouts remove the straggler wait but train on data from older policies. Limit the staleness and correct with a truncated importance weight.

</div>

### From toy to real

- **AReaL** (Fu et al., 2025) is fully asynchronous: rollout workers never stop, and training workers update when a batch is ready. A maximum staleness $\eta$ limits how old the data is ($\eta = 0$ is synchronous RL). The paper uses $\eta = 4$ for code and $\eta = 8$ for math.
- AReaL uses a **decoupled PPO objective**. The trust-region centre is a recent **proximal policy** $\pi_{\text{prox}}$ (the parameters before the update), not the behaviour policy:

    $$J(\theta) = \mathbb{E}\left[\frac{\pi_{\text{prox}}}{\pi_{\text{behav}}} \min\left(u A,\ \operatorname{clip}(u, 1 - \varepsilon, 1 + \varepsilon) A\right)\right], \qquad u = \frac{\pi_\theta}{\pi_{\text{prox}}}$$

- In its staleness test (1.5B model, math), $\eta = 4$ gave 23.3 on AIME24 without the decoupled objective, 42.2 with it, and 42.0 for synchronous RL. The paper reports training 2.77 times faster than synchronous systems.
- AReaL can interrupt a rollout to load new weights, so one trajectory can mix tokens of several versions.
- **PipelineRL** (Piché et al., 2025) uses **in-flight weight updates**: the engine loads new weights with a short pause and continues the sequences in progress. On 128 H100 GPUs it reports about 2 times faster learning, with data that stays close to on-policy.
- Even with no staleness, the rollout engine and the trainer give slightly different log-probs: see [Stable RL training](05-stable-training.md).

## 4. Systems and libraries

### Setup

An agentic RL system has five parts. A **rollout engine** generates. An **environment** runs the tools, often in a **sandbox** (an isolated container, so model code cannot harm the host). A reward function scores the episodes. A **trainer** updates the weights. A **weight sync** sends them to the rollout engine.

<figure class="diagram" data-src="agentic-rl/system.svg">
<figcaption>The parts of an agentic RL system and the data that moves between them.</figcaption>
</figure>

Assume 128 episodes for each step, a 7B policy in bf16, 30 ms for each decode step, 10 s for each update, 25 GB/s for the sync, and 16 parallel sandboxes at 0.5 s for each `check` call.

### Steps

1. **Calculate the weight sync.** The toy policy has 9 numbers. A 7B policy in bf16 has 7 × 10⁹ × 2 bytes = 14 GB. At 25 GB/s the sync takes 14 / 25 = 0.56 s.

    **Why:** each update without a sync makes the data one more version stale.

2. **Calculate the tool time.** 128 `check` calls on 16 sandboxes run in 8 rounds of 0.5 s: 4 s.

    **Why:** a slow sandbox is a straggler too. Run tools in parallel, with a timeout.

3. **Compare two layouts.** In a **colocated** layout, one set of GPUs does each phase in turn: generation (628 × 0.03 = 18.84 s, set by the longest rollout), tools (4 s), training (10 s) and sync (0.56 s): 33.4 s for each step. In a **pipelined** layout, a second set (assume the same speed) generates during training: $\max(8.715 + 4,\ 10) + 0.56 = 13.275$ s, or 26.55 GPU-set seconds.

    **Why:** the pipelined layout removes the idle phases. Even with two sets, it uses less GPU time here (26.55 vs 33.4).

<div class="key-point">

Agentic RL is a systems problem. The layout (colocated or pipelined) and the staleness limit decide the GPU use more than the loss formula.

</div>

### From toy to real

- **verl** implements **HybridFlow** (Sheng et al., 2024): FSDP or Megatron-LM to train, vLLM or SGLang to generate. The paper reports 1.53 to 20.57 times more throughput than earlier systems. The documentation covers multi-turn tool rollouts, an agent loop and asynchronous training.
- **OpenRLHF** (Hu et al., 2024) is built on Ray, vLLM, DeepSpeed and Hugging Face Transformers. Its paper reports speedups of 1.22 to 1.68 times over other frameworks.
- **TRL `GRPOTrainer`** takes `tools=` and an experimental `environment_factory=` (one stateful environment for each rollout). `vllm_mode="colocate"` (the default) shares the GPUs with the trainer; `vllm_mode="server"` runs vLLM on separate GPUs.
- To choose a library, read [Training libraries](../scientist/10-training-libraries.md#4-how-to-choose).

## Minimal code

This code gives the main numbers of this page.

```python
import numpy as np

logp = np.array([-0.417, -1.417, -0.417, -0.105,      # turn 1 + <check> (model)
                 -1.6, -0.9, -2.3, -1.2,              # tool output (environment)
                 -2.317, -1.417, -0.417])             # turn 2 (model)
mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=float)
A = 1.0
masked = -A * (mask * logp).sum() / mask.sum()        # 0.930
unmasked = -A * logp.mean()                           # 1.137 (wrong)

def norm(x):
    return (x - x.mean()) / x.std()

r = np.array([[2/3, 1/3], [2/3, 1/3], [2/3, 0], [1/3, 1/3]])  # per-turn rewards
gamma = 0.9
G = np.stack([r[:, 0] + gamma * r[:, 1], r[:, 1]], axis=1)    # discounted returns
A_turn = np.stack([norm(G[:, 0]), norm(G[:, 1])], axis=1)     # E4: [-1.102, 0.577]

lengths = np.array([100, 147, 186, 227, 275, 335, 426, 628])
idle_sync = 1 - lengths.mean() / lengths.max()                # 0.537
```

## Comparison

| Choice | Option A | Option B | Toy numbers | Main risk |
|---|---|---|---|---|
| Loss over tokens | Masked | Unmasked | 0.930 vs 1.137 | Unmasked: learns to write tool output |
| Credit | Trajectory-level | Turn-level | E4 turn 2: −1 vs +0.577 | Turn-level: needs per-turn rewards |
| Rollouts | Synchronous | Asynchronous | 53.7% vs 0% idle | Async: data 2 versions stale |
| Off-policy fix | Raw weight | Truncated | k = 4: 4.159 vs 2.000 | Raw: variance. Truncated: bias |
| Layout | Colocated | Pipelined | 33.4 s vs 13.275 s | Pipelined: two GPU sets, staleness |

## Checks and common misconceptions

- **"The chat template masks the tool output for me."** Check it: print one example with its mask. Tool and user tokens need 0, model tokens (tool calls included) need 1.
- **"Asynchronous RL is just faster RL."** It is also off-policy. Log the staleness and the importance weights. If many weights hit the truncation, lower the staleness limit.
- **"One reward at the end is enough for long episodes."** It works, but the signal for each turn is weak. If episodes like E4 (a good turn with a negative advantage) are common, add per-turn rewards or step-level groups.

## Sources

This lesson is an addition. The [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0) has no section on this topic. The text and the example are new. Each statement about a real system was read in the linked source on 25 November 2026.

- [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978), Feng et al., 2025 (step-level groups by anchor states, results on ALFWorld and WebShop).
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298), Fu et al., 2025 (staleness limit and values, decoupled objective, staleness table, 2.77 times).
- [PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation](https://arxiv.org/abs/2509.19128), Piché et al., 2025 (in-flight weight updates, 2 times on 128 H100).
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256), Sheng et al., 2024 (verl design, throughput numbers).
- [verl documentation](https://verl.readthedocs.io/en/latest/) and its [multi-turn rollout page](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) (backends, the assistant loss mask, async pages).
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), Hu et al., 2024 (components, speedups).
- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer), Hugging Face (tool and environment options, vLLM modes, defaults).
