"""Numbers and plots for content/rl/05-stable-training.md.

Run: uv run python -m examples.rl.stable_training

Four parts, all on the shared toy of examples/rl/toy_rl.py:
1. KL to the reference: the exact KL and the estimators k1, k2, k3 at one position (exact enumeration).
2. Entropy collapse: a simulated GRPO run (fixed seeds) with and without an entropy bonus or clip-higher.
3. Training-inference mismatch: importance weights and truncated importance weights at one position.
4. Monitoring: a dashboard of the signals of the simulated run.
"""

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import LENGTH, POLICY_LOGITS, TARGET, token_ids
from examples.toy import LOGITS, VOCAB, softmax

PAGE = "stable-training"
TARGET_IDS = np.array(token_ids(TARGET))


def entropy(p: np.ndarray) -> np.ndarray:
    """Entropy (nats) of each row of probabilities."""
    return -(p * np.log(p)).sum(axis=-1)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Exact KL(p || q) of each row."""
    return (p * np.log(p / q)).sum(axis=-1)


# --- 1. KL to the reference model ------------------------------------------------------------

REF_LOGITS = LOGITS                          # reference = the toy start policy (the SFT model)
TRAINED_LOGITS = np.array([2.5, 1.0, 2.0])   # assumed policy at position 1 after some updates
KL_BETA = 0.04                               # KL coefficient of DeepSeekMath (GRPO paper)


def kl_estimators(policy_logits: np.ndarray = TRAINED_LOGITS,
                  ref_logits: np.ndarray = REF_LOGITS) -> dict[str, np.ndarray | float]:
    """k1, k2, k3 for each possible sampled token, their exact expectations and the exact KL.

    The token is sampled from the policy pi. r = pi_ref(a) / pi(a) (Schulman 2020).
    """
    pi, ref = softmax(policy_logits), softmax(ref_logits)
    r = ref / pi
    k1 = -np.log(r)
    k2 = np.log(r) ** 2 / 2
    k3 = (r - 1) - np.log(r)
    mean = {name: float(pi @ k) for name, k in (("k1", k1), ("k2", k2), ("k3", k3))}
    std = {name: float(np.sqrt(pi @ (k - mean[name]) ** 2)) for name, k in (("k1", k1), ("k2", k2), ("k3", k3))}
    return {"pi": pi, "ref": ref, "r": r, "k1": k1, "k2": k2, "k3": k3,
            "mean": mean, "std": std, "exact": float(kl(pi, ref))}


# --- 2. Entropy collapse: a simulated GRPO run -----------------------------------------------

GROUP_SIZE = 8      # assumed answers per step
LR = 4.0            # assumed step size on the logits
INNER_STEPS = 2     # assumed updates on each batch (so the PPO clip can act)
STEPS = 100         # steps of the main run
SEED = 0
# Variant: assume position 2 has already collapsed on "sat" (the target is "cat").
COLLAPSED_LOGITS = POLICY_LOGITS.copy()
COLLAPSED_LOGITS[1] = [0.1, -3.0, 3.0]
VARIANT_STEPS = 150
VARIANT_SEEDS = 40


def simulate(start_logits: np.ndarray = POLICY_LOGITS, steps: int = STEPS, seed: int = SEED,
             entropy_coef: float = 0.0, eps_low: float = 0.2, eps_high: float = 0.2,
             group_size: int = GROUP_SIZE, lr: float = LR, inner_steps: int = INNER_STEPS) -> dict:
    """GRPO-style training of the tabular policy with the partial reward (fraction of correct tokens).

    Each step: sample a group, compute group-normalized advantages, then do `inner_steps` gradient
    ascent steps on the PPO-clipped objective (mean over all tokens) plus `entropy_coef` * entropy.
    Returns the logits after training and one value of each monitored signal for each step.
    """
    rng = np.random.default_rng(seed)
    ref = softmax(start_logits)
    z = start_logits.copy()
    log = {key: [] for key in ("reward", "entropy", "entropy_pos", "kl", "clip_frac", "zero_adv", "grad_norm")}
    onehot = np.eye(len(VOCAB))
    for _ in range(steps):
        p_old = softmax(z)
        answers = np.array([[rng.choice(len(VOCAB), p=p_old[t]) for t in range(LENGTH)]
                            for _ in range(group_size)])
        rewards = (answers == TARGET_IDS).mean(axis=1)
        std = rewards.std()
        adv = np.zeros(group_size) if std == 0 else (rewards - rewards.mean()) / std
        for inner in range(inner_steps):
            p = softmax(z)
            grad = np.zeros_like(z)
            clipped = 0
            for i in range(group_size):
                for t in range(LENGTH):
                    a = answers[i, t]
                    rho = p[t, a] / p_old[t, a]
                    if (adv[i] > 0 and rho > 1 + eps_high) or (adv[i] < 0 and rho < 1 - eps_low):
                        clipped += 1   # clipped token: the min() picks the flat branch, gradient 0
                        continue
                    grad[t] += adv[i] * rho * (onehot[a] - p[t])
            grad /= group_size * LENGTH
            h = entropy(p)
            grad += entropy_coef * (-p * (np.log(p) + h[:, None]))   # gradient of the entropy bonus
            if inner == 0:
                grad_norm = float(np.linalg.norm(grad))
            z = z + lr * grad
        p = softmax(z)
        log["reward"].append(rewards.mean())
        log["entropy_pos"].append(entropy(p))
        log["entropy"].append(entropy(p).mean())
        log["kl"].append(kl(p, ref).sum())
        log["clip_frac"].append(clipped / (group_size * LENGTH))   # from the last inner step
        log["zero_adv"].append(float(std == 0))
        log["grad_norm"].append(grad_norm)
    return {"logits": z, **{key: np.array(value) for key, value in log.items()}}


def variant_summary(entropy_coef: float = 0.0, eps_high: float = 0.2) -> dict[str, float]:
    """Run the collapsed-position variant over many seeds. Count the runs that learn the target."""
    solved, entropies = 0, []
    for seed in range(VARIANT_SEEDS):
        run = simulate(COLLAPSED_LOGITS, VARIANT_STEPS, seed, entropy_coef=entropy_coef, eps_high=eps_high)
        solved += int((softmax(run["logits"]).argmax(axis=1) == TARGET_IDS).all())
        entropies.append(run["entropy"][-1])
    return {"solved": solved, "entropy": float(np.mean(entropies))}


# --- 3. Training-inference mismatch ----------------------------------------------------------

TRAINER_LOGITS = COLLAPSED_LOGITS[1]                       # position 2 of the variant: "cat" is rare
SAMPLER_LOGITS = TRAINER_LOGITS + np.array([0.0, -0.85, 0.0])   # assumed small error of the engine
TIS_CAP = 2.0                                              # assumed cap C
GOOD = VOCAB.index("cat")                                  # assume reward 1 for "cat" here, else 0


def mismatch() -> dict[str, np.ndarray | float]:
    """Importance weights w = pi_train / pi_sampler and the effect on the policy gradient."""
    pi, mu = softmax(TRAINER_LOGITS), softmax(SAMPLER_LOGITS)
    w = pi / mu
    w_tis = np.minimum(w, TIS_CAP)
    reward = (np.arange(len(VOCAB)) == GOOD).astype(float)
    # Gradient of E[reward] on the logit of "cat": E[reward * (1 - pi_cat)] under each estimator.
    score = 1 - pi[GOOD]
    grads = {
        "on_policy": float(pi @ (reward * score)),
        "no_correction": float(mu @ (reward * score)),
        "is": float(mu @ (w * reward * score)),
        "tis": float(mu @ (w_tis * reward * score)),
    }
    relative = {name: g / grads["on_policy"] for name, g in grads.items()}
    return {"pi": pi, "mu": mu, "w": w, "w_tis": w_tis, "mean_w": float(mu @ w),
            "grads": grads, "relative": relative}


ZERO_ADV_WINDOW = 10   # the toy has one group per step, so the dashboard shows a moving share


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` values at each step (fewer at the start)."""
    return np.array([x[max(0, i - window + 1):i + 1].mean() for i in range(len(x))])


# --- Plots -----------------------------------------------------------------------------------

def plot_kl_estimators() -> None:
    est = kl_estimators()
    r = np.linspace(0.05, 4, 400)

    def draw(ax, theme):
        for name, values, color in (("k1 = -log r", -np.log(r), theme["series"][0]),
                                    ("k2 = (log r)² / 2", np.log(r) ** 2 / 2, theme["series"][1]),
                                    ("k3 = (r - 1) - log r", (r - 1) - np.log(r), theme["series"][2])):
            ax.plot(r, values, color=color, label=name)
        ax.scatter(est["r"], est["k1"], color=theme["series"][0], zorder=3)
        ax.scatter(est["r"], est["k3"], color=theme["series"][2], zorder=3)
        # "cat" and "sat" have the same ratio r here, so they share one label.
        for label, k in (("the", 0), ("cat, sat", 2)):
            ax.annotate(label, (est["r"][k], est["k1"][k]), textcoords="offset points", xytext=(6, -12),
                        color=theme["ink2"])
        ax.axhline(0, color=theme["axis"], linewidth=1)
        ax.set_xlabel("r = π_ref(a) / π(a) for the sampled token a")
        ax.set_ylabel("estimate of KL for one sample")
        ax.set_ylim(-1.5, 3)
        ax.legend(loc="upper center")

    save_figure(draw, PAGE, "kl-estimators")


def plot_entropy(plain: dict, bonus: dict) -> None:
    steps = np.arange(1, len(plain["entropy"]) + 1)

    def draw_positions(ax, theme):
        for t, token in enumerate(TARGET):
            ax.plot(steps, plain["entropy_pos"][:, t], color=theme["series"][t],
                    label=f"position {t + 1} (target '{token}')")
        ax.set_xlabel("training step")
        ax.set_ylabel("entropy (nats)")
        ax.legend()

    def draw_bonus(ax, theme):
        ax.plot(steps, plain["entropy"], color=theme["series"][0], label="no bonus")
        ax.plot(steps, bonus["entropy"], color=theme["series"][1], label="entropy bonus 0.1")
        ax.set_xlabel("training step")
        ax.set_ylabel("mean entropy per position (nats)")
        ax.legend()

    save_figure(draw_positions, PAGE, "entropy-positions")
    save_figure(draw_bonus, PAGE, "entropy-bonus")


def plot_dashboard(run: dict) -> None:
    steps = np.arange(1, len(run["reward"]) + 1)
    window = ZERO_ADV_WINDOW
    series = dict(run, zero_adv=rolling_mean(run["zero_adv"], window))
    panels = [("reward", "mean reward"), ("entropy", "entropy (nats)"), ("kl", "KL to reference"),
              ("clip_frac", "clip fraction"), ("zero_adv", f"zero-advantage share ({window} steps)"),
              ("grad_norm", "gradient norm")]

    def draw(ax, theme):
        fig = ax.figure
        ax.remove()
        for k, (key, label) in enumerate(panels):
            sub = fig.add_subplot(2, 3, k + 1)
            sub.plot(steps, series[key], color=theme["series"][k % 4], linewidth=1.5)
            sub.set_title(label, fontsize=10, color=theme["ink"])
            if k >= 3:
                sub.set_xlabel("step")

    save_figure(draw, PAGE, "dashboard", size=(7.2, 4.4))


def main() -> None:
    est = kl_estimators()
    print("1. KL to the reference (position 1)")
    print("   reference probs:", np.round(est["ref"], 3), " policy probs:", np.round(est["pi"], 3))
    print("   r     :", np.round(est["r"], 3))
    for name in ("k1", "k2", "k3"):
        print(f"   {name}    :", np.round(est[name], 3),
              f" mean {est['mean'][name]:.3f}  std {est['std'][name]:.3f}")
    print(f"   exact KL(pi || ref) = {est['exact']:.3f}")
    a = VOCAB.index("the")
    print(f"   reward penalty for 'the' = -beta * k1 = {-KL_BETA * est['k1'][a]:.4f}; "
          f"loss term beta * k3 = {KL_BETA * est['k3'][a]:.4f}")

    plain = simulate()
    bonus = simulate(entropy_coef=0.1)
    print("2. Entropy collapse (main run: seed 0, G=8, lr=4, 2 inner steps)")
    print("   start entropy per position:", np.round(entropy(softmax(POLICY_LOGITS)), 3))
    for s in (1, 5, 10, 20, 50, 100):
        print(f"   step {s:3d}: reward {plain['reward'][s - 1]:.3f}  entropy/pos",
              np.round(plain["entropy_pos"][s - 1], 3), f" mean {plain['entropy'][s - 1]:.3f}")
    first_zero = int(np.argmax(plain["zero_adv"] > 0)) + 1
    print(f"   first step with a zero-advantage group: {first_zero}; "
          f"zero-advantage steps in 51-100: {plain['zero_adv'][50:].mean():.2f}")
    print(f"   bonus 0.1: entropy at step 100 {bonus['entropy'][-1]:.3f}, "
          f"mean reward steps 51-100 {bonus['reward'][50:].mean():.3f} vs plain {plain['reward'][50:].mean():.3f}")
    print("   variant (position 2 collapsed on 'sat'): start probs", np.round(softmax(COLLAPSED_LOGITS)[1], 4))
    for label, coef, eps_high in (("plain", 0.0, 0.2), ("clip-higher 0.28", 0.0, 0.28),
                                  ("bonus 0.02", 0.02, 0.2), ("bonus 0.1", 0.1, 0.2)):
        v = variant_summary(coef, eps_high)
        print(f"   {label:17s}: solved {v['solved']}/{VARIANT_SEEDS}  entropy {v['entropy']:.3f}")

    m = mismatch()
    print("3. Training-inference mismatch (position 2 of the variant)")
    print("   trainer probs:", np.round(m["pi"], 4), " sampler probs:", np.round(m["mu"], 4))
    print("   w      :", np.round(m["w"], 3), " truncated:", np.round(m["w_tis"], 3))
    print(f"   E_mu[w] = {m['mean_w']:.3f}")
    print("   gradient on logit 'cat':", {k: round(v, 5) for k, v in m["grads"].items()})
    print("   relative to on-policy  :", {k: round(v, 3) for k, v in m["relative"].items()})

    print("4. Dashboard (main run)")
    for s in (1, 5, 10, 20, 100):
        print(f"   step {s:3d}: KL {plain['kl'][s - 1]:.3f}  clip {plain['clip_frac'][s - 1]:.3f}  "
              f"grad norm {plain['grad_norm'][s - 1]:.3f}")
    print(f"   max KL {plain['kl'].max():.3f} at step 100: {plain['kl'][-1]:.3f}; "
          f"max clip fraction {plain['clip_frac'].max():.3f}")

    plot_kl_estimators()
    plot_entropy(plain, bonus)
    plot_dashboard(plain)


if __name__ == "__main__":
    main()
