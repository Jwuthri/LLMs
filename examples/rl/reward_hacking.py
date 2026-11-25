"""Numbers and plots for content/rl/06-reward-hacking.md.

A buggy checker is the proxy reward. It counts how many answer tokens equal the FIRST target token
("the"), in place of comparing each token with the target token at the same position. Training on it
pushes the policy to "the the the". The gold reward (the full answer is correct) rises, then falls.

Run: uv run python -m examples.rl.reward_hacking
"""

import math
from itertools import product

import numpy as np

from examples.plotstyle import save_figure
from examples.rl.toy_rl import (
    LENGTH,
    POLICY_LOGITS,
    TARGET,
    partial_reward,
    sparse_reward,
)
from examples.toy import VOCAB, softmax

PAGE = "reward-hacking"
START_PROBS = softmax(POLICY_LOGITS[0])            # [0.099, 0.242, 0.659] at each position
ALL_ANSWERS = list(product(VOCAB, repeat=LENGTH))  # the 27 possible answers
HACK = ("the", "the", "the")                       # full proxy score, wrong answer

# Assumed settings of the simulated run.
SEED = 0
STEPS = 300
GROUP = 32     # answers per step
LR = 1.0       # step size on the logits
KL_BETA = 0.2  # KL coefficient of the KL run
ALARM = 0.1    # the alarm fires when the gap rose this much above its value at step 0


# --- 1. The proxy (buggy checker) and the gold rewards ------------------------------------

def proxy_reward(answer: tuple[str, ...]) -> float:
    """Buggy checker: compares each token with TARGET[0] in place of TARGET[pos]."""
    return sum(token == TARGET[0] for token in answer) / LENGTH


def fixed_reward(answer: tuple[str, ...]) -> float:
    """The fixed checker: position by position (the partial reward of the shared toy)."""
    return partial_reward(answer)


def overpaid() -> list[tuple[str, ...]]:
    """Answers that the buggy checker pays more than the fixed checker, best proxy first."""
    rows = [a for a in ALL_ANSWERS if proxy_reward(a) > fixed_reward(a)]
    return sorted(rows, key=lambda a: (-proxy_reward(a), a))


def checker_test(checker) -> bool:
    """Unit test of a checker on a known-bad answer: it must pay the hack less than the target."""
    return checker(HACK) < checker(TARGET)


def expected(reward, logits: np.ndarray) -> float:
    """Exact expected reward under the tabular policy: sum over the 27 answers."""
    probs = softmax(logits)
    total = 0.0
    for answer in ALL_ANSWERS:
        p = np.prod([probs[pos, VOCAB.index(tok)] for pos, tok in enumerate(answer)])
        total += p * reward(answer)
    return float(total)


def answer_prob(answer: tuple[str, ...], logits: np.ndarray) -> float:
    probs = softmax(logits)
    return float(np.prod([probs[pos, VOCAB.index(tok)] for pos, tok in enumerate(answer)]))


def kl_to_start(logits: np.ndarray) -> float:
    """KL(pi || pi_start), summed over the 3 positions (positions are independent)."""
    probs = softmax(logits)
    return float(np.sum(probs * np.log(probs / START_PROBS)))


# --- Simulated run: REINFORCE with the group mean as baseline -----------------------------

def advantages(rewards: np.ndarray) -> np.ndarray:
    """r - mean(r) over the group: the GRPO advantage without the division by the std.
    Without the division, the reward keeps its scale, so beta means the same as in kl_optimum."""
    return rewards - rewards.mean()


def kl_gradient(logits: np.ndarray) -> np.ndarray:
    """Exact gradient of KL(pi || pi_start) with respect to the logits, for each position."""
    probs = softmax(logits)
    log_ratio = np.log(probs / START_PROBS)
    kl_pos = np.sum(probs * log_ratio, axis=1, keepdims=True)
    return probs * (log_ratio - kl_pos)


def train(reward, beta: float = 0.0, steps: int = STEPS, seed: int = SEED) -> dict[str, np.ndarray]:
    """Sample GROUP answers, compute the advantages, take one policy-gradient step.
    Record exact expected values of the policy after each step (step 0 = start)."""
    rng = np.random.default_rng(seed)
    logits = POLICY_LOGITS.copy()
    hist = {key: [] for key in ("proxy", "partial", "gold", "kl", "hack")}

    def record():
        hist["proxy"].append(expected(proxy_reward, logits))
        hist["partial"].append(expected(partial_reward, logits))
        hist["gold"].append(expected(sparse_reward, logits))
        hist["kl"].append(kl_to_start(logits))
        hist["hack"].append(answer_prob(HACK, logits))

    record()
    for _ in range(steps):
        probs = softmax(logits)
        ids = np.array([[rng.choice(3, p=probs[pos]) for pos in range(LENGTH)] for _ in range(GROUP)])
        answers = [tuple(VOCAB[i] for i in row) for row in ids]
        adv = advantages(np.array([reward(a) for a in answers]))
        grad = np.zeros_like(logits)
        for row, a in zip(ids, adv):
            grad += a * (np.eye(3)[row] - probs)  # gradient of log pi(answer) for each position
        logits = logits + LR * (grad / GROUP - beta * kl_gradient(logits))
        record()
    return {key: np.array(values) for key, values in hist.items()}


# --- Closed form: the best policy for proxy - beta * KL ------------------------------------

def kl_optimum(beta: float) -> np.ndarray:
    """Maximiser of E[proxy] - beta * KL(pi || pi_start). Each position gets reward 1/3 for "the",
    so pi(a) is proportional to pi_start(a) * exp(r(a) / beta)."""
    bonus = np.array([1 / LENGTH, 0.0, 0.0]) / beta
    return np.tile(np.log(START_PROBS) + bonus, (LENGTH, 1))


def best_beta() -> float:
    """Beta of the closed-form policy with the highest gold. Gold = q * P(cat) * P(sat) with
    P(cat), P(sat) proportional to (1 - q), so it peaks at q = P(the) = 1/3."""
    q = 1 / 3
    return 1 / (LENGTH * math.log(q / (1 - q) * (1 - START_PROBS[0]) / START_PROBS[0]))


# --- 3. Detection --------------------------------------------------------------------------

def gap(run: dict[str, np.ndarray]) -> np.ndarray:
    """Proxy minus gold. Both are fractions of tokens, so the partial reward is the gold here."""
    return run["proxy"] - run["partial"]


def alarm_step(run: dict[str, np.ndarray], threshold: float = ALARM) -> int:
    """First step where the gap is more than `threshold` above its value at step 0."""
    g = gap(run)
    return int(np.argmax(g - g[0] > threshold))


def seed_spread(seeds=range(5)) -> tuple[float, float, int, int]:
    """Peak gold of the buggy run, and the step of the peak, for several seeds: (min, max) of each."""
    runs = [train(proxy_reward, seed=s) for s in seeds]
    peaks = [float(r["gold"].max()) for r in runs]
    steps = [int(r["gold"].argmax()) for r in runs]
    return min(peaks), max(peaks), min(steps), max(steps)


def detect_prob(hack_rate: float, n: int) -> float:
    """Probability that n random samples contain at least one hack."""
    return 1 - (1 - hack_rate) ** n


def spot_checks(hack_rate: float, confidence: float = 0.95) -> int:
    """Smallest n with 1 - (1 - h)^n >= confidence."""
    return math.ceil(math.log(1 - confidence) / math.log(1 - hack_rate))


# --- Figures -------------------------------------------------------------------------------

def plot_run(run: dict[str, np.ndarray], alarm: int) -> None:
    steps = np.arange(len(run["proxy"]))

    def draw(ax, theme):
        ax.plot(steps, run["proxy"], color=theme["series"][0], label="proxy (buggy checker)")
        ax.plot(steps, run["partial"], color=theme["series"][1], label="gold: partial (fixed checker)")
        ax.plot(steps, run["gold"], color=theme["series"][2], label="gold: full answer correct")
        ax.axvline(alarm, color=theme["muted"], linestyle="--", linewidth=1)
        ax.text(alarm + 4, 0.9, f"alarm, step {alarm}", color=theme["ink2"], fontsize=9)
        ax.set_xlabel("training step")
        ax.set_ylabel("expected reward")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="center right", fontsize=9)

    save_figure(draw, PAGE, "proxy-vs-gold")


def plot_kl_frontier() -> None:
    betas = np.geomspace(0.03, 20, 200)
    d = np.array([math.sqrt(kl_to_start(kl_optimum(b))) for b in betas])
    gold = np.array([expected(sparse_reward, kl_optimum(b)) for b in betas])
    marks = [(b, math.sqrt(kl_to_start(kl_optimum(b))), expected(sparse_reward, kl_optimum(b)))
             for b in (1.0, KL_BETA, 0.1)]

    def draw(ax, theme):
        ax.plot(d, gold, color=theme["series"][2], label="gold: P(full answer correct)")
        for b, x, y in marks:
            ax.plot([x], [y], "o", color=theme["series"][0])
            ax.annotate(f"beta {b}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9,
                        color=theme["ink2"])
        ax.set_xlabel("distance from the start policy, sqrt(KL)")
        ax.set_ylabel("P(full answer correct)")
        ax.set_ylim(0, 0.035)
        ax.legend(loc="upper right", fontsize=9)

    save_figure(draw, PAGE, "kl-frontier")


def plot_mitigations(runs: dict[str, dict[str, np.ndarray]]) -> None:
    def draw(ax, theme):
        for color, (name, run) in zip(theme["series"], runs.items()):
            ax.plot(np.arange(len(run["gold"])), run["gold"], color=color, label=name)
        ax.set_xlabel("training step")
        ax.set_ylabel("P(full answer correct)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="center right", fontsize=9)

    save_figure(draw, PAGE, "mitigations")


def main() -> None:
    r3 = lambda x: round(float(x), 3)

    print("== 1. proxy vs gold on single answers")
    for answer in [TARGET, HACK, ("sat", "sat", "sat"), ("the", "cat", "the")]:
        print(" ".join(answer), "| proxy", r3(proxy_reward(answer)), "| partial", r3(partial_reward(answer)),
              "| full correct", sparse_reward(answer))
    bad = overpaid()
    print("answers over-paid by the buggy checker:", len(bad), "| top:", " ".join(bad[0]))
    print("checker test on the hack: buggy passes?", checker_test(proxy_reward),
          "| fixed passes?", checker_test(fixed_reward))
    print("start: proxy", r3(expected(proxy_reward, POLICY_LOGITS)), "partial", r3(expected(partial_reward, POLICY_LOGITS)),
          "gold", round(expected(sparse_reward, POLICY_LOGITS), 4), "P(hack)", round(answer_prob(HACK, POLICY_LOGITS), 5))

    run = train(proxy_reward)
    peak = int(np.argmax(run["gold"]))
    print("buggy run: peak gold", r3(run["gold"][peak]), "at step", peak, "| KL there", r3(run["kl"][peak]))
    print("seed spread (seeds 0-4): peak gold min, max, peak step min, max:",
          [r3(x) for x in seed_spread()])
    for s in (0, 10, 20, 50, 100, 200, STEPS):
        print(f"  step {s}: proxy {r3(run['proxy'][s])} partial {r3(run['partial'][s])} "
              f"gold {r3(run['gold'][s])} kl {r3(run['kl'][s])} P(hack) {r3(run['hack'][s])}")

    print("closed form (best policy for proxy - beta KL):")
    for beta in (0.1, KL_BETA, 1.0):
        z = kl_optimum(beta)
        print(f"  beta {beta}: P(the) {r3(softmax(z)[0, 0])} proxy {r3(expected(proxy_reward, z))} "
              f"gold {r3(expected(sparse_reward, z))} kl {r3(kl_to_start(z))} P(hack) {r3(answer_prob(HACK, z))}")

    b = best_beta()
    print("best beta", r3(b), "| gold there", r3(expected(sparse_reward, kl_optimum(b))),
          "| sqrt KL there", r3(math.sqrt(kl_to_start(kl_optimum(b)))))

    print("== 3. detection")
    g = gap(run)
    alarm = alarm_step(run)
    print("gap at step 0", r3(g[0]), "| alarm step", alarm, "| gap there", r3(g[alarm]),
          "| gold there", r3(run["gold"][alarm]), "| P(hack) there", r3(run["hack"][alarm]))
    print("spot checks for h = P(hack) at the alarm:", spot_checks(run["hack"][alarm]))
    print("P(see a hack) with 50 samples at h = 0.01:", r3(detect_prob(0.01, 50)))
    for h in (0.01, 0.05, 0.1):
        print(f"  h = {h}: n = {spot_checks(h)}")

    print("== 4. mitigation")
    kl_run = train(proxy_reward, beta=KL_BETA)
    fixed = train(fixed_reward)
    for name, r in (("buggy", run), ("buggy + KL", kl_run), ("fixed checker", fixed)):
        print(f"  {name}: final proxy {r3(r['proxy'][-1])} partial {r3(r['partial'][-1])} "
              f"gold {r3(r['gold'][-1])} kl {r3(r['kl'][-1])} max gold {r3(r['gold'].max())}")

    plot_run(run, alarm)
    plot_kl_frontier()
    plot_mitigations({"buggy checker": run, f"buggy checker + KL (beta {KL_BETA})": kl_run,
                      "fixed checker": fixed})


if __name__ == "__main__":
    main()
