import csv
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from Game import generals_game
from AIs.Kal_AI.AI import PolicyGradientAI, PolicyNet, Adam

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

GAMMA = 0.99          # discount: how much later rewards count compared to immediate ones
ENTROPY_BETA = 0.01   # strength of the "stay a little random" bonus
CHECKPOINT = "pg_latest.pkl"
BEST = "pg_best.pkl"
LOG_FILE = "pg_training_log.csv"


# ============================================================
# GAME PLAYING (runs inside worker processes)
# ============================================================
def random_ai_move(env, player_id):
    """Opponent used for training/eval: picks any legal action (including pass) at random."""
    legal = env.get_legal_actions(player_id, include_pass=True)
    if legal:
        env.submit_action(player_id, random.choice(legal))


def play_game(args):
    """
    Play one full game and return what each learning agent experienced.

    args = (weights, seed, mode, training)
      mode "random": learner is player 0, players 1 and 2 are random bots
                     (same setup as main.py).
      mode "self":   all three players share the same weights (self-play).
      training:      True samples moves from the policy (exploration);
                     False plays more greedily for evaluation.
    """
    params, seed, mode, training = args
    env = generals_game.GeneralsEnvironment(seed=seed)   # same seed => same map
    net = PolicyNet(params)
    pids = [0] if mode == "random" else [0, 1, 2]
    agents = [PolicyGradientAI(net, pid, training=training, seed=seed * 3 + pid) for pid in pids]
    random.seed(seed)   # makes the random bots reproducible too

    # `guard` stops an endless loop if something ever fails to advance the turn.
    guard = 0
    while not env.game_over and guard < env.max_turns * 3 + 10:
        guard += 1
        if mode == "random":
            random_ai_move(env, 1)
            random_ai_move(env, 2)
        for agent in agents:
            agent.take_turn(env)       # the last submission resolves the turn

    for agent in agents:
        agent.finish_game(env)         # pay terminal reward, record outcome

    # Return plain data (lists/arrays) so it can travel back from the worker process.
    episodes = []
    for agent in agents:
        episodes.append({
            "F": [s["F"] for s in agent.steps], "h": [s["h"] for s in agent.steps],
            "p": [s["p"] for s in agent.steps], "a": [s["a"] for s in agent.steps],
            "r": [s["r"] for s in agent.steps],
            "outcome": agent.outcome, "fitness": agent.get_fitness(env),
            "share": agent.final_territory_share,
        })
    return episodes


# ============================================================
# LEARNING (main process)
# ============================================================
def discounted_returns(rewards, gamma=GAMMA):
    """G[t] = r[t] + gamma*r[t+1] + gamma^2*r[t+2] + ...  (computed backwards)."""
    G = np.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def policy_gradient_step(net, optimizer, episodes):
    """
    One REINFORCE update from a batch of finished games.

    Each decision is weighted by how much better or worse its return was than
    the batch average ("advantage"): good outcomes make those moves more likely,
    bad outcomes make them less likely. Returns the average policy entropy so
    you can watch whether the agent is still exploring.
    """
    returns = [discounted_returns(ep["r"]) for ep in episodes]
    flat = np.concatenate([g for g in returns if len(g)])
    if len(flat) < 2:
        return 0.0
    # Normalising keeps the update size stable no matter how big the rewards are.
    mean, std = flat.mean(), flat.std() + 1e-8

    total = {k: np.zeros_like(v) for k, v in net.params.items()}
    n_steps = 0
    entropy_sum = 0.0
    for ep, G in zip(episodes, returns):
        for t in range(len(G)):
            adv = (G[t] - mean) / std
            g = net.grad_log_prob(ep["F"][t], ep["h"][t], ep["p"][t], ep["a"][t], adv, ENTROPY_BETA)
            for k in total:
                total[k] += g[k]
            p = ep["p"][t]
            entropy_sum += -(p * np.log(p + 1e-12)).sum()
            n_steps += 1
    for k in total:
        total[k] /= max(1, n_steps)     # average over all decisions in the batch
    optimizer.ascend(net.params, total)
    return entropy_sum / max(1, n_steps)


def evaluate(pool, net, seeds):
    """Score the current weights vs random bots on fixed seeds (not used for learning)."""
    results = pool.map(play_game, [({k: v for k, v in net.params.items()}, s, "random", False) for s in seeds])
    eps = [r[0] for r in results]
    return {
        "win_rate": float(np.mean([e["outcome"] == "win" for e in eps])),
        "fitness": float(np.mean([e["fitness"] for e in eps])),
        "land_share": float(np.mean([e["share"] for e in eps])),
    }


def train(iterations=500, games_per_iter=12, mode="random", eval_every=10, eval_games=20, lr=0.005, resume=True):
    # Resume from the last checkpoint if there is one, otherwise start fresh.
    net = PolicyNet(PolicyNet.load(CHECKPOINT).params) if resume and os.path.exists(CHECKPOINT) else PolicyNet(seed=0)
    optimizer = Adam(net.params, lr=lr)
    eval_seeds = list(range(10_000, 10_000 + eval_games))   # fixed seeds => comparable learning curve
    best = -1.0
    new_log = not os.path.exists(LOG_FILE)

    with Pool() as pool, open(LOG_FILE, "a", newline="") as log:
        writer = csv.writer(log)
        if new_log:
            writer.writerow(["iter", "train_win_rate", "train_land_share", "mean_entropy",
                             "eval_win_rate", "eval_fitness", "eval_land_share"])
        for it in range(iterations):
            t0 = time.time()
            # 1) play a batch of games in parallel with the current weights
            seeds = [random.randrange(1 << 30) for _ in range(games_per_iter)]
            batches = pool.map(play_game, [(net.params, s, mode, True) for s in seeds])
            episodes = [ep for batch in batches for ep in batch]
            # 2) learn from them
            entropy = policy_gradient_step(net, optimizer, episodes)

            train_win = float(np.mean([e["outcome"] == "win" for e in episodes]))
            train_share = float(np.mean([e["share"] for e in episodes]))
            row = [it, train_win, train_share, entropy, "", "", ""]

            # 3) every few iterations, measure progress on the fixed eval seeds
            if it % eval_every == 0:
                ev = evaluate(pool, net, eval_seeds)
                row[4:] = [ev["win_rate"], ev["fitness"], ev["land_share"]]
                print(f"[{it:4d}] EVAL win={ev['win_rate']:.2f} fitness={ev['fitness']:.3f} "
                      f"land={ev['land_share']:.3f}")
                if ev["fitness"] > best:
                    best = ev["fitness"]
                    net.save(BEST)
            net.save(CHECKPOINT)
            writer.writerow(row)
            log.flush()
            print(f"[{it:4d}] train win={train_win:.2f} land={train_share:.3f} "
                  f"entropy={entropy:.3f} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    train()
