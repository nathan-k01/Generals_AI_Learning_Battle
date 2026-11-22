import pickle
import numpy as np

# The game file lives in the Game package in your project. The fallback lets
# this file also run when generals_game.py sits in the same folder.
try:
    from Game.generals_game import Action
except ImportError:
    from generals_game import Action

# ---------------- model / search sizes ----------------
N_FEATURES = 24          # numbers describing ONE candidate move (see build_features)
HIDDEN = 16              # hidden-layer width of the policy network
MAX_CANDIDATES = 48      # pass + up to 47 sampled moves are scored each turn
TOP_BY_ARMY = 24         # of those, the moves from the biggest tiles are always kept

# ---------------- values the engine uses in observations ----------------
NEUTRAL = -1             # owner id of unclaimed tiles
UNKNOWN = -2             # owner/terrain id of fogged or never-seen tiles
T_MOUNTAIN, T_CASTLE, T_GENERAL = 1, 2, 3   # terrain ids (plain = 0)


# ============================================================
# 1. POLICY NETWORK
# ============================================================
class PolicyNet:
    """
    Tiny two-layer network shared by all moves:
        features (K moves x d) -> tanh hidden layer -> one score per move -> softmax
    The softmax output is the probability of choosing each candidate move.
    """

    def __init__(self, params=None, seed=None):
        # Fresh network: small random weights. Otherwise copy the given weights
        # (this is how worker processes and saved checkpoints rebuild a net).
        if params is None:
            rng = np.random.default_rng(seed)
            params = {
                "W1": rng.normal(0, 0.3, (N_FEATURES, HIDDEN)),   # features -> hidden
                "b1": np.zeros(HIDDEN),                           # hidden bias
                "w2": rng.normal(0, 0.3, HIDDEN),                 # hidden -> move score
            }
        self.params = {k: np.array(v, dtype=np.float64) for k, v in params.items()}

    def forward(self, F, temperature=1.0):
        """
        F: (K, N_FEATURES) matrix, one row per candidate move.
        Returns (hidden activations, probability of each move).
        Lower temperature makes the choice greedier (used when evaluating).
        """
        h = np.tanh(F @ self.params["W1"] + self.params["b1"])
        z = (h @ self.params["w2"]) / temperature      # raw score per move
        z = z - z.max()                                # numerical stability
        p = np.exp(z)
        p /= p.sum()                                   # softmax -> probabilities
        return h, p

    def grad_log_prob(self, F, h, p, a, weight, entropy_beta):
        """
        Gradient of   weight * log pi(a)  +  entropy_beta * H(pi)   w.r.t. the weights.

        `weight` is the normalised advantage: positive makes the chosen move
        more likely, negative makes it less likely. The entropy term nudges the
        policy to stay a bit random so it keeps exploring.
        Returns gradients for gradient ASCENT (the optimizer adds them).
        """
        logp = np.log(p + 1e-12)
        entropy = -(p * logp).sum()
        # d/dz of log pi(a) is (onehot(a) - p); scaled by the advantage
        dz = -p * weight
        dz[a] += weight
        # d/dz of the entropy bonus is -p * (log p + H)
        dz += entropy_beta * (-p * (logp + entropy))
        # backpropagate through the hidden layer
        w2 = self.params["w2"]
        dh = np.outer(dz, w2)
        dpre = dh * (1.0 - h * h)          # derivative of tanh
        return {"W1": F.T @ dpre, "b1": dpre.sum(axis=0), "w2": h.T @ dz}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))


class Adam:
    """Adam optimizer, written for gradient ASCENT (we maximise expected reward)."""

    def __init__(self, params, lr=0.005, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}   # running mean of grads
        self.v = {k: np.zeros_like(v) for k, v in params.items()}   # running mean of grad^2
        self.t = 0                                                  # step counter

    def ascend(self, params, grads):
        # Gradient clipping: shrink the whole update if its norm is too large,
        # so one unlucky batch can't wreck the network.
        norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)       # bias-corrected averages
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] += self.lr * mh / (np.sqrt(vh) + self.eps)   # in place, so the net updates


# ============================================================
# 2. PERCEPTION  (observation -> arrays -> per-move features)
# ============================================================
def _shift_sum(mask, include_diagonals):
    """
    For every cell, count how many neighbouring cells are True in `mask`.
    Pads the edges with zeros. With include_diagonals=True it counts all 8
    neighbours, otherwise just up/down/left/right.
    """
    p = np.pad(mask.astype(np.float32), 1)
    h, w = mask.shape
    total = (p[0:h, 1:w + 1] + p[2:h + 2, 1:w + 1] + p[1:h + 1, 0:w] + p[1:h + 1, 2:w + 2])
    if include_diagonals:
        total += (p[0:h, 0:w] + p[0:h, 2:w + 2] + p[2:h + 2, 0:w] + p[2:h + 2, 2:w + 2])
    return total


def parse_board(obs):
    """Convert the observation's list-of-dicts board into four numpy grids [y][x]."""
    board = obs["board"]
    return {
        "owner": np.array([[c["owner"] for c in row] for row in board]),
        "army": np.array([[c["army"] for c in row] for row in board]),
        "terrain": np.array([[c["terrain_id"] for c in row] for row in board]),
        "visible": np.array([[c["visible"] for c in row] for row in board]),
    }


def build_features(obs, board, actions, player_id):
    """
    Build one row of N_FEATURES numbers per action. actions[0] must be the pass action.
    Everything here comes from the fog-of-war observation, so the agent never
    sees more than a human player would.

    Feature index guide:
       0 army sent (log)            1 half-move flag
       2 target is friendly         3 target is visible neutral
       4 target is visible enemy    5 target is in fog
       6 target is a castle         7 target is a non-own general
       8 target is own general      9 attack would win (visible hostile)
      10 win margin                11 defender size (log)
      12 unexplored tiles near target
      13 source is own general     14 army left behind (log)
      15 moves closer to enemy     16 moves closer to unexplored land
      17 friendly target's share of the merged army
      18 target touches an enemy   19 source touches an enemy
      20 pass flag                 21-23 game context (turn, own army, own land)
    """
    owner, army, terrain, visible = board["owner"], board["army"], board["terrain"], board["visible"]
    height, width = owner.shape

    # --- whole-board helper masks, computed once per turn ---
    enemy_mask = visible & (owner >= 0) & (owner != player_id)        # enemy tiles we can see now
    enemy_general_mask = (terrain == T_GENERAL) & (owner != player_id)  # remembered generals that aren't ours
    target_mask = enemy_mask | enemy_general_mask
    targets = np.argwhere(target_mask)                  # (row=y, col=x) pairs to move toward
    unknown_mask = terrain == UNKNOWN                   # never-seen tiles
    unknowns = np.argwhere(unknown_mask)
    enemy_adjacent = _shift_sum(enemy_mask, False) > 0  # tiles touching a visible enemy
    unknown_3x3 = _shift_sum(unknown_mask, True) + unknown_mask   # unexplored tiles in each 3x3 block

    # --- context numbers identical for every move this turn ---
    stats = obs["stats"]
    turn_frac = obs["turn"] / max(1, obs["max_turns"])
    own_army_ctx = np.log1p(stats["total_army"]) / 8.0
    own_land_ctx = stats["territory"] / float(width * height)
    context = (turn_frac, own_army_ctx, own_land_ctx)

    def dist(points, pool):
        """Manhattan distance from each point to its nearest point in `pool`."""
        if len(pool) == 0:
            return None
        return np.abs(points[:, None, :] - pool[None, :, :]).sum(axis=2).min(axis=1)

    moves = actions[1:]
    F = np.zeros((len(actions), N_FEATURES), dtype=np.float32)
    F[0, 20] = 1.0            # row 0 is the pass action: only the pass flag + context are set
    F[0, 21:24] = context
    if not moves:
        return F

    # source/destination coordinates as (y, x) so they index the numpy grids directly
    src = np.array([[a.source[1], a.source[0]] for a in moves])
    dst = np.array([[a.destination[1], a.destination[0]] for a in moves])

    # "Does this move get closer to an enemy / to unexplored land?" (+1 closer, -1 farther)
    enemy_gain = None
    d_s, d_d = dist(src, targets), dist(dst, targets)
    if d_s is not None:
        enemy_gain = np.clip(d_s - d_d, -1, 1)
    unk_gain = None
    u_s, u_d = dist(src, unknowns), dist(dst, unknowns)
    if u_s is not None:
        unk_gain = np.clip(u_s - u_d, -1, 1)

    for i, a in enumerate(moves, start=1):
        sy, sx = src[i - 1]
        dy, dx = dst[i - 1]
        src_army = army[sy, sx]
        # mirrors the engine: half sends army//2, otherwise all but one unit
        sent = src_army // 2 if a.half else src_army - 1
        vis_d = visible[dy, dx]          # can we currently see the destination?
        o = owner[dy, dx]
        t = terrain[dy, dx]
        da = army[dy, dx] if vis_d else 0    # defender size is only known if visible
        hostile_visible = vis_d and o != player_id

        F[i, 0] = np.log1p(sent) / 5.0
        F[i, 1] = float(a.half)
        F[i, 2] = float(o == player_id)
        F[i, 3] = float(vis_d and o == NEUTRAL)
        F[i, 4] = float(vis_d and o >= 0 and o != player_id)
        F[i, 5] = float(not vis_d)
        F[i, 6] = float(t == T_CASTLE)
        F[i, 7] = float(t == T_GENERAL and o != player_id)
        F[i, 8] = float(t == T_GENERAL and o == player_id)
        if hostile_visible:
            F[i, 9] = float(sent > da)                                   # would we capture it?
            F[i, 10] = np.clip((sent - da) / (sent + da + 1.0), -1, 1)   # by how much
            F[i, 11] = np.log1p(da) / 5.0
        F[i, 12] = unknown_3x3[dy, dx] / 9.0
        F[i, 13] = float(terrain[sy, sx] == T_GENERAL)
        F[i, 14] = np.log1p(src_army - sent) / 5.0
        if enemy_gain is not None:
            F[i, 15] = enemy_gain[i - 1]
        if unk_gain is not None:
            F[i, 16] = unk_gain[i - 1]
        if o == player_id:
            F[i, 17] = army[dy, dx] / (sent + army[dy, dx] + 1.0)
        F[i, 18] = float(enemy_adjacent[dy, dx])
        F[i, 19] = float(enemy_adjacent[sy, sx])
        F[i, 21:24] = context
    return F


# ============================================================
# 3. THE AGENT
# ============================================================
class PolicyGradientAI:
    # Reward shaping: points earned per turn from changes in our own stats.
    LAND_W, CASTLE_W, ARMY_W = 1.0, 4.0, 0.01
    # Terminal reward added to the final decision of the game.
    WIN_BONUS, LOSS_PENALTY = 30.0, -30.0

    def __init__(self, net, player_id=0, training=True, seed=None, eval_temperature=0.25):
        self.net = net
        self.player_id = player_id
        self.training = training                    # True: sample from the policy (explore)
        self.eval_temperature = eval_temperature    # used instead when training=False
        self.rng = np.random.default_rng(seed)
        self.steps = []              # one dict per decision: features F, hidden h, probs p, choice a, reward r
        self.prev_stats = None       # our stats at the previous turn, to compute reward
        self.final_territory_share = 0.0
        self.outcome = None          # "win" / "loss" / "draw" / "eliminated"

    # ---------- candidate selection ----------
    def _select_candidates(self, legal):
        """
        Late in the game there can be hundreds of legal moves. Keep the moves from
        the largest tiles (usually the important ones) plus a random sample of the
        rest, so small frontier tiles still get a chance.
        """
        if len(legal) <= MAX_CANDIDATES - 1:
            return legal
        order = sorted(range(len(legal)), key=lambda i: self._src_army(legal[i]), reverse=True)
        keep = order[:TOP_BY_ARMY]
        rest = order[TOP_BY_ARMY:]
        extra = self.rng.choice(len(rest), size=MAX_CANDIDATES - 1 - TOP_BY_ARMY, replace=False)
        keep += [rest[j] for j in extra]
        return [legal[i] for i in keep]

    def _src_army(self, action):
        # Source tiles are always our own, so their army is visible in the observation.
        x, y = action.source
        return self._board["army"][y, x]

    # ---------- one turn ----------
    def take_turn(self, env):
        """Pick one move, submit it, and remember the decision for learning."""
        pid = self.player_id
        # Nothing to do if the game is over, we're eliminated, or we already submitted.
        if env.game_over or pid in env.eliminated_players or env.has_player_submitted(pid):
            return None

        obs = env.get_observation(pid)               # fog-of-war view only
        self._credit_reward(obs["stats"])            # reward for what the PREVIOUS move achieved
        self._board = parse_board(obs)

        legal = env.get_legal_actions(pid, include_pass=False)
        candidates = self._select_candidates(legal)
        actions = [Action.pass_action()] + candidates    # index 0 is always "pass"

        # No legal moves: submit a pass so the game never stalls waiting for us.
        if not candidates:
            env.submit_action(pid, actions[0])
            return actions[0]

        F = build_features(obs, self._board, actions, pid)
        temp = 1.0 if self.training else self.eval_temperature
        h, p = self.net.forward(F, temp)
        a = int(self.rng.choice(len(p), p=p))        # sample a move from the policy
        env.submit_action(pid, actions[a])
        # Store everything needed to compute the gradient later.
        self.steps.append({"F": F, "h": h, "p": p, "a": a, "r": 0.0})
        return actions[a]

    def _credit_reward(self, stats):
        """
        Shaped reward = change in our land, castles and army since the last turn,
        credited to the previous decision (that's the move that caused it).
        """
        if self.prev_stats is not None and self.steps:
            r = (self.LAND_W * (stats["territory"] - self.prev_stats["territory"])
                 + self.CASTLE_W * (stats["castles"] - self.prev_stats["castles"])
                 + self.ARMY_W * (stats["total_army"] - self.prev_stats["total_army"]))
            self.steps[-1]["r"] += r
        self.prev_stats = dict(stats)

    def finish_game(self, env):
        """Call once when the game is over: pays the last reward and records the outcome."""
        obs = env.get_observation(self.player_id)
        self._credit_reward(obs["stats"])
        if obs["winner"] == self.player_id:
            self.outcome, bonus = "win", self.WIN_BONUS
        elif obs["eliminated"]:
            self.outcome, bonus = "eliminated", self.LOSS_PENALTY
        elif obs["draw"]:
            self.outcome, bonus = "draw", 0.0
        else:
            self.outcome, bonus = "loss", self.LOSS_PENALTY
        if self.steps:
            self.steps[-1]["r"] += bonus
        board_cells = obs["width"] * obs["height"]
        self.final_territory_share = obs["stats"]["territory"] / board_cells

    # ---------- reporting (comparable across models) ----------
    def get_fitness(self, env):
        """Simple cross-model score: win = 1.0, draw/timeout = final land share, otherwise 0."""
        if self.outcome == "win":
            return 1.0
        if self.outcome == "draw":
            return self.final_territory_share
        return 0.0

    def print_fitness_breakdown(self):
        total = sum(s["r"] for s in self.steps)
        print("\n========== POLICY GRADIENT AI ==========")
        print(f"Outcome:            {self.outcome}")
        print(f"Decisions made:     {len(self.steps)}")
        print(f"Shaped return:      {total:.2f}")
        print(f"Final land share:   {self.final_territory_share:.3f}")
        print("========================================\n")
