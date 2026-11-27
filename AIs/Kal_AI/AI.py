"""
Kal_AI  -  a Greedy AI for Generals Arena.

Every turn it scores each legal move with a simple hand-written rule and plays
the highest-scoring one. It has no neural network. Its scoring weights can be tuned by
Kal_AI/trainer.py (evolution); with the default weights it is a fixed baseline.

Scoring, from most to least valuable:
  1. capture an enemy general we can beat       (huge)
  2. capture an enemy tile / castle we can beat (high)
  3. capture a neutral tile / castle we can beat (medium)
  4. step into fog to explore                   (small)
  5. move our own army toward the enemy / unexplored land (small)
Attacks that would LOSE are penalised so the AI doesn't throw armies away.

It only uses get_observation() and get_legal_actions(), so it respects fog of war.

Public interface (same as the other AIs in this project):
    from AIs.Kal_AI.AI import AI
    ai = AI(weights=None, player_id=1)   # weights: optional dict from the trainer
    ai.take_turn(env)
    ai.finish_game(env)
    ai.get_fitness(env)
    ai.print_fitness_breakdown()
"""
import numpy as np

# The game file lives in the Game package in your project. The fallback lets
# this file also run when generals_game.py sits in the same folder.
try:
    from Game.generals_game import Action
except ImportError:
    from Game.generals_game import Action

NEUTRAL = -1             # owner id of unclaimed tiles
UNKNOWN = -2             # owner/terrain id of fogged or never-seen tiles
T_CASTLE, T_GENERAL = 2, 3   # terrain ids (plain = 0, mountain = 1)

# ---------------- scoring weights (the trainer tunes these) ----------------
DEFAULT_WEIGHTS = {
    "general": 1000.0,       # capturing an enemy general (wins the game)
    "enemy": 20.0,           # capturing any enemy tile
    "neutral": 10.0,         # capturing a neutral tile
    "castle": 15.0,          # extra for capturing a castle
    "explore": 3.0,          # moving into fog
    "losing_attack": -8.0,   # attack that would not win
    "toward_enemy": 0.6,     # friendly move that gets closer to an enemy
    "toward_unknown": 0.3,   # friendly move that gets closer to unexplored land
}
MIN_TRANSFER = 3         # don't shuffle stacks smaller than this between own tiles


def _parse_board(obs):
    """Convert the observation's list-of-dicts board into numpy grids [y][x]."""
    board = obs["board"]
    return {
        "owner": np.array([[c["owner"] for c in row] for row in board]),
        "army": np.array([[c["army"] for c in row] for row in board]),
        "terrain": np.array([[c["terrain_id"] for c in row] for row in board]),
        "visible": np.array([[c["visible"] for c in row] for row in board]),
    }


def _nearest_distance(points, pool):
    """Manhattan distance from each point to its closest point in `pool` (None if pool is empty)."""
    if len(pool) == 0:
        return None
    return np.abs(points[:, None, :] - pool[None, :, :]).sum(axis=2).min(axis=1)


class GreedyAI:
    def __init__(self, weights=None, player_id=0, seed=None):
        # weights=None uses the hand-picked defaults; the trainer passes tuned ones in
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.player_id = player_id
        self.rng = np.random.default_rng(seed)   # only used to break ties between equal moves
        self.decisions = 0
        self.final_territory_share = 0.0
        self.outcome = None          # "win" / "loss" / "draw" / "eliminated"



    # ---------- one turn ----------
    def take_turn(self, env):
        """Score every legal move, submit the best one."""
        pid = self.player_id
        # Nothing to do if the game is over, we're eliminated, or we already submitted.
        if env.game_over or pid in env.eliminated_players or env.has_player_submitted(pid):
            return None

        obs = env.get_observation(pid)               # fog-of-war view only


        legal = env.get_legal_actions(pid, include_pass=False)
        if not legal:                                # no moves: pass so the game never stalls
            action = Action.pass_action()
            env.submit_action(pid, action)
            return action

        scores = self._score_moves(obs, legal)
        best = np.flatnonzero(scores == scores.max())
        action = legal[int(self.rng.choice(best))]   # random choice among exact ties
        env.submit_action(pid, action)
        self.decisions += 1
        return action

    def _score_moves(self, obs, legal):
        pid = self.player_id
        b = _parse_board(obs)
        owner, army, terrain, visible = b["owner"], b["army"], b["terrain"], b["visible"]

        # Tiles to move toward: visible enemy tiles and remembered enemy generals.
        enemy_mask = visible & (owner >= 0) & (owner != pid)
        general_mask = (terrain == T_GENERAL) & (owner != pid)
        targets = np.argwhere(enemy_mask | general_mask)      # (y, x) pairs
        unknowns = np.argwhere(terrain == UNKNOWN)

        src = np.array([[a.source[1], a.source[0]] for a in legal])        # (y, x)
        dst = np.array([[a.destination[1], a.destination[0]] for a in legal])

        # +1 if the move gets closer to an enemy / unexplored tile, -1 if farther
        enemy_gain = unknown_gain = None
        d_s, d_d = _nearest_distance(src, targets), _nearest_distance(dst, targets)
        if d_s is not None:
            enemy_gain = np.clip(d_s - d_d, -1, 1)
        u_s, u_d = _nearest_distance(src, unknowns), _nearest_distance(dst, unknowns)
        if u_s is not None:
            unknown_gain = np.clip(u_s - u_d, -1, 1)

        w = self.w
        scores = np.zeros(len(legal))
        for i, a in enumerate(legal):
            sy, sx = src[i]
            dy, dx = dst[i]
            src_army = army[sy, sx]
            sent = src_army // 2 if a.half else src_army - 1     # same rule as the engine
            o, t = owner[dy, dx], terrain[dy, dx]

            if not visible[dy, dx]:
                # Fogged destination: we can't see its owner or army, so just explore.
                scores[i] = w["explore"] + 0.01 * sent
            elif o == pid:
                # Friendly transfer: only worth it for decent stacks heading somewhere useful.
                s = 0.0
                if enemy_gain is not None:
                    s += w["toward_enemy"] * enemy_gain[i]
                if unknown_gain is not None:
                    s += w["toward_unknown"] * unknown_gain[i]
                if sent < MIN_TRANSFER:
                    s -= 1.0
                scores[i] = s + 0.01 * sent
            else:
                # Neutral or enemy tile: attack only if we'd actually win the fight.
                defenders = army[dy, dx]
                if sent > defenders:
                    s = (w["neutral"] if o == NEUTRAL else w["enemy"])
                    if t == T_CASTLE:
                        s += w["castle"]
                    if t == T_GENERAL:
                        s += w["general"]
                    s -= 0.1 * defenders                 # cheaper captures first
                    if a.half:
                        s -= 1.0                         # sending half is usually wasteful here
                    scores[i] = s
                else:
                    scores[i] = w["losing_attack"]
        return scores

    # ---------- end of game / reporting ----------
    def finish_game(self, env):
        """Call once when the game is over: records the outcome and final land share."""
        obs = env.get_observation(self.player_id)

        if obs["winner"] == self.player_id:
            self.outcome = "win"
        elif obs["eliminated"]:
            self.outcome = "eliminated"
        elif obs["draw"]:
            self.outcome = "draw"
        else:
            self.outcome = "loss"
        self.final_territory_share = obs["stats"]["territory"] / (obs["width"] * obs["height"])

    def get_fitness(self, env):
        """Same cross-model score as before: win = 1.0, draw/timeout = final land share, otherwise 0."""
        if self.outcome == "win":
            return 1.0
        if self.outcome == "draw":
            return self.final_territory_share
        return 0.0

    def print_fitness_breakdown(self):
        print("\n========== GREEDY AI (Kal_AI) ==========")
        print(f"Outcome:            {self.outcome}")
        print(f"Decisions made:     {self.decisions}")
        print(f"Final land share:   {self.final_territory_share:.3f}")
        print("========================================\n")


# Short name so it can be imported like the other AIs:  from AIs.Kal_AI.AI import AI
AI = GreedyAI