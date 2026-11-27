import numpy as np


try:
    from Game.generals_game import Action
except ImportError:
    from Game.generals_game import Action

NEUTRAL = -1          
UNKNOWN = -2            
T_CASTLE, T_GENERAL = 2, 3   

DEFAULT_WEIGHTS = {
    "general": 1000.0,      
    "enemy": 20.0,           
    "neutral": 10.0,        
    "castle": 15.0,          
    "explore": 3.0,          
    "losing_attack": -8.0,   
    "toward_enemy": 0.6,     
    "toward_unknown": 0.3,   
}
MIN_TRANSFER = 3      


def _parse_board(obs):
    """Convert the observation's list-of-dicts board into numpy grids [y][x] (single pass)."""
    board = obs["board"]
    cells = np.array([(c["owner"], c["army"], c["terrain_id"], c["visible"]) for row in board for c in row])
    cells = cells.reshape(len(board), len(board[0]), 4)
    return {
        "owner": cells[:, :, 0],
        "army": cells[:, :, 1],
        "terrain": cells[:, :, 2],
        "visible": cells[:, :, 3].astype(bool),
    }


def _distance_field(shape, pool):

    if len(pool) == 0:
        return None
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    cells = np.stack([ys.ravel(), xs.ravel()], axis=1)
    d = np.abs(cells[:, None, :] - pool[None, :, :]).sum(axis=2).min(axis=1)
    return d.reshape(shape)


class GreedyAI:
    def __init__(self, weights=None, player_id=0, seed=None):
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.player_id = player_id
        self.rng = np.random.default_rng(seed)  
        self.decisions = 0
        self.final_territory_share = 0.0
        self.outcome = None       

    # ---------- one turn ----------
    def take_turn(self, env):
        """Score every legal move, submit the best one."""
        pid = self.player_id
        if env.game_over or pid in env.eliminated_players or env.has_player_submitted(pid):
            return None

        obs = env.get_observation(pid)
        legal = env.get_legal_actions(pid, include_pass=False)
        if not legal:
            action = Action.pass_action()
            env.submit_action(pid, action)
            return action

        scores = self._score_moves(obs, legal)
        best = np.flatnonzero(scores == scores.max())
        action = legal[int(self.rng.choice(best))]  
        env.submit_action(pid, action)
        self.decisions += 1
        return action

    def _score_moves(self, obs, legal):
        pid = self.player_id
        b = _parse_board(obs)
        owner, army, terrain, visible = b["owner"], b["army"], b["terrain"], b["visible"]
        w = self.w

        enemy_mask = visible & (owner >= 0) & (owner != pid)
        general_mask = (terrain == T_GENERAL) & (owner != pid)
        targets = np.argwhere(enemy_mask | general_mask)    
        unknowns = np.argwhere(terrain == UNKNOWN)
        enemy_dist = _distance_field(owner.shape, targets)
        unknown_dist = _distance_field(owner.shape, unknowns)

        sx = np.fromiter((a.source[0] for a in legal), int, len(legal))
        sy = np.fromiter((a.source[1] for a in legal), int, len(legal))
        dx = np.fromiter((a.destination[0] for a in legal), int, len(legal))
        dy = np.fromiter((a.destination[1] for a in legal), int, len(legal))
        half = np.fromiter((a.half for a in legal), bool, len(legal))

        src_army = army[sy, sx]
        sent = np.where(half, src_army // 2, src_army - 1)     # same rule as the engine
        o, t, vis, defenders = owner[dy, dx], terrain[dy, dx], visible[dy, dx], army[dy, dx]

        move_value = np.zeros(len(legal))
        if enemy_dist is not None:
            move_value += w["toward_enemy"] * np.clip(enemy_dist[sy, sx] - enemy_dist[dy, dx], -1, 1)
        if unknown_dist is not None:
            move_value += w["toward_unknown"] * np.clip(unknown_dist[sy, sx] - unknown_dist[dy, dx], -1, 1)

        fog_score = w["explore"] + 0.01 * sent
        friendly_score = move_value - 1.0 * (sent < MIN_TRANSFER) + 0.01 * sent
        capture = (np.where(o == NEUTRAL, w["neutral"], w["enemy"])
                   + w["castle"] * (t == T_CASTLE)
                   + w["general"] * (t == T_GENERAL)
                   - 0.1 * defenders
                   - 1.0 * half)                              
        attack_score = np.where(sent > defenders, capture, w["losing_attack"])

        return np.where(~vis, fog_score, np.where(o == pid, friendly_score, attack_score))

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

AI = GreedyAI