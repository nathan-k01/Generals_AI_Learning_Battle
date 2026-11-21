import math
from dataclasses import dataclass
import numpy as np

PLAYER_ID = 0


@dataclass
class TurnRecord:
    turn: int
    strategy_id: int
    action: object
    territory_before: int
    territory_after: int
    army_before: int
    army_after: int
    castles_before: int
    castles_after: int
    enemy_land_before: int
    enemy_land_after: int
    enemy_army_before: int
    enemy_army_after: int
    eliminations_before: int
    eliminations_after: int


# GAME METRICS
class GameMetrics:
    def __init__(self):
        self.records = []
        self.expand_fitness = 0.0
        self.economy_fitness = 0.0
        self.attack_fitness = 0.0
        self.defense_fitness = 0.0
        self.assassin_fitness = 0.0
        self.overlord_fitness = 0.0
        self.strategy_usage = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.strategy_history = []


# GENERAL HELPERS
def get_enemy_stats(env, player_id):
    enemy_land = []
    enemy_army = []
    for pid in env.get_active_players():
        if pid == player_id:
            continue
        territory, army, _ = env.get_stats(pid)
        enemy_land.append(territory)
        enemy_army.append(army)
    return max(enemy_land, default=0), max(enemy_army, default=0)


def get_visible_enemy_tiles(obs):
    count = 0
    owner_id = obs["player_id"]
    for row in obs["board"]:
        for cell in row:
            if not cell["visible"]:
                continue
            owner = cell["owner"]
            if owner not in (-1, None, owner_id):
                count += 1
    return count


def get_neutral_tiles(obs):
    count = 0
    for row in obs["board"]:
        for cell in row:
            if not cell["visible"]:
                continue
            if cell["owner"] is None:
                count += 1
    return count


def get_visible_cities(obs):
    count = 0
    for row in obs["board"]:
        for cell in row:
            if cell["visible"] and cell["terrain"] == "castle":
                count += 1
    return count


# SNAPSHOTS
def create_snapshot(env, player_id):

    territory, army, castles = env.get_stats(player_id)
    enemy_land = 0
    enemy_army = 0
    for pid in env.get_active_players():
        if pid == player_id:
            continue
        t, a, _ = env.get_stats(pid)
        enemy_land += t
        enemy_army += a

    return {"territory": territory, "army": army, "castles": castles, "enemy_land": enemy_land,
            "enemy_army": enemy_army, "eliminations": len(env.eliminated_players)}


# CANDIDATE ACTION FEATURES
def get_action_features(env, player_id, action):
    if action.is_pass:
        return [0, 0, 0, 0]

    source_tile = env.tile(action.source)
    target_tile = env.tile(action.destination)

    if target_tile.owner == player_id:
        owner_type = 0

    elif target_tile.owner is None:
        owner_type = 1

    else:
        owner_type = 2

    return [float(source_tile.army), float(owner_type), float(int(target_tile.terrain)), float(int(action.half))]


def build_candidate_action_block(env, player_id, actions, limit=8):
    features = []
    for action in actions:
        features.extend(get_action_features(env, player_id, action))

    while len(features) < limit * 4:
        features.extend([0, 0, 0, 0])

    return features


def get_candidate_actions(env, player_id, limit=8):
    actions = env.get_legal_actions(player_id, include_pass=False)

    def score(action):
        source_tile = env.tile(action.source)
        target_tile = env.tile(action.destination)
        value = source_tile.army

        if target_tile.owner != player_id:
            value += 25

        return value

    actions.sort(key=score, reverse=True)
    return actions[:limit]


# AI CONTAINER
class AI:
    def __init__(self, genome_package):
        self.metrics = GameMetrics()
        self.overlord = OverlordAI(genome_package["overlord"])
        self.expand_ai = ExpandAI(genome_package["expand"])
        self.economy_ai = EconomyAI(genome_package["economy"])
        self.attack_ai = AttackAI(genome_package["attack"])
        self.defense_ai = DefenseAI(genome_package["defense"])
        self.assassin_ai = AssassinAI(genome_package["assassin"])
        self.metrics = GameMetrics()

    def take_turn(self, env):
        return take_turn(env=env, metrics=self.metrics, overlord=self.overlord, expand_ai=self.expand_ai,
                         economy_ai=self.economy_ai, attack_ai=self.attack_ai, defense_ai=self.defense_ai,
                         assassin_ai=self.assassin_ai)

    def get_overlord_fitness(self, env):
        return finalize_overlord_fitness(env, self.metrics)

    def print_fitness_breakdown(self):
        print("\n========== FITNESS ==========")
        print(f"Expand:    {self.metrics.expand_fitness:.2f}")
        print(f"Economy:   {self.metrics.economy_fitness:.2f}")
        print(f"Attack:    {self.metrics.attack_fitness:.2f}")
        print(f"Defense:   {self.metrics.defense_fitness:.2f}")
        print(f"Assassin:  {self.metrics.assassin_fitness:.2f}")
        print(f"Overlord:  {self.metrics.overlord_fitness:.2f}")
        print("Strategy Usage:", self.metrics.strategy_usage)
        print("=============================\n")


# NEURAL NETWORK
class NeuralNet:
    def __init__(self, genome, input_size, hidden_size, output_size):
        idx = 0
        w1 = (input_size * hidden_size)
        b1 = hidden_size
        w2 = (hidden_size * output_size)
        b2 = output_size
        self.W1 = np.array(genome[idx:idx + w1]).reshape(hidden_size, input_size)
        idx += w1
        self.b1 = np.array(genome[idx:idx + b1])
        idx += b1
        self.W2 = np.array(genome[idx:idx + w2]).reshape(output_size, hidden_size)
        idx += w2
        self.b2 = np.array(genome[idx:idx + b2])

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        hidden = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ hidden + self.b2


# STRATEGY BASE
class StrategyAI:
    INPUT_SIZE = 40
    ACTION_CHOICES = 8

    def __init__(self, genome):
        self.np_array = None
        self.brain = NeuralNet(genome=genome, input_size=self.INPUT_SIZE,
                               hidden_size=16, output_size=self.ACTION_CHOICES)

    def choose_move(self, state, candidate_actions):
        outputs = self.brain.forward(state)
        if len(candidate_actions) == 0:
            print("CHOOSE MOVE RETURNED NONE")
            return None

        choice = int(np.argmax(outputs))
        choice = min(choice, len(candidate_actions) - 1)

        return candidate_actions[choice]


# SPECIALIST AIs
class ExpandAI(StrategyAI):
    def build_ai_state(self,env, candidate_actions):
        obs = env.get_observation(PLAYER_ID)
        territory, army, castles = env.get_stats(PLAYER_ID)
        state = [territory, get_neutral_tiles(obs), get_visible_enemy_tiles(obs),
                 army, castles, env.turn, env.width, env.height]
        state.extend(build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


class EconomyAI(StrategyAI):
    def build_ai_state(self, env, candidate_actions):
        territory, army, castles = env.get_stats(PLAYER_ID)
        largest_enemy_land, largest_enemy_army = get_enemy_stats(env, PLAYER_ID)
        state = [army, territory, castles, largest_enemy_army, army / max(1, largest_enemy_army),
                 env.turn, len(env.get_active_players()), territory + castles]
        state.extend(build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


class AttackAI(StrategyAI):
    def build_ai_state(self, env, candidate_actions):
        territory, army, castles = (env.get_stats(PLAYER_ID))
        obs = env.get_observation(PLAYER_ID)
        largest_enemy_land, largest_enemy_army = (get_enemy_stats(env,PLAYER_ID))
        state = [army, territory, largest_enemy_army, largest_enemy_land, get_visible_enemy_tiles(obs),
                 army / max(1, largest_enemy_army), len(env.get_active_players()), env.turn]
        state.extend(build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


class DefenseAI(StrategyAI):
    def build_ai_state(self, env, candidate_actions):
        territory, army, castles = (env.get_stats(PLAYER_ID))
        largest_enemy_land, largest_enemy_army = (get_enemy_stats(env, PLAYER_ID))
        state = [territory, army, castles, largest_enemy_army, largest_enemy_land,
                 territory - largest_enemy_land, army - largest_enemy_army, env.turn]
        state.extend(build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


class AssassinAI(StrategyAI):
    def build_ai_state(self, env, candidate_actions):
        territory, army, castles = env.get_stats(PLAYER_ID)
        obs = env.get_observation(PLAYER_ID)
        state = [army, territory, get_visible_enemy_tiles(obs), castles, len(env.get_active_players()),
                 env.turn, env.width, env.height]
        state.extend(build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


# OVERLORD
class OverlordAI:
    INPUT_SIZE = 9

    def __init__(self, genome):
        self.np_array = None
        self.brain = NeuralNet(genome, self.INPUT_SIZE, 20, 5)

    def build_overlord_state(self, env):
        obs = env.get_observation(PLAYER_ID)

        territory, army, castles = env.get_stats(PLAYER_ID)
        largest_enemy_land, largest_enemy_army = get_enemy_stats(env, PLAYER_ID)

        self.np_array = np.array([env.turn, territory, army, castles, len(env.get_active_players()),
                                  largest_enemy_land, largest_enemy_army, get_visible_enemy_tiles(obs),
                                  get_neutral_tiles(obs)], dtype=np.float32)

    def choose_strategy(self, state):
        outputs = self.brain.forward(state)
        return int(np.argmax(outputs))


# FITNESS SYSTEM
def process_record(metrics, record):
    land_delta = record.territory_after - record.territory_before
    army_delta = record.army_after - record.army_before
    castle_delta = record.castles_after - record.castles_before
    enemy_land_delta = record.enemy_land_before - record.enemy_land_after
    enemy_army_delta = record.enemy_army_before - record.enemy_army_after
    elimination_delta = record.eliminations_after - record.eliminations_before

    sid = record.strategy_id

    if sid == 0:
        metrics.expand_fitness += land_delta * 10
        metrics.expand_fitness += castle_delta * 50

    elif sid == 1:
        metrics.economy_fitness += army_delta * 5
        metrics.economy_fitness += castle_delta * 250

    elif sid == 2:
        metrics.attack_fitness += enemy_land_delta * 15
        metrics.attack_fitness += enemy_army_delta * 2
        metrics.attack_fitness += elimination_delta * 5000

    elif sid == 3:
        metrics.defense_fitness += max(0, land_delta)
        metrics.defense_fitness -= max(0, -land_delta) * 10

    elif sid == 4:
        metrics.assassin_fitness += elimination_delta * 10000


def finalize_overlord_fitness(env, metrics):
    specialist_score = (metrics.expand_fitness + metrics.economy_fitness + metrics.attack_fitness +
                        metrics.defense_fitness + metrics.assassin_fitness)
    fitness = specialist_score

    if env.winner == PLAYER_ID:
        fitness += 10000

    elif PLAYER_ID not in env.eliminated_players:
        fitness += 3000

    total_turns = sum(metrics.strategy_usage.values())

    if total_turns > 0:
        entropy = 0
        for turns in metrics.strategy_usage.values():
            if turns == 0:
                continue

            p = turns / total_turns
            entropy -= (p * math.log(p))
        # reward diversity
        fitness += entropy * 500

    switches = 0
    for i in range(1, len(metrics.strategy_history)):
        if metrics.strategy_history[i] != metrics.strategy_history[i - 1]:
            switches += 1

    # punish excessive switching
    fitness -= switches * 5
    metrics.overlord_fitness = fitness
    return fitness


# TURN EXECUTION
def take_turn(env, metrics, overlord, expand_ai, economy_ai, attack_ai, defense_ai, assassin_ai):
    candidate_actions = get_candidate_actions(env, PLAYER_ID, 8)

    if len(candidate_actions) == 0:
        print("TAKE TURN RETURNED NONE")
        return None

    before = create_snapshot(env, PLAYER_ID)
    overlord.build_overlord_state(env)
    strategy_id = overlord.choose_strategy(overlord.np_array)
    strategy_lookup = {0: expand_ai, 1: economy_ai, 2: attack_ai, 3: defense_ai, 4: assassin_ai}
    strategy_ai = strategy_lookup[strategy_id]
    strategy_ai.build_ai_state(env, candidate_actions)
    action = strategy_ai.choose_move(strategy_ai.np_array, candidate_actions)

    if action is None:
        print("TAKE TURN RETURNED NONE")
        return None

    metrics.strategy_usage[strategy_id] += 1
    metrics.strategy_history.append(strategy_id)
    env.submit_action(PLAYER_ID, action)
    after = create_snapshot(env, PLAYER_ID)

    record = TurnRecord(
        turn=env.turn, strategy_id=strategy_id, action=action,
        territory_before=before["territory"], territory_after=after["territory"],
        army_before=before["army"], army_after=after["army"],
        castles_before=before["castles"], castles_after=after["castles"],
        enemy_land_before=before["enemy_land"], enemy_land_after=after["enemy_land"],
        enemy_army_before=before["enemy_army"], enemy_army_after=after["enemy_army"],
        eliminations_before=before["eliminations"], eliminations_after=after["eliminations"]
    )
    metrics.records.append(record)
    process_record(metrics, record)
    return action
