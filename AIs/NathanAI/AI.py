import math
import random
from . import helpers
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
        self.explore_fitness = 0.0
        self.build_fitness = 0.0
        self.combat_fitness = 0.0
        self.overlord_fitness = 0.0
        self.strategy_usage = {0: 0, 1: 0, 2: 0}
        self.strategy_history = []
        self.pending_record = None


# AI CONTAINER
class AI:

    def __init__(self, genome_package):
        self.overlord = OverlordAI(genome_package["overlord"])
        self.explore_ai = ExplorerAI(genome_package["explore"])
        self.build_ai = BuildAI(genome_package["build"])
        self.combat_ai = CombatAI(genome_package["combat"])
        self.metrics = GameMetrics()

    def take_turn(self, env):
        return take_turn(env=env, metrics=self.metrics, overlord=self.overlord, explore_ai=self.explore_ai,
                         build_ai=self.build_ai, combat_ai=self.combat_ai)

    def get_overlord_fitness(self, env):
        return finalize_overlord_fitness(env, self.metrics)

    def print_fitness_breakdown(self):
        print("\n========== FITNESS ==========")
        print(f"Explore:  {self.metrics.explore_fitness:.2f}")
        print(f"Build:    {self.metrics.build_fitness:.2f}")
        print(f"Combat:   {self.metrics.combat_fitness:.2f}")
        print(f"Overlord: {self.metrics.overlord_fitness:.2f}")

        print(f"Strategy Usage: {self.metrics.strategy_usage}")

        total_turns = sum(self.metrics.strategy_usage.values())
        if total_turns > 0:
            print(f"Explore % : {100 * self.metrics.strategy_usage[0] / total_turns:.1f}")
            print(f"Build %   : {100 * self.metrics.strategy_usage[1] / total_turns:.1f}")
            print(f"Combat %  : {100 * self.metrics.strategy_usage[2] / total_turns:.1f}")
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
    INPUT_SIZE = 80
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
    def build_ai_state(self, env, candidate_actions):
        obs = env.get_observation(PLAYER_ID)
        territory, army, castles = env.get_stats(PLAYER_ID)
        state = [territory, helpers.get_neutral_tiles(obs), helpers.get_visible_enemy_tiles(obs),
                 army, castles, env.turn, env.width, env.height]
        state.extend(helpers.build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


class ExplorerAI(StrategyAI):
    def build_ai_state(self, env, candidate_actions):
        obs = env.get_observation(PLAYER_ID)
        territory, army, castles = (env.get_stats(PLAYER_ID))
        state = [territory, army, castles, helpers.get_neutral_tiles(obs), helpers.get_visible_enemy_tiles(obs),
                 helpers.get_visible_cities(obs), env.turn, len(env.get_active_players())]
        state.extend(helpers.build_candidate_action_block(env, PLAYER_ID, candidate_actions))
        self.np_array = np.array(state, dtype=np.float32)


class BuildAI(ExplorerAI):
    pass


class CombatAI(ExplorerAI):
    pass


# OVERLORD
class OverlordAI:
    INPUT_SIZE = 14

    def __init__(self, genome):
        self.np_array = None
        self.brain = NeuralNet(genome, self.INPUT_SIZE, 24, 3)

    def build_overlord_state(self, env):
        obs = env.get_observation(PLAYER_ID)
        territory, army, castles = (env.get_stats(PLAYER_ID))
        largest_enemy_land, largest_enemy_army = (helpers.get_enemy_stats(env, PLAYER_ID))
        army_ratio = (army / max(1, largest_enemy_army))
        territory_ratio = (territory / max(1, largest_enemy_land))

        self.np_array = np.array([
            env.turn, territory, army, castles, largest_enemy_land, largest_enemy_army, army_ratio, territory_ratio,
            helpers.get_visible_enemy_tiles(obs), helpers.get_neutral_tiles(obs), helpers.get_visible_cities(obs),
            helpers.get_frontline_tiles(env, PLAYER_ID), helpers.get_largest_stack(env, PLAYER_ID),
            len(env.get_active_players())], dtype=np.float32)

    def choose_strategy(self, state):
        if random.random() < 0.20:
            return random.randint(0, 2)

        outputs = self.brain.forward(state)
        return int(np.argmax(outputs))


# FITNESS SYSTEM
def process_record(metrics, record):
    land_delta = (record.territory_after - record.territory_before)
    castle_delta = (record.castles_after - record.castles_before)
    enemy_land_delta = (record.enemy_land_before - record.enemy_land_after)
    enemy_army_delta = (record.enemy_army_before - record.enemy_army_after)
    elimination_delta = (record.eliminations_after - record.eliminations_before)
    sid = record.strategy_id

    if sid == 0:
        metrics.explore_fitness += (land_delta * 3)
        metrics.explore_fitness += (castle_delta * 50)

    elif sid == 1:
        territory_ratio_before = record.territory_before / max(1, record.enemy_land_before)
        territory_ratio_after = record.territory_after / max(1, record.enemy_land_after)
        metrics.build_fitness += (territory_ratio_after - territory_ratio_before) * 100
        metrics.build_fitness += (castle_delta * 100)

    elif sid == 2:
        metrics.combat_fitness += enemy_land_delta * 20
        metrics.combat_fitness += enemy_army_delta * 5
        metrics.combat_fitness += elimination_delta * 1000


def finalize_overlord_fitness(env, metrics):
    fitness = (metrics.explore_fitness + metrics.build_fitness + metrics.combat_fitness)
    used_strategies = sum(1 for turns in metrics.strategy_usage.values() if turns > 20)

    fitness += used_strategies * 250

    if env.winner == PLAYER_ID:
        fitness += 10000

    elif PLAYER_ID not in env.eliminated_players:
        fitness += 500

    total_turns = sum(metrics.strategy_usage.values())

    if total_turns > 0:
        entropy = 0
        for turns in metrics.strategy_usage.values():
            if turns == 0:
                continue

            p = turns / total_turns
            entropy -= (p * math.log(p))
        # reward diversity
        fitness += entropy * 50

    switches = 0
    for i in range(1, len(metrics.strategy_history)):
        if metrics.strategy_history[i] != metrics.strategy_history[i - 1]:
            switches += 1

    # punish excessive switching
    fitness -= switches * .5
    metrics.overlord_fitness = fitness
    return fitness


# TURN EXECUTION
def take_turn(env, metrics, overlord, explore_ai, build_ai, combat_ai):
    if metrics.pending_record is not None:
        current = helpers.create_snapshot(env, PLAYER_ID)
        record = metrics.pending_record
        record.territory_after = current["territory"]
        record.army_after = current["army"]
        record.castles_after = current["castles"]
        record.enemy_land_after = current["enemy_land"]
        record.enemy_army_after = current["enemy_army"]
        record.eliminations_after = current["eliminations"]
        process_record(metrics, record)
        metrics.pending_record = None

    candidate_actions = helpers.get_candidate_actions(env, PLAYER_ID, 8)

    if len(candidate_actions) == 0:
        env.submit_action(PLAYER_ID, env.get_legal_actions(PLAYER_ID, include_pass=True))
        return None

    before = helpers.create_snapshot(env, PLAYER_ID)
    overlord.build_overlord_state(env)
    strategy_id = overlord.choose_strategy(overlord.np_array)
    strategy_lookup = {0: explore_ai, 1: build_ai, 2: combat_ai}
    strategy_ai = strategy_lookup[strategy_id]
    strategy_ai.build_ai_state(env, candidate_actions)
    action = strategy_ai.choose_move(strategy_ai.np_array, candidate_actions)

    if action is None:
        print("TAKE TURN RETURNED NONE")
        return None

    metrics.strategy_usage[strategy_id] += 1
    metrics.strategy_history.append(strategy_id)
    env.submit_action(PLAYER_ID, action)
    metrics.pending_record = TurnRecord(env.turn, strategy_id, action, before["territory"], 0, before["army"], 0,
                                        before["castles"], 0, before["enemy_land"], 0, before["enemy_army"], 0,
                                        before["eliminations"], 0)
    return
