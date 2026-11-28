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
    frontline_army_before: int
    frontline_army_after: int


class BoardAnalysis:
    def __init__(self, env, player_id):
        obs = env.get_observation(player_id)
        self.turn = env.turn
        self.territory, self.army, self.castles = env.get_stats(PLAYER_ID)
        self.enemy_tiles = helpers.get_visible_enemy_tiles(obs)
        self.neutral_tiles = helpers.get_neutral_tiles(obs)
        self.visible_cities = helpers.get_visible_cities(obs)
        self.frontline_tiles = helpers.get_frontline_tiles(env, player_id)
        self.largest_stack = helpers.get_largest_stack(env, player_id)
        self.enemy_seen = int(self.enemy_tiles > 0)
        self.dx, self.dy, self.enemy_distance = (helpers.nearest_enemy_info(env, player_id))
        self.active_players = env.get_active_players()
        self.largest_enemy_land, self.largest_enemy_army = helpers.get_enemy_stats(env, PLAYER_ID)


# GAME METRICS
class GameMetrics:
    def __init__(self):
        self.records = []
        self.explore_fitness = 0.0
        self.build_fitness = 0.0
        self.combat_fitness = 0.0
        self.overlord_fitness = 0.0
        self.strategy_usage = {0: 0, 1: 0, 2: 0}
        self.combat_opportunities = 0
        self.strategy_history = []
        self.total_eliminations = 0
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
        return take_turn(env=env, metrics=self.metrics, overlord=self.overlord,
                         explore_ai=self.explore_ai, build_ai=self.build_ai, combat_ai=self.combat_ai)

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
    INPUT_SIZE = 463
    ACTION_CHOICES = 8
    HIDDEN_SIZE = 32

    def __init__(self, genome):
        self.np_array = None
        self.brain = NeuralNet(genome=genome, input_size=self.INPUT_SIZE,
                               hidden_size=self.HIDDEN_SIZE, output_size=self.ACTION_CHOICES)

    def choose_move(self, state, candidate_actions):
        outputs = self.brain.forward(state)
        if len(candidate_actions) == 0:
            print("CHOOSE MOVE RETURNED NONE")
            return None

        choice = int(np.argmax(outputs))
        choice = min(choice, len(candidate_actions) - 1)

        return candidate_actions[choice]


# SPECIALIST AIs
class ExplorerAI(StrategyAI):
    def build_ai_state(self, analysis, candidate_actions_block):
        state = [analysis.territory, analysis.army, analysis.castles, analysis.neutral_tiles, analysis.enemy_tiles,
                 analysis.visible_cities,
                 analysis.turn, len(analysis.active_players), analysis.dx, analysis.dy, analysis.enemy_distance,
                 analysis.frontline_tiles, analysis.largest_stack]
        state.extend(candidate_actions_block)

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

    def build_overlord_state(self, analysis):
        army_ratio = analysis.army / max(1, analysis.largest_enemy_army)
        territory_ratio = analysis.territory / max(1, analysis.largest_enemy_land)
        state = [analysis.turn, analysis.territory, analysis.army, analysis.castles, analysis.largest_enemy_land,
                 analysis.largest_enemy_army, army_ratio, territory_ratio, analysis.enemy_tiles, analysis.neutral_tiles,
                 analysis.visible_cities, analysis.frontline_tiles, analysis.largest_stack,
                 len(analysis.active_players)]

        self.np_array = np.array(state, dtype=np.float32)

    def choose_strategy(self, state):
        if random.random() < 0.20:
            return random.randint(0, 2)

        outputs = self.brain.forward(state)
        return int(np.argmax(outputs))


# FITNESS SYSTEM
def process_record(metrics, analysis, record):
    land_delta = (record.territory_after - record.territory_before)
    castle_delta = (record.castles_after - record.castles_before)
    enemy_land_delta = (record.enemy_land_before - record.enemy_land_after)
    enemy_army_delta = (record.enemy_army_before - record.enemy_army_after)
    elimination_delta = (record.eliminations_after - record.eliminations_before)
    sid = record.strategy_id

    if record.territory_before < record.territory_after:
        metrics.overlord_fitness += 10

    if sid == 0:  # expand
        if analysis.turn < 20:
            metrics.overlord_fitness += 2

        if analysis.neutral_tiles > 10:
            metrics.overlord_fitness += 1

        reward = 0
        # Expanding is the primary job
        reward += land_delta * 250

        metrics.explore_fitness += reward

    elif sid == 1:
        frontline_delta = record.frontline_army_after - record.frontline_army_before
        army_delta = record.army_after - record.army_before

        reward = 0
        if army_delta > 0:
            reward += army_delta * 0.5

        if frontline_delta > 0:
            reward += frontline_delta

        reward += castle_delta * 1000
        metrics.build_fitness += reward

    elif sid == 2:
        # Once enemies are visible, combat should be more attractive.
        if analysis.enemy_seen:
            metrics.overlord_fitness += .5

        if analysis.largest_enemy_army > analysis.army:
            metrics.overlord_fitness -= 2

        if analysis.largest_enemy_army * 2 < analysis.army:
            metrics.overlord_fitness += 5

        reward = 0

        if enemy_land_delta > 0:
            reward += enemy_land_delta * 250

        reward += elimination_delta * 50000
        metrics.overlord_fitness += (elimination_delta * 100)

        metrics.combat_fitness += reward


def finalize_overlord_fitness(env, metrics):
    fitness = metrics.overlord_fitness
    used_strategies = sum(1 for turns in metrics.strategy_usage.values() if turns > 20)

    fitness += used_strategies * 100
    fitness += metrics.total_eliminations * 5000

    territory, army, castles = env.get_stats(PLAYER_ID)

    fitness += territory * 1.2
    fitness += army * 0.1
    fitness += castles * 1.5
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

    if env.winner == PLAYER_ID:
        fitness += 25000

    elif PLAYER_ID not in env.eliminated_players:
        fitness += 1000

    metrics.overlord_fitness += fitness
    metrics.overlord_fitness /= 10

    explore_turns = max(1, metrics.strategy_usage[0])
    build_turns = max(1, metrics.strategy_usage[1])
    combat_turns = max(1, metrics.strategy_usage[2])

    metrics.explore_fitness /= explore_turns
    metrics.build_fitness /= build_turns
    metrics.combat_fitness /= combat_turns

    return metrics.overlord_fitness


# TURN EXECUTION
def take_turn(env, metrics, overlord, explore_ai, build_ai, combat_ai):
    analysis = BoardAnalysis(env, PLAYER_ID)
    if metrics.pending_record is not None:
        current = helpers.create_snapshot(env, PLAYER_ID)
        record = metrics.pending_record
        record.territory_after = current["territory"]
        record.army_after = current["army"]
        record.castles_after = current["castles"]
        record.enemy_land_after = current["enemy_land"]
        record.enemy_army_after = current["enemy_army"]
        record.eliminations_after = current["eliminations"]
        process_record(metrics, analysis, record)
        metrics.pending_record = None

    candidate_actions = helpers.get_candidate_actions(env, PLAYER_ID, 50)
    candidate_actions_block = helpers.build_candidate_action_block(env, 0, candidate_actions)
    analysis.candidate_actions = candidate_actions

    if len(candidate_actions) == 0:
        env.submit_action(PLAYER_ID, env.get_legal_actions(PLAYER_ID, include_pass=True))
        return None

    before = helpers.create_snapshot(env, PLAYER_ID)
    overlord.build_overlord_state(analysis)
    strategy_id = overlord.choose_strategy(overlord.np_array)
    strategy_lookup = {0: explore_ai, 1: build_ai, 2: combat_ai}
    strategy_ai = strategy_lookup[strategy_id]
    strategy_ai.build_ai_state(analysis, candidate_actions_block)
    action = strategy_ai.choose_move(strategy_ai.np_array, candidate_actions)

    if action is None:
        print("TAKE TURN RETURNED NONE")
        return None

    metrics.strategy_usage[strategy_id] += 1
    metrics.strategy_history.append(strategy_id)
    env.submit_action(PLAYER_ID, action)
    metrics.pending_record = TurnRecord(env.turn, strategy_id, action, before["territory"], 0, before["army"], 0,
                                        before["castles"], 0, before["enemy_land"], 0, before["enemy_army"], 0,
                                        before["eliminations"], 0, before["frontline_army"], 0)
    return
