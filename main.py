import random
from Game import generals_game

import AIs.NathanAI.AI as ai1
import AIs.NathanAI.trainer as ai1_trainer

import AIs.Kal_AI.AI as ai2
import AIs.Kal_AI.Trainer as ai2_trainer

from multiprocessing import Pool
import pickle
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

KAL_ID = 1

training_history = {
    "generation": [],
    "best": [],
    "avg": [],
    "worst": [],
    "expand": [],
    "combat": [],
    "build": [],
    "win_rate": [],
    "territory": [],
    "army": [],
    "castles": [],
}

def print_fitness_breakdown(metrics):
    print("\n========== FITNESS ==========")
    print(f"Expand:   {metrics.expand_fitness:.2f}")
    print(f"Economy:  {metrics.economy_fitness:.2f}")
    print(f"Attack:   {metrics.attack_fitness:.2f}")
    print(f"Defense:  {metrics.defense_fitness:.2f}")
    print(f"Assassin: {metrics.assassin_fitness:.2f}")


def random_ai_move(environment, player_id):
    legal = environment.get_legal_actions(player_id, include_pass=True)
    if legal:
        environment.submit_action(player_id, random.choice(legal))


def evaluate_individual(individual, kal_weights):  
    turn_history = []
    env = generals_game.GeneralsEnvironment()
    nathan_ai = ai1.AI(individual)

    kal_ai = ai2.AI(weights=kal_weights, player_id=KAL_ID)  

    while not env.game_over:
        if env.turn == 0:
            for p in env.get_active_players():
                env.submit_action(p, env.get_legal_actions(p, include_pass=True)[0])

        random_ai_move(env, 2)
        kal_ai.take_turn(env)
        nathan_ai.take_turn(env)
        if (not env.game_over and not env.has_player_submitted(0)
                and 0 not in env.eliminated_players):
            env.submit_action(0, generals_game.Action.pass_action())

        if env.turn % 5 == 0:
            turn_history.append(env.get_full_state())

    fitness = nathan_ai.get_overlord_fitness(env)

    kal_ai.finish_game(env)

    return (individual, fitness, nathan_ai, kal_ai.outcome, kal_ai.final_territory_share, turn_history,
            kal_weights, kal_ai.get_fitness(env))


if __name__ == "__main__":
    nate_trainer = ai1_trainer.NathanTrainer(population_size=50)
    kal_trainer = ai2_trainer.KalTrainer(population_size=50)   
    highest_fitness = float("-inf")
    generation = 0
    if os.path.exists("best_ever.pkl"):
        with open("best_ever.pkl", "rb") as file:
            champion = pickle.load(file)

        print(f"Loaded Champion "f"(Generation {champion['generation']}) "f"Fitness={champion['fitness']}")
        nate_trainer.population[0] = (champion["genome"])

        for i in range(1, len(nate_trainer.population)):
            child = nate_trainer.mutator.crossover(champion["genome"], champion["genome"])
            nate_trainer.mutator.mutate_individual(child)
            nate_trainer.population[i] = child

        highest_fitness = (champion["fitness"])
        generation = (champion["generation"] + 1)

    else:
        highest_fitness = float("-inf")
        generation = 0

    while True:

        with Pool() as pool:
            results = pool.starmap(evaluate_individual, zip(nate_trainer.population, kal_trainer.population))

        scored_individuals = [r[:3] for r in results]

        scored_individuals.sort(key=lambda x: x[1], reverse=True)
        best_individual = scored_individuals[0][0]
        best_fitness = scored_individuals[0][1]
        best_metrics = scored_individuals[0][2]
        average_fitness = (sum(fitness for _, fitness, _ in scored_individuals) / len(scored_individuals))
        worst_fitness = scored_individuals[-1][1]

        if best_fitness > highest_fitness:
            highest_fitness = best_fitness
            with open("best_ever.pkl", "wb") as file:
                pickle.dump({"generation": generation, "fitness": best_fitness, "genome": scored_individuals[0][0]},
                            file)

        print(f"\n========== GENERATION {generation} ==========")
        print(f"Best Fitness: {best_fitness}")
        print(f"Average Fitness: {average_fitness}")
        print(f"Worst Fitness: {worst_fitness}")
        best_metrics.print_fitness_breakdown()

        kal_scored = [(r[6], r[4] + (1.0 if r[3] == "win" else 0.0)) for r in results]
        kal_best = max(kal_scored, key=lambda x: x[1])
        kal_trainer.save_best(kal_best[0], kal_best[1])
        print(f"Kal_AI: wins={[r[3] for r in results].count('win')}/{len(results)} "
              f"avg land share={sum(r[4] for r in results) / len(results):.3f} "
              f"best fitness={kal_best[1]:.3f}")

        kal_trainer.next_generation(kal_scored)
        nate_trainer.next_generation(scored_individuals)
        generation += 1

        best_result = max(results, key=lambda r: r[1])
        first_match_history = best_result[5]

        replay_packet = {
            "generation": generation,
            "best": best_fitness,
            "avg": average_fitness,
            "worst": worst_fitness,
            "history": first_match_history,

            "expand": best_metrics.metrics.expand_fitness,
            "economy": best_metrics.metrics.economy_fitness,
            "attack": best_metrics.metrics.attack_fitness,
            "defense": best_metrics.metrics.defense_fitness,
            "assassin": best_metrics.metrics.assassin_fitness,
            "overlord": best_metrics.metrics.overlord_fitness,

            "usage": best_metrics.metrics.strategy_usage.copy(),

            "kal_wins": [r[3] for r in results].count("win"),
            "kal_games": len(results),
            "kal_land": sum(r[4] for r in results) / len(results),
            "kal_best_fitness": kal_best[1],
            "kal_weights": kal_best[0],
        }

        with open("live_replay.pkl", "wb") as f:
            pickle.dump(replay_packet, f)