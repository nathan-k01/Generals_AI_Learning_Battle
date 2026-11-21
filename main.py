import random
from Game import generals_game
import AIs.NathanAI.AI as ai1
import AIs.NathanAI.trainer as ai1_trainer
from multiprocessing import Pool
import pickle

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


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


def evaluate_individual(individual):
    env = generals_game.GeneralsEnvironment()
    nathan_ai = ai1.AI(individual)

    while not env.game_over:
        if env.turn == 0:
            for p in env.get_active_players():
                env.submit_action(p, env.get_legal_actions(p, include_pass=True)[0])

        random_ai_move(env, 1)
        random_ai_move(env, 2)
        nathan_ai.take_turn(env)

    fitness = nathan_ai.get_overlord_fitness(env)
    return individual, fitness, nathan_ai


if __name__ == "__main__":
    trainer = ai1_trainer.NathanTrainer(population_size=50)

    if os.path.exists("best_ever.pkl"):
        with open("best_ever.pkl", "rb") as file:
            champion = pickle.load(file)

        print(f"Loaded Champion "f"(Generation {champion['generation']}) "f"Fitness={champion['fitness']}")
        trainer.population[0] = (champion["genome"])

        for i in range(1, len(trainer.population)):
            child = trainer.mutator.crossover(champion["genome"], champion["genome"])
            trainer.mutator.mutate_individual(child)
            trainer.population[i] = child

        highest_fitness = (champion["fitness"])
        generation = (champion["generation"] + 1)
    else:
        highest_fitness = float("-inf")
        generation = 0

    while True:

        with Pool() as pool:
            scored_individuals = pool.map(evaluate_individual, trainer.population)

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
        trainer.next_generation(scored_individuals)

        generation += 1


