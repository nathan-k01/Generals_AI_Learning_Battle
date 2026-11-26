import random
from Game import generals_game
import AIs.NathanAI.AI as ai1
import AIs.NathanAI.trainer as ai1_trainer
from AIs.Kal_AI.AI import AI as KalAI, PolicyNet, Adam
from AIs.Kal_AI.trainer import episode_gradient, apply_gradients
from functools import partial
from multiprocessing import Pool
import pickle

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

KAL_ID = 1
KAL_CHECKPOINT = "kal_latest.pkl"


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


def evaluate_individual(individual, kal_params):   
    env = generals_game.GeneralsEnvironment()
    nathan_ai = ai1.AI(individual)
    kal_net = PolicyNet(kal_params)
    kal_ai = KalAI(kal_net, player_id=KAL_ID, training=True)

    while not env.game_over:
        if env.turn == 0:
            for p in env.get_active_players():
                env.submit_action(p, env.get_legal_actions(p, include_pass=True)[0])

        kal_ai.take_turn(env)
        random_ai_move(env, 2)
        nathan_ai.take_turn(env)
        if (not env.game_over and not env.has_player_submitted(0)
                and 0 not in env.eliminated_players):
            env.submit_action(0, generals_game.Action.pass_action())

    fitness = nathan_ai.get_overlord_fitness(env)
    kal_ai.finish_game(env)
    kal_grad = episode_gradient(kal_net, kal_ai.steps)                
    return individual, fitness, nathan_ai, kal_grad, kal_ai.outcome, kal_ai.final_territory_share


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

    if os.path.exists(KAL_CHECKPOINT):
        kal_net = PolicyNet.load(KAL_CHECKPOINT)
        print("Loaded Kal_AI weights")
    else:
        kal_net = PolicyNet(seed=0)
        kal_optimizer = Adam(kal_net.params, lr=0.005)

    while True:

        with Pool() as pool:
            results = pool.map(partial(evaluate_individual, kal_params=kal_net.params), trainer.population)


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

        apply_gradients(kal_net, kal_optimizer, [r[3] for r in results])
        kal_net.save(KAL_CHECKPOINT)
        print(f"Kal_AI: wins={[r[4] for r in results].count('win')}/{len(results)} "
              f"avg land share={sum(r[5] for r in results) / len(results):.3f}")

        trainer.next_generation(scored_individuals)

        generation += 1


