import json
import os
import random

from AIs.Kal_AI.AI import DEFAULT_WEIGHTS

BEST_FILE = "kal_best.json"


class KalTrainer:
    SURVIVORS = 10       
    MUTATION_RATE = 0.3   
    SIGMA = 0.25         

    def __init__(self, population_size=50):
        start = self.load_best() or dict(DEFAULT_WEIGHTS)
        self.best_fitness = self.saved_fitness()   
        self.population = [dict(start)]
        while len(self.population) < population_size:
            self.population.append(self.mutate(start, rate=1.0, sigma=0.5))

    # ---------- evolution ----------
    def mutate(self, weights, rate=None, sigma=None):
        rate = self.MUTATION_RATE if rate is None else rate
        sigma = self.SIGMA if sigma is None else sigma
        child = {}
        for key, value in weights.items():
            if random.random() < rate:
                value = value * (1 + random.gauss(0, sigma)) + random.gauss(0, 0.05)
            child[key] = value
        return child

    def crossover(self, parent_a, parent_b):
        return {k: (parent_a[k] if random.random() < 0.5 else parent_b[k]) for k in parent_a}

    def next_generation(self, scored_individuals):
        """scored_individuals: list of (weights, fitness). Best one always survives unchanged."""
        scored = sorted(scored_individuals, key=lambda x: x[1], reverse=True)
        survivors = [item[0] for item in scored[:self.SURVIVORS]]
        new_population = [dict(scored[0][0])]
        while len(new_population) < len(self.population):
            child = self.crossover(random.choice(survivors), random.choice(survivors))
            new_population.append(self.mutate(child))
        self.population = new_population

    def save_best(self, weights, fitness):
        """Only overwrite the saved file when this is the best fitness so far."""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            with open(BEST_FILE, "w") as f:
                json.dump({"fitness": fitness, "weights": weights}, f, indent=2)
            return True
        return False

    @staticmethod
    def saved_fitness():
        if not os.path.exists(BEST_FILE):
            return float("-inf")
        with open(BEST_FILE) as f:
            return json.load(f)["fitness"]

    @staticmethod
    def load_best():
        if not os.path.exists(BEST_FILE):
            return None
        with open(BEST_FILE) as f:
            saved = json.load(f)["weights"]
        return {**DEFAULT_WEIGHTS, **saved}