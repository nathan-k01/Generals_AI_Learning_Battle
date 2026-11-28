import random
from AIs.NathanAI.AI import AI


class NathanTrainer:
    def __init__(self, population_size=50):
        self.population = [GenomeManager.create_individual() for _ in range(population_size)]
        self.mutator = Mutator()

    def build_ai(self, index):
        return AI(self.population[index])

    def mutate_population(self):
        for individual in self.population:
            self.mutator.mutate_individual(individual)

    def next_generation(self, scored_individuals):
        overlord_sorted = sorted(scored_individuals, key=lambda x: x[1]["overlord"], reverse=True)
        explore_sorted = sorted(scored_individuals, key=lambda x: x[1]["explore"], reverse=True)
        build_sorted = sorted(scored_individuals, key=lambda x: x[1]["build"], reverse=True)
        combat_sorted = sorted(scored_individuals, key=lambda x: x[1]["combat"], reverse=True)
        best_overlords = [x[0]["overlord"] for x in overlord_sorted[:10]]
        best_explorers = [x[0]["explore"] for x in explore_sorted[:10]]
        best_builders = [x[0]["build"] for x in build_sorted[:10]]
        best_combat = [x[0]["combat"] for x in combat_sorted[:10]]

        new_population = [{"overlord": best_overlords[0][:], "explore": best_explorers[0][:],
                           "build": best_builders[0][:], "combat": best_combat[0][:]}]

        while len(new_population) < len(self.population):
            child = {
                "overlord": self.mutator.crossover_genome(random.choice(best_overlords), random.choice(best_overlords)),
                "explore": self.mutator.crossover_genome(random.choice(best_explorers), random.choice(best_explorers)),
                "build": self.mutator.crossover_genome(random.choice(best_builders), random.choice(best_builders)),
                "combat": self.mutator.crossover_genome(random.choice(best_combat), random.choice(best_combat)),
            }
            self.mutator.mutate_individual(child)
            new_population.append(child)

        self.population = new_population


class GenomeManager:
    SPECIALIST_INPUTS = 463
    SPECIALIST_HIDDEN = 32
    SPECIALIST_OUTPUTS = 8

    OVERLORD_INPUTS = 14
    OVERLORD_HIDDEN = 24
    OVERLORD_OUTPUTS = 3

    @classmethod
    def specialist_size(cls):

        return (cls.SPECIALIST_INPUTS * cls.SPECIALIST_HIDDEN + cls.SPECIALIST_HIDDEN +
                cls.SPECIALIST_HIDDEN * cls.SPECIALIST_OUTPUTS + cls.SPECIALIST_OUTPUTS)

    @classmethod
    def overlord_size(cls):
        return (cls.OVERLORD_INPUTS * cls.OVERLORD_HIDDEN + cls.OVERLORD_HIDDEN + cls.OVERLORD_HIDDEN *
                cls.OVERLORD_OUTPUTS + cls.OVERLORD_OUTPUTS)

    @staticmethod
    def random_genome(length):
        return [random.uniform(-1, 1) for _ in range(length)]

    @classmethod
    def create_individual(cls):
        specialist = cls.specialist_size()
        return {"overlord": cls.random_genome(cls.overlord_size()), "explore": cls.random_genome(specialist),
                "build": cls.random_genome(specialist), "combat": cls.random_genome(specialist)}


class Mutator:
    def __init__(self):
        self.genome = None
        self.children = {}

    def set_genome(self, genome):
        self.genome = genome

    def mutate_genome(self, mutation_rate=0.10, sigma=0.25):
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] += random.gauss(0, sigma)

    def mutate_individual(self, individual, mutation_rate=0.10, sigma=0.25):

        for genome in individual.values():
            self.genome = genome
            self.mutate_genome(mutation_rate, sigma)

    def crossover_genome(self, genome_a, genome_b):
        split = random.randint(0, len(genome_a) - 1)

        return (genome_a[:split] + genome_b[split:])

    def crossover(self, parent_a, parent_b):
        child = {}
        for key in parent_a:
            genome_a = parent_a[key]
            genome_b = parent_b[key]

            split = random.randint(0, len(genome_a) - 1)

            child[key] = (genome_a[:split] + genome_b[split:])

        return child
