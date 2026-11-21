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
        scored_individuals.sort(key=lambda x: x[1], reverse=True)
        survivors = [item[0] for item in scored_individuals[:20]]

        new_population = [scored_individuals[0][0]]

        while len(new_population) < len(self.population):
            parent_a = random.choice(survivors)
            parent_b = random.choice(survivors)
            child = self.mutator.crossover(parent_a, parent_b)
            self.mutator.mutate_individual(child)
            new_population.append(child)

        self.population = new_population


class GenomeManager:
    SPECIALIST_INPUTS = 40
    SPECIALIST_HIDDEN = 16
    SPECIALIST_OUTPUTS = 8

    OVERLORD_INPUTS = 9
    OVERLORD_HIDDEN = 20
    OVERLORD_OUTPUTS = 5

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
        return {
            "overlord": cls.random_genome(cls.overlord_size()),
            "expand": cls.random_genome(specialist),
            "economy": cls.random_genome(specialist),
            "attack": cls.random_genome(specialist),
            "defense": cls.random_genome(specialist),
            "assassin": cls.random_genome(specialist)
        }


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

    def crossover(self, parent_a, parent_b):
        child = {}
        for key in parent_a:
            genome_a = parent_a[key]
            genome_b = parent_b[key]

            split = random.randint(0, len(genome_a) - 1)

            child[key] = (genome_a[:split] + genome_b[split:])

        return child
