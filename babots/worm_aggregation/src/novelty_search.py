import random
import numpy as np
from utils import *
from tsne import *
import json
def generate_initial_population(pop_size):
    return [create_individual() for _ in range(pop_size)]

# Evaluate novelty of each individual
def evaluate_novelty(population, archive, k=15):
    for individual in population:
        distances = []
        for other in population + archive:
            if individual != other:
                distances.append(np.linalg.norm(individual.vector.to_array() -other.vector.to_array()))
        distances.sort()
        individual.novelty = np.mean(distances[:k])

# Select individuals based on novelty
def select(population, num_selected):
    population.sort(key=lambda ind: ind.novelty, reverse=True)
    return population[:num_selected]

# Crossover between two individuals
def crossover(parent1, parent2):
    crossover_point = random.randint(1, parent1.genotype.get_length() - 1)
    child1_genome = parent1.genotype.to_list()[:crossover_point] + parent2.genotype.to_list()[crossover_point:]
    child2_genome = parent2.genotype.to_list()[:crossover_point] + parent1.genotype.to_list()[crossover_point:]
    return Individual(Genotype.from_list(child1_genome),None,None,None), Individual(Genotype.from_list(child2_genome),None,None,None)

# Mutate an individual
def mutate(individual, mutation_rate=0.01):
    l = individual.genotype.to_list()
    for i in range(individual.genotype.get_length()):
        if random.random() < mutation_rate:
            l[i] += random.uniform(-0.1, 0.1)
    return Individual(Genotype.from_list(l),None,None,None)

# Main genetic algorithm
def genetic_algorithm(pop_size, generations, mutation_rate=0.3):
    population = generate_initial_population(pop_size)
    archive = []

    for gen in range(generations):
        for index, individual in enumerate(population):
            individual.phenotype = create_phenotype(individual.genotype,gen,index)
            individual.vector = create_vector(individual.phenotype)
        evaluate_novelty(population, archive)
        archive.extend(population)
        #archive = select(archive, pop_size)

        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(select(population, pop_size // 2), 2)
            child1, child2 = crossover(parent1, parent2)
            child1=mutate(child1, mutation_rate)
            child2=mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Print the best novelty score in the current generation
        best_individual = max(population, key=lambda ind: ind.novelty)
        print(f"Generation {gen}: Best Novelty = {best_individual.novelty}")
        population = new_population[:pop_size]

    return archive

# Parameters
pop_size = 50
generations = 100

# Run the genetic algorithm
archive = genetic_algorithm(pop_size, generations)
archive_json = []
for ind in archive:
    archive_json.append({
            'genotype': ind.genotype.to_list(),
            'vector': ind.vector.to_list(),
            'novelty': ind.novelty
        })

with open('archive.json', 'w') as f:
    json.dump(archive_json, f)
tsne(archive)