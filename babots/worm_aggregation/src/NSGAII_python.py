import json

import os
import random
import time
import numpy as np
import subprocess




parameter_ranges = {
    "AGENT_COUNT": (1000, 60000),
    "PERSISTENCE_FACTOR": (0.1, 0.9),
    "OXYGEN_CONSUMPTION": (0.0005, 0.001),
    "OXYGEN_GLOBAL": (0.0, 0.21),
    "OXYGEN_DIFFUSION": (0.001, 0.003),
    "SENSING_RANGE": (1, 20),
    "BETA_ATTRACTANT": (0.001, 0.002),
    "BETA_REPELLENT": (0.001, 0.002),
    "ATTRACTANT_CREATION": (0.0, 0.015),
    "ALPHA_ATTRACTANT": (15, 15),
    "ALPHA_REPELLENT": (15, 15),
    "REPELLENT_CREATION": (0.0, 0.0015),
    "OXYGEN": (0, 1),  # Binary parameter
    "PHEROMONES": (0, 1),  # Binary parameter
    "H": (0, 1)
}

def create_individual():
    individual = []
    for param, (min_val, max_val) in parameter_ranges.items():
        if param in ["OXYGEN", "PHEROMONES"]:  # Handling binary parameters
            individual.append(random.randint(min_val, max_val))
        else:
            individual.append(random.uniform(min_val, max_val))
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            min_val, max_val = parameter_ranges[list(parameter_ranges.keys())[i]]
            if i in [list(parameter_ranges.keys()).index("OXYGEN"), list(parameter_ranges.keys()).index("PHEROMONES")]:
                individual[i] = random.randint(min_val, max_val)
            else:
                individual[i] = random.uniform(min_val, max_val)
    return individual

def individual_to_json(individual):
    json_data = {"environment": {}}
    for i, (param, _) in enumerate(parameter_ranges.items()):
        json_data["environment"][param] = individual[i]
    return json.dumps(json_data)

def json_to_individual(json_data):
    individual = []
    decoded_data = json.loads(json_data)
    for param in parameter_ranges.keys():
        individual.append(decoded_data["environment"][param])
    return individual
def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract 'x' and 'y' coordinates
    x_values = [agent['x'] for agent in data['agents']['worm']['default']]
    y_values = [agent['y'] for agent in data['agents']['worm']['default']]

    # Combine 'x' and 'y' values into a single array
    return np.array(list(zip(x_values, y_values)))
def objectives(gen,ind,i):
    print("gen: "+str(gen)+",ind: "+str(i))
    with open("parameters.json", "w") as f:
        f.write(individual_to_json(ind))
    subprocess.run(["./run.sh"]+[str(gen)+"_"+str(i)+".json"])


    agents_data = process_json_file("logs/"+str(gen)+"_"+str(i)+".json")
    grid_size = 128
    center_start = 54
    center_end = 73

    agents_inside_center = 0
    total_agents = len(agents_data)

    # Iterate over each agent and determine which grid cell it falls into
    for agent in agents_data:
        x, y = agent
        # Convert agent coordinates to grid indices
        grid_x = int(x / (20/grid_size))
        grid_y = int(y / (20/grid_size))
        # Check if the agent falls within the center square
        if center_start <= grid_x < center_end and center_start <= grid_y < center_end:
            agents_inside_center += 1

    percent = (agents_inside_center / total_agents) * 100
    print(percent)
    return percent

# Define the folder to save the best individuals
output_folder = "best_individuals"

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def save_individual_as_json(gen, ind, individual):
    filename = os.path.join(output_folder, f"gen{gen}_ind{ind}_best.json")
    with open(filename, "w") as f:
        f.write(individual)
def genetic_algorithm(objective_function, population_size, num_generations, crossover_rate, mutation_rate):
    population = [create_individual() for _ in range(population_size)]
    best_individuals = []
    for gen in range(num_generations):
        # Evaluate fitness of each individual
        fitness_scores = [objective_function(gen, ind,i) for i, ind in enumerate(population)]

        # Selection: Tournament selection
        selected_indices = []
        for _ in range(population_size):
            tournament_size = 3
            competitors = random.sample(range(population_size), tournament_size)
            selected_indices.append(max(competitors, key=lambda x: fitness_scores[x]))

        # Create next generation through crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = population[selected_indices[i]]
            parent2 = population[selected_indices[i + 1]]
            child1,child2 = crossover(parent1, parent2)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        best_index = max(range(population_size), key=lambda x: fitness_scores[x])
        best_individual = population[best_index]
        best_individuals.append(best_individual)


        save_individual_as_json(gen, best_index, individual_to_json(best_individual))

        population = new_population

    # Return the best individual after all generations
    best_index = max(range(population_size), key=lambda x: fitness_scores[x])
    return population[best_index], fitness_scores[best_index]

best_individual, best_fitness = genetic_algorithm(objectives, population_size=20, num_generations=50, crossover_rate=0.7, mutation_rate=0.1)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)