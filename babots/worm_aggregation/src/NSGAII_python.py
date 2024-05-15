import json
import math
import os
import random
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Utility functions
def index_locator(a, list):
    for i in range(len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_locator(min(values), values) in list1:
            sorted_list.append(index_locator(min(values), values))
        values[index_locator(min(values), values)] = math.inf
    return sorted_list

def crowding_distance(values1, values2, front):
    distance = [0] * len(front)
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = distance[-1] = float('inf')
    m1 = max(values1) - min(values1)
    m2 = max(values2) - min(values2)
    m1 = m1 if m1 != 0 else 1e-10
    m2 = m2 if m2 != 0 else 1e-10

    for k in range(1, len(front) - 1):
        distance[k] += (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / m1
        distance[k] += (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / m2
    return distance

def non_dominated_sorting_algorithm(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0] * len(values1)
    rank = [0] * len(values1)

    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    front.pop()
    return front

# Parameter ranges
parameter_ranges = {
    "AGENT_COUNT": (1, 100),
    "BABOTS": (1, 100),
    "PERSISTENCE_FACTOR": (0.1, 0.9),
    "SENSING_RANGE": (1, 30),
    "BETA_ATTRACTANT": (0.0011, 0.002),
    "BETA_REPELLENT": (0.0011, 0.002),
    "ATTRACTANT_CREATION": (0.0, 0.015),
    "ALPHA_ATTRACTANT": (15, 15),
    "ALPHA_REPELLENT": (15, 15),
    "REPELLENT_CREATION": (0.0, 0.0015),
    "OXYGEN":(0,0),
    "H": (0,0)
}

# Functions to create, mutate, and crossover individuals
def create_individual():
    return [random.uniform(min_val, max_val) for min_val, max_val in parameter_ranges.values()]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            min_val, max_val = parameter_ranges[list(parameter_ranges.keys())[i]]
            individual[i] = random.uniform(min_val, max_val)
    return individual

# JSON conversion functions
def individual_to_json(individual):
    return json.dumps({"environment": {param: individual[i] for i, param in enumerate(parameter_ranges.keys())}})

def json_to_individual(json_data):
    decoded_data = json.loads(json_data)
    return [decoded_data["environment"][param] for param in parameter_ranges.keys()]

def process_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Objective functions
def objective1(gen, ind, i):
    print(f"gen: {gen}, ind: {i}")
    with open("parameters.json", "w") as f:
        f.write(individual_to_json(ind))
    subprocess.run(["./run.sh", f"{gen}_{i}.json"])

    try:
        agents_data = process_json_file(f"logs/{gen}_{i}.json")
    except FileNotFoundError:
        print(f"Log file for generation {gen}, individual {i} not found.")
        return 0

    grid_size = 128
    center_start = 54
    center_end = 73

    agents_inside_center = 0
    total_agents = 0

    babots_data = agents_data.get('agents', {}).get('babots', {}).get('default', [])
    worm_data = agents_data.get('agents', {}).get('worm', {}).get('default', [])
    combined_data = babots_data + worm_data

    total_agents = len(combined_data)

    for agent in combined_data:
        x, y = agent['x'], agent['y']
        grid_x = int(x / (20 / grid_size))
        grid_y = int(y / (20 / grid_size))
        if center_start <= grid_x < center_end and center_start <= grid_y < center_end:
            agents_inside_center += 1

    percent = (agents_inside_center / total_agents) * 100 if total_agents > 0 else 0
    print(f"Percentage of agents inside center: {percent}")
    return -percent

def objective2(ind):
    denominator = ind[1] + ind[0]
    if denominator == 0:
        print("Warning: denominator is zero in objective2 calculation.")
        return 0
    ratio = ind[1] / denominator
    print(f"Objective2 ratio: {ratio}")
    return ratio

# NSGA-II algorithm
def nsga2(population, max_gen, mutation_rate):
    gen_no = 0
    solution = [create_individual() for _ in range(population)]

    while gen_no < max_gen:
        objective1_values = [objective1(gen_no, individual, ind) for ind, individual in enumerate(solution)]
        objective2_values = [objective2(individual) for ind, individual in enumerate(solution)]
        non_dominated_sorted_solution = non_dominated_sorting_algorithm(objective1_values[:], objective2_values[:])

        print('Best Front for Generation:', gen_no)
        for values in non_dominated_sorted_solution[0]:
            print(np.array(solution[values]).round(3), end=" ")
        print("\n")

        crowding_distance_values = []
        for i in range(len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(objective1_values[:], objective2_values[:], non_dominated_sorted_solution[i][:])
            )

        solution2 = solution[:]

        while len(solution2) < 2 * population:
            a1 = random.randint(0, population - 1)
            b1 = random.randint(0, population - 1)
            c1, c2 = crossover(solution[a1], solution[b1])
            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)
            solution2.append(c1)
            solution2.append(c2)

        objective1_values2 = [objective1(gen_no, individual, ind) for ind, individual in enumerate(solution2)]
        objective2_values2 = [objective2(individual) for ind, individual in enumerate(solution2)]
        non_dominated_sorted_solution2 = non_dominated_sorting_algorithm(objective1_values2[:], objective2_values2[:])

        crowding_distance_values2 = []
        for i in range(len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(objective1_values2[:], objective2_values2[:], non_dominated_sorted_solution2[i][:])
            )

        new_solution = []
        for i in range(len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_locator(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(len(non_dominated_sorted_solution2[i]))
            ]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == population:
                    break
            if len(new_solution) == population:
                break

        solution = [solution2[i] for i in new_solution]
        gen_no += 1

    return [objective1_values, objective2_values]

def non_dominating_curve_plotter(objective1_values, objective2_values):
    plt.figure(figsize=(15, 8))
    plt.xlabel('Objective 1: Percentage of agents in center (minimized)', fontsize=15)
    plt.ylabel('Objective 2: Ratio of second parameter to the sum of first two parameters (minimized)', fontsize=15)
    plt.scatter(objective2_values, objective1_values, c='red', s=25)

# Parameters
population = 10
max_gen = 20
mutation_rate = 0.3

# Running NSGA-II
objective1_values, objective2_values = nsga2(population, max_gen, mutation_rate)

# Plotting
non_dominating_curve_plotter(objective1_values, objective2_values)
plt.show()
