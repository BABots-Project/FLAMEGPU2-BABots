import numpy as np
import random
from scipy.spatial import ConvexHull
from itertools import combinations
import subprocess
import json
class Agent:
    def __init__(self, position, speed_x, speed_y):
        self.position = position
        self.speed_x = speed_x
        self.speed_y = speed_y

    def to_list(self):
        return [self.position, self.speed_x, self.speed_y]
class Vector:
    def __init__(self, average_speed,angular_momentum,radial_variance,scatter,group_rotation):
        self.average_speed = average_speed
        self.angular_momentum = angular_momentum
        self.radial_variance = radial_variance
        self.scatter = scatter
        self.group_rotation = group_rotation

    def to_array(self):
        return np.array(
            [self.average_speed, self.angular_momentum, self.radial_variance, self.scatter, self.group_rotation])
    def to_list(self):
        return [self.average_speed, self.angular_momentum, self.radial_variance, self.scatter, self.group_rotation]
class Genotype:
    def __init__(self, agent_count, persistence_factor,sensing_range,alhpa_attractant,alpha_repellent,beta_attractant,beta_repellent,attractant_creation,repellent_creation,oxygen,h):
        self.agent_count = agent_count
        self.persistence_factor = persistence_factor
        self.sensing_range = sensing_range
        self.alhpa_attractant = alhpa_attractant
        self.alpha_repellent = alpha_repellent
        self.beta_attractant = beta_attractant
        self.beta_repellent = beta_repellent
        self.attractant_creation = attractant_creation
        self.repellent_creation = repellent_creation
        self.oxygen = oxygen
        self.h=h
    def get_length(self):
        return len(self.__dict__)

    def to_list(self):
        return [
            self.agent_count, self.persistence_factor, self.sensing_range,
            self.alhpa_attractant, self.alpha_repellent, self.beta_attractant,
            self.beta_repellent, self.attractant_creation, self.repellent_creation,
            self.oxygen, self.h
        ]

    @staticmethod
    def from_list(genotype_list):
        return Genotype(*genotype_list)
class Individual:
    def __init__(self, genotype,phenotype,vector,novelty):
        self.genotype = genotype
        self.phenotype = phenotype
        self.vector = vector
        self.novelty = novelty

parameter_ranges = {
    "AGENT_COUNT": (18000,18000),
    "PERSISTENCE_FACTOR": (0.1, 1.0),
    "SENSING_RANGE": (1, 30),
    "ALPHA_ATTRACTANT":(10,30),
    "ALPHA_REPELLENT":(10,30),
    "BETA_ATTRACTANT": (0.0011, 0.002),
    "BETA_REPELLENT": (0.0011, 0.002),
    "ATTRACTANT_CREATION": (0.0, 0.015),
    "REPELLENT_CREATION": (0.0, 0.0015),
    "OXYGEN":(0,0),
    "H": (0,0)
}

def individual_to_json(individual):
    return json.dumps({"environment": {param: individual.to_list()[i] for i, param in enumerate(parameter_ranges.keys())}})
def process_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
def create_genotype(density):
    values = [random.uniform(min_val, max_val) for min_val, max_val in parameter_ranges.values()]
    values[0]=density*20*20
    return Genotype(*values)

def generate_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agents.append(Agent((random.uniform(0, 20), random.uniform(0, 20)),random.uniform(-10, 10),random.uniform(-10, 10)))
    return agents
def create_phenotype(genotype,gen,i,density):
    print("generation: "+ str(gen)+" individual: "+str(i))

    max_retries = 3
    retry_count = 0

    # Write the parameters to the file
    with open("parameters.json", "w") as f:
        f.write(json.dumps(individual_to_json(genotype)))

    # Run the subprocess with retry logic
    while retry_count < max_retries:
        result = subprocess.run(["./run.sh", f"{gen}_{i}.json", str(density)])
        if result.returncode == 0:
            # Subprocess succeeded
            break
        else:
            # Subprocess failed
            retry_count += 1
            print(f"Attempt {retry_count} failed. Retrying...")

    try:
        agents_data = process_json_file(f"logs_{density}/{gen}_{i}.json")
    except FileNotFoundError:
        print(f"Log file for generation {gen}, individual {i} not found.")
        return 0
    worm_data = agents_data.get('agents', {}).get('worm', {}).get('default', [])
    agents =[]
    for agent in worm_data:
        position = (agent['x'], agent['y'])
        speed_x = agent['speed_x']
        speed_y = agent['speed_y']
        agents.append(Agent(position, speed_x, speed_y))

    #agents = generate_random_agents(100)
    return agents

def create_individual(density):
    genotype = create_genotype(density)
    vector= None
    phenotype = None
    novelty = None
    return Individual(genotype,phenotype, vector, novelty)

def coverage(phenotype):
    cell_size = 20/128
    num_cells = 128
    total_cells = num_cells * num_cells
    coverage_grid = np.zeros((num_cells, num_cells))

    positions = np.array([i.position for i in phenotype])

    for i in range(num_cells):
        for j in range(num_cells):
            cell_center = np.array([(i + 0.5) * cell_size, (j + 0.5) * cell_size])
            if any(np.linalg.norm(position - cell_center) <= 0.2 for position in positions):
                coverage_grid[i, j] = 1

    covered_cells = np.sum(coverage_grid)
    coverage_ratio = covered_cells / total_cells
    return coverage_ratio
def dispersion(phenotype):
    positions = np.array([i.position for i in phenotype])

    pairwise_distances = []
    for (i, j) in combinations(positions, 2):
        distance = np.linalg.norm(i - j)
        pairwise_distances.append(distance)

    average_pairwise_distance = np.mean(pairwise_distances)
    return average_pairwise_distance
def compactness(phenotype):
    positions = np.array([i.position for i in phenotype])

    # Convex Hull Area
    hull = ConvexHull(positions)
    return hull.area
def average_speed(phenotype):

    res = []
    for i in phenotype:
        res.append(abs(np.sqrt(i.speed_x**2 + i.speed_y**2)) * 2)
    return np.sum(res) / len(phenotype)


def scatter(phenotype):
    res = 0
    mu = np.mean([i.position for i in phenotype], axis=0)
    for i in phenotype:
        res += np.linalg.norm(np.array(i.position) - mu)**2
    R = np.sqrt(800) / 2
    return res / (R**2 * len(phenotype))


def create_vector(phenotype):
    return Vector(average_speed(phenotype),compactness(phenotype),dispersion(phenotype),scatter(phenotype),coverage(phenotype))


