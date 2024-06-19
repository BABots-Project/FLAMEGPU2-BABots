import numpy as np
import random

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
    "AGENT_COUNT": (300,300),
    "PERSISTENCE_FACTOR": (0.1, 0.9),
    "SENSING_RANGE": (1, 30),
    "ALPHA_ATTRACTANT":(13,150),
    "ALPHA_REPELLENT":(13,150),
    "BETA_ATTRACTANT": (0.0011, 0.002),
    "BETA_REPELLENT": (0.0011, 0.002),
    "ATTRACTANT_CREATION": (0.0, 0.015),
    "REPELLENT_CREATION": (0.0, 0.0015),
    "OXYGEN":(0,0),
    "H": (0,0)
}

def process_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
def create_genotype():
    values = [random.uniform(min_val, max_val) for min_val, max_val in parameter_ranges.values()]
    return Genotype(*values)

def generate_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agents.append(Agent((random.uniform(0, 20), random.uniform(0, 20)),random.uniform(-10, 10),random.uniform(-10, 10)))
    return agents
def create_phenotype(genotype,gen,i):
    print("generation: "+ str(gen)+" individual: "+str(i))
    with open("parameters.json", "w") as f:
        f.write(individual_to_json(genotype))
    subprocess.run(["./run.sh", f"{gen}_{i}.json"])
    try:
        agents_data = process_json_file(f"logs/{gen}_{i}.json")
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

    return agents

def create_individual():
    genotype = create_genotype()
    vector= None
    phenotype = None
    novelty = None
    return Individual(genotype,phenotype, vector, novelty)

def average_speed(phenotype):
    res = []
    for i in phenotype:
        res.append(abs(np.sqrt(i.speed_x**2 + i.speed_y**2)) * 2)
    return np.sum(res) / len(phenotype)

def angular_momentum(phenotype):
    total_cross_product = 0
    mu = np.mean([i.position for i in phenotype], axis=0)
    for i in phenotype:
        r_i = np.array(i.position) - mu
        v_i = np.array([i.speed_x, i.speed_y])
        cross_product = np.cross(v_i, r_i)
        total_cross_product += cross_product
    R = np.sqrt(800) / 2
    angular_momentum = total_cross_product / (R * len(phenotype))
    return angular_momentum

def radial_variance(phenotype):
    mu = np.mean([i.position for i in phenotype], axis=0)
    res = 0
    for i in phenotype:
        r_i = np.linalg.norm(np.array(i.position) - mu)
        for y in phenotype:
            s = np.linalg.norm(np.array(y.position) - mu)
        res += (r_i - s) / len(phenotype)
    R = np.sqrt(800) / 2
    return res**2 / (R**2 * len(phenotype))

def scatter(phenotype):
    res = 0
    mu = np.mean([i.position for i in phenotype], axis=0)
    for i in phenotype:
        res += np.linalg.norm(np.array(i.position) - mu)**2
    R = np.sqrt(800) / 2
    return res / (R**2 * len(phenotype))

def group_rotation(phenotype):
    mu = np.mean([i.position for i in phenotype], axis=0)
    total_cross_product = 0
    for i in phenotype:
        r_i = mu / np.linalg.norm(mu)  # Assuming mu/mu was meant to be normalized
        v_i = np.array([i.speed_x, i.speed_y])
        cross_product = np.cross(v_i, r_i)
        total_cross_product += cross_product
    return total_cross_product  # Return the total cross product

def create_vector(phenotype):
    return Vector(average_speed(phenotype),angular_momentum(phenotype),radial_variance(phenotype),scatter(phenotype),group_rotation(phenotype))


