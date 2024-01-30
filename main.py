import copy
import datetime
import json
import os
import argparse
import random
import time
import uuid
from mods import brains
from mods.brains.mlp_brain import MLPBrain
from mods.environments.basic_grid.basic_agent import AgentAttributes, BasicAgent
from mods.environments.basic_grid.basic_grid import BasicGrid
from PIL import Image
from mods.brains import Brain, RandomBrain


brains = []
attributes = []

evaluations = []

def add_brain(brain: Brain):
    brains.append(brain)

def add_agent(attribute: AgentAttributes):
    attributes.append(attribute)

def evaluate(brains: 'list[Brain]', attributes: 'list[AgentAttributes]', agents_per_brain_num: int, max_iterations: int, render_mode, grid_image_path, seed: int):
    assert grid_image_path is not None
    grid_image = Image.open(grid_image_path)
    env = BasicGrid(agents_num=agents_per_brain_num*len(brains), render_mode=render_mode,
                    grid_image=grid_image, max_iterations=max_iterations)
    env.reset(seed=42)

    used_brains = []

    idx = 0
    for a in env.agents:
        a.attributes = attributes[idx % len(attributes)]
        new_brain = brains[idx % len(brains)].clone()
        new_brain.id = a.id
        used_brains.append(new_brain)
        idx += 1

    observations, infos = env.reset(seed=seed)
    # print(observations, infos)
    swarm_reward = 0

    while env.agents:
        actions = {b.id: b.act(observations[b.id], env) for b in used_brains}

        observations, rewards, terminations, truncations, infos = env.step(
            actions)
        swarm_reward += sum(rewards.values())

        env.render()
    
    env.close()

    evaluations.append({
        "brains": brains,
        "attributes": attributes,
        "swarm_reward": swarm_reward
    })

def main_old():
    evaluate(brains, attributes, agents_per_brain_num=10, max_iterations=100, render_mode='None', grid_image_path="./data/environments/0.png", seed=21)

def evaluate_brains_agents(brains: 'list[Brain]', agents: 'list[AgentAttributes]', agents_per_brain_num: int, max_iterations: int, render_mode, grid_image_path, seed: int):
    brain_instances = []
    
    grid_image = Image.open(grid_image_path)
    env = BasicGrid(agents_num=agents_per_brain_num*len(brains), render_mode=render_mode, grid_image=grid_image, max_iterations=max_iterations)
    env.reset(seed=seed)
    idx = 0
    for a in env.agents:
        # a.attributes = attributes[idx % len(attributes)]
        new_brain = brains[idx % len(brains)].clone()
        new_brain.id = a.id
        brain_instances.append(new_brain)
        a.attributes = agents[idx % len(agents)]
        idx += 1

    observations, infos = env.reset(seed=seed)
    fitness = 0

    while env.agents:
        actions = {b.id: b.act(observations[b.id], env) for b in brain_instances}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        fitness += sum(rewards.values())

        env.render()
    env.close()
    return fitness

def mutate_brain_agent_pair(brain_agent_tuple_a, random_brain_agent_tuple_b, mutation_chance=0.1):
    brain_a, agent_a = brain_agent_tuple_a
    brain_b, agent_b = random_brain_agent_tuple_b

    random_brain_cross_over_point = random.randint(0, len(brain_a)-1)
    random_agent_cross_over_point = random.randint(0, len(agent_a)-1)

    new_brain_a = brain_a[:random_brain_cross_over_point] + brain_b[random_brain_cross_over_point:]
    new_agent_a = agent_a[:random_agent_cross_over_point] + agent_b[random_agent_cross_over_point:]

    new_brain_a = [type(brain_a).deserialize(brain_a.serialize()) for brain_a in new_brain_a]
    new_agent_a = [copy.copy(agent_a) for agent_a in new_agent_a]

    if random.random() < mutation_chance:
        for brain in new_brain_a:
            brain.mutate(None)
    if random.random() < mutation_chance:
        for agent_attributes in new_agent_a:
            BasicAgent.mutate(agent_attributes)
    return (new_brain_a, new_agent_a)

def get_time_from_seconds(s):
    return str(datetime.timedelta(seconds=s))
    
def write_json(json_data, path):
    data_file = open(path, 'w')
    data_file.write(json.dumps(json_data, indent=2))
    data_file.close()

def run_simulation(grid_image_path:str, brain_type, brain_setup, hierarchy_levels, agents_per_brain_num, population_size, iterations, crossover_chance, mutation_chance, max_iterations, render_mode, seed):
    simulation_data = {
        "id": str(uuid.uuid4()),
        "start_time": int(time.time()),
        "end_time": int(time.time()),

        "research": {
            "grid_image_path": grid_image_path,
            "brain_type": str(brain_type),
            "brain_setup": brain_setup,
            "hierarchy_levels": hierarchy_levels,
            "agents_per_brain_num": agents_per_brain_num, 
        },

        "evolution": {
            "iterations": iterations,
            "population_size": population_size,
            "crossover_chance": crossover_chance,
            "mutation_chance": mutation_chance,
        },

        "evaluation": {
            "max_iterations": max_iterations,
            "render_mode": render_mode,
            "seed": seed
        }
    }

    os.makedirs('./data/simulations/' + simulation_data["id"], exist_ok=True)
    os.makedirs('./data/simulations/' + simulation_data["id"] +'/generations', exist_ok=True)
    write_json(simulation_data, './data/simulations/' + simulation_data["id"] + '/simulation_data.json')

    print(f"brain_type {simulation_data['research']['brain_type']}\n\
hierarchy_levels {simulation_data['research']['hierarchy_levels']}\n\
agents_per_brain_num {simulation_data['research']['agents_per_brain_num']}\n\
population_size {simulation_data['evolution']['population_size']}\n\
iterations {simulation_data['evolution']['iterations']}\n\
crossover_chance {simulation_data['evolution']['crossover_chance']}\n\
mutation_chance {simulation_data['evolution']['mutation_chance']}\n\
max_iterations {simulation_data['evaluation']['max_iterations']}\n\
grid_image_path {simulation_data['research']['grid_image_path']}\n\
seed {simulation_data['evaluation']['seed']}")

    # log initial state
    population = []
    population_size = simulation_data["evolution"]["population_size"]
    hierarchy_levels = simulation_data["research"]["hierarchy_levels"]
    evolution_iterations = simulation_data["evolution"]["iterations"]
    start_time = simulation_data["start_time"]
    
    # create initial population
    for _ in range(population_size):    
        brain_types = []
        agent_attributes = []
        for _ in range(hierarchy_levels):
            if brain_type == "random":
                brain_types.append(RandomBrain()) 
            elif brain_type == "mlp":
                brain_types.append(MLPBrain(simulation_data["research"]["brain_setup"])) 
            
            agent_attributes.append(BasicAgent.generate_random_attributes())
        population.append((brain_types, agent_attributes))
    
    # run evolution
    for evolution_iteration in range(1, evolution_iterations+1):
        population_with_fitness = []
        for brain_types, agent_attributes in population:
            fitness = evaluate_brains_agents(brain_types, agent_attributes, 
                                             agents_per_brain_num=simulation_data["research"]["agents_per_brain_num"],
                                             max_iterations=simulation_data["evaluation"]["max_iterations"],
                                             render_mode=simulation_data["evaluation"]["render_mode"],
                                             grid_image_path=simulation_data["research"]["grid_image_path"],
                                             seed=simulation_data["evaluation"]["seed"])
            if fitness == 0:
                fitness = 1
            population_with_fitness.append((brain_types, agent_attributes, fitness))
            # print(brain_types, agent_attributes, fitness)


        iteration_data = {
            "generation": evolution_iteration,
            "population_with_fitness": [([brain.serialize() for brain in brain_types], [aa.serialize() for aa in agent_attributes], fitness) for (brain_types, agent_attributes, fitness) in population_with_fitness]
        }
        write_json(iteration_data, './data/simulations/' + simulation_data["id"] + '/generations/' + str(evolution_iteration) + '.json')        
        # pick agents to survive
        total_fitness = sum([x[2] for x in population_with_fitness])
        average_fitness = total_fitness / len(population_with_fitness)
        best_fitness = max([x[2] for x in population_with_fitness])

        # uneven roulette wheel selection
        population_with_copy_value = [(brain_types, agent_attributes, population_size * fitness/total_fitness) for (brain_types, agent_attributes, fitness) in population_with_fitness]
        population_with_copy_value.sort(key=lambda x: x[2], reverse=True)
        # minimal_fiteness = population_with_copy_value[-1][2]
        # population_with_copy_value = [(brain_types, agent_attributes, fitness-minimal_fiteness) for (brain_types, agent_attributes, fitness) in population_with_copy_value]

        new_population_parents = []
        for i in range(population_size):
            num = random.random() * population_size
            for j in range(len(population_with_copy_value)):
                if num < population_with_copy_value[j][2]:
                    brain, agent, copy_count = population_with_copy_value[j]
                    new_population_parents.append((brain, agent))
                    break
                num -= population_with_copy_value[j][2]

        new_population = []
        for brain_types, agent_attributes in new_population_parents:
            if random.random() < simulation_data["evolution"]["crossover_chance"]: 
                random_brain_agent = new_population_parents[random.randint(0, len(new_population_parents)-1)]
                new_population.append(mutate_brain_agent_pair((brain_types, agent_attributes), random_brain_agent, mutation_chance=simulation_data["evolution"]["mutation_chance"]))
            else:
                new_population.append((brain_types, agent_attributes))
        
        population = new_population
        # check stop conditions
        # for brain_types, agent_attributes, fitness in population_with_fitness:
        #     print(f"fitness {fitness}")
        generation_end_time = int(time.time())
        eta_s = (generation_end_time - start_time) * (evolution_iterations - evolution_iteration) / (evolution_iteration)
        avg_s = (generation_end_time - start_time) / (evolution_iteration)
        print(f"elapsed {get_time_from_seconds(int(time.time()-start_time))} avg {int(avg_s)}s eta {get_time_from_seconds(int(eta_s))} generation {evolution_iteration}/{simulation_data['evolution']['iterations']} avg/max {round(average_fitness)}/{round(best_fitness)} pop_size {len(population)}")

    # log final state

def collect_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain', type=str, default='random', help='Brain type')
    parser.add_argument('--hl', type=str, default='1', help='hierarchy_levels')

    args = parser.parse_args()

    brain_type = args.brain
    brain_setups = [None]
    if brain_type == "mlp":
        brain_setups = []
        brain_setups.append([9+15,5,10])
        brain_setups.append([9+15,10,10])
        brain_setups.append([9+15,20,10])
        brain_setups.append([9+15,7, 3,10])
    for brain_setup in brain_setups:
        for grid_image_path in ["./data/environments/1.png", "./data/environments/2.png", "./data/environments/0.png"]:
            total_agent_count = 20
            hierarchy_levels = int(args.hl)
            for evolution_params in [(0.4,0.4), (0.4,0.7),(0.5,0.1)]:
                crossover_chance, mutation_chance = evolution_params
                for _ in range(3):
                    run_simulation(
                        grid_image_path=grid_image_path,
                        brain_type=brain_type,
                        brain_setup=brain_setup,
                        hierarchy_levels=hierarchy_levels,
                        agents_per_brain_num=total_agent_count//hierarchy_levels,
                        population_size=30,
                        iterations=200,
                        crossover_chance=crossover_chance,
                        mutation_chance=mutation_chance,
                        max_iterations=200,
                        render_mode=None,
                        seed=random.randint(0, 1000000))

def main():
    pass

if __name__ == "__main__":
    main()