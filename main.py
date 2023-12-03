import datetime
from http.server import SimpleHTTPRequestHandler
import json
import os
import time
import uuid
import matplotlib.pyplot as plt
from numpy import mat
from sympy import im, parallel_poly_from_expr
from mods.environments.basic_grid.basic_grid import BasicGrid
from pettingzoo.test import parallel_api_test
from PIL import Image
from mods.agents.random_agent import RandomAgent
# testing the parallel api
# env = BasicGrid(agents_num=10)
# env.reset(seed=42)
# parallel_api_test(env, num_cycles=1000)

# image = Image.open("./data/environments/0.png")
# env = BasicGrid(agents_num=200, render_mode="human", grid_image=image, max_iterations=200)
# observations, infos = env.reset(seed=21)
# # print(observations, infos)
# while env.agents:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#     print('nex')

#     observations, rewards, terminations, truncations, infos = env.step(actions)

#     env.render() 
# env.close()


'''
TODO:
- implement observations
- objective
    survive to end iteration
- implement rewards
- implement terminations
    no energy
- implement logging
    agents classes
- implement policies
'''

def run_simulation(agents_num, max_iterations, render_mode, grid_image_path, seed):
    grid_image = Image.open(grid_image_path)
    env = BasicGrid(agents_num=agents_num, render_mode=render_mode, grid_image=grid_image, max_iterations=max_iterations)
    env.reset(seed=42)

    agents = [RandomAgent(id=a.id) for a in env.agents]

    observations, infos = env.reset(seed=seed)
    # print(observations, infos)
    swarm_reward = 0
    simulation_data = {
        "id": str(uuid.uuid4()),
        "agents_num": agents_num,
        "max_iterations": max_iterations,
        "image_path": grid_image_path,
        "time": int(time.time())
    }

    os.makedirs('./data/simulations/' + simulation_data["id"], exist_ok=True)
    sim_data_file = open('./data/simulations/' + simulation_data["id"] + '/simulation_data.json', 'w')
    sim_data_file.write(json.dumps(simulation_data, indent=2))
    sim_data_file.close()
    iterations_data = []
    while env.agents:
        actions = {a.id:a.act(observations[a.id], env) for a in agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        swarm_reward += sum(rewards.values())
        
        # agents_state = [a.id:0 for a in env.agents]

        # print(observations, rewards, terminations, truncations, infos)

        iterations_data.append({
            # "actions": actions, TODO
            "observations": observations,
            "rewards": rewards,
            "swarm_reward": swarm_reward,
        })
        
        env.render()
    sim_data_file = open('./data/simulations/' + simulation_data["id"] + '/iterations_data.json', 'w')
    sim_data_file.write(json.dumps(iterations_data, indent=2))
    sim_data_file.close()
    env.close()

    plt.plot([d["swarm_reward"] for d in iterations_data])
    plt.xlabel('Iterations')
    plt.ylabel('Swarm Reward')
    plt.title("Swarm Reward over time")
    plt.show()

    

run_simulation(agents_num=10, max_iterations=10, render_mode="human", grid_image_path="./data/environments/0.png", seed=21)
