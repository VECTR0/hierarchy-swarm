from agent import Agent 
from environment import Environment 
from evaluator import Evaluator

def main():
    env = Environment(100, 200)
    agt = Agent(env)
    eva = Evaluator()
    
    env.add_agents([agt])
    env.simulate(iterations=100)

main()

