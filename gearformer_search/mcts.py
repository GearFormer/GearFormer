import math
import random
from domain_comp import DomainComp
import json
import logging
import warnings

warnings.filterwarnings("ignore")

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.dc = DomainComp()

    def is_fully_expanded(self):
        return len(self.children) == len(self.dc.get_possible_states(self.state))

    def best_child(self, exploration_weight=1.4):
        children_weights = [
            (child.reward / child.visits) + exploration_weight * math.sqrt(
                (2 * math.log(self.visits) / child.visits))
            for i, child in enumerate(self.children)
        ]
        best_child_idx = children_weights.index(max(children_weights))
        return self.children[best_child_idx]


class MCTS:
    def __init__(self, problem, max_length, hybrid_mode=False):
        self.best_solution = {
            "solution": [],
            "reward": 0,
            "results": {}
        }
        self.dc = DomainComp()
        self.problem = problem
        self.history = []
        self.max_length = max_length
        self.hybrid_mode = hybrid_mode

    def search(self, root, iterations=100):
        for i in range(0, iterations):
            print(f"\rIteration: {i}", end='')
            self.history = [root.state]
            node = self.select_node(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return root.best_child(exploration_weight=0)

    def select_node(self, node):
        while not self.dc.is_terminal(node.state):
            if node.is_fully_expanded():
                node = node.best_child()
                self.history.append(node.state)
            else:
                new_node = self.expand(node)
                self.history.append(new_node.state)
                return new_node
        return node

    def expand(self, node):
        visited_states = [child.state for child in node.children]
        new_state = random.choice([state for state in self.dc.get_possible_states(node.state)
            if state not in visited_states])
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
        return new_node

    def count_history(self):
        count = 0
        for phrase in self.history:
            for token in phrase:
                count += 1
        return count

    def simulate(self, node):
        current_state = node.state
        while (not self.dc.is_terminal(current_state)) and (self.count_history() <= self.max_length):
            current_state = random.choice(self.dc.get_possible_states(current_state))
            self.history.append(current_state)

        if self.dc.is_terminal(current_state):
            reward, results = self.dc.evaluate_reward(self.history, self.problem, self.hybrid_mode)
            if reward > self.best_solution["reward"]:
                self.best_solution["solution"] = self.dc.parse_history(self.history)
                self.best_solution["reward"] = reward
                self.best_solution["results"] = results
            return reward

        elif self.count_history() > self.max_length:
            self.history.append(('<end>',))
            reward, results = self.dc.evaluate_reward(self.history, self.problem, self.hybrid_mode)
            if reward > self.best_solution["reward"]:
                self.best_solution["solution"] = self.dc.parse_history(self.history)
                self.best_solution["reward"] = reward
                self.best_solution["results"] = results
            return reward

        return 0

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent


def run_MCTS(problem_file, results_file, hybrid_mode):
    with open(problem_file, 'r') as file:
        problems = json.load(file)

    all_results = {
        "input_motion_type": [],
        "output_motion_type": [],
        "output_motion_vector": [],
        "output_position": [],
        "output_motion_speed": [],
        "weight": []
    }
    logging.basicConfig(filename='output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')

    for j in range(0, len(problems)):
        with open("data/best_mcts.json", 'r') as fileX:
            best = json.load(fileX)
        fileX.close()
        best["reward"] = 0.0
        with open("data/best_mcts.json", 'w') as fileY:
            json.dump(best, fileY, indent=4)

        problem = problems[j]
        root_state = Node(('<start>',))
        mcts = MCTS(problem, 4)
        mcts.hybrid_mode = hybrid_mode
        mcts.search(root_state, iterations=1000)

        logging.info("===========================================")
        logging.info("Target requirements:")
        logging.info(f'input motion type: {problem["input_motion_type"]}')
        logging.info(f'output motion type: {problem["output_motion_type"]}')
        logging.info(f'output position: {problem["output_position"]}')
        logging.info(f'output motion vector: {problem["output_motion_vector"]}')
        logging.info(f'output motion speed: {problem["output_motion_speed"]}')
        logging.info(f'-------------------------------------------')
        logging.info(f'Best solution found:')
        logging.info(f'{mcts.best_solution["solution"]}')
        logging.info(f'Evaluated requirements:')
        with open("data/best_mcts.json", 'r') as fileX:
            best = json.load(fileX)
        fileX.close()
        solution = best["results"]
        print(solution)
        logging.info(f'input motion type: {solution["input_motion_type"]}')
        logging.info(f'output motion type: {solution["output_motion_type"]}')
        logging.info(f'output position: {solution["output_position"]}')
        logging.info(f'output motion vector: {solution["output_motion_vector"]}')
        logging.info(f'output motion speed: {solution["output_motion_speed"]}')
        logging.info(f'weight: {solution["weight"]}')

        all_results["input_motion_type"].append((problems[j]["input_motion_type"], solution["input_motion_type"]))
        all_results["output_motion_type"].append((problems[j]["output_motion_type"], solution["output_motion_type"]))
        all_results["output_motion_vector"].append((problems[j]["output_motion_vector"], solution["output_motion_vector"]))
        all_results["output_position"].append((problems[j]["output_position"], solution["output_position"]))
        all_results["output_motion_speed"].append((problems[j]["output_motion_speed"], solution["output_motion_speed"]))
        all_results["weight"].append(solution["weight"])

    with open(results_file) as file:
        json.dump(all_results, file, indent=4)

