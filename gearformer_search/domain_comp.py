import numpy as np
from gear_design_dl.gear_design_model.utils.helper import is_physically_feasible
from gear_train_simulator.gear_train_simulator import Simulator
import json
import itertools
import signal
from gearformer import autocomplete

class TimeoutException(Exception):
    pass

# Function to handle the timeout signal
def timeout_handler(signum, frame):
    raise TimeoutException

# Set the signal handler for SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


class DomainComp:
    def __init__(self):

        self.language_path = "gear_train_simulator/gear_train_language/language.json"
        with open(self.language_path, 'r') as file:
            self.language = json.load(file)

        self.catalogue_path = "gear_design_dl/gear_design_model/utils/catalogue.json"

        self.state_tree = self.get_state_tree()

    def get_state_tree(self):
        vocab = self.language["vocab"]
        grammar = self.language["grammar"]

        state_tree = {}

        for g in grammar:
            LHS = g["LHS"]
            lhs_enum = [t for t in vocab[LHS]]

            RHS = g["RHS"]
            rhs_enum = []
            for rhs in RHS:
                rhs_expanded = []
                for token_class in rhs:
                    token_list = [t for t in vocab[token_class]]
                    rhs_expanded.append(token_list)
                rhs_enum += list(itertools.product(*rhs_expanded))

            for lhs_i in lhs_enum:
                state_tree[lhs_i] = []
                for rhs_i in rhs_enum:
                    state_tree[lhs_i].append(rhs_i)

        return state_tree

    def get_possible_states(self, state):
        next_states = [state for state in self.state_tree[state[-1]]]
        return next_states

    def is_terminal(self, state):
        return state == ('<end>',)

    def evaluate_reward(self, history, problem, hybrid_mode):

        if hybrid_mode:
            """
            input_[0]: input_ motion type, 1 for T and 0 for R
            input_[1]: output motion type, 1 for T and 0 for R
            input_[2]: speed ratio
            input_[3], input_[4], input_[5]: x, y, z for output position
            input_[6]: output motion vector direction xyz - 0 for x, 1 for y and 2 for z
            input_[7] : output motion vector sign
            """
            input_ = [0]*8
            if problem["input_motion_type"] == 'R':
                input_[0] = 0
            elif problem["input_motion_type"] == 'T':
                input_[0] = 1
            else:
                exit()
            if problem["output_motion_type"] == 'R':
                input_[1] = 0
            elif problem["output_motion_type"] == 'T':
                input_[1] = 1
            else:
                exit()
            input_[2] = problem["output_motion_speed"]
            output_position = problem["output_position"]
            input_[3] = output_position[0]
            input_[4] = output_position[1]
            input_[5] = output_position[2]
            output_motion_vector = problem["output_motion_vector"]
            if output_motion_vector[0] == 1:
                input_[6] = 0
                input_[7] = 1
            elif output_motion_vector[0] == -1:
                input_[6] = 0
                input_[7] = -1
            elif output_motion_vector[1] == 1:
                input_[6] = 1
                input_[7] = 1
            elif output_motion_vector[1] == -1:
                input_[6] = 1
                input_[7] = -1
            elif output_motion_vector[2] == 1:
                input_[6] = 2
                input_[7] = 1
            elif output_motion_vector[2] == -1:
                input_[6] = 2
                input_[7] = -1
            else:
                exit()

            incom_seq = self.parse_history(history)
            try:
                signal.alarm(2)
                seq = autocomplete(input_, incom_seq[:-1])
                signal.alarm(0)
            except TimeoutException:
                seq = incom_seq

        else:
            seq = self.parse_history(history)

        sim = Simulator()

        input_data = {
            "id": "0",
            "gear_train_sequence": seq
        }

        if len(seq) < 5:
            return 0, {}

        try:
            if is_physically_feasible(seq, self.catalogue_path):
                try:
                    signal.alarm(2)
                    results = sim.run(input_data)
                    signal.alarm(0)

                    if results["output_motion_type"] != problem["output_motion_type"]:
                        reward = 0
                    elif results["input_motion_type"] != problem["input_motion_type"]:
                        reward = 0
                    elif results["weight"] == 0.0:
                        reward = 0
                    else:
                        obj = 0
                        rot_euclidean = np.linalg.norm(np.array(results["output_motion_vector"]) - np.array(problem["output_motion_vector"]))
                        obj += rot_euclidean

                        speed_ratio_diff = abs(problem["output_motion_speed"] - results["output_motion_speed"])
                        obj += speed_ratio_diff

                        pos_euclidean = np.linalg.norm(np.array(results["output_position"]) - np.array(problem["output_position"]))
                        obj += pos_euclidean

                        obj += results["weight"]

                        reward = 1/obj

                except TimeoutException:
                    print("simulation timed out!")
                    reward = 0
                    results = {}
                except:
                    print("simulation failed")
                    reward = 0
                    results = {}
            else:
                print("not feasible")
                reward = 0
                results = {}
        except:
            print("feasiblility test failed")
            reward = 0
            results = {}

        with open("best.json", 'r') as fileX:
            best = json.load(fileX)
        fileX.close()
        if reward > best["reward"]:
            best["reward"] = reward
            best["seq"] = seq
            best["results"] = results
        with open("best.json", 'w') as fileY:
            json.dump(best, fileY, indent=4)

        return reward, results

    def parse_history(self, history):
        sequence = []
        for state in history:
            sequence += list(state)
        return sequence

