from simulator.gear_train_simulator import Simulator
from gearformer_model.utils.helper import is_physically_feasible
from gearformer_model.utils.config_file import config
from gearformer import autocomplete
import json
import signal
import numpy as np
import itertools

class TimeoutException(Exception):
    pass

# Function to handle the timeout signal
def timeout_handler(signum, frame):
    raise TimeoutException

# Set the signal handler for SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

class EDA:

    def __init__(self, num_variables, init_pop_size, trunc_rate, problem, hybrid_mode=False):
        self.hybrid_mode = hybrid_mode

        args = config()
        self.language_path = args.language_path
        self.catalogue_path = args.catalogue_path

        with open(self.language_path, 'r') as file:
            self.language = json.load(file)

        self.int2token = {}
        self.token2int = {}

        token_idx = 0
        for key, vals in self.language["vocab"].items():
            for token in vals:
                self.int2token[token_idx] = token
                self.token2int[token] = token_idx
                token_idx += 1

        self.num_variables = num_variables
        self.var_ranges = []
        for i in range(0, self.num_variables):
            self.var_ranges.append(list(self.int2token.keys()))
        self.population_size = init_pop_size
        self.truncation_rate = trunc_rate
        self.problem = problem

        self.current_solutions = []
        self.current_best_solution = ()
        self.current_probability_model = {
            "p_grammar": {}
        }

        self._iteration_num = 0
        self._solutions_trunc = []
        self._global_best = ([], 1e9)

        self.freq_grammar = {}
        self.init_freq()
        self.init_p()

    def init_freq(self):
        grammar = self.language["grammar"]
        for g in grammar:

            LHS = g["LHS"]
            lhs_enum = [self.token2int[t] for t in self.language["vocab"][LHS]]

            RHS = g["RHS"]
            rhs_enum = []
            for rhs in RHS:
                rhs_expanded = []
                for token_class in rhs:
                    token_list = [self.token2int[t] for t in self.language["vocab"][token_class]]
                    rhs_expanded.append(token_list)
                rhs_enum += list(itertools.product(*rhs_expanded))

            for lhs_i in lhs_enum:
                self.freq_grammar[lhs_i] = {}
                for rhs_i in rhs_enum:
                    self.freq_grammar[lhs_i][rhs_i] = 0

    def init_p(self):
        p = {}
        for k1, v1 in self.freq_grammar.items():
            p[k1] = {}
            v1_length = float(len(v1))
            for k2, v2 in v1.items():
                p[k1][k2] = 1 / v1_length
        self.current_probability_model["p_grammar"] = p

    def init_pop_with_gf(self):
        problem = self.problem

        input_ = [0] * 8
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

        seq = ["<start>"]
        init_population = []
        for i in range(self.population_size):
            new_seq = autocomplete(input_, seq)
            if 'EOS' in new_seq:
                i -= 0
            else:
                end_token_chioces = range(3, self.num_variables + 1)
                end_token = np.random.choice(end_token_chioces)
                sol = [self.token2int[t] for t in new_seq[:end_token]] + [1]
                init_population.append(np.array(sol))

        return init_population

    def select(self, current_solutions, truncation_rate, population_size, iteration_num, global_best):
        # selection
        obj_rank = np.argsort(np.array(current_solutions, dtype=object)[:, 1])
        truncation_n = int(truncation_rate * population_size)
        solutions_trunc = np.array(current_solutions, dtype=object)[obj_rank][:truncation_n]

        current_best_solution = solutions_trunc[0]
        if iteration_num == 0:
            global_best = current_best_solution
        else:
            if current_best_solution[1] < global_best[1]:
                global_best = current_best_solution

        return current_solutions, solutions_trunc, current_best_solution, global_best

    def evaluate(self, current_population):

        results = []
        eval_count = 0
        for x in current_population:
            print(eval_count)
            results.append((x, self.obj_f(x)))
            eval_count += 1

        return results

    def evolve_first(self):
        current_population = self.init_pop_with_gf()

        # evaluation
        self.current_solutions= \
            self.evaluate(current_population)

        # selection
        self.current_solutions, self._solutions_trunc, self.current_best_solution, self._global_best = \
            self.select(self.current_solutions, self.truncation_rate, self.population_size,
                               self._iteration_num, self._global_best)

        population_trunc = self._solutions_trunc[:,0]

        # probability model estimation
        for sol in population_trunc:
            int_idx = 0
            while int_idx < len(sol):
                lhs_int = sol[int_idx]
                if lhs_int == 1:
                    break

                freq_counted = False
                if lhs_int in self.freq_grammar:
                    for i in range(2, 5):
                        rhs_ints = tuple([t_i for t_i in sol[int_idx+1:int_idx+i]])
                        if rhs_ints in self.freq_grammar[lhs_int]:
                            freq_counted = True
                            self.freq_grammar[lhs_int][rhs_ints] += 1
                            int_idx += len(rhs_ints)
                            break
                    if not freq_counted:
                        int_idx += 1

                else:
                    int_idx += 1

        for k1, v1 in self.freq_grammar.items():
            f_sum = sum(self.freq_grammar[k1].values())
            if f_sum > 0:
                for k2, v2 in v1.items():
                    self.current_probability_model["p_grammar"][k1][k2] = (self.freq_grammar[k1][k2] / float(f_sum))

        self._iteration_num += 1

    def evolve(self):
        # sampling
        current_population = []

        for i in range(0, self.population_size):
            new_sol = [0]
            while len(new_sol) < self.num_variables:
                p_dict = self.current_probability_model["p_grammar"][new_sol[-1]]
                p_keys = list(p_dict.keys())
                p_vals = list(p_dict.values())
                choice_range = range(0, len(p_keys))

                np.random.seed(None)
                int_to_add = p_keys[int(np.random.choice(choice_range, size=1, p=p_vals)[0])]
                new_sol += [i for i in int_to_add]
                if new_sol[-1] == 1:
                    break

            if new_sol[-1] != 1:
                new_sol.append(1)
            current_population.append(np.array(new_sol))

        # evaluations
        self.current_solutions= \
            self.evaluate(current_population)

        # selection
        self.current_solutions, self._solutions_trunc, self.current_best_solution, self._global_best = \
            self.select(self.current_solutions, self.truncation_rate, self.population_size,
                               self._iteration_num, self._global_best)
        population_trunc = self._solutions_trunc[:,0]

        # probability model estimation
        for sol in population_trunc:
            int_idx = 0
            while int_idx < len(sol):
                lhs_int = sol[int_idx]
                if lhs_int == 1:
                    break

                freq_counted = False
                if lhs_int in self.freq_grammar:
                    for i in range(2, 5):
                        rhs_ints = tuple([t_i for t_i in sol[int_idx + 1:int_idx + i]])
                        if rhs_ints in self.freq_grammar[lhs_int]:
                            freq_counted = True
                            self.freq_grammar[lhs_int][rhs_ints] += 1
                            int_idx += len(rhs_ints)
                            break
                    if not freq_counted:
                        int_idx += 1

                else:
                    int_idx += 1

        for k1, v1 in self.freq_grammar.items():
            f_sum = sum(self.freq_grammar[k1].values())
            if f_sum > 0:
                for k2, v2 in v1.items():
                    self.current_probability_model["p_grammar"][k1][k2] = (self.freq_grammar[k1][k2] / float(f_sum))

        self._iteration_num += 1

    def obj_f(self, x):

        problem = self.problem
        simulator = Simulator()

        if self.hybrid_mode:
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

            incom_seq = [self.int2token[i] for i in x]
            try:
                signal.alarm(2)
                seq = autocomplete(input_, incom_seq[:-1])
                signal.alarm(0)
                input_data = {
                    "id": "0",
                    "gear_train_sequence": seq
                }
            except TimeoutException:
                print("completion timed out!")
                obj = 1e9
            except:
                print("completion failed")
                obj = 1e9
                seq = incom_seq
        else:
            seq = [self.int2token[i] for i in x]

        try:
            if is_physically_feasible(seq, self.catalogue_path):
                try:
                    input_data = {
                        "id": "0",
                        "gear_train_sequence": seq
                    }
                    signal.alarm(2)
                    results = simulator.run(input_data)
                    signal.alarm(0)

                    if results["output_motion_type"] != problem["output_motion_type"]:
                        obj = 1e9
                    elif results["input_motion_type"] != problem["input_motion_type"]:
                        obj = 1e9
                    else:
                        obj = 0
                        rot_euclidean = np.linalg.norm(np.array(results["output_motion_vector"]) - np.array(
                            problem["output_motion_vector"]))
                        obj += rot_euclidean

                        speed_ratio_diff = abs(
                            results["output_motion_speed"] - problem["output_motion_speed"])
                        obj += speed_ratio_diff

                        pos_euclidean = np.linalg.norm(
                            np.array(results["output_position"]) - np.array(problem["output_position"]))
                        obj += pos_euclidean

                        obj += results["weight"]

                except TimeoutException:
                    print("simulation timed out!")
                    obj = 1e9
                except:
                    print("simulation failed")
                    obj = 1e9
            else:
                print("not feasible")
                obj = 1e9
        except:
            print("feasibility failed")
            obj = 1e9

        if obj != 1e9:
            with open("data/best.json", 'r') as file2:
                best = json.load(file2)
            file2.close()
            if obj < best["obj"]:
                best["obj"] = obj
                best["seq"] = seq
                best["results"] = results
                with open("data/best.json", 'w') as file3:
                    json.dump(best, file3, indent=4)
                file3.close()
        print(obj)
        input()
        return obj
