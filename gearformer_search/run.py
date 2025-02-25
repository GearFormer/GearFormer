import json
import logging
from mcts import MCTS, Node
from eda import EDA

# search settings
search_method = "MCTS"  # ["EDA", "MCTS"]
mcts_iterations = 10000
eda_iterations = 10
eda_population_size = 1000
eda_truncation_rate = 0.2
max_search_depth = 21
hybrid_mode = False  # [True, False]
hybrid_mode_search_depth = 6
problems_file = "data/benchmark_problems.json"
results_file = "data/output.json"

if __name__ == "__main__":

    with open(problems_file, 'r') as file:
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
        print("problem no.: ", j)
        problem = problems[j]

        if search_method == "MCTS":
            # file used to keep track of best solution
            with open("data/best.json", 'r') as file1:
                best = json.load(file1)
            file1.close()
            best["reward"] = 0.0
            best["seq"] = []
            with open("data/best.json", 'w') as file2:
                json.dump(best, file2, indent=4)
            file2.close()

            if hybrid_mode:
                root_state = Node(('<start>',))
                mcts = MCTS(problem, max_length=hybrid_mode_search_depth-2, hybrid_mode=True)
                mcts.search(root_state, iterations=mcts_iterations)
                best_solution = mcts.best_solution["solution"]
            else:
                root_state = Node(('<start>',))
                mcts = MCTS(problem, max_length=max_search_depth-4) # -2 for start and end & -2 for extra tokens that can be added in a single grammar "action"
                mcts.search(root_state, iterations=mcts_iterations)
                best_solution = mcts.best_solution["solution"]

        elif search_method == "EDA":
            # file used to keep track of best solution
            with open("data/best.json", 'r') as file1:
                best = json.load(file1)
            file1.close()
            best["obj"] = 1e9
            best["seq"] = []
            with open("data/best.json", 'w') as file2:
                json.dump(best, file2, indent=4)
            file2.close()

            if hybrid_mode:
                eda = EDA(num_variables=hybrid_mode_search_depth, init_pop_size=eda_population_size,
                         trunc_rate=eda_truncation_rate, problem=problem, hybrid_mode=hybrid_mode)
                for i in range(0, eda_iterations):
                    print("iteration number: ", i)
                    if hybrid_mode and i == 0:
                        eda.evolve_first()
                    else:
                        eda.evolve()
                    print()

            else:
                eda = EDA(num_variables=max_search_depth, init_pop_size=eda_population_size,
                          trunc_rate=eda_truncation_rate, problem=problem)
                for i in range(0, eda_iterations):
                    print("iteration number: ", i)
                    eda.evolve()
                    print()

        else:
            print("The search method is not supported")
            exit()

        # log the results
        logging.info("===========================================")
        logging.info("Target requirements:")
        logging.info(f'input motion type: {problem["input_motion_type"]}')
        logging.info(f'output motion type: {problem["output_motion_type"]}')
        logging.info(f'output position: {problem["output_position"]}')
        logging.info(f'output motion vector: {problem["output_motion_vector"]}')
        logging.info(f'output motion speed: {problem["output_motion_speed"]}')
        logging.info(f'-------------------------------------------')
        logging.info(f'Best solution found:')
        with open("data/best.json", 'r') as file1:
            best = json.load(file1)
        file1.close()
        logging.info(f'{best["seq"]}')
        logging.info(f'Evaluated requirements:')
        solution = best["results"]
        logging.info(f'input motion type: {solution["input_motion_type"]}')
        logging.info(f'output motion type: {solution["output_motion_type"]}')
        logging.info(f'output position: {solution["output_position"]}')
        logging.info(f'output motion vector: {solution["output_motion_vector"]}')
        logging.info(f'output motion speed: {solution["output_motion_speed"]}')
        logging.info(f'weight: {solution["weight"]}')

        # to save in the results file
        all_results["input_motion_type"].append((problems[j]["input_motion_type"], solution["input_motion_type"]))
        all_results["output_motion_type"].append(
            (problems[j]["output_motion_type"], solution["output_motion_type"]))
        all_results["output_motion_vector"].append(
            (problems[j]["output_motion_vector"], solution["output_motion_vector"]))
        all_results["output_position"].append((problems[j]["output_position"], solution["output_position"]))
        all_results["output_motion_speed"].append(
            (problems[j]["output_motion_speed"], solution["output_motion_speed"]))
        all_results["weight"].append(solution["weight"])

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)