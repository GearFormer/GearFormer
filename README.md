## GearFormer: Deep Generative Model for Mechanical System Configuration Design

Deep generative model approach to mechanical system configuration design, focusing on gear train synthesis.

Paper: https://arxiv.org/abs/2409.06016

Website: https://gearformer.github.io/

**Setup**
<pre>
docker build -t [image name] .
docker run --gpus all -it -v [GearFormer repo directory]:/app gearformer
</pre>

**GearFormer Dataset**
<pre>
cd dataset

train.csv
val.csv
test.csv
</pre>

**Evaluating the GearFormer model**

Specify the checkpoints you'd like to evaluate in /gearformer_model/utils/config_file.py

<pre>
cd ../gearformer_model
python evaluation.py --val_data_path [path to test_data.csv]
</pre>

This creates a new csv file that you will use in the next step.

<pre>
cd ../simulator
python evaluating_with_simulator.py --csv_path [path to csv file generated in the above step]
</pre>

To evaluate with a random baseline:

<pre>
cd ../gearformer_model
python evaluation_random.py --val_data_path [path to test_data.csv]

cd ../simulator
python evaluating_with_simulator.py --csv_path [path to csv file generated in the above]
</pre>


**Search (including hybrid) methods for gear train design**

Monte Carlo tree search and Estimation of Distribution Algorithm for gear train design.

To run:

<pre>
export PYTHONPATH=$PYTHONPATH:/app/
export PYTHONPATH=$PYTHONPATH:/app/gearformer_model
export PYTHONPATH=$PYTHONPATH:/app/simulator

cd gearformer_search

python run.py
</pre>

Change the search settings at the top of the run.py file.

For the paper, we ran:
1. EDA
```
search_method = "EDA"
eda_iterations = 10
eda_population_size = 1000
eda_truncation_rate = 0.2
max_search_depth = 21
hybrid_mode = False
problems_file = "data/benchmark_problems.json"
results_file = "data/output_EDA.json"
```
2. MCTS
```
search_method = "MCTS"
mcts_iterations = 10000
max_search_depth = 21
hybrid_mode = False
problems_file = "data/benchmark_problems.json"
results_file = "data/output_MCTS.json"
```
3. EDA+GF
```
search_method = "EDA"
eda_iterations = 10
eda_population_size = 100
eda_truncation_rate = 0.2
hybrid_mode = True
hybrid_mode_search_depth = 6
problems_file = "data/benchmark_problems.json"
results_file = "data/output_EDA_hybrid.json"
```
4. MCTS+GF
```
search_method = "MCTS"
mcts_iterations = 1000
hybrid_mode = True
hybrid_mode_search_depth = 6
problems_file = "data/benchmark_problems.json"
results_file = "data/output_MCTS_hybrid.json"
```

Copyright 2025, Autodesk, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software on a non-commercial research or educational basis only, the rights to use, copy, modify, merge, publish, distribute, and/or sublicense the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


