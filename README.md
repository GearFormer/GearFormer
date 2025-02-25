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

**Loading and evaluating the GearFormer model**

<pre>
cd ../simulator
python evaluating_with_simulator.py --csv_path  path/to/csv/file/for/desired/epoch/available/in/checkpoint/folder
</pre>


And to evaluate on the test set:
<pre>
cd ../gearformer_model
python evaluation.py --lr 0.0001 --model_name GearFormer --checkpoint_path path/to/checkpoint/folder --encoder_chackpoint_name name/of/the/encoder/checkpoint/you/want --decoder_chackpoint_name name/of/the/decoder/checkpoint/you/want --val_data_path path/to/test.csv/you generated --WWL 1 --BS 1024
python evaluating_with_simulator.py --csv_path path/to/csv/file/generated/in/previous/step
</pre>


### Search methods for gear train design

Monte Carlo tree search and Estimation of Distribution Algorithm for gear train design

<h4>To run:</h4>

<pre>
cd gearformer_search

python run.py
</pre>

<h4>Change the search settings at the top of the run.py file</h4>

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
