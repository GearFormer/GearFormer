# Generative models for kinematic assemblies

## Create dataset:
1. Create all the possible sequences with maximum length 10(max 10 components) that not only respect the grammar but also are physically feasible. The sequences are going to be saved in pickle files in "examples" directory.
<pre>
cd gear_design_dl_dataset
bash parallel.sh [number of cpu cores]
</pre>

2. Put the generated sequences in "yasaman_example" folder, run the simulator and the sequences with their corresponding output will be saved in "simulator_output" folder.
<pre>
cd ../gear-train-simulator
mv ../gear_design_dl_dataset/examples/* yasaman_example
source gear_simulator/bin/activate
bash parallel.sh [number of cpu cores]
</pre> 

## Run the deep learning model:
3. Put the generated pickle files in "simulator_output" in the data forlder and mv around 30% of the files to data_val for the validation.
4. To run deep learning methods you should change the config files in each folder first. You can run the LSTM models with `python train.py` and Transformer model with `python transformers.py`. 
5. To do evaluation you can run `python evaluation.py` that gives you the validity scores. To run evaluation with simulator, first move the generated csv file(created by running evaluation.py) to simulator folder and the go to the simulator folder and run "evaluation_with_simulator.py".
