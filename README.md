# XGAO

  XGAO, a novel GNN explanation framework that utilizes global and self-adaptive optimization at the model level.
## Requirement

Python 3.8

Pytorch 1.9.0

Pytorch-Geometric 2.0.0

network 2.5.1

## Data

Our method is evaluated on three datasets, As shown in following table. These datasets can be found in an open-source library [DIG](https://github.com/divelab/DIG/tree/main/dig/xgraph/datasets), which can be directly used to reproduce results of existing GNN explanation methods, develop new algorithms, and conduct evaluations for explanation results.


| Dataset    | Task                  | Data class     |
|------------|-----------------------|----------------|
| [Is_Acyclic]() | Graph classification   | Synthetic data |
| [BA-2motif]()  | Graph classification   | Synthetic data |
| [Twitch Egos](https://snap.stanford.edu/data/twitch_ego_nets.html)| Graph classification  | Real-word data |
| [MUTAG]()     | Graph classification  | Real-word data |


## How to use

Our method can be used to explain both node classification model and graph classification. For each task, you just need to enter the corresponding folder to find the main.py.

For example, when we run the method for explaining the graph classification model which is trained on the Is_Acyclic dataset, we run the following command in the console.

First, we need train a simple GNN model which will be explained:
```
python main.py --mode train
```
If needed, you can change other configuration parameters in the source file 'main.py'.

Before to run the explanation process, you should also change the configuration parameters in the source file 'main.py'. 

Then, running this command to execute the explanation process:

```
python MCTS.py
```
You can also change other configuration parameters shown in the source file 'main.py'.
The explanation results will be saved in the folder '/Img'.  


