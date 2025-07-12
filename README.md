# Long-Short Distance Graph Neural Networks and Improved Curriculum Learning for Emotion Recognition in Conversation (Accepted by ECAI2025)

Emotion Recognition in Conversation (ERC) is a practical and challenging task. This paper proposes a novel multimodal approach, the Long-Short Distance Graph Neural Network (LSDGNN). Based on the Directed Acyclic Graph (DAG), it constructs a long-distance graph neural network and a short-distance graph neural network to obtain multimodal features of distant and nearby utterances, respectively. To ensure that long- and short-distance features are as distinct as possible in representation while enabling mutual influence between the two modules, we employ a Differential Regularizer and incorporate a BiAffine Module to facilitate feature interaction. In addition, we propose an Improved Curriculum Learning (ICL) to address the challenge of data imbalance. By computing the similarity between different emotions to emphasize the shifts in similar emotions, we design a "weighted emotional shift" metric and develop a difficulty measurer, enabling a training process that prioritizes learning easy samples before harder ones. Experimental results on the IEMOCAP and MELD datasets demonstrate that our model outperforms existing benchmarks.

## Requirements
Python 3.11
CUDA 12.2

After configuring the Python environment and CUDA, you can use `pip install -r requirements.txt` to install the following libraries.

torch==2.0.0+cu117
transformers==4.46.3
numpy==1.24.2
pandas==2.1.4
matplotlib==3.7.1
scikit-learn==1.2.2
tqdm==4.67.1

### Training
GPU NVIDIA GeForce RTX 3090 

for IEMOCAP:
`python run.py --dataset_name IEMOCAP --gnn_layers 4 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.4 --emb_dim 2948 --windowpl 5 --diffloss 0.1 --curriculum --bucket_number 5`

for MELD:
`python run.py --dataset_name MELD --gnn_layers 2 --lr 0.00001 --batch_size 64 --epochs 30 --dropout 0.1  --emb_dim 1666 --windowpl 5 --diffloss 0.2  --curriculum --bucket_number  12`