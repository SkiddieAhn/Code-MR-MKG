# MR-MKG
Python tutorial of MR-MKG paper for ACL 2024: [[MR-MKG](https://aclanthology.org/2024.acl-long.579/)]  
To develop the tutorial code, I referred to the repository of the MARS dataset used in the paper: [[GitHub](https://github.com/zjunlp/MKG_Analogy)]

First, as instructed in the README of the MARS repository, I downloaded the image data and moved it to ```MKG_Analogy/MarT/dataset/MARS/images```. Then, I conducted all tasks from the ```MKG_Analogy/MarT/dataset``` directory.

## The network pipeline.  
![pipe](https://github.com/user-attachments/assets/6b19a6d2-d6b2-4c8b-bc0c-5ba51cc728a6)

## Environments  
PyTorch >= 1.13.1  
Python >= 3.8
sklearn  
opencv  
torchvision  
transformers  
Other common packages.  

## Tutorial with Jupyter Notebook
1. **Knowledge Graph** [[Google Colab](https://colab.research.google.com/drive/1om31YdESmQ4OG3c-gK_Q7XQeSI_O28nI?usp=sharing)]  
   : This tutorial covers how to load a Knowledge Graph (KG) and retrieve a subgraph corresponding to a given query.
   
2. **Multi Modal Knowledge Graph** [[Google Colab](https://colab.research.google.com/drive/15gxlp1H1hKy9fEdoUi4A_LHcvh6PUuUK?usp=sharing)]  
   : This tutorial explains how to extract and store image embeddings for Multi Modal KG (MMKG), and how to build a Sub-MMKG.
   
3. **MR-MKG Modeling** [[Google Colab](https://colab.research.google.com/drive/1kZnB-EZx16pvCCGJ5PZb5Bv8EnVcqtWe?usp=sharing)]  
   : This tutorial shows how to create a Sub-MMKG for multimodal analogical reasoning, and how to make prompt for LLM.

## Note
The provided tutorial materials were implemented based on the original paper and may contain differences from the official implementation.  
For example, instead of using RGAT[[Paper](https://aclanthology.org/2020.emnlp-main.597/)] as the KG encoder, I used NNConv[[Paper](https://arxiv.org/abs/1704.01212)][[Code](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.NNConv.html)]. This decision was made because the official code is currently unavailable, and NNConv serves as a viable alternative that supports edge embeddings in the graph.
A tutorial for training and inference may be added in the future if time permits!
