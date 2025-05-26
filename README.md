# Multimodal Reasoning with Multimodal Knowledge Graph
Python tutorial of MR-MKG paper for ACL 2024: [[Paper](https://aclanthology.org/2024.acl-long.579.pdf)]  
To develop the tutorial code, I referred to the repository of the **MARS dataset** used in the paper: [[GitHub](https://github.com/zjunlp/MKG_Analogy)]

First, as instructed in the README of the MARS repository, I downloaded the image data and moved it to ```MKG_Analogy/MarT/dataset/MARS/images```. Then, I conducted all tasks from the ```MKG_Analogy/MarT/dataset``` directory.

## The network pipeline.  
This paper explores a method for performing Multimodal Reasoning by combining **LLMs and MMKG**. By leveraging the ```strong reasoning capabilities of LLMs``` and the ```up-to-date information provided by MMKGs```, it becomes possible to effectively solve a wide range of complex, multimodal tasks.
![pipe](https://github.com/user-attachments/assets/6b19a6d2-d6b2-4c8b-bc0c-5ba51cc728a6)

## Multimodal analogical reasoning and MMKG
**Multimodal Analogical Reasoning** is a task for inferring analogous relational patterns across different modalities. For example, it addresses questions like ```lion (image) : jungle (image) = bear (text) : ? (text) -> (Answer: forest)```. 

**Multimodal Knowledge Graph (MMKG)** is a knowledge graph that represents relationships between entities using multiple modalities, such as images, textual descriptions, and other forms of data.
Since MMKG is a large-scale graph that represents relationships, if a relevant subgraph can be appropriately retrieved based on the question, it can be effectively utilized for Multimodal Analogical Reasoning.

**MARS** is a dataset designed for ```Multimodal Analogical Reasoning```, and it provides a MMKG called ```MarKG```. The following is a description of the MarKG files. More details can be found on the MARS GitHub repository.
```bash
entity2text: entity_id to entity_description
relation2text: relation_id to relation_description
wiki_tuple_ids: knowledge triplets with (head_id, rel_id, tail_id) format
```

## Multimodal analogical reasoning and LLM
Here is an example prompt for performing Multimodal Analogical Reasoning using an LLM:
```bash
Question: "croissant" and "pastry" are related by the relation "subclass of". Considering a similar relationship, what is the text that has a same relation with the given image?
Image: {soy milk}
Answer: "plant milk"
```

## Tutorial with Jupyter Notebook
**1. Knowledge Graph [[Google Colab](https://colab.research.google.com/drive/1njY_YOQ8Yllo3DgAWcJf2DvzAFUrRkFr?usp=sharing)]**  
This tutorial covers how to load a Knowledge Graph (KG) and retrieve a subgraph corresponding to a given query.
   
**2. Multi Modal Knowledge Graph [[Google Colab](https://colab.research.google.com/drive/19JOGkbsG6qhUb9curWpEpkGF8KBLXBv0?usp=sharing)]**  
This tutorial explains how to extract image embeddings for MMKG, and how to build a Sub-MMKG.
   
**3. MR-MKG Modeling [[Google Colab](https://colab.research.google.com/drive/14jKZiRs-4NCVYYLRs_GMuoPifWCfepOZ?usp=sharing)]**  
This tutorial shows how to create a Sub-MMKG for multimodal analogical reasoning, and how to make prompt for LLM.

## Environments  
PyTorch >= 1.13.1  
Python >= 3.8  
sklearn  
opencv  
torchvision  
transformers  
Other common packages.

## Note
The provided tutorial materials were implemented based on the original paper and may contain differences from the official implementation. For example, instead of using RGAT [[Paper](https://aclanthology.org/2020.emnlp-main.597/)] as the KG encoder, I used NNConv [[Paper](https://arxiv.org/abs/1704.01212)][[Code](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.NNConv.html)]. This decision was made because the official code is currently unavailable, and NNConv serves as a viable alternative that supports edge embeddings in the graph.  
A tutorial for training and inference may be added in the future if time permits!
