import os
import torch
import glob
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def get_ranges(n, interval):
    ranges = []
    for start in range(0, n, interval):
        if start + interval > n:
            end = n
        else:
            end = start + interval
        ranges.append((start, end))
    return ranges


@torch.no_grad()
def get_clip_image_embeddings(images, clip_processor, clip_model, device):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    return clip_model.get_image_features(**inputs) # [num of images, dimenstion]


@torch.no_grad()
def get_clip_text_embeddings(texts, clip_processor, clip_model, device):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    return clip_model.get_text_features(**inputs) # [num of texts, dimenstion]


def get_entity_dict(path):
    entity_dict = {}
    with open(path, "r") as enidf:
        for line in enidf.readlines():
            entity_dict[line.split('\t')[0]] = line.split('\t')[1][:-1]
    return entity_dict


def get_relation_dict(path):
    relation_dict = {}
    with open(path, "r") as enidf:
        for line in enidf.readlines():
            relation_dict[line.split('\t')[0]] = line.split('\t')[1][:-1]
    return relation_dict


def get_entity_list(graph):
    return sorted(list({h for h, _, _ in graph} | {t for _, _, t in graph}))


def get_relation_list(graph):
    return sorted(list({r for _, r, _ in graph}))


def get_triple_list(graph):
    return [f'{item[0]}-{item[1]}-{item[2]}' for item in graph]


def get_clip_text_embeddings_iter(text_list, clip_processor, clip_model, device, clip_len=1000):
    clip_ranges = get_ranges(len(text_list), clip_len) 
    embed_list = []
    for clip_range in clip_ranges:
        start, end = clip_range[0], clip_range[1]
        embed = get_clip_text_embeddings(text_list[start:end], clip_processor, clip_model, device) 
        embed_list.append(embed)
    embeddings = torch.cat(embed_list, dim=0)
    return embeddings


def get_ad_list(graph):
    ad_list = {}
    for head, rel, tail in graph:
        if head not in ad_list:
            ad_list[head] = {tail: rel}
        else:
            ad_list[head][tail] = rel
    return ad_list


def get_all_img_embed(all_img_embed_path="mars_image_embeddings"):
    loaded_all_img_embed = {}
    with h5py.File(all_img_embed_path, "r") as h5_file:
        for key in h5_file:
            loaded_all_img_embed[key] = h5_file[key][:]
    return loaded_all_img_embed


def make_subMMKG(subKG, entity_dict):
    relation = 'imagee of'
    inv_entity_dict = {v: k for k, v in entity_dict.items()}
    sub_MMKG = subKG.copy()
    sub_KG_entity = get_entity_list(subKG)
    for entity in sub_KG_entity:
        entity_id = inv_entity_dict[entity]
        head_entity = "IMG:"+entity_id
        item = (head_entity, relation, entity)
        sub_MMKG.append(item)
    new_entities = sorted(list({h for h, _, _ in sub_MMKG} | {t for _, _, t in sub_MMKG}))
    return sub_MMKG


class MARS_KG:
    def __init__(self, path, entity_dict, relation_dict, clip_model, clip_processor, device):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.clip_batch_size = 1000
        self.device = device

        # construct KG Data
        self.KG = []
        with open(path, "r") as enidf:
            for line in enidf.readlines():
                entity1 = entity_dict[line.split('\t')[0]]
                entity2 = entity_dict[line.split('\t')[2][:-1]]
                relation = relation_dict[line.split('\t')[1]]
                self.KG.append((entity1, relation, entity2))
        self.KG_size = len(self.KG)
        self.KG_entities = get_entity_list(self.KG)
        self.KG_relations = get_relation_list(self.KG)
        self.KG_ad_list = get_ad_list(self.KG)
        self.KG_triples = get_triple_list(self.KG)
        self.KG_triple_embeds = get_clip_text_embeddings_iter(self.KG_triples, self.clip_processor, self.clip_model, self.device, self.clip_batch_size)

    def get_sub_KG(self, query, top_n=10, top_N=10, query_mode=0):
        # query embedding: early fusion (many queries)
        if query_mode == 0:
            query_embed_list = []
            for query_item in query:
                if isinstance(query_item, str):
                    query_embed = get_clip_text_embeddings([query_item], self.clip_processor, self.clip_model, self.device)
                else:
                    query_embed = get_clip_image_embeddings([query_item], self.clip_processor, self.clip_model, self.device)
                query_embed_list.append(query_embed)
            query_embed_tensor = torch.cat(query_embed_list, dim=0)
            query_embed = query_embed_tensor.mean(dim=0).view(1, -1) 

        # query embedding: late fusion (one query)
        else:
            if isinstance(query, str):
                query_embed = get_clip_text_embeddings([query], self.clip_processor, self.clip_model, self.device)
            else:
                query_embed = get_clip_image_embeddings([query], self.clip_processor, self.clip_model, self.device)

        # retrieve sub-graph
        query_embed_norm = F.normalize(query_embed, dim=1)
        kg_embed_norm = F.normalize(self.KG_triple_embeds, dim=1)
        similarity = query_embed_norm @ kg_embed_norm.T

        _, topn_indices = similarity.squeeze(0).topk(top_n, largest=True)
        topn_indices_np = topn_indices.cpu().numpy()
        sub_triples = [self.KG[i] for i in topn_indices_np]
        sub_entities = get_entity_list(sub_triples)

        sub_KG = []
        for se in sub_entities:
            if se in self.KG_ad_list:
                se_connection = self.KG_ad_list[se]
                for nbr in se_connection:
                    rel = se_connection[nbr]
                    sub_KG.append((se, rel, nbr))  

        # filter sub-graph
        sg_triples = get_triple_list(sub_KG)  
        sgt_embed = get_clip_text_embeddings(sg_triples, self.clip_processor, self.clip_model, self.device)
        sgt_embed_norm =  F.normalize(sgt_embed, dim=1)
        re_similarity = query_embed_norm @ sgt_embed_norm.T

        _, re_topn_indices = re_similarity.squeeze(0).topk(top_N, largest=True)
        re_topn_indices_np = re_topn_indices.cpu().numpy()
        f_sub_KG = [sub_KG[i] for i in re_topn_indices_np]
        return f_sub_KG
    

def draw_knowledge_graph(KG, query, seed=42):
    G = nx.DiGraph()
    for h, r, t in KG:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, label=r)

    pos = nx.spring_layout(G, seed=seed)
    node_colors = ['orange' if 'IMG:' in node else 'skyblue' for node in G.nodes()]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors,
            font_size=10, edge_color='gray', arrows=True)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    title=f"Retrieved Knowledge Graph (Query: {query})"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()