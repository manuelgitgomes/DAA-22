#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import chain, combinations, groupby
import random


def generateRandomXY(seed):
    random.seed(seed)
    x = round(20 * random.random())
    seed += 1
    random.seed(seed)
    y = round(20 * random.random())
    seed += 1
    
    return x, y, seed

def generatePos(n):
    seed = 88939
    pos = {}
    node = 1
    x = None
    y = None
    while node in range(1, n + 1):
        x, y, seed = generateRandomXY(seed)
        if (x,y) not in pos.values():
            pos[node] = (x,y)
            node += 1

    return pos

def generateEdgeWeight(G):
    seed = 88939
    for (u,v,w) in G.edges(data=True):
        random.seed(seed)
        w['weight'] = random.randint(1,10)
        seed += 1

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    seed = 88939
    edges = combinations(range(1,n+1), 2)
    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        random.seed(seed)
        seed += 1
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            random.seed(seed)
            seed += 1
            if random.random() < p:
                G.add_edge(*e)
    return G


def verifyCondition(combination, G):
    nodes = list(set(chain.from_iterable(combination))) 
    all_edges = G.edges(nodes)
    if len(all_edges) == len(G.edges):
        return True
    else:
        return False

def exhaustiveSearch(G):
    min_weight = np.inf
    min_edges = None
    list_combinations = list()
    for n in range(1, len(G.edges) + 1):
        list_combinations += list(combinations(G.edges, n))

    for combination in list_combinations:
        condition = verifyCondition(combination, G)
        if condition:
            weight = 0
            for edge in combination:
                weight += G.edges[edge]['weight']
            if weight < min_weight:
                min_weight = weight
                min_edges = combination
    return min_edges

def main():
    n = 10
    p = 0.75
    G = gnp_random_connected_graph(n,p)
    generateEdgeWeight(G)
    pos = generatePos(n)
    nx.draw_networkx(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels) 
    min_edges = exhaustiveSearch(G)
    print(min_edges)
    plt.title("Practical Assignment 1")
    plt.show()

if __name__ == '__main__':
    main()