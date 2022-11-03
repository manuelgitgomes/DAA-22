#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import random


def ER(n, p):
    V = set([v for v in range(1, n + 1)])
    E = set()
    seed = 88939
    for combination in combinations(V, 2):
        random.seed(seed)
        a = random.random()
        seed += 1
        if a < p:
            E.add(combination)
            seed += 1

    g = nx.Graph()
    g.add_nodes_from(V)
    g.add_edges_from(E)

    return g

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
        w['weight'] = random.randint(0,10)
        seed += 1

def main():
    n = 400
    p = 0.25
    G = ER(n, p)
    generateEdgeWeight(G)
    pos = generatePos(n)
    nx.draw_networkx(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels) 
    print(G.edges)
    plt.title("Random Graph Generation Example")
    plt.show()

if __name__ == '__main__':
    main()