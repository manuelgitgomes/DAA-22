#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import random


def ER(n, p):
    V = set([v for v in range(n)])
    E = set()
    seed = 88939
    for combination in combinations(V, 2):
        random.seed(seed)
        a = random.random()
        seed += 1
        if a < p:
            E.add(combination)

    g = nx.Graph()
    g.add_nodes_from(V)
    g.add_edges_from(E)

    return g

def main():
    n = 400
    p = 0.25
    G = ER(n, p)
    pos = nx.spring_layout(G, seed=88939)
    pos.update((x , (y*10+10))for x, y in pos.items())
    print(pos.items())
    nx.draw_networkx(G, pos)
    print(G)
    plt.title("Random Graph Generation Example")
    plt.show()

if __name__ == '__main__':
    main()