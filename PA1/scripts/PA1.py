#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import chain, combinations, groupby
import random


def generateRandomXY(seed):
    # Generate random X and Y
    random.seed(seed)
    x = round(20 * random.random())
    seed += 1
    random.seed(seed)
    y = round(20 * random.random())
    seed += 1
    
    return x, y, seed

def generatePos(n):
    # Generate valid position
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
    # Generate random weight graph
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
    """Verify if this combination of edges of graph G is an edge dominating set

    Args:
        combination (tuple): combination of edges
        G (networkx.graph): graph

    Returns:
       Bool: return True or False if combination is or not an edge dominating set
    """
    # List every node present in the combination
    nodes = list(set(chain.from_iterable(combination))) 
    # Find every edge connected to those nodes
    all_edges = G.edges(nodes)
    # If these edges are every edge present in the graph, return True. If not, return False.
    if len(all_edges) == len(G.edges):
        return True
    else:
        return False

def exhaustiveSearch(G):
    """Exhaustive search of a graph to find a minimum weighted solution

    Args:
        G (networkx.graph): graph

    Returns:
        min_edges (tuple): edges who verify the condition
    """
    min_weight = np.inf
    min_edges = None
    # Create a list of every combination of edges in the graph G
    list_combinations = list()
    for n in range(1, len(G.edges) + 1):
        list_combinations += list(combinations(G.edges, n))

    # Verify if every combination passes the condition, if so, compare their weight with the current minimum
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
    # create logger
    logger = logging.getLogger("PA1")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # Define argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nodes", help='Number of nodes to use.', type=int, default=5)
    ap.add_argument("-p", "--probability", help='Probabilty of connecting two nodes.', type=float, default=0.5)
    
    # Defining args
    args = vars(ap.parse_args())
    logger.info(f'The inputs of the script are: {args}')

    n = args['nodes']
    p = args['probability']

    # Generate graph
    G = gnp_random_connected_graph(n,p)
    generateEdgeWeight(G)
    pos = generatePos(n)
    
    # Prepare exhaustive search
    min_edges = exhaustiveSearch(G)
    logger.info(f'After exhaustive search, the best solution for the problem is {min_edges}')

    # Prepare plot
    nx.draw_networkx(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels) 
    plt.title("Practical Assignment 1")
    plt.show()

if __name__ == '__main__':
    main()