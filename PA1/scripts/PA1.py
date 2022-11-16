#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import chain, combinations, groupby
import random
from math import dist

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

def generateRandomXY():
    # Generate random X and Y
    x = round(20 * random.random())
    y = round(20 * random.random())
    
    return x, y

def generatePos(n):
    # Generate valid position
    pos = {}
    node = 1
    x = None
    y = None
    while node in range(1, n + 1):
        x, y = generateRandomXY()
        if (x,y) not in pos.values():
            pos[node] = (x,y)
            node += 1

    return pos

def generateEdgeWeight(G, pos):
    # Generate weight graph
    for (u,v,w) in G.edges(data=True):
        w['weight'] = round(dist(pos[u],pos[v]))

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(1,n+1), 2)
    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
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
        logger.debug(f'Edges {combination} are an edge dominating set')
        return True
    else:
        logger.debug(f'Edges {combination} are not an edge dominating set')
        return False

def exhaustiveSearch(G):
    """Exhaustive search of a graph to find a minimum weighted solution

    Args:
        G (networkx.graph): graph

    Returns:
        min_edges (tuple): edges who verify the condition
    """
    counter = 0
    min_weight = np.inf
    min_edges = None
    # Create a list of every combination of edges in the graph G
    list_combinations = list()
    for n in range(1, len(G.edges) + 1):
        list_combinations += list(combinations(G.edges, n))

    # Verify if every combination passes the condition, if so, compare their weight with the current minimum
    for combination in list_combinations:
        counter += 1
        condition = verifyCondition(combination, G)
        if condition:
            weight = 0
            for edge in combination:
                weight += G.edges[edge]['weight']
            if weight < min_weight:
                logger.debug(f'Weight of current combination ({weight}) is smaller than current minimum weight ({min_weight}). Placing current combination as best.')
                logger.info(f'New best weight found: ({weight}).')
                min_weight = weight
                min_edges = combination
            else:
                logger.debug(f'Weight of current combination ({weight}) is equal or larger than current minimum weight ({min_weight}).')
    
    return min_edges, min_weight, counter


def greedyHeuristicsMinWeight(G):
    counter = 0
    min_weight = 0 
    weight_list = []
    min_edges = []
    
    # Get edges weight
    for edge in G.edges():
        weight_list.append(G.edges[edge]['weight'])
    
    # Order edges by weight
    edges_list = zip(weight_list, G.edges())
    sorted_edges_list = sorted(edges_list)

    for weight, edge in sorted_edges_list:
        counter += 1
        min_edges.append(edge)
        min_weight += weight
        
        condition = verifyCondition(min_edges, G)
        
        if condition:
            break
    
    return min_edges, min_weight, counter


def greedyHeuristicsMaxConnection(G):
    counter = 0
    min_weight  = 0 
    nodes_edges_list = []
    edges_list = []
    connection_list = []
    min_edges = []
    weight_list = []
    
    # Get number of edges per node
    for node in G.nodes():
        nodes_edges_list.append(list(G.edges(node)))
    
    # Get edges weight
    for edge in G.edges():
        weight_list.append(G.edges[edge]['weight'])

    # Get number of connecting edges to a certain edge
    for edge in G.edges():
        node_connecting_list = []
        node_connecting_list.extend(nodes_edges_list[edge[0]-1])
        node_connecting_list.extend(nodes_edges_list[edge[1]-1])
        node_connecting_list = [*set(tuple(sorted(t)) for t in node_connecting_list)]
        connection_list.append(len(node_connecting_list))

    def sortWeight(tuple):
        return tuple[2]
    
    # Order edges by connection
    edges_list = list(zip(connection_list, G.edges(), weight_list))
    edges_list.sort(key=lambda x: (-x[0], x[2], x[1][0], x[1][1]))

    for connection, edge, weight in edges_list:
        counter += 1
        min_edges.append(edge)
        min_weight += weight
        
        condition = verifyCondition(min_edges, G)
        
        if condition:
            break
    
    return min_edges, min_weight, counter



def main():
    # Define argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nodes", help='Number of nodes to use.', type=int, default=5)
    ap.add_argument("-p", "--probability", help='Probabilty of connecting two nodes.', type=float, default=0.5)
    ap.add_argument("-pl", "--plot", help='Plot graph', action='store_true')
    ap.add_argument("-s", "--save", help='Save plots', action='store_true')
    ap.add_argument("-seed", "--seed", help='Seed to use.', type=int, default=88939)
    
    # Defining args
    args = vars(ap.parse_args())
    logger.info(f'The inputs of the script are: {args}')

    n = args['nodes']
    p = args['probability']
    random.seed(args['seed'])

    # Generate graph
    logger.debug('Generating graph')
    G = gnp_random_connected_graph(n,p)
    logger.debug('Graph generated')
    pos = generatePos(n)
    logger.debug('Positions generated')
    generateEdgeWeight(G, pos)
    logger.debug('Weights generated')
    logger.info(f'Graph generated is: {G}')
    
    # Prepare exhaustive search
    logger.debug('Exhaustive search starting')
    min_edges, min_weight, counter = exhaustiveSearch(G)
    logger.info(f'After exhaustive search with {counter} simple operations, the best solution for the problem is {min_edges} with weight {min_weight}.')

    # Greedy heuristics
    logger.debug('Greedy heuristics starting')
    min_weight_greedy_edges, min_weight_greedy_weight, min_weight_greedy_counter = greedyHeuristicsMinWeight(G)
    logger.info(f'After minimum weight greedy heuristics with {min_weight_greedy_counter} simple operations, the best solution for the problem is {min_weight_greedy_edges} with weight {min_weight_greedy_weight}.')
    max_connection_greedy_edges, max_connection_greedy_weight, max_connection_greedy_counter = greedyHeuristicsMaxConnection(G)
    logger.info(f'After maximum connection greedy heuristics with {max_connection_greedy_counter} simple operations, the best solution for the problem is {max_connection_greedy_edges} with weight {max_connection_greedy_weight}.')

    # Prepare plots
    if args['plot'] or args['save']:
        edge_labels = nx.get_edge_attributes(G, "weight")
        px = 1/plt.rcParams['figure.dpi']

        f1 = plt.figure(1, figsize=(2560*px, 1440*px))
        plt.title("Exhaustive Search")
        nx.draw_networkx(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=list(min_edges), edge_color='r', width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels) 

        f2 = plt.figure(2, figsize=(2560*px, 1440*px))
        plt.title("Minimum Weight Greedy Heuristics")
        nx.draw_networkx(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=list(min_weight_greedy_edges), edge_color='r', width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        f3 = plt.figure(3, figsize=(2560*px, 1440*px))
        plt.title("Maximum Connections Greedy Heuristics")
        nx.draw_networkx(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=list(max_connection_greedy_edges), edge_color='r', width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        if args['save']:
            f1.savefig(f'../report/figs/fig-{n}-{p}-exhaustive.png')
            f2.savefig(f'../report/figs/fig-{n}-{p}-min-weight-greedy.png')
            f3.savefig(f'../report/figs/fig-{n}-{p}-max-connection-greedy.png')
            logger.info('Figures saved')
        elif args['plot']:
            f1.show()
            f2.show()
            f3.show()
            input()


if __name__ == '__main__':
    main()