#!/usr/bin/env python3

import argparse
import csv
import logging
import math
import time
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
ch.setLevel(logging.INFO)

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

def generateRandomEdges(G, edges, p, desired_edges):
   
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            rand = random.random()
            if rand< p:
                G.add_edge(*e)
            # if len(G.edges()) >= desired_edges:
                # return G

    if len(G.edges()) < desired_edges:
        generateRandomEdges(G, edges, p, desired_edges)

    return G
            

def graphGenerator(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is connected
    """
    max_edges = (n * (n-1))/2
    desired_edges = math.floor(p * max_edges)
    edges = combinations(range(1,n+1), 2)
    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)

    G = generateRandomEdges(G, edges, p, desired_edges)

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
                min_edges = list(combination)
            else:
                logger.debug(f'Weight of current combination ({weight}) is equal or larger than current minimum weight ({min_weight}).')
    
    return min_edges, min_weight, counter


def greedyHeuristicsMinWeight(G):
    solutions_counter = 0
    min_weight = 0 
    weight_list = []
    min_edges = []
    
    # Get edges weight
    for edge in G.edges():
        weight_list.append(G.edges[edge]['weight'])
    
    # Order edges by weight
    edges_list = list(zip(weight_list, G.edges()))
    edges_list.sort()
    logger.debug(f'edges list: {edges_list}')

    for weight, edge in edges_list:
        solutions_counter += 1
        min_edges.append(edge)
        min_weight += weight
        
        condition = verifyCondition(min_edges, G)
        
        if condition:
            break
    
    return min_edges, min_weight, solutions_counter


def greedyHeuristicsMaxConnection(G):
    connection_list = []
    counter = 0
    min_weight  = 0 
    nodes_edges_list = []
    connection_number_list = []
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
        connection_list.append(node_connecting_list)
        connection_number_list.append(len(node_connecting_list))
    
    # Order edges by connection
    edges_list = list(zip(connection_number_list, G.edges(), weight_list, connection_list))
    edges_list.sort(key=lambda x: (-x[0], x[2], x[1][0], x[1][1]))
    logger.debug(f'edges list: {edges_list}')

    remove_list = []

    for _ , edge, weight, connections in edges_list:
        if edge not in remove_list:
            counter += 1
            min_edges.append(edge)
            min_weight += weight
            
            condition = verifyCondition(min_edges, G)
            
            # remove already neighbouring edges
            remove_list.extend(connections)
            logger.debug(f'Removed the following edges: {connections}')

            if condition:
                break
    
    return min_edges, min_weight, counter


def greedyHeuristicsChaurasia(G):
    counter = 0
    min_weight  = 0 
    nodes_edges_list = []
    connection_list = []
    min_edges = []
    weight_list = []
    weight_ratio_list = []
    
    # Get number of edges per node
    for node in G.nodes():
        nodes_edges_list.append(list(G.edges(node)))
    
    # Get edges weight
    for edge in G.edges():
        weight_list.append(G.edges[edge]['weight'])

    edges_list = list(zip(G.edges(), weight_list))
    # Get number of connecting edges to a certain edge
    for edge, weight in edges_list:
        node_connecting_list = []
        node_connecting_list.extend(nodes_edges_list[edge[0]-1])
        node_connecting_list.extend(nodes_edges_list[edge[1]-1])
        node_connecting_list = [*set(tuple(sorted(t)) for t in node_connecting_list)] 
        neighbour_weight = [weight1 for edge1, weight1 in edges_list if edge1 in node_connecting_list]
        weight_ratio_list.append(sum(neighbour_weight)/weight)
        connection_list.append(node_connecting_list)


    # Create list with every information needed
    edges_list = list(zip(connection_list, G.edges(), weight_list, weight_ratio_list))
    edges_list.sort(key=lambda x: (-x[3], x[2], x[1][0], x[1][1]))
    logger.debug(f'edges list: {edges_list}')


    remove_list = []
    for connections, edge, weight, weight_ratio in edges_list:
        if edge not in remove_list:
            counter += 1
            min_edges.append(edge)
            min_weight += weight

            # remove already neighbouring edges
            remove_list.extend(connections)
            logger.debug(f'Removed the following edges: {connections}')

            condition = verifyCondition(min_edges, G)
            
            if condition:
                break
    
    return min_edges, min_weight, counter


def csvWriter(filename, fields, rows):
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)


def main():
    # Define argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nodes", help='Number of nodes to use.', type=list, default=[4, 5, 6, 7, 8, 9])
    ap.add_argument("-p", "--probability", help='Probabilty of connecting two nodes.', type=list, default=[0.125, 0.25, 0.5, 0.75])
    ap.add_argument("-pl", "--plot", help='Plot graph', action='store_true')
    ap.add_argument("-s", "--save", help='Save plots', action='store_true')
    ap.add_argument("-csv", "--csv", help='Save data to csv', action='store_true')
    ap.add_argument("-seed", "--seed", help='Seed to use.', type=int, default=88939)
    
    # Defining args
    args = vars(ap.parse_args())
    logger.info(f'The inputs of the script are: {args}')
    n_list = args['nodes']
    p_list = args['probability']

    # Defining rows for csv
    rows = []
    
    for n in n_list:
        for p in p_list:

            random.seed(args['seed'])
            # Generate graph
            logger.debug('Generating graph')
            G = graphGenerator(n,p)
            logger.debug('Graph generated')
            pos = generatePos(n)
            logger.debug('Positions generated')
            generateEdgeWeight(G, pos)
            logger.debug('Weights generated')
            logger.info(f'Graph generated is: {G}')
            
            # Prepare exhaustive search
            logger.debug('Exhaustive search starting')
            min_start_time = time.time()
            min_edges, min_weight, counter = exhaustiveSearch(G)
            min_execution_time = (time.time() - min_start_time)
            logger.info(f'After exhaustive search with {counter} simple operations and taking {min_execution_time} seconds, the best solution for the problem is {min_edges} with weight {min_weight}.')

            # Greedy heuristics
            logger.debug('Minimum weight greedy heuristics starting')
            min_weight_greedy_start_time = time.time()
            min_weight_greedy_edges, min_weight_greedy_weight, min_weight_greedy_counter = greedyHeuristicsMinWeight(G)
            min_weight_greedy_execution_time = (time.time() - min_weight_greedy_start_time)
            logger.info(f'After minimum weight greedy heuristics with {min_weight_greedy_counter} simple operations and taking {min_weight_greedy_execution_time} seconds, the best solution for the problem is {min_weight_greedy_edges} with weight {min_weight_greedy_weight}.')
            
            logger.debug('Maximum connection greedy heuristics starting')
            max_connection_greedy_start_time = time.time()
            max_connection_greedy_edges, max_connection_greedy_weight, max_connection_greedy_counter = greedyHeuristicsMaxConnection(G)
            max_connection_greedy_execution_time = (time.time() - max_connection_greedy_start_time)
            logger.info(f'After maximum connection greedy heuristics with {max_connection_greedy_counter} simple operations and taking {max_connection_greedy_execution_time} seconds, the best solution for the problem is {max_connection_greedy_edges} with weight {max_connection_greedy_weight}.')
            
            logger.debug('Chaurasia greedy heuristics starting')
            chaurasia_greedy_start_time = time.time()
            chaurasia_greedy_edges, chaurasia_greedy_weight, chaurasia_greedy_counter = greedyHeuristicsChaurasia(G)
            chaurasia_greedy_execution_time = (time.time() - chaurasia_greedy_start_time)
            logger.info(f'After Chaurasia greedy heuristics with {chaurasia_greedy_counter} simple operations and taking {chaurasia_greedy_execution_time} seconds, the best solution for the problem is {chaurasia_greedy_edges} with weight {chaurasia_greedy_weight}.')

            # Save CSV
            if args['csv']:
                filename = '../report/data/full_data.csv'
                fields = ['n', 'p', 'E','Algorithm', 'Edges', 'Weight', 'Number of Solutions Tested', 'Execution Time', 'Number of Basic Operations', 'Relative Error', 'Accuracy Ratio']
                rows.append([n, p, len(G.edges()), 'Exhaustive Search', min_edges, min_weight, counter, min_execution_time, counter, 'NA', 'NA'])
                rows.append([n, p, len(G.edges()), 'Minimum Weight Greedy Heuristics', min_weight_greedy_edges, min_weight_greedy_weight, min_weight_greedy_counter, min_weight_greedy_execution_time, min_weight_greedy_counter, (min_weight_greedy_weight-min_weight)/min_weight, min_weight_greedy_weight/min_weight])
                rows.append([n, p, len(G.edges()), 'Maximum Connection Greedy Heuristics', max_connection_greedy_edges, max_connection_greedy_weight, max_connection_greedy_counter, max_connection_greedy_execution_time, max_connection_greedy_counter, (max_connection_greedy_weight-min_weight)/min_weight, max_connection_greedy_weight/min_weight])
                rows.append([n, p, len(G.edges()), 'Chaurasia Greedy Heuristics', chaurasia_greedy_edges, chaurasia_greedy_weight, chaurasia_greedy_counter, chaurasia_greedy_execution_time, chaurasia_greedy_counter, (chaurasia_greedy_weight-min_weight)/min_weight, chaurasia_greedy_weight/min_weight])
                csvWriter(filename, fields, rows)
                logger.info('CSV saved')


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

                f4 = plt.figure(4, figsize=(2560*px, 1440*px))
                plt.title("Chaurasia Greedy Heuristics")
                nx.draw_networkx(G, pos)
                nx.draw_networkx_edges(G, pos, edgelist=list(chaurasia_greedy_edges), edge_color='r', width=2)
                nx.draw_networkx_edge_labels(G, pos, edge_labels)

                if args['save']:
                    f1.savefig(f'../report/figs/fig-{n}-{p}-exhaustive.png')
                    f2.savefig(f'../report/figs/fig-{n}-{p}-min-weight-greedy.png')
                    f3.savefig(f'../report/figs/fig-{n}-{p}-max-connection-greedy.png')
                    f4.savefig(f'../report/figs/fig-{n}-{p}-chaurasia-greedy.png')
                    logger.info('Figures saved')
                    f1.clear(True)
                    f2.clear(True)
                    f3.clear(True)
                    f4.clear(True)
                if args['plot']:
                    f1.show()
                    f2.show()
                    f3.show()
                    f4.show()
                    input()

                


if __name__ == '__main__':
    main()