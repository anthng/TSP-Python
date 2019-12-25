"""
    1. Read a csv file
    2. Get x, y into a list
    3. Build a graph (create a distance matrix)
    4. Run algorithm, the result is index city (0,1,2, .. n_cities)

    To plot maps, you need to get coords X, y
        and then plot_maps(x,y, './maps/dj.png')

    To run this file in terminal:
        -   Open command line and change direction to this project (TSP folder).
            $   cd <your>/<path>/TSP
        -   Run with python 3;
            $   python  src/nearest_neighbour.py
"""


from utils import *
import random
import numpy as np
import pandas as pd
import math
import time

def nearest_neighbour(dist_matrix, cities = None, fname = 'default.txt'):
    '''
        :Compute nearest neighbor.

        :Input:
        -   distance matrix: weight matrix
        -   cities: coordinates to get coords
        -   fname: a path to folder contains output
        :Output: Return route (city, 5->6->8->1...)

    '''
    start = time.time()
    #initalize a node to start
    node = random.randrange(len(dist_matrix))
    result = [node]
    #cost = []

    nodes_to_visit = list(range(len(dist_matrix)))

    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])

        #print("Nearest ", nearest_node)

        distance ,node = nearest_node
        nodes_to_visit.remove(node)

        #cost.append(distance)
        #print(distance)
        result.append(node)

    delta_time = time.time() - start
    print("--- %s seconds ---" % delta_time)

    #get element results to calculate total didstance
    coords = [cities[elem-1] for elem in result]
    cost = total_distance(coords)
    print("Final distance: ", total_distance(coords))

    #write down result file
    with open(fname, 'a') as f:
        # for i in result:
        #     f.write("%d " %i)
        f.write('\nCOST: %f' %cost)
        f.write('\nTIME EXECUTION: %f' %delta_time)
        f.write('\n\n=========================================\n\n')

    return result, coords, cost

if __name__ == "__main__":

    data = read_file('./datasets/qatar194.tsp')
    #plot_maps(data, './maps/dj.png')

    #get x, y into a tuple-list [(0,1), (0,2), (1,2)...]
    cities = [(x, y) for x, y in zip(data['x'], data['y'])]


    #build an euclidean-distance matrix
    matrix = build_graph(cities)

    #print(matrix)
    #run algorithm
    results, coords,cost = nearest_neighbour(matrix, cities,'./outputs/nn_test_194.output')

    print(results)
    #plot_pop(coords,'./img/nn.png')
