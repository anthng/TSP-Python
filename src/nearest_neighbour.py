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


def nearest_neighbour(dist_matrix, cities = None, fname = 'default.txt'):
    '''
        :Compute nearest neighbor.

        :Input: 
        -   distance matrix: weight matrix
        -   cities: coordinates to visulize
        -   fname: a path to folder contains output
        :Output: Return route (city, 5->6->8->1...)

    '''
    #get x and y
    #x,y = get_coords(cities)
    
    #initalize a node to start
    node = random.randrange(len(dist_matrix))
    result = [node]
    cost = []

    nodes_to_visit = list(range(len(dist_matrix)))

    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        
        print("Nearest ", nearest_node)

        distance ,node = nearest_node
        nodes_to_visit.remove(node)

        cost.append(distance)
        result.append(node)
        

    #write result file
    with open(fname, 'w') as f:
        for i in result:
            f.write("%d " %i)
        f.write('\nCOST: %f' %sum(cost))

    return result, sum(cost)

if __name__ == "__main__":
    
    data = read_file('./datasets/test.tsp')
    #plot_maps(data, './maps/dj.png')

    #get x, y into a tuple-list [(0,1), (0,2), (1,2)...]
    cities = [(x, y) for x, y in zip(data['x'], data['y'])]


    #plot maps following x and y, uncomment if wanna plot maps.
    #plot_maps(x,y, './maps/dj.png')
    
    #build an euclidean-distance matrix
    matrix = build_graph(cities)
    
    #print(matrix)
    #run algorithm
    result, cost = nearest_neighbour(matrix, cities,'./outputs/test.output')
    print(cost)
    print(result)
    print("DONE")
