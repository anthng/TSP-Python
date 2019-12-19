import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from operator import itemgetter

def read_file(fname):
    """
        Read a *tsp file into a DataFrame.
        The *tsp file is downloaded http://www.math.uwaterloo.ca/tsp/world/countries.html

        Params:
            -   Input:  fname: file name (*tsp)
            -   Output: dataframe
    """

    with open(fname) as f:
        
        lines = f.readlines()

        i = 0
        node = 0
        dim = 0
        #print(lines)
        while True:
            line = lines[i]
            #print(line)
            if line.startswith('DIMENSION :'):
                dim = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node = i+1
                break
            i+=1

        #print("seeker: ",node)

        f.seek(0)

        cities = pd.read_csv(
            f,
            skiprows=node,
            sep=' ',
            names=['city', 'y', 'x'],
            dtype={'city': str, 'y': np.float64, 'x': np.float64},
            header=None,
            nrows=dim
        )
        print(cities.info())
        #cities.to_csv(fname+'.csv', index=False)

        return cities

def dist(a,b):
    """
        euclidean distance: sqrt( (x1^2 -x2^2) + (y2^2 - y1^2) )
    """
    #return math.sqrt( ((x1 - x2) ** 2 + (y1 - y2) ** 2) )
    return np.linalg.norm(a-b)

def get_coords(cities):
    cities = np.array(cities)
    y = cities[:,0]
    x = cities[:,1]
                
    return x,y

def build_graph(data):
    graph = {}
    data = np.array(data)
    for this_point in range(len(data)):
        for another_point in range(len(data)):
            if this_point != another_point:
                if this_point not in graph:
                    graph[this_point] = {}
                graph[this_point][another_point] = dist(data[this_point], data[another_point])

    return graph


def plot_pop(cities, fname):
    
    fig = plt.figure(figsize=(6,8))
    x = [i[0] for i in cities]
    y = [i[1] for i in cities]
    x1=[x[0],x[-1]]
    y1=[y[0],y[-1]]
    plt.plot(x, y, 'b', x1, y1, 'b')
    plt.scatter(x, y)
    fig.savefig(fname, bbox_inches = 'tight', dpi=300)
    plt.show()

def plot_maps(x,y, fname = 'maps.png'):
    fig = plt.figure(figsize=(6, 8))
    plt.scatter(x,y, color = 'red')
    plt.show()
    fig.savefig(fname, bbox_inches = 'tight', dpi=300)
