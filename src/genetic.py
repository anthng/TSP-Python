from utils import *
import random
import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
import time


def random_init_population(cities, size):
    """
        Initalize population
        :Input:
            - cities: a list
            - size:   contains solutions
        :Output:
            Return: size of solutions which contain all coords
    """
    return [random.sample(cities, len(cities)) for i in range(size)]

def fitness(cities):
    """
        quality of gene
    """
    return 1.0/(total_distance(cities))

def rank_route(population):
    result = {}
    n = len(population)
    for i in range(n):
        #print("{} Population {}".format(i,population[i]))
        result[i] = fitness(population[i])
        #print("Result ", result[i])

    return sorted(result.items(), key = operator.itemgetter(1),reverse = True)

def mapping_gen_population(population, rank):
    new_gen_fittes = []
    for i in range(len(rank)):
        #print(rank)
        index = rank[i][0]
        new_gen_fittes.append(population[index])

    return new_gen_fittes

def selection(rank):
    """
        Return: a list of route arcording to rank_route function
    """
    route = []
    for i in range(len(rank)):
        route.append(rank[i][0])
    return route

def selection2population(population, selected_values):
    """
        :Mapping index of selected_values to population
            selected_value = [2,1,3,0]
        :Input
        0    [(104766.6667, 8600.0), (105066.6667, 8650.0), (104733.3333, 8600.0), (104833.3333, 8600.0)]
        1    [(104766.6667, 8600.0), (104733.3333, 8600.0), (104833.3333, 8600.0), (105066.6667, 8650.0)]
        2    [(104833.3333, 8600.0), (104733.3333, 8600.0), (104766.6667, 8600.0), (105066.6667, 8650.0)]
        3    [(104733.3333, 8600.0), (104766.6667, 8600.0), (105066.6667, 8650.0), (104833.3333, 8600.0)]

        :Output
        [[(104766.6667, 8600.0), (104733.3333, 8600.0), (104833.3333, 8600.0), (105066.6667, 8650.0)],
        [(104833.3333, 8600.0), (104733.3333, 8600.0), (104766.6667, 8600.0), (105066.6667, 8650.0)],
        [(104733.3333, 8600.0), (104766.6667, 8600.0), (105066.6667, 8650.0), (104833.3333, 8600.0)],
        [(104766.6667, 8600.0), (105066.6667, 8650.0), (104733.3333, 8600.0), (104833.3333, 8600.0)]]

    """
    mapping = [population[selected_values[i]] for i in range(len(selected_values))]
    return mapping

def crossover(father, mother):
    """Run the crossover step if needed, otherwise return a copy of
        the father"""

    crossover_index = random.randint(0, len(father) - 1)

    start_father = father[:crossover_index]
    end_father = father[crossover_index:]

    start_mother = mother[:crossover_index]
    end_mother = mother[crossover_index:]

    child = start_father + end_mother
    #print("\nChild: ",child)

    intersection = list(set(end_father) & set(start_mother))


    dups = []

    for i in range(len(child)):
        if child[i] in dups:
            #print("Child nam trong dups: ", child[i])
            try:
                child[i] = intersection.pop()
            except:
                print(child[i])
        else:
            dups.append(child[i])

    return child

def off_spring_population(mate, elite_size):
    """
        :Create the offspring population

        :Input:
            - mate: selection2population
            - elite_size: keep the best solution found, and used to build next generation
    """
    children = []
    n = len(mate) - elite_size
    sample = random.sample(mate, len(mate))

    children = [mate[i] for i in range(elite_size)]

    for i in range(n):
        #print("Sample: ",sample[len(mate)-i-1])
        child = crossover(mate[i], sample[len(mate)-i-1])
        children.append(child)
        #print("Child ", child)

    return children

def muate(child, mutation_rate):
    """
        :Mutate the child according to the mutation_rate
            mutation_rate: probability of mutation, float between 0 and 1
    """
    n = len(child)
    for i in range(n):
        r = random.random()
        #print(r)
        if r <= mutation_rate:
            #index = random.randint(0, len(cities)-1)
            index = random.randint(0, n-1)
            #print("Random: ",r)
            #print("Index: ", index)
            child[i], child[index] = child[index], child[i]
            #print("\nSwap: ", (child[i], child[index]))

    return child

def next_generations(current_gene, elite_size, mutate_rate):
    #rank route: index - fitness
    route = rank_route(current_gene)
    #get route base on rank_route
    selected_values = selection(route)
    mapping = selection2population(current_gene, selected_values)
    children = off_spring_population(mapping, elite_size)

    new_generation = muate(children,mutate_rate)
    return new_generation


def GA(citites,fname,population_size = 10, elite_size = 20, mutate_rate = 1e-3, iterations = 500):
    start = time.time()
    #initialize population
    population = random_init_population(cities, population_size)

    progress = []
    progress.append(1 / rank_route(population)[0][1])
    for i in range(iterations):
        population = next_generations(population, elite_size, mutate_rate)
        #print('population: ', population)
        #break
        progress.append(1 / rank_route(population)[0][1])
        if i % 100 == 0:
            print("Iteration: %d"%i)
            #print(population)

    #print("Initial weight: ",progress[0])
    #print("Final weight: " ,progress[-1])

    index_best_route = rank_route(population)[0][0]
    best_route = population[index_best_route]
    delta_time = time.time() - start

    cost = total_distance(best_route)
    print("--- %s seconds ---" % delta_time)
    print("Final distance: ", cost)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generations')
    fig.savefig(fname + '.png', bbox_inches = 'tight', dpi=300)

    plt.show()
    return (best_route, index_best_route, cost, delta_time)

if __name__ == "__main__":
    PATH = './img/'
    fname = 'gen_test_194'

    population_size = 300
    mutate_rate = 0.001
    elite_size = 70
    iterations = 200

    data = read_file('./datasets/qatar194.tsp')

    #get x, y into a tuple-list [(0,1), (0,2), (1,2)...]
    cities = [(x, y) for x, y in zip(data['x'], data['y'])]
    #x,y = get_coords(cities)
    #plot_maps(x,y, './maps/'+ fname)

    result, index, cost, delta_time = GA(cities,PATH + 'coverage/' + fname, population_size, elite_size,mutate_rate, iterations)

    with open('./outputs/genetic/' + fname + '300300.output', 'a') as f:
        f.write('\nCOST: %f' %cost)
        f.write('\nTIME EXECUTION: %f' %delta_time)
        f.write('\n\n=========================================\n\n')

    plot_pop(result,PATH+fname)
