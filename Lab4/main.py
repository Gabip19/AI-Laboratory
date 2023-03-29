import os
import sys
import warnings

from GeneticAlgorithm import GA
from MyChromosome import MyChromosome
from MyExtraChromosome import MyExtraChromosome

warnings.simplefilter('ignore')


def read_graph_from_file(file_name):
    fin = open(file_name, "r")
    n = int(fin.readline())
    matrix = []
    for i in range(n):
        matrix.append([])
        line = fin.readline()
        elems = line.split(",")
        for j in range(n):
            matrix[-1].append(int(elems[j]))
    no_edges = 0
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                if j > i:
                    no_edges += 1
            else:
                matrix[i][j] = INF
    graph = {'noNodes': n,
             'matrix': matrix,
             'noEdges': no_edges,
             'source': int(fin.readline()),
             'destination': int(fin.readline())}
    fin.close()
    return graph


def shortest_path_all(path, param):
    matrix = param['matrix']
    length = 0
    for index in range(len(path) - 1):
        length += matrix[path[index] - 1][path[index + 1] - 1]
    length += matrix[path[0] - 1][path[-1] - 1]
    return length


def shortest_path(path, param):
    matrix = param['matrix']
    length = 0
    for index in range(len(path) - 1):
        length += matrix[path[index] - 1][path[index + 1] - 1]
    return length


def call_ga(graph, population_size=500, no_of_gens=100, chromosome_type=MyChromosome, function=shortest_path_all):
    gaParam = {'popSize': population_size, 'noGen': no_of_gens, 'chromosome': chromosome_type}
    problems_param = graph
    problems_param['function'] = function

    best_chromosomes = []
    ga = GA(gaParam, problems_param)
    ga.initialisation()
    ga.evaluation()
    gens = []

    for generation in range(gaParam['noGen']):
        ga.one_generation_elitism()
        best_chromosome = ga.best_chromosome()
        gens = ga.population
        print(generation, best_chromosome)
        best_chromosomes.append(best_chromosome)

    the_best = best_chromosomes[0]
    for good in best_chromosomes:
        if the_best.fitness > good.fitness:
            the_best = good
    the_bests = [the_best]
    for possible in gens:
        if the_best.fitness == possible.fitness and possible not in the_bests:
            the_bests.append(possible)
    return the_bests


if __name__ == '__main__':
    INF = sys.maxsize
    crtDir = os.getcwd()
    # tests()
    file = input("Give the name of the file: ")
    file += ".txt"
    file_path = os.path.join(crtDir, 'data', file)
    graph_ = read_graph_from_file(file_path)

    population = int(input("Population size: "))
    generations = int(input("Number of generations: "))

    bests = call_ga(graph_, population, generations)
    shortest = call_ga(graph_, population, generations, MyExtraChromosome, shortest_path)

    print("Solutions: \n")
    for chromosome in bests:
        print(chromosome)
    print('Number of solutions: ', len(bests), '  Fit: ', bests[0].fitness)

    print('\n\nShortest paths from: ', graph_['source'], ' to ', graph_['destination'], ' are: ')
    for chromosome in shortest:
        print(chromosome)
    print('Number of solutions: ', len(shortest), ' Fit:', shortest[0].fitness)
