from random import randint


class GA:
    def __init__(self, param=None, problems_param=None):
        self.__param = param
        self.__problemsParam = problems_param
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for _ in range(0, self.__param['popSize']):
            c = self.__param['chromosome'](self.__problemsParam)
            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problemsParam['function'](c.representation, self.__problemsParam)

    def best_chromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if c.fitness < best.fitness:
                best = c
        return best

    def selection(self):
        pop = self.__population
        size = len(self.__population)
        pop = sorted(pop, key=lambda x : x.fitness)[:size//5]
        pos1 = randint(0, len(pop) - 1)
        pos2 = randint(0, len(pop) - 1)
        if self.__population[pos1].fitness < self.__population[pos2].fitness:
            return pos1
        else:
            return pos2

    def one_generation(self):
        newPop = []
        for _ in range(self.__param['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()

    def one_generation_elitism(self):
        newPop = [self.best_chromosome()]
        for _ in range(self.__param['popSize'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()
