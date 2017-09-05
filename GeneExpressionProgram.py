import numpy as np

from Chromosome import Chromosome

class GeneExpressionProgram:

    ### Hyperparameters ###
    NUM_RUNS = 100
    NUM_GENERATIONS = 50
    POPULATION_SIZE = 30
    NUM_FITNESS_CASES = 10
    ERROR_TOLERANCE = 0.01

    ### Reproduction ###
    MUTATION_RATE = 0.051
    ONE_POINT_CROSSOVER_RATE, TWO_POINT_CROSSOVER_RATE, GENE_CROSSOVER_RATE = 0.2, 0.5, 0.1
    IS_TRANSPOSITION_RATE, IS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    RIS_TRANSPOSITION_RATE, RIS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    GENE_TRANSPOSITION_RATE = 0.1

    ### Selection ###
    SELECTION_RANGE = 100

    ### Fitness Evaluation ###
    OBJECTIVE_FUNCTION = staticmethod(lambda a: a ** 4 + a ** 3 + a ** 2 + a)
    OBJECTIVE_MIN, OBJECTIVE_MAX = 0, 20
    FUNCTION_Y_RANGE = None


    def __init__(self):
        pass


    @staticmethod
    def evolve() -> Chromosome:

        if GeneExpressionProgram.FUNCTION_Y_RANGE is None:
            raise ValueError("Class variable FUNCTION_Y_RANGE must be set to calculate fitness.")

        Chromosome.max_fitness = GeneExpressionProgram.FUNCTION_Y_RANGE * GeneExpressionProgram.NUM_FITNESS_CASES

        # create initial population
        population = [Chromosome.generate_random_individual() for _ in range(GeneExpressionProgram.POPULATION_SIZE)]

        generation = 0
        best_fit_individual = None
        while generation < GeneExpressionProgram.NUM_GENERATIONS and best_fit_individual.fitness() < Chromosome.max_fitness:


            ### EVALUATION ###

            # calcluate fitnesses for population
            population_fitnesses = Chromosome.absolute_fitness(GeneExpressionProgram.FUNCTION_Y_RANGE, *population)

            # find best fit individual
            # noinspection PyTypeChecker
            best_fit_generation = population[np.argmax(population_fitnesses)]
            if best_fit_individual is None or best_fit_individual.fitness() < best_fit_generation.fitness():
                best_fit_individual = best_fit_generation

            # skip rest of loop if we have found optimal solution
            if best_fit_individual.fitness() >= Chromosome.max_fitness:
                continue

            next_generation = list()


            ### SELECTION (roulette wheel with simple elitism) ###

            # copy best individual to next generation
            next_generation.append(best_fit_individual)

            # select the rest of the next generation with roulette wheel selection
            


        return best_fit_individual