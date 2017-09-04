from Chromosome import Chromosome

class GeneExpressionProgram:

    # Hyperparameters
    NUM_RUNS = 100
    NUM_GENERATIONS = 50
    POPULATION_SIZE = 30
    NUM_FITNESS_CASES = 10
    ERROR_TOLERANCE = 0.01

    # Reproduction
    MUTATION_RATE = 0.051
    ONE_POINT_CROSSOVER_RATE, TWO_POINT_CROSSOVER_RATE, GENE_CROSSOVER_RATE = 0.2, 0.5, 0.1
    IS_TRANSPOSITION_RATE, IS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    RIS_TRANSPOSITION_RATE, RIS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    GENE_TRANSPOSITION_RATE = 0.1

    # Selection
    SELECTION_RANGE = 100

    # Fitness Evaluation
    OBJECTIVE_FUNCTION = lambda a : a ** 4 + a ** 3 + a ** 2 + a    # ground truth function, only used to generate fitness cases
    OBJECTIVE_MIN, OBJECTIVE_MAX = 0, 20                            # bounds of objective function
    FUNCTION_Y_RANGE = None


    def __init__(self):
        GeneExpressionProgram.FUNCTION_Y_RANGE = GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MAX) -\
                                                 GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MIN)


    def evolve(self):
        population = [Chromosome.generate_random_individual() for _ in range(GeneExpressionProgram.POPULATION_SIZE)]