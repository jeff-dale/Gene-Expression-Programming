from Chromosome import Chromosome

from lib.anytree import RenderTree, AsciiStyle
import numpy as np

# Define GEP parameters.
NUM_RUNS = 100
NUM_GENERATIONS = 50
POPULATION_SIZE = 30
NUM_FITNESS_CASES = 10
HEAD_LENGTH = 6
NUM_GENES = 3
CHROMOSOME_LENGTH = 39
MUTATION_RATE = 0.051
ONE_POINT_CROSSOVER_RATE, TWO_POINT_CROSSOVER_RATE, GENE_CROSSOVER_RATE = 0.2, 0.5, 0.1
IS_TRANSPOSITION_RATE, IS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
RIS_TRANSPOSITION_RATE, RIS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
GENE_TRANSPOSITION_RATE = 0.1
SELECTION_RANGE = 100
ERROR_TOLERANCE = 0.01


# Objective function, fitness cases, fitness function
objective_function = lambda a : a**4 + a**3 + a**2 + a  # ground truth function, only used to generate fitness cases
objective_min, objective_max = 0, 20                    # bounds of objective function

# generate fitness cases of random points in objective function
fitness_cases = [
    ({"a": a}, objective_function(a))
        for a in np.random.rand(NUM_FITNESS_CASES)*(objective_max - objective_min) + objective_min
]

# fitness function, absolute error in this case
fitness_function = lambda x : x


# Define functions and terminators
FUNCTIONS = {
    "+": {"args": 2, "f": lambda x,y : x + y},
    "-": {"args": 2, "f": lambda x,y : x - y},
    "*": {"args": 2, "f": lambda x,y : x * y},
    "/": {"args": 2, "f": lambda x,y : x / y},
    "Q": {"args": 1, "f": lambda x : x**0.5}
    #"I": {"args": 3, "f": lambda x,y,z : y if x else z}
}
TERMINALS = ["a"]


def get_random_individual():
    raise NotImplementedError


def evolve():
    population = [get_random_individual() for _ in range(POPULATION_SIZE)]


Chromosome.functions = FUNCTIONS
Chromosome.terminals = TERMINALS
Chromosome.fitness_cases = fitness_cases
Chromosome.head_length = 13 # HEAD_LENGTH

c1 = Chromosome(["Q*Q+bbaaa"])
print(c1.evaluate(terminal_values={
    "a": 3,
    "b": 1
}))

pass