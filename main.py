from Chromosome import Chromosome
from GeneExpressionProgram import GeneExpressionProgram

from math import factorial
import numpy as np


def a4_a3_a2_a1():

    # define objective function
    GeneExpressionProgram.OBJECTIVE_FUNCTION = staticmethod(lambda a: a ** 4 + a ** 3 + a ** 2 + a)
    GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX = 0, 20

    # Define terminals and functions
    Chromosome.terminals = ["a"]
    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y}
    }
    Chromosome.fitness_cases = [
        ({"a": a}, GeneExpressionProgram.OBJECTIVE_FUNCTION(a))
        for a in np.random.rand(GeneExpressionProgram.NUM_FITNESS_CASES) * (
        GeneExpressionProgram.OBJECTIVE_MAX - GeneExpressionProgram.OBJECTIVE_MIN) + GeneExpressionProgram.OBJECTIVE_MIN
    ]

    GeneExpressionProgram.FUNCTION_Y_RANGE = \
        GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MAX) - \
        GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MIN)

    Chromosome.linking_function = "+"
    Chromosome.max_fitness = 1 #GeneExpressionProgram.FUNCTION_Y_RANGE * GeneExpressionProgram.NUM_FITNESS_CASES

    ans = GeneExpressionProgram.evolve()
    ans.print_tree()


def sinx_polynomial():
    # define objective function
    GeneExpressionProgram.OBJECTIVE_FUNCTION = staticmethod(lambda a: np.sin(a))
    GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX = -1, 1
    GeneExpressionProgram.FITNESS_FUNCTION = Chromosome.inv_squared_error
    GeneExpressionProgram.FITNESS_FUNCTION_ARGS = []

    Chromosome.head_length = 15
    Chromosome.num_genes = 5
    Chromosome.length = 45

    # Define terminals and functions
    Chromosome.terminals = ["x", "6", "120"]
    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y}
        #"^": {"args": 2, "f": lambda x, y: x ** y}
    }
    Chromosome.fitness_cases = [
        ({"x": x}, GeneExpressionProgram.OBJECTIVE_FUNCTION(x))
        for x in np.random.rand(GeneExpressionProgram.NUM_FITNESS_CASES) * (
        GeneExpressionProgram.OBJECTIVE_MAX - GeneExpressionProgram.OBJECTIVE_MIN) + GeneExpressionProgram.OBJECTIVE_MIN
    ]

    GeneExpressionProgram.FUNCTION_Y_RANGE = 2
    Chromosome.linking_function = "+"
    Chromosome.max_fitness = 1 #GeneExpressionProgram.FUNCTION_Y_RANGE * GeneExpressionProgram.NUM_FITNESS_CASES

    ans = GeneExpressionProgram.evolve()
    ans.print_tree()


def euclidean_distance():

    def random_fitness_case(f, function_min: float, function_max: float) -> (dict, float):
        x = np.random.rand() * (function_max - function_min) + function_min
        y = np.random.rand() * (function_max - function_min) + function_min
        key = {"x": x, "y": y}
        value = f(x, y)
        return key, value

    # define objective function
    GeneExpressionProgram.OBJECTIVE_FUNCTION = staticmethod(lambda x, y: (x**2 + y**2)**0.5)
    GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX = 0, 20

    # Define terminals and functions
    Chromosome.terminals = ["x", "y"]
    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y},
        "Q": {"args": 1, "f": lambda x: x**0.5}
    }

    Chromosome.fitness_cases = [
        random_fitness_case(GeneExpressionProgram.OBJECTIVE_FUNCTION, GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX)
        for _ in range(GeneExpressionProgram.NUM_FITNESS_CASES)
    ]

    Chromosome.linking_function = "+"
    Chromosome.num_genes = 3
    Chromosome.length = 39
    Chromosome.head_length = 10
    Chromosome.max_fitness = 1

    GeneExpressionProgram.FUNCTION_Y_RANGE = GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MAX, GeneExpressionProgram.OBJECTIVE_MAX)

    ans = GeneExpressionProgram.evolve()
    ans.print_tree()


if __name__ == "__main__":

    #a4_a3_a2_a1()
    sinx_polynomial()
    #euclidean_distance()