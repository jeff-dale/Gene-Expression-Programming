from Chromosome import Chromosome
from GeneExpressionProgram import GeneExpressionProgram

import numpy as np


if __name__ == "__main__":

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
        for a in np.random.rand(GeneExpressionProgram.NUM_FITNESS_CASES) * (GeneExpressionProgram.OBJECTIVE_MAX - GeneExpressionProgram.OBJECTIVE_MIN) + GeneExpressionProgram.OBJECTIVE_MIN
    ]
    Chromosome.linking_function = "+"

    GeneExpressionProgram.FUNCTION_Y_RANGE = \
        GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MAX) - \
        GeneExpressionProgram.OBJECTIVE_FUNCTION(GeneExpressionProgram.OBJECTIVE_MIN)

    GeneExpressionProgram.evolve()