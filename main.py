from Chromosome import Chromosome
from GeneExpressionProgram import GeneExpressionProgram

import numpy as np


if __name__ == "__main__":
    GEP = GeneExpressionProgram()
    # Define terminals and functions
    Chromosome.terminals = ["a"]
    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y}
    }
    Chromosome.fitness_cases = [
        ({"a": a}, GEP.OBJECTIVE_FUNCTION(a))
        for a in np.random.rand(GEP.NUM_FITNESS_CASES) * (GEP.OBJECTIVE_MAX - GEP.OBJECTIVE_MIN) + GEP.OBJECTIVE_MIN
    ]
