from Chromosome import Chromosome
from GeneExpressionProgram import GeneExpressionProgram

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
    GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX = -5, 5
    GeneExpressionProgram.FITNESS_FUNCTION = Chromosome.centralized_inv_squared_error
    GeneExpressionProgram.FITNESS_FUNCTION_ARGS = [0, "x"]
    GeneExpressionProgram.NUM_FITNESS_CASES = 10
    GeneExpressionProgram.NUM_GENERATIONS = 150
    GeneExpressionProgram.POPULATION_SIZE = 200
    GeneExpressionProgram.MUTATION_RATE = 0.05

    Chromosome.head_length = 8
    Chromosome.num_genes = 3
    Chromosome.length = 24

    # Define terminals and functions
    Chromosome.terminals = ["x", "6", "!"]
    Chromosome.constants = {"!": 120}
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
    Chromosome.linking_function = "*"
    Chromosome.max_fitness = 1 #GeneExpressionProgram.FUNCTION_Y_RANGE * GeneExpressionProgram.NUM_FITNESS_CASES

    ans, average_fitnesses, best_fitnesses = GeneExpressionProgram.evolve()
    ans.print_tree()
    ans.plot_solution(GeneExpressionProgram.OBJECTIVE_FUNCTION,
                      GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX,
                      average_fitnesses, best_fitnesses)


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


def cart_pole_bool():

    import gym
    env = gym.make('CartPole-v1')

    def fitness(num_trials: int, *args) -> np.ndarray:
        fitnesses = []
        for chromosome_index in range(len(args)):
            chromosome = args[chromosome_index]
            total_reward = 0
            #print("\tCalculating chromosome:\t%d" % chromosome_index, end="\t")
            for trial in range(num_trials):
                x, theta, dx, dtheta = env.reset()
                t = 0
                while True:
                    #env.render()
                    action = chromosome.evaluate({"x": x > 0, "v": dx > 0, "t": theta > 0, "u": dtheta > 0})
                    #action = chromosome.evaluate({"v": dx > 0, "t": theta > 0, "u": dtheta > 0})
                    observation, reward, done, info = env.step(action)
                    x, theta, dx, dtheta = observation
                    total_reward += reward
                    if done:
                        break
                    t += 1
            chromosome._fitness_ = total_reward / float(num_trials)
            fitnesses.append(total_reward / float(num_trials))
            #print("Fitness: %.5f" % (total_reward / float(num_trials)))
        return np.asarray(fitnesses)

    Chromosome.functions = {

        # if function
        "?": {"args": 3, "f": lambda x, y, z:  y if x else z},

        # and/or functions
        "&": {"args": 2, "f": lambda x, y: x and y},
        "|": {"args": 2, "f": lambda x, y: x or y},

        # not function
        "!": {"args": 1, "f": lambda x: not x},

        # xor/nor functions
        "^": {"args": 2, "f": lambda x, y: (x and not y) or (y and not x)},
        "#": {"args": 2, "f": lambda x, y: not (x and y)}
    }

    Chromosome.terminals = [
        "x",    # cart x-value
        "v",    # cart velocity
        "t",    # pole angle
        "u"     # rate of change of pole angle
    ]

    GeneExpressionProgram.FITNESS_FUNCTION = fitness
    GeneExpressionProgram.FITNESS_FUNCTION_ARGS = [10]
    GeneExpressionProgram.POPULATION_SIZE = 10
    GeneExpressionProgram.ERROR_TOLERANCE = 5

    Chromosome.max_fitness = 500
    Chromosome.num_genes = 3
    Chromosome.length = 45
    Chromosome.head_length = 15
    Chromosome.linking_function = "&"

    ans, average_fitnesses, best_fitnesses = GeneExpressionProgram.evolve()
    ans.print_tree()
    ans.plot_solution(None, 0, 0, average_fitnesses, best_fitnesses)
    print(ans.genes)


def cart_pole_real():

    import gym
    env = gym.make('CartPole-v1')

    def fitness(num_trials: int, render: bool = False, doPrint: bool = False, *args) -> np.ndarray:
        fitnesses = []
        for chromosome_index in range(len(args)):
            chromosome = args[chromosome_index]
            total_reward = 0
            if doPrint:
                print("\tCalculating chromosome %d across %d trials." % (chromosome_index, num_trials), end="\t")
            for trial in range(num_trials):
                x, theta, dx, dtheta = env.reset()
                t = 0
                while True:
                    if render: env.render()
                    action = chromosome.evaluate({"x": x, "v": dx, "t": theta, "u": dtheta})
                    observation, reward, done, info = env.step(action > 0)
                    x, theta, dx, dtheta = observation
                    total_reward += reward
                    if done:
                        break
                    t += 1
            chromosome._fitness_ = total_reward / float(num_trials)
            fitnesses.append(total_reward / float(num_trials))
            if doPrint:
                print("Fitness: %.5f" % (total_reward / float(num_trials)))
        return np.asarray(fitnesses)

    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y}
    }

    Chromosome.terminals = [
        "x",    # cart x-value
        "v",    # cart velocity
        "t",    # pole angle
        "u"     # rate of change of pole angle
    ]

    GeneExpressionProgram.FITNESS_FUNCTION = fitness
    GeneExpressionProgram.FITNESS_FUNCTION_ARGS = [10, False, False]
    GeneExpressionProgram.POPULATION_SIZE = 10
    GeneExpressionProgram.ERROR_TOLERANCE = 0
    GeneExpressionProgram.NUM_RUNS = 1
    GeneExpressionProgram.NUM_GENERATIONS = 15

    Chromosome.max_fitness = 5000
    Chromosome.num_genes = 3
    Chromosome.length = 30
    Chromosome.head_length = 10
    Chromosome.linking_function = "+"

    average_fitnesses = []
    best_fitnesses = []

    for rep in range(GeneExpressionProgram.NUM_RUNS):
        ans, gen_average_fitnesses, gen_best_fitnesses = GeneExpressionProgram.evolve()
        average_fitnesses.append(gen_average_fitnesses)
        best_fitnesses.append(gen_best_fitnesses)
        ans.print_tree()
        print(ans.genes)

        fitness(100, False, True, ans)

    random_search_individual, random_search_avg, random_search_best = GeneExpressionProgram.random_search(
        GeneExpressionProgram.NUM_GENERATIONS,
        GeneExpressionProgram.FITNESS_FUNCTION,
        GeneExpressionProgram.FITNESS_FUNCTION_ARGS
    )

    GeneExpressionProgram.plot_reps(average_fitnesses, best_fitnesses, random_search_avg, random_search_best)

    # Very good genes (500 fitness over 100 attempts):
    # ['-u-uvx*txtxxutuutvtvtvttttutvu', '+*vxu/x/-/tutxtxxxuvuuxvtuvvtu', '+ut*vv*vtvxvxvuxvtxvtvuutvtttt']
    # ['t/v*/xtttvuxtvvttutxxtuvxuvttt', '-v*+v*+t+tuxtvvttuuvxvtuvuvxtt', 'u++tu+uvtvtxtvtvtuvxxttvxvxvv']
    # ['-t-*+u*u-*tuutxxuutxtxttxttvxu', '+-u**+-/vxvxxxvtxvxuxvxvvxxtvx', '+xv-xvx-xtuutxxtxxxtxxuttvxxut']


if __name__ == "__main__":

    #a4_a3_a2_a1()
    #sinx_polynomial()
    cart_pole_real()