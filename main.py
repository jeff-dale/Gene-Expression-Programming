from copy import deepcopy

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


def hard_regression():
    # TODO - crossover and mutation of constants
    # define objective function
    GeneExpressionProgram.OBJECTIVE_FUNCTION = staticmethod(lambda a: 4.251*np.power(a, 2) + np.log(np.power(a, 2)) + 7.243*np.exp(a))
    GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX = -1, 1
    GeneExpressionProgram.NUM_FITNESS_CASES = 20
    GeneExpressionProgram.FITNESS_FUNCTION = Chromosome.inv_squared_error
    GeneExpressionProgram.NUM_GENERATIONS = 5000
    GeneExpressionProgram.NUM_RUNS = 5
    GeneExpressionProgram.MUTATION_RATE = 0.041

    # Define terminals and functions
    Chromosome.terminals = ["a", "?"]
    Chromosome.ephemeral_random_constants_range = (-1, 1)
    Chromosome.functions = {
        "+": {"args": 2, "f": lambda x, y: x + y},
        "-": {"args": 2, "f": lambda x, y: x - y},
        "*": {"args": 2, "f": lambda x, y: x * y},
        "/": {"args": 2, "f": lambda x, y: x / y if y != 0 else np.nan},
        "L": {"args": 1, "f": lambda x: np.log(x) if x > 0 else np.nan},
        "E": {"args": 1, "f": lambda x: np.exp(x)},
        #"K": {"args": 1, "f": lambda x: np.log10(x) if x > 0 else np.nan},
        #"~": {"args": 1, "f": lambda x: np.power(10, x)},
        #"S": {"args": 1, "f": lambda x: np.sin(x)},
        #"C": {"args": 1, "f": lambda x: np.cos(x)}
    }
    Chromosome.fitness_cases = [
        ({"a": a}, GeneExpressionProgram.OBJECTIVE_FUNCTION(a))
        for a in np.random.rand(GeneExpressionProgram.NUM_FITNESS_CASES) * (
        GeneExpressionProgram.OBJECTIVE_MAX - GeneExpressionProgram.OBJECTIVE_MIN) + GeneExpressionProgram.OBJECTIVE_MIN
    ]

    Chromosome.linking_function = "+"
    Chromosome.length = 60
    Chromosome.head_length = 20
    Chromosome.max_fitness = 1 #GeneExpressionProgram.FUNCTION_Y_RANGE * GeneExpressionProgram.NUM_FITNESS_CASES

    average_fitnesses = []
    best_fitnesses = []
    answers = []

    ans = None
    for rep in range(GeneExpressionProgram.NUM_RUNS):
        print("====================================== Rep %d ======================================" % rep)
        ans, gen_average_fitnesses, gen_best_fitnesses = GeneExpressionProgram.evolve()
        average_fitnesses.append(gen_average_fitnesses)
        best_fitnesses.append(gen_best_fitnesses)
        ans.print_tree()
        print(ans.genes)
        answers.append(deepcopy(ans))

    print("====================================== Random Search ======================================")
    random_search_individual, random_search_avg, random_search_best = GeneExpressionProgram.random_search(
        GeneExpressionProgram.NUM_GENERATIONS,
        GeneExpressionProgram.FITNESS_FUNCTION,
        GeneExpressionProgram.FITNESS_FUNCTION_ARGS
    )

    # Good genes:
    #ans = Chromosome(['+/aE?aa?aaEa*Ea+E/?L?aa??aaaaa???aaaaa???aa?a??a?aa???aaa??a?a??a?aaaaaa?aaa??aa', '?a???L*Laa+++L-?///aaa??????a?aa?aaa?aaa?aaa?aa??a?a?a?aa??aaa?a?aaaaaa?aa???aaa', '+L**E*aaEaa+E??*/a*-???a?aa?aaaa?aaaaaaa?a?aaaa?aa?aaaaaaa???aa?aaaa?aa??aaa????'])
    #ans.ephemeral_random_constants = [0.15711028370174618, 0.8582913293794245, 0.08834476189713181, -0.6507871955790874, -0.5001482762755984, -0.36339831251011967, 0.49239731282122023, -0.7707384044151064, 0.33957437653782097, -0.050909069821311936, 0.40688394042469067, 0.8838615430620207, -0.03950637280673064, -0.8398663459127116, -0.9701111175669401, 0.16516078130630563, -0.5163031755060181, 0.26930803528455916, -0.7833159749333989, -0.7075160969083776, -0.6751546227334948, 0.1505636368911023, -0.16805240822390255, -0.34424168190370485, -0.8544980338428079, -0.01217703484745547, -0.24005751860391533, -0.5077198074421936, -0.47443544577074226, 0.5247967085947582, -0.22543318048008576, 0.4938865002308659, -0.6465093618701077, -0.19098460727467326, -0.2944062401077585, 0.7016839377380519, -0.14341637591100542, 0.23227088210671476, 0.36051772215302, 0.6509343605188611, -0.332150327502315, 0.3171544096208032, 0.2676850827993431, -0.46506262502073414, 0.843996276413379, -0.5614960005334659, 0.47891344757490373, -0.5575325624206526, -0.8156525364269045, -0.31652263727243746, 0.06884531540832572, 0.8836222097723032, 0.6601557412383419, -0.7869161076217597, -0.16110865846423783, -0.06702662866877018, -0.22568995470545228, 0.9438977429740325, -0.6410133183089668, 0.045353882523733624, 0.16786587502585948, -0.9008872395302681, 0.004355424045595413, 0.9179463218466064, 0.4105708444235643, -0.001799675111191501, -0.4201794697378056, -0.37672122028632216, 0.5906938113771141, 0.6004718433032417, -0.4698494772153414, 0.04238505345795085, 0.1657146428146341, 0.5148585050576182, 0.6955837854892111, -0.08812485635975653, -0.4101965485774235, 0.7234363214796748, 0.14285945798729927, 0.6450352601522822]
    #ans.plot_solution(GeneExpressionProgram.OBJECTIVE_FUNCTION, GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX, [], [], "a")

    #ans = Chromosome([0.10325095527275208, -0.6353207023355358, 0.3087002846077995, -0.646063772861873, -0.7604999404652737, -0.7026365243175745, -0.8546384172276964, -0.9210656698793773, -0.92473371383499159, 0.38475054329790037, 0.011392225394319944, -0.18840805047095377, 0.40103873272978197, 0.06328742279657229, 0.7176441901359412, -0.26941389839428265, 0.8330939983953638, -0.3291581824265808, -0.6081511326274629, 0.9437496769675557, 0.14649019501924365, -0.026586568971478375, 0.6942711369955727, -0.4952751886918654, 0.49348107956565457, 0.6373366861063929, -0.4180513467203788, 0.37222546291909575, 0.09407582333744235, -0.7976697480943422, -0.04875126785507633, 0.0738340898048051, 0.4574594498229325, 0.6176526203069368, -0.8316660105318336, 0.328772037314927, 0.854958551073552, -0.3935592840377773, -0.1523385500807699, -0.8272148377421589, 0.8580710774374207, -0.18672644199209687, 0.5393722078778671, -0.6906345459192962, -0.22443846331305117, 0.34512085908760115, 0.21516058072431332, -0.8499353974445536, 0.26989917691422316, -0.44201601204751584, 0.00887253706744362, -0.7836088648008288, -0.8679234491746306, -0.07209104779770104, 0.16000238129327515, 0.8890857194001152, -0.6325595149575627, -0.11804832230841544, -0.018168330413408817, -0.6561487673112327, 0.18667288624065526, -0.21171583904469493, -0.7074160949096726, -0.5647179047402893, 0.6466522462832285, 0.17157152187819658, -0.55051406791476754, -0.32459344245456756, -0.73568042741430184, 0.39543504084587644, -0.9332865781958235, 0.9903268434472272, 0.05041449572068979, 0.0798497529910911, -0.66617358746171074, 0.5071192429292173, 0.38014218205802264, -0.929868126429702, -0.23615736970440504, 0.5939936181262968])
    #ans.ephemeral_random_constants = ['+EEEaa/L-+L-EL-L?aLEaa??????aa?aa?????aa??a???aaaaaaa????aa?aa??aa??a?a??aaa?aaa', '+EEE*aa?+?*/LLa/*+*-a?a?aaaa??a?aa?a??aa???aaaaa????a???????a?aaa????????aaaaaa?', '-L***aaa--+*+*+?+?--?aa??aa?aa?aa??a??a?aa??aaaaa?a??a?aa?aa?????aaaa??aa?aaa?a?']

    best = max(answers, key=lambda c:c.fitness())
    ans.plot_solution(GeneExpressionProgram.OBJECTIVE_FUNCTION, GeneExpressionProgram.OBJECTIVE_MIN, GeneExpressionProgram.OBJECTIVE_MAX, average_fitnesses[0], best_fitnesses[0], "a")
    GeneExpressionProgram.plot_reps(average_fitnesses, best_fitnesses, random_search_avg, random_search_best)


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
        "/": {"args": 2, "f": lambda x, y: x / y if y != 0 else np.inf}
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
    GeneExpressionProgram.NUM_RUNS = 5
    GeneExpressionProgram.NUM_GENERATIONS = 15
    GeneExpressionProgram.MUTATION_RATE = 0.05

    Chromosome.max_fitness = 500
    Chromosome.num_genes = 3
    Chromosome.length = 30
    Chromosome.head_length = 10
    Chromosome.linking_function = "+"

    average_fitnesses = []
    best_fitnesses = []

    ans = None
    for rep in range(GeneExpressionProgram.NUM_RUNS):
        print("====================================== Rep %d ======================================" % rep)
        ans, gen_average_fitnesses, gen_best_fitnesses = GeneExpressionProgram.evolve()
        average_fitnesses.append(gen_average_fitnesses)
        best_fitnesses.append(gen_best_fitnesses)
        #ans.print_tree()
        #print(ans.genes)

    print("====================================== Random Search ======================================")
    random_search_individual, random_search_avg, random_search_best = GeneExpressionProgram.random_search(
        GeneExpressionProgram.NUM_GENERATIONS,
        GeneExpressionProgram.FITNESS_FUNCTION,
        GeneExpressionProgram.FITNESS_FUNCTION_ARGS
    )

    GeneExpressionProgram.plot_reps(average_fitnesses, best_fitnesses, random_search_avg, random_search_best)

    fitness(100, True, True, ans)

    # Very good genes (500 fitness over 100 attempts):
    # ['-u-uvx*txtxxutuutvtvtvttttutvu', '+*vxu/x/-/tutxtxxxuvuuxvtuvvtu', '+ut*vv*vtvxvxvuxvtxvtvuutvtttt']
    # ['t/v*/xtttvuxtvvttutxxtuvxuvttt', '-v*+v*+t+tuxtvvttuuvxvtuvuvxtt', 'u++tu+uvtvtxtvtvtuvxxttvxvxvv']
    # ['-t-*+u*u-*tuutxxuutxtxttxttvxu', '+-u**+-/vxvxxxvtxvxuxvxvvxxtvx', '+xv-xvx-xtuutxxtxxxtxxuttvxxut']


if __name__ == "__main__":

    #a4_a3_a2_a1()
    #sinx_polynomial()
    #cart_pole_real()
    hard_regression()
