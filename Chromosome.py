from lib.anytree.node import Node
from lib.anytree.render import RenderTree

import numpy as np
from random import randint


class Chromosome:

    # Functions and Terminals are shared by all chromosomes
    functions = dict()
    terminals = list()
    linking_function = None

    # length of head of chromosome
    num_genes = 3
    head_length = 6
    length = 39

    # list of real-valued tuples of the form (x, f(x))
    fitness_cases = []
    max_fitness = None


    def __init__(self, genes: list):

        # do not let chromosomes be defined without first defining their functions, terminals, and head length
        if not Chromosome.functions:
            raise ValueError("Chromosome class has no functions associated with it.")
        if len(Chromosome.terminals) == 0:
            raise ValueError("Chromosome class has no terminals associated with it.")
        if Chromosome.length is None:
            raise ValueError("Chromosome class has no length defined.")
        if Chromosome.head_length is None:
            raise ValueError("Chromosome class has no head length defined.")
        if Chromosome.linking_function is None and len(genes) > 1:
            raise ValueError("Multigenic chromosome defined with no linking function.")
        if len(genes) != Chromosome.num_genes:
            raise ValueError("Number of genes does not match excpected value in class level variable.")

        # initialize chromosomes
        self.genes = genes
        self.trees = []
        self._values_ = {}
        self._fitness_ = None

    # TODO - put informative error message when terminal_values doesn't have enough entries
    def evaluate(self, terminal_values: dict) -> float:
        """
        Returns the result of evaluating the given chromosome for specified fitness cases.

        :param terminal_values: dictionary mapping all present terminal symbols to real values
        :return: real valued result of evaluating the chromosome
        """

        if len(Chromosome.fitness_cases) == 0:
            raise ValueError("Chromosome class has no fitness cases.")

        # memoize value in case the chromosome was already evaluated
        value_fingerprint = tuple(sorted(terminal_values.items()))
        if value_fingerprint in self._values_:
            return self._values_[value_fingerprint]

        # build expression trees for each gene if not already built
        if len(self.trees) == 0:
            self.trees = [Chromosome.build_tree(gene) for gene in self.genes]

        # link expression trees if the chromosome is multigenic, otherwise use first tree
        if self.num_genes > 1:
            expression_tree = Chromosome.link(*self.trees)
        else:
            expression_tree = self.trees[0]

        # recursive inorder tree traversal
        def inorder(start: Node) -> float:
            nonlocal terminal_values
            if start.name in Chromosome.terminals:
                return terminal_values[start.name]
            if start.name in Chromosome.functions:
                return Chromosome.functions[start.name]["f"](*[inorder(node) for node in start.children])

        self._values_[value_fingerprint] = inorder(expression_tree)

        # noinspection PyTypeChecker
        return self._values_[value_fingerprint]


    def fitness(self):
        if self._fitness_ is not None:
            return self._fitness_
        raise ValueError("Fitness of chromosome has not been properly calculated.")


    def print_tree(self):
        for t in range(len(self.trees)):
            print("Tree %d" % t)
            for pre, _, node in RenderTree(self.trees[t]):
                print("\t%s%s" % (pre, node.name))


    @staticmethod
    def build_tree(gene: str) -> Node:

        # shortcut to get the number of arguments to a function
        def args(f: str) -> int:
            return Chromosome.functions[f]["args"] if f in Chromosome.functions else 0

        # recursively build chromosome tree
        def grab_children(parent: Node, current_level = 1):
            nonlocal levels
            if current_level < len(levels):
                nargs = args(parent.name)
                for i in range(nargs):
                    current_node = Node(levels[current_level][i], parent=parent)
                    grab_children(parent=current_node, current_level=current_level + 1)
                    if current_level < len(levels) - 1:
                        levels[current_level + 1] = levels[current_level + 1][args(current_node.name):]

        # build each level of the tree
        levels = [gene[0]]
        index = 0
        while index < len(gene) and sum([args(f) for f in levels[-1]]) != 0:
            nargs = sum([args(f) for f in levels[-1]])
            levels.append(gene[index + 1: index + 1 + nargs])
            index += nargs

        # intialize tree and parse
        tree = Node(gene[0])
        grab_children(tree)
        return tree


    @staticmethod
    # TODO - verify recursive linking with non-commutative linking functions (e.g. -)
    def link(*args) -> Node:
        """
        Links two trees at their roots using the specified linking function.
        Linking function must take as many arguments as number of args provided.

        :param args: expression trees to link. Must be at least as many expression trees as linking function has arguments.
        :return: expression tree with tree1 and tree2 as subtrees
        """
        if Chromosome.linking_function not in Chromosome.functions:
            raise ValueError("Linking function is not defined in Chromosome.functions.")
        if not all([isinstance(arg, Node) for arg in args]):
            raise TypeError("Can only link expression trees.")

        nargs = Chromosome.functions[Chromosome.linking_function]["args"]

        def link_recursive(*args) -> Node:
            root = Node(Chromosome.linking_function)
            if len(args) == nargs:
                for tree in args:
                    tree.parent = root
                return root
            else:
                return link_recursive(link_recursive(*args[:nargs]), *args[nargs:])

        return link_recursive(*args)


    @staticmethod
    # TODO - calculate using numpy arrays for speed
    def absolute_fitness(M: float, *args) -> np.ndarray:
        """
        Calculate absolute fitness of an arbitrary number of Chromosomes.

        :param M: range of fitness function over domain
        :param args: any number of gene objects
        :return: list of fitnesses of corresponding chromosomes
        """
        fitnesses = []
        for chromosome in args:
            # memoize fitness values
            if chromosome._fitness_ is not None:
                fitnesses.append(chromosome._fitness_)
            else:
                fitness = 0
                for j in range(len(Chromosome.fitness_cases)):
                    C_ij = chromosome.evaluate(Chromosome.fitness_cases[j][0])
                    # assign any chromosome that divides by zero a fitness value of zero
                    if np.isnan(C_ij) or np.isinf(C_ij) or np.isneginf(C_ij):
                        fitness = 0
                        break
                    T_j = Chromosome.fitness_cases[j][1]
                    fitness += M - abs(C_ij - T_j)
                chromosome._fitness_ = fitness
                fitnesses.append(fitness)
        return np.asarray(fitnesses)


    @staticmethod
    # TODO - calculate using numpy arrays for speed
    def relative_fitness(M: float, *args) -> np.ndarray:
        """
        Calculate relative fitness of an arbitrary number of genes.

        :param M: range of fitness function over domain
        :param args: any number of gene objects
        :return: list of fitnesses of corresponding genes
        """
        fitnesses = []
        for chromosome in args:
            # memoize fitness values
            if chromosome._fitness_ is not None:
                fitnesses.append(chromosome._fitness_)
            else:
                fitness = 0
                for j in range(len(Chromosome.fitness_cases)):
                    C_ij = chromosome.evaluate(Chromosome.fitness_cases[j][0])
                    T_j = Chromosome.fitness_cases[j][1]
                    fitness += M - 100*abs(C_ij / T_j - 1)
                chromosome._fitness_ = fitness
                fitnesses.append(fitness)
        return np.asarray(fitnesses)


    @staticmethod
    def generate_random_gene() -> str:
        """
        Generates one random gene based on settings specified in Chromosome class.
        :return: string of valid characters
        """
        possible_chars = list(Chromosome.functions.keys()) + Chromosome.terminals
        head = "".join([possible_chars[randint(0, len(possible_chars) - 1)] for _ in range(Chromosome.head_length)])
        tail = "".join([Chromosome.terminals[randint(0, len(Chromosome.terminals) - 1)] for _ in range(Chromosome.length - Chromosome.head_length)])
        return head + tail


    @staticmethod
    def generate_random_individual() -> 'Chromosome':
        """
        Generates one random individual based on settings specified in Chromosome class.
        :return: new Chromosome
        """
        return Chromosome([Chromosome.generate_random_gene() for _ in range(Chromosome.num_genes)])