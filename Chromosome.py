from lib.anytree.node import Node


class Chromosome:

    # Functions and Terminals are shared by all chromosomes
    functions = dict()
    terminals = list()
    linking_function = None

    # length of head of chromosome
    head_length = None

    # list of real-valued tuples of the form (x, f(x))
    fitness_cases = []


    def __init__(self, genes: list):

        # do not let chromosomes be defined without first defining their functions, terminals, and head length
        if not Chromosome.functions:
            raise ValueError("Chromosome class has no functions associated with it.")
        if len(Chromosome.terminals) == 0:
            raise ValueError("Chromosome class has no terminals associated with it.")
        if Chromosome.head_length is None:
            raise ValueError("Chromosome class has no head length defined.")
        if Chromosome.linking_function is None and len(genes) > 1:
            raise ValueError("Multigenic chromosome defined with no linking function.")

        # initialize chromosomes
        self.genes = genes
        self.num_genes = len(self.genes)
        self.trees = []
        self.value = None


    def evaluate(self, terminal_values: dict) -> float:
        """
        Returns the result of evaluating the given chromosome for specified fitness cases.

        :param terminal_values: dictionary mapping all present terminal symbols to real values
        :return: real valued result of evaluating the chromosome
        """

        if len(Chromosome.fitness_cases) == 0:
            raise ValueError("Chromosome class has no fitness cases.")

        # memoize value in case the chromosome was already evaluated
        if self.value is not None:
            return self.value

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

        self.value = inorder(expression_tree)

        # noinspection PyTypeChecker
        return self.value


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
    def link(*args) -> Node:
        """
        Links two trees at their roots using the specified linking function.
        Linking function must take as many arguments as number of args provided.

        :param args: expression trees to link. Must be as many expression trees as linking function has arguments.
        :return: expression tree with tree1 and tree2 as subtrees
        """
        if Chromosome.linking_function not in Chromosome.functions:
            raise ValueError("Linking function is not defined in Chromosome.functions.")
        if len(args) != Chromosome.functions[Chromosome.linking_function]["args"]:
            raise ValueError("Invalid number of arguments to linking function.")
        root = Node(Chromosome.linking_function)
        for tree in args:
            assert(isinstance(tree, Node))
            tree.parent = root
        return root
