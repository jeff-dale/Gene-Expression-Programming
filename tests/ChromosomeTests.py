import numpy as np
import unittest

from Chromosome import Chromosome


# noinspection PyMethodMayBeStatic
class ChromosomeTests(unittest.TestCase):

    def setUp(self):
        self.F = {
            "+": {"args": 2, "f": lambda x, y: x + y},
            "-": {"args": 2, "f": lambda x, y: x - y},
            "*": {"args": 2, "f": lambda x, y: x * y},
            "/": {"args": 2, "f": lambda x, y: x / y},
            "Q": {"args": 1, "f": lambda x: x ** 0.5},
            "I": {"args": 3, "f": lambda x, y, z: y if x else z}
        }
        self.T = ["a", "b", "2", "3", "1", "u", "c"]
        self.HEAD_LENGTH = 6
        self.objective_function = lambda a: a ** 4 + a ** 3 + a ** 2 + a
        Chromosome.functions = self.F
        Chromosome.terminals = self.T
        Chromosome.head_length = self.HEAD_LENGTH
        Chromosome.num_genes = 1
        Chromosome.fitness_cases = [(a, self.objective_function(a)) for a in np.random.rand(10) * 20]

    def test_build_tree1(self):

        c1 = Chromosome(["Q*Q+bbaaa"])
        t = Chromosome.build_tree(c1.genes[0])
        assert(
            str(t.descendants).replace("\\", "") ==
            r"(Node(\'/Q/*\'), Node(\'/Q/*/Q\'), Node(\'/Q/*/Q/b\'), Node(\'/Q/*/+\'), Node(\'/Q/*/+/b\'), Node(\'/Q/*/+/a\'))".replace("\\", "")
        )

    def test_build_tree2(self):
        c2 = Chromosome(["IIIIIIIIIIIII131u3ab2ubab23c3ua31a333au3"])
        t = Chromosome.build_tree(c2.genes[0])
        assert(
            str(t.descendants).replace("\\", "") ==
            r"(Node('/I/I'), Node('/I/I/I'), Node('/I/I/I/1'), Node('/I/I/I/3'), Node('/I/I/I/1'), Node('/I/I/I'), Node('/I/I/I/u'), Node('/I/I/I/3'), Node('/I/I/I/a'), Node('/I/I/I'), Node('/I/I/I/b'), Node('/I/I/I/2'), Node('/I/I/I/u'), Node('/I/I'), Node('/I/I/I'), Node('/I/I/I/b'), Node('/I/I/I/a'), Node('/I/I/I/b'), Node('/I/I/I'), Node('/I/I/I/2'), Node('/I/I/I/3'), Node('/I/I/I/c'), Node('/I/I/I'), Node('/I/I/I/3'), Node('/I/I/I/u'), Node('/I/I/I/a'), Node('/I/I'), Node('/I/I/I'), Node('/I/I/I/3'), Node('/I/I/I/1'), Node('/I/I/I/a'), Node('/I/I/I'), Node('/I/I/I/3'), Node('/I/I/I/3'), Node('/I/I/I/3'), Node('/I/I/I'), Node('/I/I/I/a'), Node('/I/I/I/u'), Node('/I/I/I/3'))".replace("\\", "")
        )

    def test_build_tree3(self):
        c3 = Chromosome(["/a*+b-cbabaccbac"])
        t = Chromosome.build_tree(c3.genes[0])
        assert(
            str(t.descendants).replace("\\", "") ==
            r"(Node(\\'///a\\'), Node(\\'///*\\'), Node(\\'///*/+\\'), Node(\\'///*/+/-\\'), Node(\\'///*/+/-/b\\'), Node(\\'///*/+/-/a\\'), Node(\\'///*/+/c\\'), Node(\\'///*/b\\'))".replace("\\", "")
        )

    def test_evaluate1(self):
        c1 = Chromosome(["Q*Q+bbaaa"])
        assert(
            c1.evaluate(terminal_values={
                "a": 3,
                "b": 1
            }) == 2
        )

    def test_evaluate2(self):
        Chromosome.linking_function = "+"
        Chromosome.num_genes = 2
        c1 = Chromosome(["Q*Q+bbaaa", "*-babaabb"])
        assert(
            c1.evaluate(terminal_values={
                "a": 3,
                "b": 1
            }) == 4
        )

    def test_evaluate3(self):
        Chromosome.linking_function = "-"
        Chromosome.num_genes = 2
        c1 = Chromosome(["Q*Q+bbaaa", "*-babaabb"])
        assert (
            c1.evaluate(terminal_values={
                "a": 3,
                "b": 1
            }) == 0
        )


if __name__ == "__main__":
    unittest.main()