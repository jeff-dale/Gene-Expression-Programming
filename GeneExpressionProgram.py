from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from random import random, randint, shuffle

from Chromosome import Chromosome

class GeneExpressionProgram:

    ### Hyperparameters ###
    NUM_RUNS = 5
    NUM_GENERATIONS = 500
    POPULATION_SIZE = 100
    NUM_FITNESS_CASES = 10
    ERROR_TOLERANCE = 0.0000001

    ### Reproduction ###
    MUTATION_RATE = 0.051
    ONE_POINT_CROSSOVER_RATE, TWO_POINT_CROSSOVER_RATE, GENE_CROSSOVER_RATE = 0.2, 0.5, 0.1
    IS_TRANSPOSITION_RATE, IS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    RIS_TRANSPOSITION_RATE, RIS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    GENE_TRANSPOSITION_RATE = 0.1

    ### Fitness Evaluation ###
    OBJECTIVE_FUNCTION = None
    FITNESS_FUNCTION = None
    FITNESS_FUNCTION_ARGS = list()
    OBJECTIVE_MIN, OBJECTIVE_MAX = None, None
    FUNCTION_Y_RANGE = None


    def __init__(self):
        pass


    @staticmethod
    def evolve() -> (Chromosome, list, list):
        """
        Execute Gene Expression Programming algorithm

        :return: tuple of:
            the best fit chromosome,
            list of average fitness by generation (for plotting),
            list of best fitness by generation (for plotting
        """

        # create initial population
        population = [Chromosome.generate_random_individual() for _ in range(GeneExpressionProgram.POPULATION_SIZE)]

        generation = 0
        best_fit_individual = None
        average_fitness_by_generation = []
        best_fitness_by_generation = []
        while generation < GeneExpressionProgram.NUM_GENERATIONS:

            ### EVALUATION ###

            # calcluate fitnesses for population
            population_fitnesses = GeneExpressionProgram.FITNESS_FUNCTION(*GeneExpressionProgram.FITNESS_FUNCTION_ARGS, *population)

            # find best fit individual
            # noinspection PyTypeChecker
            best_fit_generation = population[np.argmax(population_fitnesses)]
            if generation == 0 or best_fit_individual.fitness() < best_fit_generation.fitness():
                best_fit_individual = deepcopy(best_fit_generation)

            # skip rest of loop if we have found optimal solution
            if abs(best_fit_individual.fitness() - Chromosome.max_fitness) <= GeneExpressionProgram.ERROR_TOLERANCE:
                average_fitness_generation = float(np.mean(population_fitnesses))
                average_fitness_by_generation.append(average_fitness_generation)
                best_fitness_by_generation.append(best_fit_individual.fitness())
                break

            next_generation = list()


            ### SELECTION (roulette wheel with simple elitism) ###

            # copy best individual to next generation
            next_generation.append(deepcopy(best_fit_individual))

            # select the rest of the next generation with roulette wheel selection
            all_parents = GeneExpressionProgram.roulette_wheel_selection(population, len(population))

            # Mutation
            all_parents = list(map(GeneExpressionProgram.mutate, all_parents))

            # IS Transposition
            all_parents = list(map(GeneExpressionProgram.is_transposition, all_parents))

            # RIS Transposition
            all_parents = list(map(GeneExpressionProgram.ris_transposition, all_parents))

            # Gene Transposition
            all_parents = list(map(GeneExpressionProgram.gene_transposition, all_parents))

            # Recombination
            shuffle(all_parents)
            for i in range(1, GeneExpressionProgram.POPULATION_SIZE, 2):

                # in case we don't have a pair to check for crossover, avoid index error
                if i + 1 >= GeneExpressionProgram.POPULATION_SIZE:
                    next_generation.append(all_parents[i])
                    break

                child1, child2 = all_parents[i], all_parents[i+1]

                # One-point Recombination
                if random() < GeneExpressionProgram.ONE_POINT_CROSSOVER_RATE:
                    child1, child2 = GeneExpressionProgram.one_point_recombination(child1, child2)

                # Two-point Recombination
                elif random() < GeneExpressionProgram.TWO_POINT_CROSSOVER_RATE:
                    child1, child2 = GeneExpressionProgram.two_point_recombination(child1, child2)

                # Gene Recombination
                elif random() < GeneExpressionProgram.GENE_CROSSOVER_RATE:
                    child1, child2 = GeneExpressionProgram.gene_recombination(child1, child2)

                # Include children in next generation
                next_generation.append(child1)
                next_generation.append(child2)

            # prepare for next iteration
            population = next_generation
            generation += 1

            average_fitness_generation = float(np.mean(population_fitnesses))
            average_fitness_by_generation.append(average_fitness_generation)
            best_fitness_by_generation.append(best_fit_individual.fitness())
            print("Generation: %d\tPopulation Size: %d\tAverage Fitness: %.5f\tBest Fitness (overall): %.5f" %
                  (generation, len(population), average_fitness_generation, best_fit_individual.fitness()))

        return best_fit_individual, average_fitness_by_generation, best_fitness_by_generation


    @staticmethod
    def random_search(num_generations: int, fitness_function: callable, fitness_function_args: list) -> (Chromosome, list, list):
        best = None
        best_fitness = 0
        average_fitnesses = []
        best_fitnesses = []
        for gen in range(num_generations):
            generation_fitnesses = []
            for individual in range(GeneExpressionProgram.POPULATION_SIZE):
                current = Chromosome.generate_random_individual()
                current_fitness = fitness_function(*fitness_function_args, current)
                generation_fitnesses.append(current_fitness)
                if best is None or best_fitness <= current_fitness:
                    best = deepcopy(current)
                    best_fitness = best.fitness()
            average_fitnesses.append(np.mean(generation_fitnesses))
            best_fitnesses.append(best_fitness)
            if best_fitness >= Chromosome.max_fitness:
                break
        return best, average_fitnesses, best_fitnesses


    @staticmethod
    def roulette_wheel_selection(chromosomes: list, n: int) -> list:
        """
        Returns n randomly selected chromosomes using roulette wheel selection.
        Adapted from:
        https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights

        :param chromosomes: list of chromosomes to select from
        :param n: number of samples to retrieve
        :return: list of chosen chromosome(s)
        """
        total = float(sum(c.fitness() for c in chromosomes))
        i = 0
        fitness = chromosomes[0].fitness()
        while n:
            x = total * (1 - random() ** (1.0 / n))
            total -= x
            while x > fitness:
                x -= fitness
                i += 1
                fitness = chromosomes[i].fitness()
            fitness -= x
            yield chromosomes[i]
            n -= 1


    @staticmethod
    def mutate(chromosome: Chromosome) -> Chromosome:
        """
        Randomly mutate genes in a chromosome.

        :param chromosome: chromosome to mutate
        :return: mutated chromosome
        """

        head_characters = list(Chromosome.functions.keys()) + Chromosome.terminals

        new_genes = []
        # for each gene (multigenic chromosomes)
        for gene in range(Chromosome.num_genes):
            new_gene = ""
            # for each character in the gene
            for i in range(len(chromosome.genes[gene])):
                # if we are to mutate it
                if random() < GeneExpressionProgram.MUTATION_RATE:
                    # if we are mutating the head
                    if i < Chromosome.head_length:
                        new_gene += head_characters[randint(0, len(head_characters) - 1)]
                    else:
                        new_gene += Chromosome.terminals[randint(0, len(Chromosome.terminals) - 1)]
                else:
                    new_gene += chromosome.genes[gene][i]
            new_genes.append(new_gene)

        # create new chromosome to ensure memoized fitness values are recalculated.
        new_chromosome = Chromosome(new_genes)
        new_chromosome.ephemeral_random_constants = chromosome.ephemeral_random_constants

        # mutate ephemeral random constants
        for constant in range(len(new_chromosome.ephemeral_random_constants)):
            if random() < GeneExpressionProgram.MUTATION_RATE:
                new_chromosome.ephemeral_random_constants[constant] = np.random.uniform(*Chromosome.ephemeral_random_constants_range)

        return new_chromosome


    @staticmethod
    def is_transposition(chromosome: Chromosome) -> Chromosome:
        """
        Insertion Sequence transposition.

        :param chromosome: chromosome to perform IS transposition on
        :return: new chromosome
        """

        # TODO - properly transpose ephemeral random constants

        if random() < GeneExpressionProgram.IS_TRANSPOSITION_RATE:

            # determine parameters of transposition
            length = np.random.choice(GeneExpressionProgram.IS_ELEMENTS_LENGTH)
            source_gene = randint(0, len(chromosome.genes) - 1)
            target_gene = randint(0, len(chromosome.genes) - 1)
            target_position = randint(1, Chromosome.head_length - length)
            sequence_start = randint(0, len(chromosome.genes[source_gene]))

            transposition_string = chromosome.genes[source_gene][sequence_start:min(Chromosome.length, sequence_start + length)]

            # make substitution
            new_chromosome = Chromosome(chromosome.genes)
            new_chromosome.ephemeral_random_constants = chromosome.ephemeral_random_constants
            new_chromosome.genes[target_gene] = new_chromosome.genes[target_gene][:target_position] + \
                                                transposition_string + \
                                                new_chromosome.genes[target_gene][target_position + length:]

            return new_chromosome

        else:

            return chromosome


    @staticmethod
    def ris_transposition(chromosome: Chromosome) -> Chromosome:
        """
        Root Insertion Sequence transposition.

        :param chromosome: chromosome to perform RIS transposition on
        :return: new chromosome
        """

        # TODO - properly transpose ephemeral random constants

        start_point = randint(0, Chromosome.head_length - 1)
        gene = randint(0, Chromosome.num_genes - 1)
        while start_point < Chromosome.head_length and chromosome.genes[gene][start_point] not in Chromosome.functions:
            start_point += 1

        if random() < GeneExpressionProgram.RIS_TRANSPOSITION_RATE and chromosome.genes[gene][start_point] in Chromosome.functions:
            ris_length = np.random.choice(GeneExpressionProgram.RIS_ELEMENTS_LENGTH)
            ris_string = chromosome.genes[gene][start_point:start_point+ris_length]

            new_chromosome = Chromosome(chromosome.genes)
            new_chromosome.ephemeral_random_constants = chromosome.ephemeral_random_constants
            old_head = new_chromosome.genes[gene][:Chromosome.head_length]
            new_head = old_head[:start_point] + ris_string + old_head[start_point:]
            new_chromosome.genes[gene] = new_head[:Chromosome.head_length] + new_chromosome.genes[gene][Chromosome.head_length:]

            return new_chromosome

        else:

            return chromosome


    @staticmethod
    def gene_transposition(chromosome: Chromosome) -> Chromosome:
        """
        Gene Insertion Sequence transposition.

        :param chromosome: chromosome to perform gene transposition on
        :return: new chromosome
        """

        # TODO - properly transpose ephemeral random constants

        if Chromosome.num_genes > 1 and random() < GeneExpressionProgram.GENE_TRANSPOSITION_RATE:

            index = randint(0, Chromosome.num_genes - 1)
            temp = chromosome.genes[index]
            chromosome.genes[index] = chromosome.genes[0]
            chromosome.genes[0] = temp
            new_chromosome = Chromosome(chromosome.genes)
            new_chromosome.ephemeral_random_constants = chromosome.ephemeral_random_constants
            return new_chromosome

        else:

            return chromosome


    @staticmethod
    def one_point_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> (Chromosome, Chromosome):
        """
        Classical one point recombination.

        :param chromosome1: parent 1
        :param chromosome2: parent 2
        :return: offspring 1, offspring 2
        """
        gene = randint(0, Chromosome.num_genes - 1)
        position = randint(0, Chromosome.length)

        child1_split_gene = chromosome1.genes[gene][:position] + chromosome2.genes[gene][position:]
        child2_split_gene = chromosome2.genes[gene][:position] + chromosome1.genes[gene][position:]

        child1_genes = chromosome1.genes[:gene] + [child1_split_gene] + (chromosome2.genes[gene+1:] if gene < Chromosome.num_genes - 1 else [])
        child2_genes = chromosome2.genes[:gene] + [child2_split_gene] + (chromosome1.genes[gene + 1:] if gene < Chromosome.num_genes - 1 else [])

        child1, child2 = Chromosome(child1_genes), Chromosome(child2_genes)

        constants_split_position = randint(0, Chromosome.length - 1)
        child1.ephemeral_random_constants = chromosome1.ephemeral_random_constants[:constants_split_position] + \
                                            chromosome2.ephemeral_random_constants[constants_split_position:]

        child2.ephemeral_random_constants = chromosome2.ephemeral_random_constants[:constants_split_position] + \
                                            chromosome1.ephemeral_random_constants[constants_split_position:]

        return child1, child2


    @staticmethod
    def two_point_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> (Chromosome, Chromosome):
        """
        Classical two point recombination.

        :param chromosome1: parent 1
        :param chromosome2: parent 2
        :return: offspring 1, offsprint 2
        """

        # generate crossover points
        position1, position2 = sorted([randint(0, Chromosome.length*Chromosome.num_genes - 1), randint(0, Chromosome.length*Chromosome.num_genes - 1)])

        # join genes into single string for ease of manipulation
        child1_genes_str = "".join(chromosome1.genes)
        child2_genes_str = "".join(chromosome2.genes)

        # perform crossover
        child1_genes = child1_genes_str[:position1] + child2_genes_str[position1:position2] + child1_genes_str[position2:]
        child2_genes = child2_genes_str[:position1] + child1_genes_str[position1:position2] + child2_genes_str[position2:]

        # split genes from string into list
        child1_genes = [child1_genes[i:i + Chromosome.length] for i in range(0, Chromosome.num_genes * Chromosome.length, Chromosome.length)]
        child2_genes = [child2_genes[i:i + Chromosome.length] for i in range(0, Chromosome.num_genes * Chromosome.length, Chromosome.length)]

        child1, child2 = Chromosome(child1_genes), Chromosome(child2_genes)
        split_positions = sorted([randint(0, Chromosome.length - 1), randint(0, Chromosome.length - 1)])

        child1.ephemeral_random_constants = chromosome1.ephemeral_random_constants[:split_positions[0]] + \
                                            chromosome2.ephemeral_random_constants[split_positions[0]:split_positions[1]] + \
                                            chromosome1.ephemeral_random_constants[split_positions[1]:]
        child2.ephemeral_random_constants = chromosome2.ephemeral_random_constants[:split_positions[0]] + \
                                            chromosome1.ephemeral_random_constants[split_positions[0]:split_positions[1]] + \
                                            chromosome2.ephemeral_random_constants[split_positions[1]:]
        return child1, child2


    @staticmethod
    def gene_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> (Chromosome, Chromosome):
        """
        Two point recombination that occurs along gene boundaries for multigenic chromosomes.

        :param chromosome1: parent 1
        :param chromosome2: parent 2
        :return: offspring 1, offspring 2
        """

        # choose gene to swap
        gene = randint(0, Chromosome.num_genes - 1)

        # initialize children genes
        child1_genes = chromosome1.genes
        child2_genes = chromosome2.genes

        # perform swap
        child1_genes[gene] = chromosome2.genes[gene]
        child2_genes[gene] = chromosome1.genes[gene]

        return Chromosome(child1_genes), Chromosome(child2_genes)


    @staticmethod
    def plot_reps(avg_fitnesses: list, best_fitnesses: list, random_search_avg: list = None, random_search_best: list = None) -> None:
        """
        Plot all reps with global best solutions.

        :param avg_fitnesses: list of lists containing average fitnesses by generation for each rep
        :param best_fitnesses: same as avg_fitnesses but best fitnesses
        :param random_search_avg: average fitness value for random search across generations
        :param random_search_best: best fitness value for random search by generation
        :return: void
        """

        is_random_search = not (random_search_avg is None or random_search_best is None)

        plt.subplots(1, 2, figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.title("Average Fitness by Generation")
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")

        # plot each rep
        for rep in range(GeneExpressionProgram.NUM_RUNS):
            plt.plot(range(len(avg_fitnesses[rep])), avg_fitnesses[rep], label="Rep %d Average" % (rep + 1))

        if is_random_search:
            plt.plot(range(len(random_search_avg)), random_search_avg, label="Random Search Average")
            #plt.plot(range(len(random_search_best)), random_search_best, label="Random Search Best")

        plt.legend(loc="upper left")

        plt.subplot(1, 2, 2)
        plt.title("Best Fitness by Generation")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")

        # plot each rep
        for rep in range(GeneExpressionProgram.NUM_RUNS):
            plt.plot(range(len(best_fitnesses[rep])), best_fitnesses[rep], label="Rep %d Best" % (rep + 1))

        if is_random_search:
            plt.plot(range(len(random_search_best)), random_search_best, label="Random Search Best")

        plt.legend(loc="upper left")
        plt.show()
