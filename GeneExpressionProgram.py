import numpy as np
from random import random, randint, shuffle

from Chromosome import Chromosome

class GeneExpressionProgram:

    ### Hyperparameters ###
    NUM_RUNS = 100
    NUM_GENERATIONS = 50
    POPULATION_SIZE = 30
    NUM_FITNESS_CASES = 10
    ERROR_TOLERANCE = 0.01

    ### Reproduction ###
    MUTATION_RATE = 0.051
    ONE_POINT_CROSSOVER_RATE, TWO_POINT_CROSSOVER_RATE, GENE_CROSSOVER_RATE = 0.2, 0.5, 0.1
    IS_TRANSPOSITION_RATE, IS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    RIS_TRANSPOSITION_RATE, RIS_ELEMENTS_LENGTH = 0.1, [1, 2, 3]
    GENE_TRANSPOSITION_RATE = 0.1

    ### Selection ###
    SELECTION_RANGE = 100

    ### Fitness Evaluation ###
    OBJECTIVE_FUNCTION = staticmethod(lambda a: a ** 4 + a ** 3 + a ** 2 + a)
    OBJECTIVE_MIN, OBJECTIVE_MAX = 0, 20
    FUNCTION_Y_RANGE = None


    def __init__(self):
        pass


    @staticmethod
    def evolve() -> Chromosome:

        if GeneExpressionProgram.FUNCTION_Y_RANGE is None:
            raise ValueError("Class variable FUNCTION_Y_RANGE must be set to calculate fitness.")

        Chromosome.max_fitness = GeneExpressionProgram.FUNCTION_Y_RANGE * GeneExpressionProgram.NUM_FITNESS_CASES

        # create initial population
        population = [Chromosome.generate_random_individual() for _ in range(GeneExpressionProgram.POPULATION_SIZE)]

        generation = 0
        best_fit_individual = None
        while generation < GeneExpressionProgram.NUM_GENERATIONS:

            ### EVALUATION ###

            # calcluate fitnesses for population
            population_fitnesses = Chromosome.absolute_fitness(GeneExpressionProgram.FUNCTION_Y_RANGE, *population)

            # find best fit individual
            # noinspection PyTypeChecker
            best_fit_generation = population[np.argmax(population_fitnesses)]
            if best_fit_individual is None or best_fit_individual.fitness() < best_fit_generation.fitness():
                best_fit_individual = best_fit_generation

            # skip rest of loop if we have found optimal solution
            if best_fit_individual.fitness() >= Chromosome.max_fitness:
                break

            next_generation = list()


            ### SELECTION (roulette wheel with simple elitism) ###

            # copy best individual to next generation
            next_generation.append(best_fit_individual)

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

            print("Generation: %d\tBest Fitness: %.5f" % (generation, best_fit_individual.fitness()))

        return best_fit_individual


    @staticmethod
    def roulette_wheel_selection(chromosomes: list, n: int) -> list:
        """
        Returns n randomly selected chromosomes using roulette wheel selection.
        Adapted from:
        https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights

        :param chromosomes: list of chromosomes to select from
        :param n: number of samples to retrieve
        :return:
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

        return Chromosome(new_genes)


    @staticmethod
    def is_transposition(chromosome: Chromosome) -> Chromosome:

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
            new_chromosome.genes[target_gene] = new_chromosome.genes[target_gene][:target_position] + \
                                                transposition_string + \
                                                new_chromosome.genes[target_gene][target_position + length:]

            return new_chromosome

        else:

            return chromosome


    @staticmethod
    def ris_transposition(chromosome: Chromosome) -> Chromosome:

        start_point = randint(0, Chromosome.head_length - 1)
        gene = randint(0, Chromosome.num_genes - 1)
        while start_point < Chromosome.head_length and chromosome.genes[gene][start_point] not in Chromosome.functions:
            start_point += 1

        if random() < GeneExpressionProgram.RIS_TRANSPOSITION_RATE and chromosome.genes[gene][start_point] in Chromosome.functions:
            ris_length = np.random.choice(GeneExpressionProgram.RIS_ELEMENTS_LENGTH)
            ris_string = chromosome.genes[gene][start_point:start_point+ris_length]

            new_chromosome = Chromosome(chromosome.genes)
            old_head = new_chromosome.genes[gene][:Chromosome.head_length]
            new_head = old_head[:start_point] + ris_string + old_head[start_point:]
            new_chromosome.genes[gene] = new_head[:Chromosome.head_length] + new_chromosome.genes[gene][Chromosome.head_length:]

            return new_chromosome

        else:

            return chromosome


    @staticmethod
    def gene_transposition(chromosome: Chromosome) -> Chromosome:
        if Chromosome.num_genes > 1 and random() < GeneExpressionProgram.GENE_TRANSPOSITION_RATE:

            index = randint(0, Chromosome.num_genes - 1)
            temp = chromosome.genes[index]
            chromosome.genes[index] = chromosome.genes[0]
            chromosome.genes[0] = temp
            return Chromosome(chromosome.genes)

        else:

            return chromosome


    @staticmethod
    def one_point_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> (Chromosome, Chromosome):
        gene = randint(0, Chromosome.num_genes - 1)
        position = randint(0, Chromosome.length)

        child1_split_gene = chromosome1.genes[gene][:position] + chromosome2.genes[gene][position:]
        child2_split_gene = chromosome2.genes[gene][:position] + chromosome1.genes[gene][position:]

        child1_genes = chromosome1.genes[:gene] + [child1_split_gene] + (chromosome2.genes[gene+1:] if gene < Chromosome.num_genes - 1 else [])
        child2_genes = chromosome2.genes[:gene] + [child2_split_gene] + (chromosome1.genes[gene + 1:] if gene < Chromosome.num_genes - 1 else [])

        return Chromosome(child1_genes), Chromosome(child2_genes)


    @staticmethod
    def two_point_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> (Chromosome, Chromosome):

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

        return Chromosome(child1_genes), Chromosome(child2_genes)


    @staticmethod
    def gene_recombination(chromosome1: Chromosome, chromosome2: Chromosome) -> (Chromosome, Chromosome):

        # choose gene to swap
        gene = randint(0, Chromosome.num_genes - 1)

        # initialize children genes
        child1_genes = chromosome1.genes
        child2_genes = chromosome2.genes

        # perform swap
        child1_genes[gene] = chromosome2.genes[gene]
        child2_genes[gene] = chromosome1.genes[gene]

        return Chromosome(child1_genes), Chromosome(child2_genes)
