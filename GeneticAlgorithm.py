# If Requirement already satisfied error then create virtual environment because it may be conflicting with global environment of python
# For creating virtual environment
# command1- python -m venv env
# command2-.\env\Scripts\activate
# Installing packge numpy command - pip install numpy
# For verifying installed package command- pip list
# Run program- python GeneticAlgorithm.py  


import numpy as np  #powerful library for handling arrays and mathematical operations in Python.
import random  #generate random numbers for mutation, crossover points, and random decisions within the algorithm.

# Step 1: Initialize Population
def generate_population(size, chromosome_length):
    population = np.random.randint(2, size=(size, chromosome_length))
    print(f"Initial Population:\n{population}")
    return population

# Step 2: Evaluate Fitness
def evaluate_fitness(population):
    fitness = np.sum(population, axis=1)
    print(f"Fitness of Population: {fitness}")
    return fitness

# Step 3: Selection (Roulette Wheel)
def select_parents(population, fitness):
    fitness_sum = np.sum(fitness)
    selection_prob = fitness / fitness_sum

    # randomly select element based on their selection probability,
    #len no, of rows the function will return
    # size - no. of individual you want to selct
    #p- probabiltiy of seleciton
    selected_indices = np.random.choice(len(population), size=len(population), p=selection_prob) 
    print(f"Selected Parents (indices): {selected_indices}")
    print(f"Selected Parents (chromosomes):\n{population[selected_indices]}")
    return population[selected_indices]

# Step 4: Crossover
def crossover(parent1, parent2, crossover_rate=0.7):
    #if random no, is less then rate , the crossover will happen; if not, it won't. This makes crossover a random event, just like in nature, where sometimes parents mix their traits and sometimes they don’t.
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        print(f"Crossover point: {point}")
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        print(f"Offspring 1: {child1}")
        print(f"Offspring 2: {child2}")
        return child1, child2
    print("No crossover occurred")
    return parent1, parent2

# Step 5: Mutation
def mutate(chromosome, mutation_rate=0.01):
    print(f"Original Chromosome Before Mutation: {chromosome}")
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 if chromosome[i] == 0 else 0
            print(f"Mutation occurred at position {i}")
    print(f"Chromosome After Mutation: {chromosome}")
    return chromosome

# Main GA Loop
def genetic_algorithm(pop_size, chromosome_length, generations):
    population = generate_population(pop_size, chromosome_length)

    for generation in range(generations):
        print(f"\n--- Generation {generation} ---")

        # Step 2: Fitness Evaluation
        fitness = evaluate_fitness(population)

        # Step 3: Selection
        parents = select_parents(population, fitness)

        # Step 4: Crossover
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            print(f"\nParent 1: {parent1}\nParent 2: {parent2}")
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        
        population = np.array(next_generation)

        # Display the fittest chromosome in the current generation
        fittest_index = np.argmax(fitness)
        print(f"Generation {generation}: Best Fitness = {fitness[fittest_index]}, Best Chromosome = {population[fittest_index]}")

# Running the Genetic Algorithm
genetic_algorithm(pop_size=8, chromosome_length=8, generations=3)