import numpy as np
import random
import itertools

# Step 1: Initialize Population
def generate_population(pop_size, num_cities):
    # Each individual in the population is a random permutation of city indices (0 to num_cities-1)
    population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]
    print(f"Initial Population:\n{population}")
    return population

# Step 2: Calculate Total Distance (Fitness Function)
def calculate_total_distance(route, distance_matrix):
    # Sum up the distances between consecutive cities in the route and return to the starting city
    total_distance = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    total_distance += distance_matrix[route[-1]][route[0]]  # Returning to the start city
    print(f"Route: {route}, Total Distance: {total_distance}")
    return total_distance

# Step 3: Evaluate Fitness for Each Individual
def evaluate_population(population, distance_matrix):
    fitness_scores = []
    for individual in population:
        total_distance = calculate_total_distance(individual, distance_matrix)
        fitness_scores.append(1 / total_distance)  # We use 1/total_distance because smaller distance is better (minimization problem)
    print(f"Fitness Scores of Population: {fitness_scores}")
    return fitness_scores

# Step 4: Selection (Roulette Wheel Selection)
def select_parents(population, fitness_scores):
    fitness_sum = sum(fitness_scores)
    selection_prob = [f / fitness_sum for f in fitness_scores]  # Normalized probabilities
    selected_indices = np.random.choice(len(population), size=len(population), p=selection_prob)  # Select parents based on their fitness probability
    selected_parents = [population[i] for i in selected_indices]
    print(f"Selected Parents (indices): {selected_indices}")
    print(f"Selected Parents (routes):\n{selected_parents}")
    return selected_parents

# Step 5: Crossover (Ordered Crossover)
def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        # Choose two crossover points randomly
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        print(f"Crossover points: {point1}, {point2}")
        # Child inherits the segment from parent1 between the two points
        child1 = [None]*len(parent1)
        child1[point1:point2] = parent1[point1:point2]
        # Fill the remaining cities from parent2 in the order they appear
        fill_values = [city for city in parent2 if city not in child1]
        child1 = [fill_values.pop(0) if city is None else city for city in child1]
        
        # The same for child2 but switch parent roles
        child2 = [None]*len(parent2)
        child2[point1:point2] = parent2[point1:point2]
        fill_values = [city for city in parent1 if city not in child2]
        child2 = [fill_values.pop(0) if city is None else city for city in child2]
        
        print(f"Offspring 1: {child1}")
        print(f"Offspring 2: {child2}")
        return child1, child2
    else:
        print("No crossover occurred")
        return parent1, parent2

# Step 6: Mutation (Swap Mutation)
def mutate(route, mutation_rate=0.01):
    print(f"Original Route Before Mutation: {route}")
    for i in range(len(route)):
        if random.random() < mutation_rate:
            # Swap two cities randomly
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
            print(f"Mutation occurred between position {i} and {j}")
    print(f"Route After Mutation: {route}")
    return route

# Step 7: Genetic Algorithm Main Loop
def genetic_algorithm_tsp(pop_size, num_cities, generations, mutation_rate):
    # Initialize a random distance matrix (symmetric matrix with random distances)
    distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)  # Distance from a city to itself is 0
    print(f"Distance Matrix:\n{distance_matrix}")

    # Step 1: Generate initial population
    population = generate_population(pop_size, num_cities)

    for generation in range(generations):
        print(f"\n--- Generation {generation} ---")

        # Step 2: Evaluate fitness of the population
        fitness_scores = evaluate_population(population, distance_matrix)

        # Step 3: Selection
        parents = select_parents(population, fitness_scores)

        # Step 4: Crossover and Mutation to create the next generation
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            print(f"\nParent 1: {parent1}\nParent 2: {parent2}")
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate))
            next_generation.append(mutate(child2, mutation_rate))
        
        population = next_generation

        # Display the best route in the current generation
        fitness_scores = evaluate_population(population, distance_matrix)
        best_fitness_index = np.argmax(fitness_scores)
        best_route = population[best_fitness_index]
        best_distance = 1 / fitness_scores[best_fitness_index]
        print(f"Generation {generation}: Best Route = {best_route}, Best Distance = {best_distance}")

# Run the TSP Genetic Algorithm
genetic_algorithm_tsp(pop_size=6, num_cities=5, generations=10, mutation_rate=0.05)
