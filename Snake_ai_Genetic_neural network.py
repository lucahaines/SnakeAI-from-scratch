#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:26:30 2025

@author: lucahaines
"""
import pygame
import random
import numpy as np
import pickle
from snake import SnakeEnvironment

# Constants for game
width, height = 10, 10
cell_dims = 30
screen_width = width * cell_dims
screen_height = height * cell_dims
fps = 10

black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 180, 0)
red = (180, 0, 0)
grey = (40, 40, 40)
blue = (0, 120, 200)

# directions
up = (0, -1)
down = (0, 1)
left = (-1, 0)
right = (1, 0)

# 8-direction vectors
directions = [
    (0, -1),   # up
    (1, -1),   # up-right
    (1, 0),    # right
    (1, 1),    # down-right
    (0, 1),    # down
    (-1, 1),   # down-left
    (-1, 0),   # left
    (-1, -1)   # up-left
    ]


# Constants for genetic algorithm
mutation_rate=0.15
min_mutation_rate = 0.01
mutation_std = 0.25
elite_percent=0.2
decay_rate = 0.99
mutation_probability = 0.8

# Constants for training loop
hiddensize1 = 32
hiddensize2 = 16
outputsize = 4
populationsize = 400
generations = 250000



# file management functions
def save_to_file(network, filename):
    with open(filename, 'wb') as f:
        pickle.dump(network, f)

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
    
# softmax function for output layer
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def convert_action_to_direction(action, current_direction):
    
    """
    Inputs a requested action and direction the snake is moving, the 
    output is the requested direction so long as the requested direction does
    not make the snake turn in onto itself
    
    """
    action_map = {
        0: up,
        1: down,
        2: left,
        3: right
    }
    
    requested_direction = action_map[action]
    dx, dy = requested_direction
    cdx, cdy = current_direction
    
    if (dx, dy) == (-cdx, -cdy):
        return current_direction
    return requested_direction


# Initialize neural network that will be trained for. Note that the neural
# network must have two hidden layers with positive sizes 
# (non-zero hidden layers)

class NeuralNetwork:
    
    # TODO make the neural network have arbitrary topology (perhaps implement NEAT)
    
    def __init__(self, inputsize, hiddensize1, hiddensize2, outputsize):
        
        self.inputsize = inputsize
        self.hiddensize1 = hiddensize1
        self.hiddensize2 = hiddensize2
        self.outputsize = outputsize
        
        # initialization of weights with Kaiming initialization
        self.weights1 = np.random.randn(inputsize, hiddensize1) * \
            np.sqrt(2 / inputsize)
        self.weights2 = np.random.randn(hiddensize1, hiddensize2) * \
            np.sqrt(2 / hiddensize1)
        self.weights3 = np.random.randn(hiddensize2, outputsize) * \
            np.sqrt(2 / hiddensize2)
        
        # biases and fitneds
        self.biases1 = np.zeros(hiddensize1)
        self.biases2 = np.zeros(hiddensize2)
        self.biases3 = np.zeros(outputsize)
        self.fitness = 0
     
        
    def forwardprop(self, x):
        
        # TODO give option to change activation functions between sigmoid,
        # tanh, etc.
        
        """
        Forward propagation with ReLu activation function on hidden layers and
        softmax on output layer
        """
        
        hidden1 = np.maximum(np.dot(x, self.weights1) + self.biases1, 0)
        hidden2 = np.maximum(np.dot(hidden1, self.weights2) + self.biases2, 0)
        output = np.dot(hidden2, self.weights3) + self.biases3
        return softmax(output)
    
    
    def getinfo(self):
        
        # TODO would also need to change to change topology
        
        """
        transforms all weights and biases into a single vector
        """
        return np.concatenate([
            self.weights1.flatten(),
            self.biases1.flatten(),
            self.weights2.flatten(),
            self.biases2.flatten(),
            self.weights3.flatten(),
            self.biases3.flatten()
            ])
        
    def setweights(self, flatweights):
        
        # TODO ^^^
        
        """
        transforms a single vector of all weights and biases organized as 
        weight1, biases1, ... weightsn, biasesn into the correct shaped
        matrices to be used in forwardprop
        """
        
        sizes = [
            self.weights1.size, self.biases1.size,
            self.weights2.size, self.biases2.size,
            self.weights3.size, self.biases3.size
        ]
        sections = np.cumsum(sizes)[:-1]
        parts = np.split(flatweights, sections)
        
        self.weights1 = parts[0].reshape(self.weights1.shape)
        self.biases1 = parts[1]
        self.weights2 = parts[2].reshape(self.weights2.shape)
        self.biases2 = parts[3]
        self.weights3 = parts[4].reshape(self.weights3.shape)
        self.biases3 = parts[5]
        
        
        
class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden1_size, hidden2_size,
                 output_size):
        
        self.population_size = population_size
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        
        # population is a list of neural networks with the same topologies
        # TODO: change each population to have different random topologies
        
        self.population = [NeuralNetwork(input_size, hidden1_size, hidden2_size, 
                            output_size) for _ in range(population_size)]
        
    
    def calculate_fitness(self, score, steps):
        """
        Calculates fitness based on score and the number of steps it took
        to get there. The current fitness equation comes from Chrispresso
        on YouTube but I will look to find a better fitness function
        """

        exponential_reward = 2 ** score
        other_reward = (score ** 2.1) * 500
        penalty = (score ** 1.2) * ((0.25 * steps) ** 1.3)
        
        fitness = steps + exponential_reward + other_reward - penalty
        
        return max(0, fitness)
        
        
        
    def fitness(self, game_environment):
        """"
        Finds game score and steps then plugs info into a sep calculate fitness
        function and updates the fitness and score of the neural network also
        breaks if the snake takes too long to score
        """
        
        maxsteps = 2000
        headless = True
        
        for nn in self.population:
            game = SnakeEnvironment(headless = headless)
            steps = 0
            last_score = 0
            stagnant = 0
            
            while not game.game_over and steps < maxsteps:
                 inputs = game.get_neural_network_inputs()
                 outputs = nn.forwardprop(inputs)
                 action = np.argmax(outputs)
                 game.next_direction = convert_action_to_direction(action, 
                                                    game.direction)
                 game.move()
                 steps += 1
                 
                 if game.score > last_score:
                     last_score = game.score
                     stagnant = 0
                 else:
                     stagnant += 1
                 
                # stop if the snake takes too long to score
                 if stagnant > 100:
                     break
                     
            # Calculate fitness
            nn.fitness = self.calculate_fitness(game.score, steps)
            nn.last_score = game.score
            
        
    def sort_by_fitness(self):
        """"sort networks from fittest to least fit"""
        return sorted(self.population, key=lambda nn: nn.fitness, reverse=True)
    
    
    def selection(self):
        """
        Selects the top elite ratio percent of neural networks that can then
        be used to breed future generations of nns
        """
        
        sorted_nets = self.sort_by_fitness()
        elite_count = max(2, int(len(self.population) * elite_percent))
        return sorted_nets[:elite_count], elite_count
    
            
    def roulette_wheel_selection(self, tot_fitness):
        """
        Use roulette wheel selection to choose individual parents for the next
        generation that fills up the rest of the next generation that does
        not get filled by the top elite percent. Neural networks with higher
        fitnesses will have a greater chance of being selected as parents but
        other networks will also have a chance, this encourages exploration.
        """
        if tot_fitness <= 0:
            return random.choice(self.population)
        
        r = random.uniform(0, tot_fitness)
        current = 0
        for nn in self.population:
            current += nn.fitness
            if current >= r:
                return nn
        return random.choice(self.population)
    
    
    def crossover(self, parent1, parent2):
        
        # TODO make a more sophisticated breeding algorithm
        
        """
        Given two parent neural networks, this function creates a child that
        inherits weights and biases from both parents. It chooses 2 random
        indices in the weights-and-biases vector and all weights and biases in
        the child's vector before index2 will be inherited from parent1, 
        between indices 1 and 2 will be from parent2 and after index2 will be
        from parent1 again.
        """
        child = NeuralNetwork(self.input_size, self.hidden1_size, 
                              self.hidden2_size, self.output_size)
        weights1 = parent1.getinfo()
        weights2 = parent2.getinfo()
        
        # Two-point crossover
        size = len(weights1)
        index1, index2 = sorted(random.sample(range(size), 2))
        
        child_weights = np.concatenate([
            weights1[:index1],
            weights2[index1:index2],
            weights1[index2:]
        ])
                
        child.setweights(child_weights)
        return child
    
    
    def mutate(self, individual, generation):
        """"
        Random mutation that changes weights to have some other value in the
        gaussian distribution with mean 0 and std of mutation_std. This is
        done to ensure diverse models. The rate of mutation decreases with
        the more generations that are bred.
        """
        current_rate = max(min_mutation_rate, mutation_rate * \
                           (decay_rate ** generation))
        
        weights = individual.getinfo()
        for i in range(len(weights)):
            if np.random.random() < current_rate:
                weights[i] += np.random.normal(0, mutation_std)
        individual.setweights(weights)
        return individual
    
    
    def evolve(self, generation):
        """
        Evolves a new population composed of elite_percent % elite snakes and
        1 - elite_percent % crossover and mutated offspring. 
        Updates self.population accordingly
        """
        new_population = []
        total_fitness = sum(max(0, nn.fitness) for nn in self.population)
        
        # keep the best guys with no mutation
        elites, elite_count = self.selection()
        new_population.extend(elites[:elite_count])
        
        # breed offspring and mutate them MWAHAHAHAHA!!!
        while len(new_population) < self.population_size:
            parent1 = self.roulette_wheel_selection(total_fitness)
            parent2 = self.roulette_wheel_selection(total_fitness)
            
            child = self.crossover(parent1, parent2)
            
            if np.random.random() < mutation_probability:
                child = self.mutate(child, generation)
            
            new_population.append(child)
        
        self.population = new_population
    

# Main training loop

def train_snake_ai():
    """
    Trains an ai either starting from scratch or starting from mutated versions
    of an inputted ai
    """
    
    input_size = 24 
    
    # initialize genetic algorithm
    ga = GeneticAlgorithm(populationsize, input_size, hiddensize1, 
                          hiddensize2, outputsize)
    
    # Track best performer
    all_time_best = None
    best_fitness = 0
    
    try:
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}")
            print("-" * 30)
            
            # Evaluate fitness
            ga.fitness(SnakeEnvironment)
            
            # Find best performer
            sorted_pop = ga.sort_by_fitness()
            gen_best = sorted_pop[0]
            
            scores = [getattr(nn, 'last_score', 0) for nn in ga.population]
            avg_score = np.mean(scores)
            max_score = max(scores)
            avg_fitness = np.mean([nn.fitness for nn in ga.population])
            
            
            # Update all-time best
            if gen_best.fitness > best_fitness:
                ran_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
                filename = f"best_snake_ai_{ran_id}.pkl"
                best_fitness = gen_best.fitness
                all_time_best = gen_best
                print(f"NEW BEST! Fitness: {gen_best.fitness:.1f}, Score: {getattr(gen_best, 'last_score', 0)}")
                save_to_file(all_time_best, filename)
                print("Best AI saved to: ", filename)
            
            print(f"Best Score: {max_score}, Avg Score: {avg_score:.1f}")
            print(f"Best Fitness: {gen_best.fitness:.1f}, Avg Fitness: {avg_fitness:.1f}")
            
            ga.evolve(generation)
        
    except KeyboardInterrupt:
        print("\nYou ended it")
    except Exception as e:
        print(e)
    
    print("\nTraining done")
    
    if all_time_best:
        final_score = getattr(all_time_best, 'last_score', 0)
        print(f"Best AI achieved score: {final_score}")
        print(f"Best fitness: {best_fitness:.1f}")
        print("AI saved as: ", filename)
    else:
        print("No successful AI was trained.")
    
    return all_time_best
    

def visualize_ai_performance(ai):
    print("Visualizing AI performance...")
    pygame.init()
    pygame.font.init()
    
    network = load_from_file(ai)
    
    try:
        game = SnakeEnvironment(headless=False)
        clock = pygame.time.Clock()
        running = True
        
        print("AI is now playing! Press 'R' to restart, 'ESC' to quit.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                        print("Game restarted!")
            
            if not game.game_over:
                inputs = game.get_neural_network_inputs()
                outputs = network.forwardprop(inputs)
                action = np.argmax(outputs)
                game.next_direction = convert_action_to_direction(action, game.direction)
                game.move()
            
            game.draw()
            clock.tick(fps)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
    finally:
        pygame.quit()
    
    
    
    
