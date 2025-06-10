#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:30:19 2025

@author: lucahaines
"""

import pygame
import random
import sys
import numpy as np

pygame.init()

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

class SnakeEnvironment:
    def __init__(self, headless = False):
        self.headless = headless
        if not headless:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Times New Roman', 16)
        self.reset()
    
    def reset(self):
        center_x, center_y = width / 2, height / 2
        self.snake = [(center_x, center_y)]
        self.direction = random.choice([up, down, left, right])
        self.next_direction = self.direction
        self.fruit = self.place_fruit()
        self.score = 0
        self.game_over = False
        self.steps = 0
    
    def place_fruit(self):
        attempts = 0
        while attempts < 100:
            position = (random.randint(0, width - 1), 
                        random.randint(0, width - 1))
            if position not in self.snake:
                return position
            attempts += 1
        return (random.randint(0, width - 1), random.randint(0, height - 1))
    
    def distance_to_obstacle(self, head, direction):
        hx, hy = head
        dx, dy = direction
        
        # avoid errors
        if dx == 0 and dy == 0:
            return 1
            
        distance = 0
        current_x, current_y = hx, hy
        
        for step in range(1, max(width, height)):
            current_x += dx
            current_y += dy
            distance += 1
            
            # Distance to walls
            if current_x < 0 or current_x >= width or current_y < 0 \
                or current_y >= height:
                return distance
            
            # Distance to body
            if (current_x, current_y) in self.snake[1:]:
                return distance
        
        return distance
    
    def distance_to_fruit(self, head, direction):
        # Not actually distance but directional alignment
        hx, hy = head
        fx, fy = self.fruit
        dx, dy = direction
        
        if dx == 0 and dy == 0:
            return 0.0
        
        # vector to fruit
        vec_x = fx - hx
        vec_y = fy - hy
        
        # Normalize direction vector
        mag = np.sqrt(dx**2 + dy**2)
        if mag == 0:
            return 0.0
        unit_dx = dx / mag
        unit_dy = dy / mag
        
        projection = vec_x * unit_dx + vec_y * unit_dy
        if projection < 0:
            return 0.0
        
        # Normalize by maximum possible distance
        max_dist = np.sqrt(width**2 + height**2)
        return projection / max_dist
    
    def get_tail_direction(self):
        """Get tail direction as one-hot vector"""
        if len(self.snake) < 2:
            return [0, 0, 0, 0]
        
        tail = self.snake[-1]
        prev_segment = self.snake[-2]
        dx = prev_segment[0] - tail[0]
        dy = prev_segment[1] - tail[1]
        
        # Normalize to cardinal directions
        if dx != 0 and dy != 0:
            if abs(dx) > abs(dy):
                dy = 0
            else:
                dx = 0
        
        # Map to direction constants
        if dx == 1:
            return [0, 0, 0, 1]  # right
        elif dx == -1:
            return [0, 0, 1, 0]  # left
        elif dy == 1:
            return [0, 1, 0, 0]  # down
        elif dy == -1:
            return [1, 0, 0, 0]  # up
        else:
            return [0, 0, 0, 0]
    
    def get_neural_network_inputs(self):
        head = self.snake[0]
        
        # Obstacle distances in 8 directions
        obstacle_distances = []
        max_distance = max(width, height)
        for d in directions:
            dist = self.distance_to_obstacle(head, d)
            normalized_dist = dist / max_distance
            obstacle_distances.append(normalized_dist)
        
        # Fruit distances in 8 directions
        fruit_distances = [self.distance_to_fruit(head, d) for d in directions]
        
        # Current head direction as one-hot
        head_direction = [
            1 if self.direction == up else 0,
            1 if self.direction == down else 0,
            1 if self.direction == left else 0,
            1 if self.direction == right else 0
        ]
        
        # Tail direction as one-hot
        tail_direction = self.get_tail_direction()
    
        return np.array(obstacle_distances + fruit_distances + \
                        head_direction + tail_direction, dtype=np.float32)
    
    def move(self):
        if self.game_over:
            return
            
        self.direction = self.next_direction
        dx, dy = self.direction
        
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= width or 
            new_head[1] < 0 or new_head[1] >= height or
            new_head in self.snake):
            self.game_over = True
            return
            
        self.snake.insert(0, new_head)
        
        # Check fruit collision
        if new_head == self.fruit:
            self.score += 1
            self.fruit = self.place_fruit()
        else:
            self.snake.pop()
            
        self.steps += 1
    
    def draw(self):
        if self.headless:
            return
            
        self.screen.fill(black)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = green
            pygame.draw.rect(self.screen, color, 
                            (x * cell_dims, y * cell_dims, cell_dims, cell_dims))
            pygame.draw.rect(self.screen, green, 
                            (x * cell_dims, y * cell_dims, cell_dims, cell_dims), 1)
        
        # Draw fruit
        pygame.draw.rect(self.screen, red, 
                        (self.fruit[0] * cell_dims, self.fruit[1] * cell_dims, 
                         cell_dims, cell_dims))
        
        # Draw score and info
        score_text = self.font.render(f'Score: {self.score}', True, white)
        self.screen.blit(score_text, (5, 5))
        
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_UP and self.direction != down:
                    self.next_direction = up
                if event.key == pygame.K_DOWN and self.direction != up:
                    self.next_direction = down
                if event.key == pygame.K_LEFT and self.direction != right:
                    self.next_direction = left
                if event.key == pygame.K_RIGHT and self.direction != left:
                    self.next_direction = right
                    
                    
def play_manually():
    print("Playing manually. Use arrow keys to control the snake.")
    print("Press 'R' to restart, 'ESC' to quit.")
    
    pygame.init()
    pygame.font.init()
    
    try:
        game = SnakeEnvironment(headless=False)
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                    elif event.key == pygame.K_UP and game.direction != down:
                        game.next_direction = up
                    elif event.key == pygame.K_DOWN and game.direction != up:
                        game.next_direction = down
                    elif event.key == pygame.K_LEFT and game.direction != right:
                        game.next_direction = left
                    elif event.key == pygame.K_RIGHT and game.direction != left:
                        game.next_direction = right
            
            if not game.game_over:
                game.move()
            
            game.draw()
            clock.tick(fps)
            
    except Exception as e:
        print(e)
    finally:
        pygame.quit()
                    
