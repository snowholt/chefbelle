import pygame
import numpy as np
import math

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
BALL_COLOR = (255, 100, 100)
PENTAGON_COLOR = (100, 255, 200)

# Ball properties
ball_pos = np.array([WIDTH / 2, HEIGHT / 2])
ball_vel = np.array([3.0, -4.0])
ball_radius = 10

# Pentagon properties
center = np.array([WIDTH / 2, HEIGHT / 2])
radius = 250
sides = 5
angle = 0
rotation_speed = 0.01  # radians per frame

def get_pentagon_points(center, radius, angle_offset):
    return [
        (
            center[0] + radius * math.cos(2 * math.pi * i / sides + angle_offset),
            center[1] + radius * math.sin(2 * math.pi * i / sides + angle_offset)
        )
        for i in range(sides)
    ]

def reflect_vector(v, normal):
    normal = normal / np.linalg.norm(normal)
    return v - 2 * np.dot(v, normal) * normal

def check_collision_and_reflect(pos, vel, polygon):
    next_pos = pos + vel
    for i in range(len(polygon)):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[(i + 1) % len(polygon)])
        edge = p2 - p1
        edge_normal = np.array([-edge[1], edge[0]])
        # Check if crossing the edge
        to_p1 = p1 - pos
        to_p2 = p2 - pos
        def scalar_cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        if scalar_cross(to_p1, vel) * scalar_cross(to_p2, vel) < 0:

            # Reflect and boost speed
            new_vel = reflect_vector(vel, edge_normal)
            new_vel *= 1.1  # speed boost
            return new_vel
    return vel

# Main loop
running = True
while running:
    screen.fill(BLACK)
    angle += rotation_speed

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get pentagon points
    pentagon = get_pentagon_points(center, radius, angle)

    # Check collision and update velocity
    ball_vel = check_collision_and_reflect(ball_pos, ball_vel, pentagon)

    # Update ball position
    ball_pos += ball_vel

    # Draw pentagon
    pygame.draw.polygon(screen, PENTAGON_COLOR, pentagon, 3)

    # Draw ball
    pygame.draw.circle(screen, BALL_COLOR, tuple(ball_pos.astype(int)), ball_radius)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
