import pygame
import numpy as np
import math
import sys

# --- Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PENTAGON_COLOR = WHITE
BALL_COLOR = RED

# Pentagon properties
center = np.array([WIDTH / 2, HEIGHT / 2])
radius = 250
rotation_speed = 0.01  # Radians per frame
angle = 0

# Ball properties
ball_radius = 10
ball_pos = center.copy() # Start at the center
initial_speed = 3
# Give it a small initial random velocity
ball_vel = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
ball_vel = ball_vel / np.linalg.norm(ball_vel) * initial_speed if np.linalg.norm(ball_vel) > 0 else np.array([initial_speed, 0.0])

speed_boost_factor = 1.1 # Increase speed by 10% on bounce

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball in Spinning Pentagon")
clock = pygame.time.Clock()

# --- Helper Functions ---

def get_pentagon_points(center_pos, pent_radius, angle_offset):
    """Calculates the vertices of a regular pentagon."""
    points = []
    for i in range(5):
        angle_rad = angle_offset + math.pi * 2 * i / 5
        x = center_pos[0] + pent_radius * math.cos(angle_rad)
        y = center_pos[1] + pent_radius * math.sin(angle_rad)
        points.append(np.array([x, y]))
    return points

def reflect_vector(v, normal):
    """Reflects vector v across the normal vector."""
    normal = normal / np.linalg.norm(normal) # Ensure normal is a unit vector
    return v - 2 * np.dot(v, normal) * normal

def check_collision_and_reflect(pos, vel, polygon_points):
    """Checks for collision with polygon edges and reflects velocity."""
    next_pos = pos + vel
    for i in range(len(polygon_points)):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % len(polygon_points)]
        edge = p2 - p1
        # Calculate edge normal pointing inwards (assuming clockwise vertices)
        edge_normal = np.array([edge[1], -edge[0]])
        edge_normal = edge_normal / np.linalg.norm(edge_normal)

        # Vector from edge start point (p1) to ball position (pos)
        p1_to_pos = pos - p1
        # Vector from edge start point (p1) to next ball position (next_pos)
        p1_to_next_pos = next_pos - p1

        # Check if the ball is currently outside or on the edge
        dist_to_edge_plane = np.dot(p1_to_pos, edge_normal)

        # Check if the ball is moving towards the edge plane from outside
        # and if the next position crosses the edge line segment
        # A simple check: if the sign of the distance to the edge plane changes
        # This is a simplified check and might not be perfectly robust for fast objects/rotations
        # A more robust check involves line segment intersection tests.

        # Check if ball is outside or very close to the edge and moving towards it
        if dist_to_edge_plane < ball_radius and np.dot(vel, edge_normal) < 0:
             # Check if the projected point onto the edge line is within the segment
             edge_len_sq = np.dot(edge, edge)
             proj_param = np.dot(p1_to_pos, edge) / edge_len_sq
             if 0 <= proj_param <= 1:
                 # Reflect and boost speed
                 new_vel = reflect_vector(vel, edge_normal)
                 new_vel *= speed_boost_factor # speed boost
                 # Move ball slightly away from edge to prevent sticking
                 pos += edge_normal * (ball_radius - dist_to_edge_plane + 1)
                 return pos, new_vel # Return updated position and velocity

    # Check if ball is outside the bounds (failsafe, might need refinement)
    # This basic check prevents ball escaping completely but isn't physically accurate boundary interaction
    if np.linalg.norm(pos - center) > radius - ball_radius:
         normal_to_center = (center - pos) / np.linalg.norm(center - pos)
         if np.dot(vel, normal_to_center) < 0: # Moving away from center
             new_vel = reflect_vector(vel, normal_to_center)
             new_vel *= speed_boost_factor
             return pos, new_vel


    return pos, vel # No collision detected

# --- Main Game Loop ---
running = True
while running:
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # --- Updates ---
    # Rotate pentagon
    angle += rotation_speed

    # Get current pentagon points
    pentagon_vertices = get_pentagon_points(center, radius, angle)

    # Check collision and update velocity/position
    ball_pos, ball_vel = check_collision_and_reflect(ball_pos, ball_vel, pentagon_vertices)

    # Update ball position
    ball_pos += ball_vel

    # --- Drawing ---
    screen.fill(BLACK)

    # Draw pentagon
    pygame.draw.polygon(screen, PENTAGON_COLOR, [tuple(p.astype(int)) for p in pentagon_vertices], 3)

    # Draw ball
    pygame.draw.circle(screen, BALL_COLOR, tuple(ball_pos.astype(int)), ball_radius)

    # --- Display Update ---
    pygame.display.flip()

    # --- Frame Rate Control ---
    clock.tick(FPS)

# --- Cleanup ---
pygame.quit()
sys.exit()
