import pygame
import sys
import math
import numpy as np
from pygame.locals import QUIT

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball in Spinning Pentagon")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Pentagon properties
center = (WIDTH // 2, HEIGHT // 2)
radius = 200
num_sides = 5
rotation_speed = 0.5  # degrees per frame

# Ball properties
ball_radius = 15
ball_pos = np.array([center[0] + radius // 2, center[1]])
ball_vel = np.array([2.0, 3.0])
speed_increase_factor = 1.05
gravity = np.array([0, 0.1])
damping = 0.99  # Energy loss

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

def get_pentagon_vertices(angle_offset):
    """Calculate the vertices of the pentagon based on current rotation."""
    vertices = []
    for i in range(num_sides):
        angle = math.radians(angle_offset + i * 360 / num_sides)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return vertices

def get_pentagon_edges(vertices):
    """Get the edges of the pentagon as line segments."""
    edges = []
    for i in range(num_sides):
        edges.append((vertices[i], vertices[(i + 1) % num_sides]))
    return edges

def distance_point_to_line(point, line):
    """Calculate the perpendicular distance from a point to a line segment."""
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    
    # Line segment length squared
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0:  # Line segment is actually a point
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # Calculate projection proportion
    t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2
    t = max(0, min(1, t))  # Clamp to [0, 1]
    
    # Projection point
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    # Distance
    return math.sqrt((x0 - projection_x)**2 + (y0 - projection_y)**2)

def get_line_normal(line):
    """Get unit normal vector for a line segment."""
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx**2 + dy**2)
    if length == 0:
        return np.array([0, 0])
    # Rotate 90 degrees to get normal
    return np.array([-dy / length, dx / length])

def is_point_inside_pentagon(point, vertices):
    """Check if a point is inside the pentagon using ray casting algorithm."""
    x, y = point
    inside = False
    j = len(vertices) - 1
    
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
        j = i
        
    return inside

def main():
    # Initialize these variables in the main function to fix the "possibly unbound" errors
    global ball_pos, ball_vel
    angle_offset = 0
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        screen.fill(BLACK)
        
        # Update pentagon rotation
        angle_offset += rotation_speed
        vertices = get_pentagon_vertices(angle_offset)
        edges = get_pentagon_edges(vertices)
        
        # Draw pentagon
        pygame.draw.polygon(screen, WHITE, vertices, 2)
        
        # Update ball position
        ball_vel += gravity
        ball_vel *= damping
        new_ball_pos = ball_pos + ball_vel
        
        # Check for collision with pentagon edges
        collision_occurred = False
        closest_edge = None
        min_distance = float('inf')
        
        # Find the closest edge for more accurate collision detection
        for edge in edges:
            distance = distance_point_to_line(new_ball_pos, edge)
            if distance < min_distance:
                min_distance = distance
                closest_edge = edge
                
        # Check if collision with the closest edge
        if min_distance < ball_radius and closest_edge is not None:
            # Get normal vector of the edge
            normal = get_line_normal(closest_edge)
            
            # Reflect velocity vector
            dot_product = np.dot(ball_vel, normal)
            if dot_product < 0:  # Only reflect if ball is moving towards the edge
                ball_vel = ball_vel - 2 * dot_product * normal
                # Increase speed after bounce
                ball_vel *= speed_increase_factor
                # Adjust position to prevent sticking to the edge
                overlap = ball_radius - min_distance
                ball_pos = ball_pos + overlap * normal
                collision_occurred = True
                
        # Move the ball if it would stay inside the pentagon or has handled collision
        if is_point_inside_pentagon(new_ball_pos, vertices) or collision_occurred:
            if not collision_occurred:  # Only update position if no collision was handled
                ball_pos = new_ball_pos
        else:
            # If ball would exit the pentagon, find proper bounce direction
            # First, find the edge we're trying to cross
            for edge in edges:
                normal = get_line_normal(edge)
                
                # Check if this is the edge we're crossing (simplified check)
                if np.dot(normal, ball_vel) < 0:
                    # Reflect velocity vector off this edge
                    dot_product = np.dot(ball_vel, normal)
                    ball_vel = ball_vel - 2 * dot_product * normal
                    # Increase speed after bounce
                    ball_vel *= speed_increase_factor
                    break
            
            # Keep the ball inside the pentagon
            if not is_point_inside_pentagon(ball_pos, vertices):
                # Move towards center as a last resort
                direction_to_center = np.array([center[0] - ball_pos[0], center[1] - ball_pos[1]])
                direction_length = np.linalg.norm(direction_to_center)
                if direction_length > 0:
                    direction_to_center = direction_to_center / direction_length
                    ball_pos = np.array([center[0] - radius/2 * direction_to_center[0], 
                                         center[1] - radius/2 * direction_to_center[1]])
        
        # Draw ball
        pygame.draw.circle(screen, RED, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)

if __name__ == "__main__":
    main()
