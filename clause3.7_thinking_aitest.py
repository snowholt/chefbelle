import pygame
import sys
import math

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAVITY = 0.2
FRICTION = 0.99
SPEED_INCREASE = 1.1  # Factor by which speed increases after bounce

# Create the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball in Spinning Pentagon")
clock = pygame.time.Clock()


class Pentagon:
    def __init__(self, center_x, center_y, radius, rotation_speed):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.rotation = 0
        self.rotation_speed = rotation_speed
        self.vertices = self._calculate_vertices()
        self.edges = self._calculate_edges()

    def _calculate_vertices(self):
        vertices = []
        for i in range(5):
            angle = self.rotation + (2 * math.pi * i / 5)
            x = self.center_x + self.radius * math.cos(angle)
            y = self.center_y + self.radius * math.sin(angle)
            vertices.append((x, y))
        return vertices

    def _calculate_edges(self):
        edges = []
        for i in range(5):
            edges.append((self.vertices[i], self.vertices[(i + 1) % 5]))
        return edges

    def update(self):
        self.rotation += self.rotation_speed
        self.vertices = self._calculate_vertices()
        self.edges = self._calculate_edges()

    def draw(self, surface):
        pygame.draw.polygon(surface, BLUE, self.vertices, 2)


def distance_point_to_segment(p, s1, s2):
    """Calculate the distance from point p to line segment (s1, s2)
    Returns (distance, closest_point)"""
    # Vector from s1 to s2
    v = (s2[0] - s1[0], s2[1] - s1[1])
    # Vector from s1 to p
    w = (p[0] - s1[0], p[1] - s1[1])
    
    # Length of segment squared
    c1 = v[0] * w[0] + v[1] * w[1]
    if c1 <= 0:
        # Point is before the segment start
        return math.sqrt((p[0] - s1[0])**2 + (p[1] - s1[1])**2), s1
    
    c2 = v[0]**2 + v[1]**2
    if c2 <= c1:
        # Point is after the segment end
        return math.sqrt((p[0] - s2[0])**2 + (p[1] - s2[1])**2), s2
    
    # Point projects onto the segment
    b = c1 / c2
    pb = (s1[0] + b * v[0], s1[1] + b * v[1])
    return math.sqrt((p[0] - pb[0])**2 + (p[1] - pb[1])**2), pb


class Ball:
    def __init__(self, x, y, radius, speed_x, speed_y):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.bounced = False

    def update(self):
        self.speed_y += GRAVITY
        self.x += self.speed_x
        self.y += self.speed_y
        self.bounced = False

    def draw(self, surface):
        pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, pentagon):
        for edge in pentagon.edges:
            start, end = edge
            distance, closest_point = distance_point_to_segment((self.x, self.y), start, end)
            
            if distance <= self.radius:
                # Collision detected!
                edge_vector = (end[0] - start[0], end[1] - start[1])
                self.handle_bounce(closest_point, edge_vector)
                return True
        
        return False

    def handle_bounce(self, collision_point, edge_vector):
        # Calculate normal vector (perpendicular to edge)
        edge_length = math.sqrt(edge_vector[0]**2 + edge_vector[1]**2)
        edge_normalized = (edge_vector[0] / edge_length, edge_vector[1] / edge_length)
        normal = (-edge_normalized[1], edge_normalized[0])
        
        # Ensure normal points toward the ball
        dot = (self.x - collision_point[0]) * normal[0] + (self.y - collision_point[1]) * normal[1]
        if dot < 0:
            normal = (-normal[0], -normal[1])
            
        # Calculate reflection
        dot_product = self.speed_x * normal[0] + self.speed_y * normal[1]
        
        # Reflect velocity vector
        self.speed_x = self.speed_x - 2 * dot_product * normal[0]
        self.speed_y = self.speed_y - 2 * dot_product * normal[1]
        
        # Increase speed after bounce
        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        if speed > 0:  # Avoid division by zero
            self.speed_x *= SPEED_INCREASE
            self.speed_y *= SPEED_INCREASE
        
        # Apply friction
        self.speed_x *= FRICTION
        self.speed_y *= FRICTION
        
        # Move ball slightly away from edge to prevent sticking
        self.x += normal[0] * 2
        self.y += normal[1] * 2
        
        self.bounced = True


def main():
    # Create pentagon and ball
    pentagon = Pentagon(WIDTH // 2, HEIGHT // 2, 200, 0.005)
    ball = Ball(WIDTH // 2, HEIGHT // 2 - 100, 15, 2, 0)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update
        pentagon.update()
        ball.update()
        
        # Check collision
        ball.check_collision(pentagon)
        
        # Clear the screen
        screen.fill(BLACK)
        
        # Draw
        pentagon.draw(screen)
        ball.draw(screen)
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

