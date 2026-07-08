import sys
import random
import math
import pygame
import time
from pygame.math import Vector2
import numpy as np

from AstarPathPlanning import AStarPathPlanner

# ---------- Parameters ----------
WIDTH, HEIGHT = 1400, 700 #2400, 1600
NUM_BOIDS = 10
NUM_LEADERS = 1
MAX_SPEED = 1.3
MAX_FORCE = 0.05
L_MAX_SPEED = 0

SEPARATION_RADIUS = 40
ALIGNMENT_RADIUS = 80
COHESION_RADIUS = 200
LEADER_RADIUS = 300
LEADER_SHADOW = 20

SEPARATION_WEIGHT = 5.0
ALIGNMENT_WEIGHT = 1.5
COHESION_WEIGHT = 1
LEADER_WEIGHT = 1

BOID_SIZE = 8
BG_COLOR = (30, 30, 30)
BOID_COLOR = (200, 200, 220)
LEADER_COLOR = (255, 0, 0)
YELLOW = (255,255,0)
REWARD_RAD = 130
FPS = 60

# --------------------------------


# --------------------------------



class Leader:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        angle = random.uniform(0, math.tau)
        self.vel = Vector2(math.cos(angle), math.sin(angle)) * L_MAX_SPEED
        self.max_speed = L_MAX_SPEED
        self.Corners = [(200,200),(WIDTH-200,HEIGHT-200)] #box
        

    def edges(self):
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        #Triangle
        #angle = math.atan2(self.vel.y, self.vel.x)
        #p1 = Vector2(BOID_SIZE, 0).rotate_rad(angle) + self.pos
        #p2 = Vector2(-BOID_SIZE * 0.6, BOID_SIZE * 0.6).rotate_rad(angle) + self.pos
        #p3 = Vector2(-BOID_SIZE * 0.6, -BOID_SIZE * 0.6).rotate_rad(angle) + self.pos
        #pygame.draw.polygon(surface, LEADER_COLOR, [p1, p2, p3])
        pygame.draw.circle(surface, LEADER_COLOR, self.pos, 8, 0)  # Draw reward radius

    def update_pathplan(self):
        center = Vector2(WIDTH / 2, HEIGHT / 2)
        radius = 150
        angle = math.atan2(self.pos.y - center.y, self.pos.x - center.x)
        #angle += 0.01  # rotation speed
        #self.pos.x = center.x + radius * math.cos(angle)
        #self.pos.y = center.y + radius * math.sin(angle)
        #self.vel = Vector2(math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2)) * L_MAX_SPEED
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
    
    def follow_path(self, path, iteration):
        self.pos = path[iteration]
        print(path[iteration], end=" ")
        print(iteration)
        
           

class Boid:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        angle = random.uniform(0, math.tau)
        self.vel = Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1.0, MAX_SPEED)
        self.acc = Vector2(0, 0)
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE

    def edges(self):
        # Wrap-around
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

    def apply_force(self, force):
        self.acc += force

    def update(self):
        self.vel += self.acc
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        self.pos += self.vel
        self.acc = Vector2(0, 0)

    def seek(self, target):
        desired = (target - self.pos)
        if desired.length() == 0:
            return Vector2(0, 0)
        desired.scale_to_length(self.max_speed)
        steer = desired - self.vel
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    def behaviors(self, boids, lead_boids=[None],path=None, iteration=None):
        sep = self.separation(boids,lead_boids) * SEPARATION_WEIGHT
        ali = self.alignment(boids) * ALIGNMENT_WEIGHT
        coh = self.cohesion(boids) * COHESION_WEIGHT

        if NUM_LEADERS > 0:
            led = self.leader_force(lead_boids,path,iteration) * LEADER_WEIGHT
            self.apply_force(led)

        self.apply_force(sep)
        self.apply_force(ali)
        self.apply_force(coh)
        

    def separation(self, boids,lead_boid):
        steer = Vector2(0, 0)
        total = 0
        for other in boids:
            if other is self:
                continue
            d = self.pos.distance_to(other.pos)
            if d < SEPARATION_RADIUS and d > 0:
                diff = (self.pos - other.pos)
                if diff.length() > 0:
                    diff /= d  # weight by distance
                steer += diff
                total += 1
        for lead in lead_boid:
            d = self.pos.distance_to(lead.pos)
            if d < SEPARATION_RADIUS and d > 0:
                diff = (self.pos - lead.pos)
                if diff.length() > 0:
                    diff /= d  # weight by distance
                steer += diff
                total += 1
        if total > 0:
            steer /= total
            if steer.length() > 0.01: # why doesnt this work, should be 0
                steer.scale_to_length(self.max_speed)
                steer -= self.vel
                if steer.length() > self.max_force:
                    steer.scale_to_length(self.max_force)
        return steer

    def alignment(self, boids):
        avg_vel = Vector2(0, 0)
        total = 0
        for other in boids:
            if other is self:
                continue
            d = self.pos.distance_to(other.pos)
            if d < ALIGNMENT_RADIUS:
                avg_vel += other.vel
                total += 1
        if total > 0:
            avg_vel /= total
            if avg_vel.length() > 0.01:#should be 0
                avg_vel.scale_to_length(self.max_speed)#throws error cus cant scale vector w/ 0 length
                steer = avg_vel - self.vel
                if steer.length() > self.max_force:
                    steer.scale_to_length(self.max_force)
                return steer
        return Vector2(0, 0)

    def cohesion(self, boids):
        center = Vector2(0, 0)
        total = 0
        for other in boids:
            if other is self:
                continue
            d = self.pos.distance_to(other.pos)
            if d < COHESION_RADIUS:
                center += other.pos
                total += 1
        if total > 0:
            center /= total
            return self.seek(center)
        return Vector2(0, 0)
    
    def leader_force(self, lead_boid, path, iteration):
        steer = Vector2(0,0)
        total = 0
        if iteration != None and iteration > LEADER_SHADOW:
            for lother in lead_boid:
                d = self.pos.distance_to(path[iteration-LEADER_SHADOW])
                if d < LEADER_RADIUS and d > 0:
                    diff = (self.pos - path[iteration-LEADER_SHADOW])
                    total += 1
                    steer += diff
                else:
                    continue
        else:
            for lother in lead_boid:
                d = self.pos.distance_to(lother.pos)
                if d < LEADER_RADIUS and d > 0:
                    diff = (self.pos - lother.pos)
                    total += 1
                    steer += diff
                else:
                    continue
        if total > 0:
            steer /= total 
            if steer.length() > MAX_FORCE:
                steer.scale_to_length(self.max_force)
        return (-steer)

        




    def draw(self, surface):
        # Draw a triangle pointing in direction of velocity
        angle = math.atan2(self.vel.y, self.vel.x)
        p1 = Vector2(BOID_SIZE, 0).rotate_rad(angle) + self.pos
        p2 = Vector2(-BOID_SIZE * 0.6, BOID_SIZE * 0.6).rotate_rad(angle) + self.pos
        p3 = Vector2(-BOID_SIZE * 0.6, -BOID_SIZE * 0.6).rotate_rad(angle) + self.pos
        pygame.draw.polygon(surface, BOID_COLOR, [p1, p2, p3])

distance_mx = []
def update_distance_mx(boids,lead_boid):
    distance_mx.clear()
    for i in range(len(boids)):
        
        for j in range(len(boids)):
            if i != j and i < j:
                distance_mx.append(boids[i].pos.distance_to(boids[j].pos))
        lead_dis = boids[i].pos.distance_to(lead_boid[0].pos)
        distance_mx.append(lead_dis)
    matrix_min = np.min(distance_mx)
    matrix_max = np.max(distance_mx)
    for i in range(len(distance_mx)):
        distance_mx[i] = (distance_mx[i] - matrix_min) / (matrix_max - matrix_min)  # Normalize to [0, 1]
                
velocity_mx = []
def update_velocity_mx(boids):
    velocity_mx.clear()
    for i in range(len(boids)):
        velocity_mx.append(boids[i].vel)

def create_boids(n,lead_boid):
    return [Boid(random.uniform((lead_boid[0].pos.x - 50), (lead_boid[0].pos.x + 50)), random.uniform((lead_boid[0].pos.y - 50), (lead_boid[0].pos.y + 50))) for _ in range(n)]

def create_leaders(n):
    return [Leader(140,140) for _ in range(n)]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    global NUM_BOIDS
    
    lead_boid = create_leaders(NUM_LEADERS)
    boids = create_boids(NUM_BOIDS,lead_boid)
    
    paused = False
    running = True

    start = time.time()
    print("started")

    # For Pathfinding ---------------------------------
    grid = np.zeros((WIDTH, HEIGHT))   
    x,y = lead_boid[0].pos
    x = math.floor(x)
    y = math.floor(y)
    start_pos = (x,y)
    corner1_pos = (1260),(140)
    corner2_pos = (1260),(560)
    corner3_pos = (140),(560)
    goal_pos = (140),(140)
    
    planner = AStarPathPlanner(grid, allow_diagonal=True)
    path1, _ = planner.find_path(start_pos,corner1_pos)
    path2, _ = planner.find_path(corner1_pos,corner2_pos)
    path3, _ = planner.find_path(corner2_pos,corner3_pos)
    path4, _ = planner.find_path(corner3_pos,goal_pos)


    path = path1 + path2 + path3 + path4

    #--------------------------------------------------
    num_loops = 0
    while running:
        clock.tick(FPS)  # seconds per frame
        
        num_loops = num_loops + 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    boids = create_boids(NUM_BOIDS)
                    if NUM_LEADERS > 0:
                        lead_boid =create_leaders(NUM_LEADERS)
                elif event.key == pygame.K_UP:
                    NUM_BOIDS += 10
                    boids.extend(create_boids(10))
                elif event.key == pygame.K_DOWN:
                    if NUM_BOIDS > 10:
                        NUM_BOIDS = max(10, NUM_BOIDS - 10)
                        boids = boids[:NUM_BOIDS]

        if not paused:
            for boid in boids:
                boid.behaviors(boids,lead_boid,path,num_loops)
            for boid in boids:
                boid.update()
                boid.edges()
            if NUM_LEADERS > 0:
                for leader in lead_boid:
                    leader.follow_path(path,num_loops)
                    leader.update()
                    leader.edges()
            #print(leader.pos)

        screen.fill(BG_COLOR)
        for x, y in path:
            screen.set_at((x, y), YELLOW)

        for boid in boids:
            boid.draw(screen)
        if NUM_LEADERS > 0:
            for leader in lead_boid:
                leader.draw(screen)
                if PATH_PLAN:
                    leader.update_pathplan()
        
            


    
        pygame.display.flip()
        end = time.time()
        #print(end)
        if TESTING and (end - start) > 5:
            running = False
        

    pygame.quit()
    if not TESTING:
        update_distance_mx(boids,lead_boid)
        update_velocity_mx(boids)
        #print(distance_mx)
        #print(len(distance_mx))
        #print(velocity_mx)
        #print(len(velocity_mx))

        sys.exit()
    
TESTING = False
PATH_PLAN = False
if __name__ == "__main__":
    if TESTING:
        for i in range(7):
            ALIGNMENT_WEIGHT = float(i/2)
            print(i)
            main()
    else:
        main()
