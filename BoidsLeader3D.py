import sys
import random
import math
import time
from pygame.math import Vector3
import numpy as np
from AstarPathPlanning import AStarPathPlanner
from vpython import canvas, box, sphere, cone, vector, color, rate

# ---------- Parameters ----------
WIDTH, HEIGHT, DEPTH = 1400, 500, 1400
NUM_BOIDS = 10
NUM_LEADERS = 1
MAX_SPEED = 2
MAX_FORCE = 0.02

SEPARATION_RADIUS = 70
ALIGNMENT_RADIUS = 100
COHESION_RADIUS = 200
LEADER_RADIUS = 1000

L_MAX_SPEED = MAX_SPEED - 0.2
L_MAX_FORCE = 0.1
WAYPOINT_THRESHOLD = 10

SEPARATION_WEIGHT = 1.0
ALIGNMENT_WEIGHT = 0.0
COHESION_WEIGHT = 0.0
LEADER_WEIGHT = 1.0

BOID_SIZE = 8
BG_COLOR = (114, 180, 235)
BOID_COLOR = (200, 200, 220)
LEADER_COLOR = (255, 0, 0)
FPS = 60

CENTER = Vector3(WIDTH / 2, HEIGHT / 2, DEPTH / 2)

# -------------------------------- #PATH PLANNING
mission_state = {"Takeoff": True, "cruise": False, "leveling": False, "landing": False}
START_POS = Vector3(200,1,200)
PATH_HEIGHT = 300
MID_POS = Vector3(1100,1,400)
END_POS = Vector3(1200,1,1100)

grid = np.zeros((WIDTH, DEPTH))
planner = AStarPathPlanner(grid, allow_diagonal=True)
# --------------------------------


def vp(v):
    """pygame Vector3 -> vpython vector."""
    return vector(v.x, v.y, v.z)


def vp_color(c):
    return vector(c[0] / 255, c[1] / 255, c[2] / 255)


def random_unit_vector3():
    # Uniformly distributed direction on the unit sphere.
    u = random.uniform(-1, 1)
    theta = random.uniform(0, math.tau)
    r = math.sqrt(max(0.0, 1 - u * u))
    return Vector3(r * math.cos(theta), r * math.sin(theta), u)


class Leader:
    """Steering follower: seeks each waypoint along the path in turn."""

    def __init__(self, x, y, z):
        self.pos = Vector3(x, y, z)
        self.vel = Vector3(0, 0, 0)
        self.acc = Vector3(0, 0, 0)
        self.max_speed = L_MAX_SPEED
        self.max_force = L_MAX_FORCE
        self.waypoint_idx = 0
        self.state_complete = 1
        self.shape = sphere(pos=vp(self.pos), radius=12,
                            color=vp_color(LEADER_COLOR),
                            make_trail=False)

    def apply_force(self, force):
        self.acc += force

    def seek(self, target):
        desired = target - self.pos
        if desired.length() == 0:
            return Vector3(0, 0, 0)
        desired.scale_to_length(self.max_speed)
        steer = desired - self.vel
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    def update(self, path):
        if path:
            # Skip past any waypoints already inside the arrival radius.
            while (self.waypoint_idx < len(path) - 1
                   and self.pos.distance_to(path[self.waypoint_idx]) < WAYPOINT_THRESHOLD):
                self.waypoint_idx += 1
            if self.waypoint_idx != len(path) - 1:
                self.apply_force(self.seek(path[self.waypoint_idx]))
            else:
                self.vel = self.vel * 0.9
                if (self.vel.x < 0.01 or self.vel.z < 0.01) and self.state_complete < 3:
                    self.vel = Vector3(0,0,0)
                    self.state_complete = 2

        self.vel += self.acc
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        self.pos += self.vel
        self.acc = Vector3(0, 0, 0)

    def edges(self):
        pass

    def sync_visual(self):
        self.shape.pos = vp(self.pos)

    def remove_visual(self):
        self.shape.clear_trail()
        self.shape.visible = False


class Boid:
    def __init__(self, x, y, z):
        self.pos = Vector3(x, y, z)
        self.vel = Vector3(0,0,0)
        self.acc = Vector3(0, 0, 0)
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE
        self.shape = cone(pos=vp(self.pos),
                          axis=vector(0, 0, 1) * (BOID_SIZE * 2.5),
                          radius=BOID_SIZE * 0.6,
                          color=vp_color(BOID_COLOR))

    def edges(self):
        # Wrap-around in 3D
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT
        if self.pos.z > DEPTH:
            self.pos.z = 0
        elif self.pos.z < 0:
            self.pos.z = DEPTH

    def apply_force(self, force):
        self.acc += force

    def update(self):
        self.vel += self.acc
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        self.pos += self.vel
        self.acc = Vector3(0, 0, 0)

    def seek(self, target):
        desired = (target - self.pos)
        if desired.length() == 0:
            return Vector3(0, 0, 0)
        desired.scale_to_length(self.max_speed)
        steer = desired - self.vel
        if steer.length() > self.max_force:
            steer.scale_to_length(self.max_force)
        return steer

    def behaviors(self, boids, lead_boids):
        sep = self.separation(boids, lead_boids) * SEPARATION_WEIGHT
        ali = self.alignment(boids) * ALIGNMENT_WEIGHT
        coh = self.cohesion(boids) * COHESION_WEIGHT

        if NUM_LEADERS > 0:
            led = self.leader_force(lead_boids) * LEADER_WEIGHT
            self.apply_force(led)

        self.apply_force(sep)
        self.apply_force(ali)
        self.apply_force(coh)

    def separation(self, boids, lead_boid):
        steer = Vector3(0, 0, 0)
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
            if steer.length() > 0.01:
                steer.scale_to_length(self.max_speed)
                steer -= self.vel
                if steer.length() > self.max_force:
                    steer.scale_to_length(self.max_force)
        return steer

    def alignment(self, boids):
        avg_vel = Vector3(0, 0, 0)
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
            if avg_vel.length() > 0.01:
                avg_vel.scale_to_length(self.max_speed)
                steer = avg_vel - self.vel
                if steer.length() > self.max_force:
                    steer.scale_to_length(self.max_force)
                return steer
        return Vector3(0, 0, 0)

    def cohesion(self, boids):
        center = Vector3(0, 0, 0)
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
        return Vector3(0, 0, 0)

    def leader_force(self, lead_boid):
        steer = Vector3(0, 0, 0)
        total = 0
        for lother in lead_boid:
            d = self.pos.distance_to(lother.pos)
            if d < LEADER_RADIUS and d > 0:
                diff = (self.pos - lother.pos)
                total += 1
                steer += diff
        if total > 0:
            steer /= total
            if steer.length() > MAX_FORCE:
                steer.scale_to_length(self.max_force)
        return (-steer)

    def sync_visual(self):
        self.shape.pos = vp(self.pos)
        if self.vel.length() > 0.001:
            d = self.vel.normalize()
            self.shape.axis = vector(d.x, d.y, d.z) * (BOID_SIZE * 2.5)

    def remove_visual(self):
        self.shape.visible = False


distance_mx = []
def update_distance_mx(boids, lead_boid):
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
        distance_mx[i] = (distance_mx[i] - matrix_min) / (matrix_max - matrix_min)


velocity_mx = []
def update_velocity_mx(boids):
    velocity_mx.clear()
    for i in range(len(boids)):
        velocity_mx.append(boids[i].vel)


def create_boids(n, lead_boid):
    lp = lead_boid[0].pos
    return [Boid(random.uniform(lp.x - 75, lp.x + 75),
                 lp.y,
                 random.uniform(lp.z - 75, lp.z + 75)) for _ in range(n)]


def create_leaders(n):
    return [Leader(START_POS.x, START_POS.y, START_POS.z) for _ in range(n)] #######Starting

def mission_update(lead_boid,boids):
    global mission_state
    global SEPARATION_WEIGHT
    global ALIGNMENT_WEIGHT
    global COHESION_WEIGHT
    global LEADER_WEIGHT

    if mission_state["Takeoff"] == True:
        if lead_boid[0].pos.y > PATH_HEIGHT - 5:
            mission_state["Takeoff"] = False
            mission_state["cruise"] = True

    if mission_state["cruise"] == True:
        SEPARATION_WEIGHT = 10.0
        ALIGNMENT_WEIGHT = 3.5
        COHESION_WEIGHT = 3.0
        LEADER_WEIGHT = 7.0
        if lead_boid[0].state_complete == 2:
            mission_state["cruise"] = False
            mission_state["leveling"] = True
    
    if mission_state["leveling"] == True:
        SEPARATION_WEIGHT = 10.0
        ALIGNMENT_WEIGHT = 0
        COHESION_WEIGHT = 0.5
        LEADER_WEIGHT = 0.5
        next_state = True
        for boid in boids:
            if boid.pos.y < (PATH_HEIGHT - 2):
                boid.acc.y = abs(boid.acc.y)
                next_state = False
            elif boid.pos.y > (PATH_HEIGHT + 2):
                boid.acc.y = -abs(boid.acc.y)
                next_state = False
            else:
                boid.pos.y = PATH_HEIGHT
                boid.acc.y = 0.0
                boid.vel.y = 0
        if next_state == True:
            lead_boid[0].state_complete = 3
            mission_state["landing"] = True
            mission_state["leveling"] = False
            for boid in boids:
                boid.vel = Vector3(0,0,0)

    if mission_state["landing"]:
        SEPARATION_WEIGHT = 0.0
        ALIGNMENT_WEIGHT = 0
        COHESION_WEIGHT = 0.0
        LEADER_WEIGHT = 0.0
            
        if lead_boid[0].pos.y < 1:
            lead_boid[0].vel = Vector3(0,0,0)
            lead_boid[0].state_complete = 4
        else:
            lead_boid[0].vel = Vector3(0,-0.5,0)
        for boid in boids:
            if boid.pos.y < 1:
                boid.vel = Vector3(0,0,0)
            else:
                boid.vel = Vector3(0,-0.5,0)


def main():
    global NUM_BOIDS
    
    
    scene = canvas(title="3D Boids (VPython)  -  space: pause  |  r: reset  |  up/down: add/remove  |  q: quit\n",
                   width=1200, height=700, background=vp_color(BG_COLOR))
    scene.center = vp(CENTER)
    scene.range = max(WIDTH, HEIGHT, DEPTH) * 0.6
    scene.forward = vector(-1, -0.4, -1)

    # Translucent bounding volume so the 3D area is visible.
    box(pos=vp(CENTER), size=vector(WIDTH, HEIGHT, DEPTH),
        color=color.white, opacity=0.05)
    box(pos=vp(Vector3(WIDTH/2,0,DEPTH/2)), size=vector(WIDTH, 0.1, DEPTH), color=color.green)

    lead_boid = create_leaders(NUM_LEADERS)
    boids = create_boids(NUM_BOIDS, lead_boid)
    
    path1, _ = planner.find_path((int(START_POS.x),int(START_POS.z)),(int(MID_POS.x),int(MID_POS.z)))
    path2, _ = planner.find_path((int(MID_POS.x),int(MID_POS.z)),(int(END_POS.x),int(END_POS.z)))
    path = path1 + path2
    path3d = [Vector3(p[0], PATH_HEIGHT, p[1]) for p in path]
    for p in path:
        sphere(pos=vector(p[0], PATH_HEIGHT, p[1]), radius=1, color=color.yellow, make_trail=False)

    state = {"paused": False, "running": True, "reset": False, "add": 0}

    def keydown(evt):
        k = evt.key
        if k == " ":
            state["paused"] = not state["paused"]
        elif k == "r":
            state["reset"] = True
        elif k == "up":
            state["add"] += 10
        elif k == "down":
            state["add"] -= 10
        elif k == "q":
            state["running"] = False

    scene.bind("keydown", keydown)

    start = time.time()
    print("started")

    while state["running"]:
        rate(FPS)
        
        
        
        if state["reset"]:
            for b in boids:
                b.remove_visual()
            boids = create_boids(NUM_BOIDS, lead_boid)
            state["reset"] = False

        if state["add"] != 0:
            if state["add"] > 0:
                boids.extend(create_boids(state["add"], lead_boid))
            else:
                remove = min(-state["add"], max(0, len(boids) - 10))
                for _ in range(remove):
                    boids.pop().remove_visual()
            NUM_BOIDS = len(boids)
            state["add"] = 0

        if not state["paused"]:
            
            for boid in boids:
                boid.behaviors(boids, lead_boid)###########
                mission_update(lead_boid,boids)##################
                boid.update()
                boid.edges()
            if NUM_LEADERS > 0:
                for leader in lead_boid:
                    leader.update(path3d)
                    leader.edges()
        
        for boid in boids:
            boid.sync_visual()
        if NUM_LEADERS > 0:
            for leader in lead_boid:
                leader.sync_visual()

        if TESTING and (time.time() - start) > 5:
            state["running"] = False

    if not TESTING:
        update_distance_mx(boids, lead_boid)
        update_velocity_mx(boids)
        sys.exit()


TESTING = False
if __name__ == "__main__":
    if TESTING:
        for i in range(7):
            ALIGNMENT_WEIGHT = float(i / 2)
            print(i)
            main()
    else:
        main()
