import sys
import random
import math
import time
from pygame.math import Vector3
import numpy as np
from AstarPathPlanning import AStarPathPlanner
from vpython import canvas, box, sphere, cone, cylinder, vector, color, rate

# ---------- Parameters ----------
WIDTH, HEIGHT, DEPTH = 2000, 500, 2000
NUM_BOIDS = 10
NUM_LEADERS = 1
MAX_SPEED = 1
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

OBSTACLE_WEIGHT = 20.0
OBSTACLE_RADIUS = 80          # physical radius of an obstacle
OBSTACLE_BUFFER = 60          # extra avoidance margin around an obstacle

LINE_WEIGHT = 5.0             # pull toward a boid's assigned slot on the line
LINE_SPACING = 80             # even gap between boids along the formation line

BOID_SIZE = 8
BG_COLOR = (114, 180, 235)
BOID_COLOR = (200, 200, 220)
LEADER_COLOR = (255, 0, 0)
OBSTACLE_COLOR = (200, 90, 40)   # distinct orange so obstacles stand out
FPS = 60

CENTER = Vector3(WIDTH / 2, HEIGHT / 2, DEPTH / 2)

# -------------------------------- #PATH PLANNING


grid = np.zeros((WIDTH, DEPTH))
# --------------- #OBSTACLES
cx, cz = WIDTH // 2, DEPTH // 2
xx, zz = np.ogrid[:WIDTH, :DEPTH]
grid[(xx - cx) ** 2 + (zz - cz) ** 2 <= 100 ** 2] = 1

planner = AStarPathPlanner(grid, allow_diagonal=True)


START_POS = Vector3(200,1,200)
PATH_HEIGHT = 300
MID_POS = Vector3(300,1,900)
MID2_POS = Vector3(700,1,900)
END_POS = Vector3(200,1,200)

path1, _ = planner.find_path((int(START_POS.x),int(START_POS.z)),(int(MID_POS.x),int(MID_POS.z)))
op_path, _ = planner.find_path((int(MID_POS.x),int(MID_POS.z)),(int(MID2_POS.x),int(MID2_POS.z)))
path3d1 = [Vector3(p[0], PATH_HEIGHT, p[1]) for p in path1]
op_path3d = [Vector3(p[0], PATH_HEIGHT, p[1]) for p in op_path]


mission_index = 0
mission_state = [
    {"step": 0, "name": "Takeoff", "path": None},
    {"step": 1, "name": "Cruise", "path": path3d1, "Dest": MID_POS},
    {"step": 2, "name": "Operation", "path": None, "Type": "Sweep"},
    {"step": 3, "name": "Operation", "path": op_path3d, "Type": "Match"},
    {"step": 4, "name": "Cruise", "path": None, "Dest": END_POS},
    {"step": 5, "name": "Level", "path": None},
    {"step": 6, "name": "Landing", "path": None}
]
current_step = mission_state[mission_index]
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

def generate_path(start,end):
    new_path, _ = planner.find_path((int(start.x),int(start.z)),(int(end.x),int(end.z)))
    path3d = [Vector3(p[0], PATH_HEIGHT, p[1]) for p in new_path]
    #draw_generated_path(path3d)
    return path3d

def draw_generated_path(path3d):
    for p in path3d:
        sphere(pos=vector(p[0], PATH_HEIGHT, p[1]), radius=1, color=color.yellow, make_trail=False)



class Leader:
    """Steering follower: seeks each waypoint along the path in turn."""

    def __init__(self, x, y, z):
        self.pos = Vector3(x, y, z)
        self.vel = Vector3(0, 0, 0)
        self.acc = Vector3(0, 0, 0)
        self.max_speed = L_MAX_SPEED
        self.max_force = L_MAX_FORCE
        self.waypoint_idx = 0
        self.path_done = False
        self.shape = cone(pos=vp(self.pos),
                          axis=vector(0, 0, 1) * (BOID_SIZE * 3.5),
                          radius=BOID_SIZE * 0.9,
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

    def update(self, path=None):
        if path:
            # Skip past any waypoints already inside the arrival radius.
            while (self.waypoint_idx < len(path) - 1 and self.pos.distance_to(path[self.waypoint_idx]) < WAYPOINT_THRESHOLD):
                self.waypoint_idx += 1
                self.path_done = False
            if self.waypoint_idx == len(path) - 1:
                self.path_done = True
                #print("trips1")
            else:
                self.apply_force(self.seek(path[self.waypoint_idx]))
                #print("trips2")
                #print(self.waypoint_idx, end=" ")
                #print(len(path) - 1)
                           

        self.vel += self.acc
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        self.pos += self.vel
        self.acc = Vector3(0, 0, 0)

    def edges(self):
        pass

    def sync_visual(self):
        self.shape.pos = vp(self.pos)
        if self.vel.length() > 0.001:
            d = self.vel.normalize()
            self.shape.axis = vector(d.x, d.y, d.z) * (BOID_SIZE * 3.5)

    def remove_visual(self):
        self.shape.clear_trail()
        self.shape.visible = False


class Obstacle:
    """Stationary vertical pillar. Never moves; only the Boid class reacts to it.

    Defined purely by its x/z footprint (radius); it spans the full height, so
    boids avoid it in the horizontal plane regardless of their altitude.
    """

    def __init__(self, x, z, radius=OBSTACLE_RADIUS):
        self.x = x
        self.z = z
        self.radius = radius
        # Solid core: the physical pillar, from the floor to the ceiling.
        self.shape = cylinder(pos=vector(x, 0, z),
                              axis=vector(0, HEIGHT, 0),
                              radius=radius,
                              color=vp_color(OBSTACLE_COLOR),
                              opacity=1.0)
        # Faint outer shell: the influence zone where boids start steering away.
        self.zone = cylinder(pos=vector(x, 0, z),
                             axis=vector(0, HEIGHT, 0),
                             radius=radius + OBSTACLE_BUFFER,
                             color=vp_color(OBSTACLE_COLOR),
                             opacity=0.12)

    def remove_visual(self):
        self.shape.visible = False
        self.zone.visible = False


class Boid:
    def __init__(self, x, y, z):
        self.pos = Vector3(x, y, z)
        self.vel = Vector3(0,0,0)
        self.acc = Vector3(0, 0, 0)
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE
        self.line_target = None   # assigned slot on the formation line, if any
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

    def behaviors(self, boids, lead_boids, obstacles=()):
        sep = self.separation(boids, lead_boids) * SEPARATION_WEIGHT
        ali = self.alignment(boids) * ALIGNMENT_WEIGHT
        coh = self.cohesion(boids) * COHESION_WEIGHT
        obs = self.avoid_obstacles(obstacles) * OBSTACLE_WEIGHT

        if NUM_LEADERS > 0:
            led = self.leader_force(lead_boids) * LEADER_WEIGHT
            self.apply_force(led)

        self.apply_force(sep)
        self.apply_force(ali)
        self.apply_force(coh)
        self.apply_force(obs)

        # Hold position in the line formation, if one has been assigned.
        if self.line_target is not None:
            self.apply_force(self.seek(self.line_target) * LINE_WEIGHT)

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

    def avoid_obstacles(self, obstacles):
        # Steer away from each cylindrical pillar within its influence zone.
        # Distance is measured in the horizontal x-z plane only, so altitude
        # never matters - the pillars act as full-height walls.
        steer = Vector3(0, 0, 0)
        total = 0
        for obs in obstacles:
            dx = self.pos.x - obs.x
            dz = self.pos.z - obs.z
            d = math.hypot(dx, dz)
            influence = obs.radius + OBSTACLE_BUFFER
            if 0 < d < influence:
                # Horizontal push only; harder the deeper inside the zone.
                diff = Vector3(dx, 0, dz)
                diff /= (d * d)
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

def assign_line_targets(leader, boids):
    """Spread the boids over a horizontal line passing through the leader.

    The line runs perpendicular to the leader's heading (a sweep line) and the
    boids are spaced evenly along it. Slots are handed out in the boids' current
    left-to-right order along the line, so no two assignment paths ever cross -
    combined with the separation force, the boids cannot collide on the way in.

    Sets each boid's ``line_target`` and returns True once all are in place.
    """
    n = len(boids)
    if n == 0:
        return True

    # Horizontal heading of the leader; default to the x-axis when stationary.
    fwd = Vector3(leader.vel.x, 0, leader.vel.z)
    if fwd.length() < 1e-6:
        fwd = Vector3(1, 0, 0)
    fwd = fwd.normalize()
    # Rotate 90 degrees in the x-z plane to get the sideways line direction.
    line_dir = Vector3(-fwd.z, 0, fwd.x)

    # Even offsets centered on the leader: ..., -1.5s, -0.5s, 0.5s, 1.5s, ...
    offsets = [(k - (n - 1) / 2.0) * LINE_SPACING for k in range(n)]

    # Order boids by where they already sit along the line, then assign slots in
    # that same order so their straight-line paths to the targets never cross.
    order = sorted(range(n), key=lambda i: line_dir.dot(boids[i].pos - leader.pos))

    formed = True
    for slot, i in enumerate(order):
        target = leader.pos + line_dir * offsets[slot]
        target.y = leader.pos.y          # keep the line perfectly horizontal
        boids[i].line_target = target
        if boids[i].pos.distance_to(target) > WAYPOINT_THRESHOLD:
            formed = False
    return formed


def mission_update(lead_boid,boids):
    global mission_index
    global mission_state
    global current_step
    global SEPARATION_WEIGHT
    global ALIGNMENT_WEIGHT
    global COHESION_WEIGHT
    global LEADER_WEIGHT

    current_step = mission_state[mission_index]
    next_step = False
    
    
    if current_step["name"] == "Takeoff":
        lead_boid[0].vel = Vector3(0,1,0)
        if lead_boid[0].pos.y > (PATH_HEIGHT - 5):
            next_step = True

    if current_step["name"] == "Cruise":
        SEPARATION_WEIGHT = 10.0
        ALIGNMENT_WEIGHT = 3.5
        COHESION_WEIGHT = 3.0
        LEADER_WEIGHT = 7.0
        
        if lead_boid[0].path_done:
            print("Path done")
            next_step = True
            lead_boid[0].vel = Vector3(0,0,0)
        if current_step["path"] == None:
            current_step["path"] = generate_path(lead_boid[0].pos,current_step["Dest"])

            
    
    if current_step["name"] == "Level":
        SEPARATION_WEIGHT = 10.0
        ALIGNMENT_WEIGHT = 0
        COHESION_WEIGHT = 0.5
        LEADER_WEIGHT = 0.0
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
                boid.vel.y = 0.0
        if next_state == True:
            next_step = True
            for boid in boids:
                boid.vel = Vector3(0,0,0)

    if current_step["name"] == "Landing":
        SEPARATION_WEIGHT = 0.0
        ALIGNMENT_WEIGHT = 0
        COHESION_WEIGHT = 0.0
        LEADER_WEIGHT = 0.0
            
        if lead_boid[0].pos.y < 1:
            lead_boid[0].vel = Vector3(0,0,0)
            
        else:
            lead_boid[0].vel = Vector3(0,-0.5,0)
        for boid in boids:
            if boid.pos.y < 1:
                boid.vel = Vector3(0,0,0)
            else:
                boid.vel = Vector3(0,-0.5,0)
    
    if current_step["name"] == "Operation":
        # Strong personal space so boids never collide; drop the flocking pulls
        # so the line holds its shape while each boid seeks its own slot.
        
        if current_step["Type"] == "Sweep":
            SEPARATION_WEIGHT = 3.0
            ALIGNMENT_WEIGHT = 0.0
            COHESION_WEIGHT = 0.0
            LEADER_WEIGHT = 0.0
            formed = assign_line_targets(lead_boid[0], boids)
            if formed:
                next_step = True
                print("Formed")
                for boid in boids:
                    boid.line_target = None
                    boid.vel = Vector3(0, 0, 0)
        if current_step["Type"] == "Match":
            SEPARATION_WEIGHT = 3.0
            ALIGNMENT_WEIGHT = 1.0
            COHESION_WEIGHT = 1.0
            LEADER_WEIGHT = 1.0
            if lead_boid[0].path_done:
                next_step = True
                print("ITS SHOULD BE DONE BUT WHY NOT")
            else:
                for boid in boids:
                    # Copy the leader's velocity by value; assigning the object
                    # directly would alias every boid to one shared Vector3, so
                    # the cloning could never be undone in later phases.
                    boid.vel = Vector3(lead_boid[0].vel)
                    boid.acc = Vector3(0,0,0)




    if next_step:
        mission_index = mission_index + 1
        print("mission_index", end= " ")
        print(mission_index)
        # re-arm path tracking for the next leg
        if mission_index < len(mission_state):# and mission_state[mission_index]["name"] == "Cruise":
            for leader in lead_boid:
                leader.waypoint_idx = 0
                leader.path_done = False
                


def main():
    global NUM_BOIDS
    
    
    scene = canvas(title="3D Boids (VPython)  -  space: pause  |  r: reset  |  up/down: add/remove  |  q: quit\n",
                   width=1200, height=700, background=vp_color(BG_COLOR))
    scene.center = vp(CENTER)
    scene.range = max(WIDTH, HEIGHT, DEPTH) * 0.6
    scene.forward = vector(1, -0.4, 1)

    # Translucent bounding volume so the 3D area is visible.
    box(pos=vp(CENTER), size=vector(WIDTH, HEIGHT, DEPTH),
        color=color.white, opacity=0.05)
    box(pos=vp(Vector3(WIDTH/2,0,DEPTH/2)), size=vector(WIDTH, 0.1, DEPTH), color=color.green)

    # Grey cylinder: radius 100, 400 tall, centered in the scene.
    # cylinder(pos=vp(Vector3(WIDTH / 2, HEIGHT / 2 - 200, DEPTH / 2)),
    #          axis=vector(0, 400, 0), radius=100, color=color.gray(0.5))

    lead_boid = create_leaders(NUM_LEADERS)
    boids = create_boids(NUM_BOIDS, lead_boid)

    # Stationary pillars (x, z footprint) the boids must steer around.
    obstacles = [
        Obstacle(WIDTH / 2, DEPTH / 2),
        #Obstacle(WIDTH * 0.35, DEPTH * 0.65),
    ]
    
    # path = path1
    # for p in path:
    #     sphere(pos=vector(p[0], PATH_HEIGHT, p[1]), radius=1, color=color.yellow, make_trail=False)

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
                boid.behaviors(boids, lead_boid, obstacles)
                mission_update(lead_boid,boids)
                boid.update()
                boid.edges()
            if NUM_LEADERS > 0:
                for leader in lead_boid:
                    leader.update(current_step["path"])
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
