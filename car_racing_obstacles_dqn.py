# CS530 Group 8 CarRacing Agent 

import os
import cv2
import gymnasium as gym
import imageio
import random
import json
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_ID = "CarRacing-v3"
SEED = 7

FRAME_STACK = 4
IMAGE_SIZE = 84

TOTAL_EPISODES = 1000
MAX_STEPS = 2000

BATCH_SIZE = 32
BUFFER_SIZE = 80_000
GAMMA = 0.99
LR = 1e-4

WARMUP_STEPS = 15000
START_LEARNING_AFTER = 3000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1000

EPSILON_START = 0.35
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 120_000

MODEL_PATH = "dqn_carracing_visible_obstacles_v4.pt"
VIDEO_PATH = "dqn_visible_obstacles_run_v4.mp4"

BEST_EPISODE_VIDEO_PATH = "best_episode_dqn_visible_obstacles_v4.mp4"
BEST_METADATA_PATH = "best_episode_metadata_v4.json"

DEBUG_ENV = True
DEBUG_EVERY = 50
DEBUG_FIRST_STEPS = 250

# Strong safety fixes
TERMINATE_ON_OBSTACLE_HIT = True
TERMINATE_ON_WRONG_WAY = True
TERMINATE_ON_OFF_TRACK = True

# Collision tuning
CAR_COLLISION_RADIUS = 0.90
OBSTACLE_RADIUS = 0.38
COLLISION_PENALTY = -80.0

# Wrong-way tuning
WRONG_WAY_DOT_THRESHOLD = -0.25
WRONG_WAY_COUNTER_LIMIT = 18
BACKWARD_INDEX_COUNTER_LIMIT = 12

# Off-track tuning
SOFT_OFFTRACK_LATERAL = 2.20
HARD_OFFTRACK_LATERAL = 2.80
TERMINAL_OFFTRACK_LATERAL = 3.25
OFFTRACK_COUNTER_LIMIT = 12

# DQN chooses from a fixed discrete list of continuous CarRacing actions, with each action being one of the three: [steering, gas, brake].
# Negative steer = left, positive steer = right.
ACTIONS = [
    np.array([0.00, 0.32, 0.00], dtype=np.float32),
    np.array([0.00, 0.22, 0.00], dtype=np.float32),
    np.array([0.00, 0.08, 0.20], dtype=np.float32),

    np.array([-0.08, 0.30, 0.00], dtype=np.float32),
    np.array([0.08, 0.30, 0.00], dtype=np.float32),

    np.array([-0.16, 0.26, 0.00], dtype=np.float32),
    np.array([0.16, 0.26, 0.00], dtype=np.float32),

    np.array([-0.28, 0.18, 0.04], dtype=np.float32),
    np.array([0.28, 0.18, 0.04], dtype=np.float32),

    np.array([-0.42, 0.08, 0.12], dtype=np.float32),
    np.array([0.42, 0.08, 0.12], dtype=np.float32),

    np.array([-0.55, 0.02, 0.25], dtype=np.float32),
    np.array([0.55, 0.02, 0.25], dtype=np.float32),
]

NUM_ACTIONS = len(ACTIONS)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Remove bottom dashboard area.
    gray = gray[:84, :]

    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        self.frames.clear()
        p = preprocess_frame(frame)

        for _ in range(self.k):
            self.frames.append(p)

        return np.stack(self.frames, axis=0)

    def step(self, frame):
        p = preprocess_frame(frame)
        self.frames.append(p)
        return np.stack(self.frames, axis=0)


def angle_normalize(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class VisibleTrackObstacleCarRacing(gym.Wrapper):
    def __init__(
        self,
        env,
        num_obstacles=12,
        obstacle_radius=OBSTACLE_RADIUS,
        collision_penalty=COLLISION_PENALTY,
        obstacle_step_penalty=-0.0005,
        debug=False,
    ):
        super().__init__(env)

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.collision_penalty = collision_penalty
        self.obstacle_step_penalty = obstacle_step_penalty
        self.debug = debug

        self.obstacles = []
        self._added_to_road = False

        self.offtrack_counter = 0
        self.wrong_way_counter = 0
        self.backward_index_counter = 0

        self.prev_progress = 0.0
        self.prev_track_idx = None
        self.forward_index_steps = 0
        self.backward_index_steps = 0

        self.near_finish_bonus_given = False
        self.lap_bonus_given = False

        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.obstacles = []
        self._added_to_road = False

        self.offtrack_counter = 0
        self.wrong_way_counter = 0
        self.backward_index_counter = 0

        self.prev_progress = 0.0
        self.prev_track_idx = None
        self.forward_index_steps = 0
        self.backward_index_steps = 0

        self.near_finish_bonus_given = False
        self.lap_bonus_given = False

        self.step_count = 0

        self._spawn_fixed_world_obstacles()
        self._add_obstacles_to_road_render()

        obs = self.env.render()

        current_idx = self._nearest_track_index()
        self.prev_track_idx = current_idx

        if self.debug:
            base = self.env.unwrapped
            print("\n[RESET DEBUG]")
            print(f"[RESET DEBUG] fixed obstacles: {len(self.obstacles)}")
            print(f"[RESET DEBUG] road_poly length: {len(getattr(base, 'road_poly', []))}")
            print(f"[RESET DEBUG] initial_track_idx={self.prev_track_idx}")

            for i, ob in enumerate(self.obstacles[:12]):
                print(
                    f"[RESET DEBUG] obstacle {i:02d}: "
                    f"x={ob['x']:.2f}, y={ob['y']:.2f}, "
                    f"offset={ob['offset']:.2f}, r={ob['r']:.2f}, "
                    f"track_index={ob['track_index']}"
                )

        return obs, info

    def step(self, action):
        self.step_count += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.env.render()

        current_idx = self._nearest_track_index()
        lateral_signed = self._track_lateral_position(current_idx)
        lateral = abs(lateral_signed)
        speed = self._car_speed()
        heading_error_signed = self._heading_error_to_track(current_idx)
        heading_error = abs(heading_error_signed)
        progress = self._track_progress()

        index_delta, moving_forward_by_index = self._track_index_delta(current_idx)
        heading_dot = self._heading_alignment_dot(current_idx)
        wrong_way = self._is_wrong_way(index_delta, moving_forward_by_index, heading_dot, speed)

        progress_delta = progress - self.prev_progress

        # If progress wraps around at lap completion, do not count it as negative.
        if progress_delta < -0.50:
            progress_delta = 0.0

        # Do not reward negative progress.
        progress_delta = max(0.0, progress_delta)
        self.prev_progress = progress

        hit, hit_obstacle = self._check_collision(return_obstacle=True)

        info["lateral"] = lateral
        info["lateral_signed"] = lateral_signed
        info["speed"] = speed
        info["heading_error"] = heading_error
        info["heading_error_signed"] = heading_error_signed
        info["heading_dot"] = heading_dot
        info["progress"] = progress
        info["progress_delta"] = progress_delta
        info["track_index"] = -1 if current_idx is None else int(current_idx)
        info["track_index_delta"] = int(index_delta)
        info["moving_forward_by_index"] = bool(moving_forward_by_index)
        info["wrong_way"] = bool(wrong_way)
        info["wrong_way_counter"] = int(self.wrong_way_counter)
        info["backward_index_counter"] = int(self.backward_index_counter)
        info["lap_complete"] = progress >= 0.95
        info["hit_obstacle"] = False
        info["off_track_terminated"] = False
        info["wrong_way_terminated"] = False
        info["collision_terminated"] = False

        reward = float(reward)

    
        # Progress reward / penalty
        reward += 8.0 * progress_delta

        if speed > 3.0 and progress_delta <= 0.00001:
            reward -= 0.04

    
        # Collision handling
        if hit:
            reward += self.collision_penalty
            info["hit_obstacle"] = True
            info["collision_terminated"] = True

            if hit_obstacle is not None:
                info["hit_obstacle_track_index"] = int(hit_obstacle["track_index"])
                info["hit_obstacle_x"] = float(hit_obstacle["x"])
                info["hit_obstacle_y"] = float(hit_obstacle["y"])

            if TERMINATE_ON_OBSTACLE_HIT:
                terminated = True

        else:
            reward += self.obstacle_step_penalty

    
        # Off-track handling
        if lateral > TERMINAL_OFFTRACK_LATERAL:
            self.offtrack_counter += 2
            reward -= 4.0

        elif lateral > HARD_OFFTRACK_LATERAL:
            self.offtrack_counter += 1
            reward -= 1.5

        elif lateral > SOFT_OFFTRACK_LATERAL:
            self.offtrack_counter += 1
            reward -= 0.50

        else:
            self.offtrack_counter = max(0, self.offtrack_counter - 2)

        info["offtrack_counter"] = int(self.offtrack_counter)

        if self.offtrack_counter >= OFFTRACK_COUNTER_LIMIT:
            reward -= 40.0
            info["off_track_terminated"] = True

            if TERMINATE_ON_OFF_TRACK:
                terminated = True

        
        # Centering reward
        if lateral < 0.90:
            reward += 0.08
        elif lateral < 1.40:
            reward += 0.04
        elif lateral < 2.00:
            reward += 0.01
        elif lateral > 2.40:
            reward -= 0.25

        
        # Direction / wrong-way handling
        if wrong_way:
            self.wrong_way_counter += 1
            reward -= 1.0
        else:
            self.wrong_way_counter = max(0, self.wrong_way_counter - 2)

        info["wrong_way_counter"] = int(self.wrong_way_counter)

        if self.wrong_way_counter >= WRONG_WAY_COUNTER_LIMIT:
            reward -= 50.0
            info["wrong_way_terminated"] = True

            if TERMINATE_ON_WRONG_WAY:
                terminated = True

        if heading_dot > 0.75:
            reward += 0.06
        elif heading_dot > 0.40:
            reward += 0.025
        elif heading_dot < 0.0:
            reward -= 0.30

       
        # Heading error shaping
        if heading_error < 0.25:
            reward += 0.04
        elif heading_error > 0.95:
            reward -= 0.25

      
        # Speed shaping
        # Safer speed range because obstacles exist.
        if 8.0 <= speed <= 26.0:
            reward += 0.04
        elif 26.0 < speed <= 36.0:
            reward -= 0.03
        elif speed > 36.0:
            reward -= 0.25
        elif speed < 2.5:
            reward -= 0.04

    
        # Obstacle proximity handling
        near_ob = self.nearest_forward_obstacle(lookahead=28)

        if near_ob is not None:
            car = getattr(self.env.unwrapped, "car", None)

            if car is not None:
                car_x = car.hull.position[0]
                car_y = car.hull.position[1]
                dist_to_ob = np.sqrt((car_x - near_ob["x"]) ** 2 + (car_y - near_ob["y"]) ** 2)
                info["nearest_obstacle_dist"] = float(dist_to_ob)
                info["nearest_obstacle_offset"] = float(near_ob["offset"])
                info["nearest_obstacle_track_index"] = int(near_ob["track_index"])

                if dist_to_ob < 8.0 and speed > 26.0:
                    reward -= 0.40
                elif dist_to_ob < 12.0 and speed > 32.0:
                    reward -= 0.25

        else:
            info["nearest_obstacle_dist"] = None
            info["nearest_obstacle_offset"] = None
            info["nearest_obstacle_track_index"] = None

   
        # One-time finish bonuses  
        if progress >= 0.90 and not self.near_finish_bonus_given:
            reward += 10.0
            self.near_finish_bonus_given = True

        if progress >= 0.95 and not self.lap_bonus_given:
            reward += 100.0
            self.lap_bonus_given = True
            info["lap_complete"] = True

        # Update previous index after all calculations.
        self.prev_track_idx = current_idx

        if self.debug and (
            self.step_count <= DEBUG_FIRST_STEPS
            or self.step_count % DEBUG_EVERY == 0
            or hit
            or info["off_track_terminated"]
            or info["wrong_way_terminated"]
        ):
            self._print_step_debug(
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def _print_step_debug(self, action, reward, terminated, truncated, info):
        print(
            f"[STEP DEBUG] step={self.step_count:04d} | "
            f"reward={reward:+8.3f} | "
            f"term={terminated} trunc={truncated} | "
            f"idx={info.get('track_index')} d_idx={info.get('track_index_delta')} | "
            f"forward_idx={info.get('moving_forward_by_index')} | "
            f"progress={info.get('progress', 0.0) * 100:6.2f}% | "
            f"d_prog={info.get('progress_delta', 0.0) * 100:7.4f}% | "
            f"lat={info.get('lateral', 0.0):5.2f} signed={info.get('lateral_signed', 0.0):+5.2f} | "
            f"speed={info.get('speed', 0.0):5.2f} | "
            f"head_err={info.get('heading_error', 0.0):5.2f} | "
            f"head_dot={info.get('heading_dot', 0.0):+5.2f} | "
            f"wrong={info.get('wrong_way')} wc={info.get('wrong_way_counter')} | "
            f"offc={info.get('offtrack_counter')} | "
            f"hit={info.get('hit_obstacle')} | "
            f"offterm={info.get('off_track_terminated')} | "
            f"wrongterm={info.get('wrong_way_terminated')} | "
            f"action=[{action[0]:+.2f},{action[1]:.2f},{action[2]:.2f}] | "
            f"near_ob_dist={info.get('nearest_obstacle_dist')}"
        )

    def _spawn_fixed_world_obstacles(self):
        base = self.env.unwrapped
        track = getattr(base, "track", None)

        if track is None or len(track) < 100:
            return

        n = len(track)

        start = int(0.16 * n)
        end = int(0.88 * n)

        obstacle_indices = np.linspace(start, end, self.num_obstacles, dtype=int)

        offset_pattern = [
            -1.15, 1.15,
            -1.35, 1.35,
            -0.95, 0.95,
            -1.25, 1.25,
        ]

        used_positions = []

        for k, idx in enumerate(obstacle_indices):
            _, beta, x, y = track[idx]

            nx = -np.sin(beta)
            ny = np.cos(beta)

            offset = offset_pattern[k % len(offset_pattern)]

            ox = x + offset * nx
            oy = y + offset * ny

            too_close = False

            for px, py in used_positions:
                if np.sqrt((ox - px) ** 2 + (oy - py) ** 2) < 4.0:
                    too_close = True
                    break

            if too_close:
                continue

            used_positions.append((ox, oy))

            self.obstacles.append({
                "track_index": int(idx),
                "x": float(ox),
                "y": float(oy),
                "r": float(self.obstacle_radius),
                "offset": float(offset),
            })

    def _make_circle_polygon(self, cx, cy, r, points=24):
        poly = []

        for i in range(points):
            angle = 2.0 * np.pi * i / points
            poly.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))

        return poly

    def _make_barbed_wire_segments(self, cx, cy, beta, length=1.25, width=0.32):
        tx = np.cos(beta)
        ty = np.sin(beta)

        nx = -np.sin(beta)
        ny = np.cos(beta)

        x1 = cx - length * tx
        y1 = cy - length * ty
        x2 = cx + length * tx
        y2 = cy + length * ty

        segments = []
        segments.append(([(x1, y1), (x2, y2)], (40, 40, 40)))

        for s in [-0.65, -0.25, 0.20, 0.60]:
            bx = cx + s * length * tx
            by = cy + s * length * ty

            p1 = (bx - width * nx, by - width * ny)
            p2 = (bx + width * nx, by + width * ny)

            q1 = (
                bx - width * 0.75 * (tx + nx),
                by - width * 0.75 * (ty + ny),
            )
            q2 = (
                bx + width * 0.75 * (tx + nx),
                by + width * 0.75 * (ty + ny),
            )

            segments.append(([p1, p2], (25, 25, 25)))
            segments.append(([q1, q2], (25, 25, 25)))

        return segments

    def _add_obstacles_to_road_render(self):
        if self._added_to_road:
            return

        base = self.env.unwrapped

        if not hasattr(base, "road_poly"):
            print("[WARNING] road_poly not found.")
            return

        track = getattr(base, "track", None)

        for ob in self.obstacles:
            beta = 0.0

            if track is not None and len(track) > ob["track_index"]:
                _, beta, _, _ = track[ob["track_index"]]

            warning = self._make_circle_polygon(ob["x"], ob["y"], ob["r"] * 1.55)
            base.road_poly.append((warning, (180, 40, 40)))

            segments = self._make_barbed_wire_segments(
                ob["x"],
                ob["y"],
                beta + np.pi / 2.0,
                length=1.25,
                width=0.30,
            )

            for line, color in segments:
                if len(line) == 2:
                    (x1, y1), (x2, y2) = line

                    dx = x2 - x1
                    dy = y2 - y1
                    norm = np.sqrt(dx * dx + dy * dy) + 1e-8

                    px = -dy / norm * 0.045
                    py = dx / norm * 0.045

                    thick_line = [
                        (x1 + px, y1 + py),
                        (x2 + px, y2 + py),
                        (x2 - px, y2 - py),
                        (x1 - px, y1 - py),
                    ]

                    base.road_poly.append((thick_line, color))

        self._added_to_road = True

    def _nearest_track_index(self):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return None

        car_x = car.hull.position[0]
        car_y = car.hull.position[1]

        best_idx = 0
        best_dist = float("inf")

        for i, (_, _, x, y) in enumerate(track):
            d = (x - car_x) ** 2 + (y - car_y) ** 2

            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def _track_index_delta(self, current_idx):
        if current_idx is None or self.prev_track_idx is None:
            return 0, True

        track = getattr(self.env.unwrapped, "track", None)

        if track is None or len(track) == 0:
            return 0, True

        n = len(track)

        forward_delta = (current_idx - self.prev_track_idx) % n
        backward_delta = (self.prev_track_idx - current_idx) % n

        # If the car barely moved or nearest tile jitters, we treat it as neutral
        if forward_delta == 0 or min(forward_delta, backward_delta) <= 1:
            return 0, True

        if forward_delta < backward_delta:
            self.forward_index_steps += 1
            self.backward_index_counter = max(0, self.backward_index_counter - 1)
            return int(forward_delta), True

        self.backward_index_steps += 1
        self.backward_index_counter += 1
        return -int(backward_delta), False

    def _track_lateral_position(self, idx=None):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return 0.0

        if idx is None:
            idx = self._nearest_track_index()

        if idx is None:
            return 0.0

        car_x = car.hull.position[0]
        car_y = car.hull.position[1]

        _, beta, tx, ty = track[idx]

        nx = -np.sin(beta)
        ny = np.cos(beta)

        return float((car_x - tx) * nx + (car_y - ty) * ny)

    def _heading_error_to_track(self, idx=None):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return 0.0

        if idx is None:
            idx = self._nearest_track_index()

        if idx is None:
            return 0.0

        _, beta, _, _ = track[idx]
        car_angle = car.hull.angle

        diff = beta - car_angle
        diff = angle_normalize(diff)

        return float(diff)

    def _heading_alignment_dot(self, idx=None):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return 1.0

        if idx is None:
            idx = self._nearest_track_index()

        if idx is None:
            return 1.0

        _, beta, _, _ = track[idx]

        # Car forward vector from hull angle
        car_angle = car.hull.angle
        car_fx = np.cos(car_angle)
        car_fy = np.sin(car_angle)

        # Track the forward vector
        track_fx = np.cos(beta)
        track_fy = np.sin(beta)

        return float(car_fx * track_fx + car_fy * track_fy)

    def _is_wrong_way(self, index_delta, moving_forward_by_index, heading_dot, speed):
        if speed < 3.0:
            return False

        heading_wrong = heading_dot < WRONG_WAY_DOT_THRESHOLD
        index_wrong = (not moving_forward_by_index) and index_delta < 0

        if index_wrong and self.backward_index_counter >= 3:
            return True

        if heading_wrong and self.backward_index_counter >= 2:
            return True

        if self.backward_index_counter >= BACKWARD_INDEX_COUNTER_LIMIT:
            return True

        return False

    def _car_speed(self):
        base = self.env.unwrapped
        car = getattr(base, "car", None)

        if car is None:
            return 0.0

        v = car.hull.linearVelocity
        return float(np.sqrt(v[0] ** 2 + v[1] ** 2))

    def _track_progress(self):
        base = self.env.unwrapped
        track = getattr(base, "track", None)

        if track is None or len(track) == 0:
            return 0.0

        visited = getattr(base, "tile_visited_count", 0)
        total = len(track)

        return float(visited / max(1, total))

    def _check_collision(self, return_obstacle=False):
        base = self.env.unwrapped
        car = getattr(base, "car", None)

        if car is None:
            return (False, None) if return_obstacle else False

        car_x = car.hull.position[0]
        car_y = car.hull.position[1]

        for ob in self.obstacles:
            d = np.sqrt((car_x - ob["x"]) ** 2 + (car_y - ob["y"]) ** 2)

            # We use a slightly bigger collision radius to avoid visually passing through obstacles
            if d < CAR_COLLISION_RADIUS + ob["r"]:
                return (True, ob) if return_obstacle else True

        return (False, None) if return_obstacle else False

    def nearest_forward_obstacle(self, lookahead=26):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return None

        nearest_idx = self._nearest_track_index()

        if nearest_idx is None:
            return None

        n = len(track)
        best_ob = None
        best_delta = 10**9

        for ob in self.obstacles:
            delta = (ob["track_index"] - nearest_idx) % n

            if 0 < delta <= lookahead and delta < best_delta:
                best_delta = delta
                best_ob = ob

        return best_ob


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_index, reward, next_state, done):
        reward = float(np.clip(reward, -80.0, 120.0))
        self.buffer.append((state, action_index, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE)
            conv_dim = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(NUM_ACTIONS).to(DEVICE)
        self.target_net = DQN(NUM_ACTIONS).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.steps_done = 0

    def epsilon(self):
        return EPSILON_END + (EPSILON_START - EPSILON_END) * max(
            0.0,
            (EPSILON_DECAY_STEPS - self.steps_done) / EPSILON_DECAY_STEPS
        )

    def select_action(self, state, training=True):
        if training:
            eps = self.epsilon()
            self.steps_done += 1

            if random.random() < eps:
                return random.randrange(NUM_ACTIONS)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def train_step(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * (1.0 - dones) * next_q
            target_q = torch.clamp(target_q, -100.0, 150.0)

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.steps_done = ckpt.get("steps_done", 0)


def make_env(debug=False):
    base_env = gym.make(
        ENV_ID,
        render_mode="rgb_array",
        continuous=True,
        domain_randomize=False,
        lap_complete_percent=0.95,
    )

    env = VisibleTrackObstacleCarRacing(
        base_env,
        num_obstacles=12,
        obstacle_radius=OBSTACLE_RADIUS,
        collision_penalty=COLLISION_PENALTY,
        obstacle_step_penalty=-0.0005,
        debug=debug,
    )

    return env


def closest_action_index(action):
    distances = [np.linalg.norm(action - a) for a in ACTIONS]
    return int(np.argmin(distances))


def simple_centerline_action(env):
    base = env.unwrapped
    car = getattr(base, "car", None)
    track = getattr(base, "track", None)

    if car is None or track is None or len(track) == 0:
        return np.array([0.0, 0.25, 0.0], dtype=np.float32)

    car_x = car.hull.position[0]
    car_y = car.hull.position[1]
    car_angle = car.hull.angle

    dists = []

    for _, _, x, y in track:
        dists.append((x - car_x) ** 2 + (y - car_y) ** 2)

    nearest = int(np.argmin(dists))

    speed = env._car_speed() if hasattr(env, "_car_speed") else 0.0
    lookahead = int(np.clip(8 + speed * 0.25, 8, 18))

    target = (nearest + lookahead) % len(track)
    _, _, tx, ty = track[target]

    desired_angle = np.arctan2(ty - car_y, tx - car_x)
    angle_diff = angle_normalize(desired_angle - car_angle)

    steering = float(np.clip(angle_diff * 0.85, -0.55, 0.55))

    lateral_signed = 0.0

    if hasattr(env, "_track_lateral_position"):
        lateral_signed = env._track_lateral_position(nearest)

    if lateral_signed > 1.20:
        steering -= 0.25
    elif lateral_signed < -1.20:
        steering += 0.25

    if lateral_signed > 1.80:
        steering -= 0.45
    elif lateral_signed < -1.80:
        steering += 0.45

    nearest_ob = None

    if hasattr(env, "nearest_forward_obstacle"):
        nearest_ob = env.nearest_forward_obstacle(lookahead=34)

    obstacle_close = False

    if nearest_ob is not None:
        dx = nearest_ob["x"] - car_x
        dy = nearest_ob["y"] - car_y
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 12.0:
            obstacle_close = True
            ob_offset = nearest_ob["offset"]

            if ob_offset > 0:
                steering -= 0.28
            else:
                steering += 0.28

            if lateral_signed > 1.60:
                steering -= 0.55
            elif lateral_signed < -1.60:
                steering += 0.55

    steering = float(np.clip(steering, -0.55, 0.55))

    if obstacle_close:
        gas = 0.10
        brake = 0.18
    elif abs(steering) < 0.12:
        gas = 0.30
        brake = 0.00
    elif abs(steering) < 0.32:
        gas = 0.22
        brake = 0.03
    else:
        gas = 0.08
        brake = 0.12

    return np.array([steering, gas, brake], dtype=np.float32)


def upscale_frames(frames, scale=4):
    high_quality = []

    for frame in frames:
        enlarged = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        high_quality.append(enlarged)

    return high_quality


def save_video(frames, path, fps=30, scale=4):
    if len(frames) == 0:
        print(f"[WARNING] No frames to save for {path}")
        return

    frames = upscale_frames(frames, scale=scale)
    imageio.mimsave(path, frames, fps=fps, quality=9, macro_block_size=16)
    print(f"[VIDEO SAVE] Saved video to {path}")


def save_best_metadata(metadata, path=BEST_METADATA_PATH):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[METADATA SAVE] Saved best episode metadata to {path}")


def warmup_replay_buffer(env, stacker, replay_buffer, steps=WARMUP_STEPS):
    print(f"[WARMUP] Collecting {steps} obstacle-aware expert steps...")

    obs, info = env.reset(seed=SEED)
    state = stacker.reset(obs)

    collected = 0
    episode_seed = SEED

    while collected < steps:
        expert_action = simple_centerline_action(env)
        action_index = closest_action_index(expert_action)
        action = ACTIONS[action_index]

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = stacker.step(next_obs)

        done = terminated or truncated

        replay_buffer.push(state, action_index, reward, next_state, done)

        state = next_state
        collected += 1

        if collected % 5000 == 0:
            print(f"[WARMUP] collected={collected}/{steps}")

        if done:
            episode_seed += 1
            obs, info = env.reset(seed=episode_seed)
            state = stacker.reset(obs)

    print(f"[WARMUP] Done. Buffer size: {len(replay_buffer)}")


def should_save_episode(
    episode_reward,
    max_progress,
    lap_complete,
    hit_obstacle,
    off_track,
    wrong_way_terminated,
    best_progress,
    best_clean_progress,
):
    clean_episode = (not off_track) and (not hit_obstacle) and (not wrong_way_terminated)
    valid_episode = (not off_track) and (not wrong_way_terminated)

    save_score = (
        12000.0 * max_progress
        + episode_reward
        - (1200.0 if hit_obstacle else 0.0)
        - (3000.0 if off_track else 0.0)
        - (2500.0 if wrong_way_terminated else 0.0)
    )

    if not valid_episode:
        return False, "not saved: invalid episode", save_score

    if lap_complete and clean_episode:
        return True, "clean lap complete", save_score

    if clean_episode and max_progress > best_clean_progress:
        return True, "best clean progress", save_score

    if (not hit_obstacle) and max_progress > best_progress + 0.05:
        return True, "best progress no hit", save_score

    return False, "not saved: no improvement", save_score


def train():
    env = make_env(debug=False)
    stacker = FrameStack(FRAME_STACK)

    agent = DQNAgent()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    warmup_replay_buffer(env, stacker, replay_buffer)

    best_score = -float("inf")
    best_progress = 0.0
    best_clean_progress = 0.0
    best_episode = None

    global_step = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        episode_seed = SEED + episode

        obs, info = env.reset(seed=episode_seed)
        state = stacker.reset(obs)

        episode_frames = []
        episode_frames.append(obs.copy())

        episode_reward = 0.0
        hit_obstacle = False
        off_track = False
        wrong_way_terminated = False
        last_loss = None
        lap_complete = False
        max_progress = 0.0

        lateral_sum = 0.0
        speed_sum = 0.0
        wrong_way_steps = 0

        action_counts = np.zeros(NUM_ACTIONS, dtype=np.int32)

        for step in range(MAX_STEPS):
            global_step += 1

            action_index = agent.select_action(state, training=True)
            action = ACTIONS[action_index]
            action_counts[action_index] += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = stacker.step(next_obs)

            episode_frames.append(next_obs.copy())

            done = terminated or truncated

            replay_buffer.push(state, action_index, reward, next_state, done)

            state = next_state
            episode_reward += reward

            lateral_sum += info.get("lateral", 0.0)
            speed_sum += info.get("speed", 0.0)
            max_progress = max(max_progress, info.get("progress", 0.0))

            if info.get("wrong_way", False):
                wrong_way_steps += 1

            if info.get("lap_complete", False):
                lap_complete = True

            if global_step > START_LEARNING_AFTER and global_step % TRAIN_EVERY == 0:
                last_loss = agent.train_step(replay_buffer)

            if global_step % TARGET_UPDATE_EVERY == 0:
                agent.update_target()

            if info.get("hit_obstacle", False):
                hit_obstacle = True

            if info.get("off_track_terminated", False):
                off_track = True

            if info.get("wrong_way_terminated", False):
                wrong_way_terminated = True

            if done:
                break

        avg_lateral = lateral_sum / max(1, step + 1)
        avg_speed = speed_sum / max(1, step + 1)

        should_save, save_reason, save_score = should_save_episode(
            episode_reward=episode_reward,
            max_progress=max_progress,
            lap_complete=lap_complete,
            hit_obstacle=hit_obstacle,
            off_track=off_track,
            wrong_way_terminated=wrong_way_terminated,
            best_progress=best_progress,
            best_clean_progress=best_clean_progress,
        )

        if should_save:
            best_score = save_score
            best_progress = max(best_progress, max_progress)

            if not hit_obstacle and not off_track and not wrong_way_terminated:
                best_clean_progress = max(best_clean_progress, max_progress)

            best_episode = episode

            agent.save(MODEL_PATH)

            save_video(
                episode_frames,
                BEST_EPISODE_VIDEO_PATH,
                fps=30,
                scale=4,
            )

            metadata = {
                "episode": episode,
                "seed": episode_seed,
                "reward": episode_reward,
                "save_score": save_score,
                "progress_percent": max_progress * 100.0,
                "lap_complete": lap_complete,
                "hit_obstacle": hit_obstacle,
                "off_track": off_track,
                "wrong_way_terminated": wrong_way_terminated,
                "wrong_way_steps": wrong_way_steps,
                "steps": step + 1,
                "avg_lateral": avg_lateral,
                "avg_speed": avg_speed,
                "epsilon": agent.epsilon(),
                "save_reason": save_reason,
                "model_path": MODEL_PATH,
                "best_episode_video_path": BEST_EPISODE_VIDEO_PATH,
                "action_counts": action_counts.tolist(),
            }

            save_best_metadata(metadata)

            print(
                f"[SAVE] reason={save_reason} | "
                f"episode={episode} | "
                f"seed={episode_seed} | "
                f"progress={max_progress * 100:.2f}% | "
                f"reward={episode_reward:.2f} | "
                f"save_score={save_score:.2f} | "
                f"hit={hit_obstacle} | "
                f"offtrack={off_track} | "
                f"wrongterm={wrong_way_terminated} | "
                f"best_video={BEST_EPISODE_VIDEO_PATH}"
            )

        print(
            f"Episode {episode:04d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"SaveScore: {save_score:8.2f} | "
            f"BestScore: {best_score:8.2f} | "
            f"BestProg: {best_progress * 100:6.2f}% | "
            f"BestCleanProg: {best_clean_progress * 100:6.2f}% | "
            f"BestEp: {best_episode} | "
            f"Steps: {step + 1:4d} | "
            f"Progress: {max_progress * 100:6.2f}% | "
            f"LapDone: {lap_complete} | "
            f"Epsilon: {agent.epsilon():.3f} | "
            f"Hit: {hit_obstacle} | "
            f"OffTrack: {off_track} | "
            f"WrongTerm: {wrong_way_terminated} | "
            f"WrongSteps: {wrong_way_steps} | "
            f"AvgLat: {avg_lateral:.2f} | "
            f"AvgSpeed: {avg_speed:.2f} | "
            f"Buffer: {len(replay_buffer)} | "
            f"Loss: {last_loss} | "
            f"SaveDecision: {save_reason}"
        )

    env.close()

    if best_episode is None:
        print("[WARNING] No good model was saved. Saving final model instead.")
        agent.save(MODEL_PATH)

    print("Training complete.")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Best episode video saved to: {BEST_EPISODE_VIDEO_PATH}")
    print(f"Best metadata saved to: {BEST_METADATA_PATH}")


def record_debug_video(video_path="debug_visible_obstacles_v4.mp4"):
    env = make_env(debug=True)
    frames = []

    obs, info = env.reset(seed=SEED)
    frames.append(obs.copy())

    total_reward = 0.0
    hit_obstacle = False
    off_track = False
    wrong_way_terminated = False
    max_progress = 0.0

    for step in range(900):
        action = simple_centerline_action(env)
        obs, reward, terminated, truncated, info = env.step(action)

        frames.append(obs.copy())

        total_reward += reward
        max_progress = max(max_progress, info.get("progress", 0.0))

        if info.get("hit_obstacle", False):
            hit_obstacle = True

        if info.get("off_track_terminated", False):
            off_track = True

        if info.get("wrong_way_terminated", False):
            wrong_way_terminated = True

        if terminated or truncated:
            break

    env.close()

    save_video(frames, video_path, fps=30, scale=4)

    print(
        f"[DEBUG VIDEO] reward={total_reward:.2f}, "
        f"steps={step + 1}, "
        f"progress={max_progress * 100:.2f}%, "
        f"hit_obstacle={hit_obstacle}, "
        f"off_track={off_track}, "
        f"wrong_way_terminated={wrong_way_terminated}"
    )


def record_trained_video(video_path=VIDEO_PATH, seed=SEED):
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
        print("[WARNING] Skipping trained video.")
        return

    env = make_env(debug=True)
    stacker = FrameStack(FRAME_STACK)

    agent = DQNAgent()
    agent.load(MODEL_PATH)

    frames = []

    obs, info = env.reset(seed=seed)
    state = stacker.reset(obs)

    frames.append(obs.copy())

    total_reward = 0.0
    hit_obstacle = False
    off_track = False
    wrong_way_terminated = False
    max_progress = 0.0
    lap_complete = False

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int32)

    for step in range(MAX_STEPS):
        action_index = agent.select_action(state, training=False)
        action = ACTIONS[action_index]
        action_counts[action_index] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        state = stacker.step(obs)

        frames.append(obs.copy())

        total_reward += reward
        max_progress = max(max_progress, info.get("progress", 0.0))

        if info.get("lap_complete", False):
            lap_complete = True

        if info.get("hit_obstacle", False):
            hit_obstacle = True

        if info.get("off_track_terminated", False):
            off_track = True

        if info.get("wrong_way_terminated", False):
            wrong_way_terminated = True

        if terminated or truncated:
            break

    env.close()

    save_video(frames, video_path, fps=30, scale=4)

    print(
        f"[VIDEO REPLAY] reward={total_reward:.2f}, "
        f"steps={step + 1}, "
        f"progress={max_progress * 100:.2f}%, "
        f"lap_complete={lap_complete}, "
        f"hit_obstacle={hit_obstacle}, "
        f"off_track={off_track}, "
        f"wrong_way_terminated={wrong_way_terminated}, "
        f"seed={seed}, "
        f"action_counts={action_counts.tolist()}"
    )

    print(f"Saved trained replay video to {video_path}")
    print(f"Actual best training episode video is: {BEST_EPISODE_VIDEO_PATH}")


if __name__ == "__main__":
    set_seeds(SEED)

    print("Using device:", DEVICE)

    record_debug_video()
    train()

    # This is only a fresh replay of the saved model, the actual best episode video is saved during the training itself
    record_trained_video()
