# car_racing_obstacles_dqn_fixed_visible_v2.py

import os
import cv2
import gymnasium as gym
import imageio
import random
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

TOTAL_EPISODES = 400
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

MODEL_PATH = "dqn_carracing_visible_obstacles_v2.pt"
VIDEO_PATH = "dqn_visible_obstacles_run_v2.mp4"


# DQN still chooses from fixed continuous actions.
# Expanded slightly, but still safe/smooth. Obstacles are unchanged.
ACTIONS = [
    np.array([0.00, 0.36, 0.00], dtype=np.float32),
    np.array([0.00, 0.24, 0.00], dtype=np.float32),
    np.array([0.00, 0.10, 0.18], dtype=np.float32),

    np.array([-0.08, 0.34, 0.00], dtype=np.float32),
    np.array([0.08, 0.34, 0.00], dtype=np.float32),

    np.array([-0.16, 0.30, 0.00], dtype=np.float32),
    np.array([0.16, 0.30, 0.00], dtype=np.float32),

    np.array([-0.28, 0.22, 0.02], dtype=np.float32),
    np.array([0.28, 0.22, 0.02], dtype=np.float32),

    np.array([-0.42, 0.12, 0.08], dtype=np.float32),
    np.array([0.42, 0.12, 0.08], dtype=np.float32),

    np.array([-0.55, 0.04, 0.18], dtype=np.float32),
    np.array([0.55, 0.04, 0.18], dtype=np.float32),
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


class VisibleTrackObstacleCarRacing(gym.Wrapper):
    def __init__(
        self,
        env,
        num_obstacles=12,
        obstacle_radius=0.32,
        collision_penalty=-35.0,
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
        self.collision_cooldown = 0
        self.prev_progress = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.obstacles = []
        self._added_to_road = False
        self.offtrack_counter = 0
        self.collision_cooldown = 0
        self.prev_progress = 0.0

        self._spawn_fixed_world_obstacles()
        self._add_obstacles_to_road_render()

        # Important: return rendered frame after adding visible obstacles.
        obs = self.env.render()

        if self.debug:
            base = self.env.unwrapped
            print("\n[DEBUG] fixed obstacles:", len(self.obstacles))
            print("[DEBUG] road_poly length:", len(getattr(base, "road_poly", [])))

            for i, ob in enumerate(self.obstacles[:8]):
                print(
                    f"[DEBUG] obstacle {i}: "
                    f"x={ob['x']:.2f}, y={ob['y']:.2f}, "
                    f"offset={ob['offset']:.2f}, r={ob['r']:.2f}, "
                    f"track_index={ob['track_index']}"
                )

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Always use current rendered frame after physics step.
        obs = self.env.render()

        lateral_signed = self._track_lateral_position()
        lateral = abs(lateral_signed)
        speed = self._car_speed()
        heading_error = abs(self._heading_error_to_track())
        progress = self._track_progress()

        progress_delta = progress - self.prev_progress
        if progress_delta < -0.50:
            progress_delta = 0.0
        progress_delta = max(0.0, progress_delta)
        self.prev_progress = progress

        hit = self._check_collision()

        info["lateral"] = lateral
        info["lateral_signed"] = lateral_signed
        info["speed"] = speed
        info["heading_error"] = heading_error
        info["progress"] = progress
        info["lap_complete"] = progress >= 0.95
        info["hit_obstacle"] = False
        info["off_track_terminated"] = False

        # Reward real progress directly. This is the main fix.
        reward += 3.0 * progress_delta

        # Collision should hurt, but should not instantly end training every time.
        # Cooldown prevents one collision from applying the penalty on 20 frames in a row.
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1

        if hit and self.collision_cooldown == 0:
            reward += self.collision_penalty
            self.collision_cooldown = 20
            info["hit_obstacle"] = True
        else:
            reward += self.obstacle_step_penalty

        # Give recovery chance instead of instant death.
        if lateral > 3.25:
            self.offtrack_counter += 1
            reward -= 2.0
        elif lateral > 2.75:
            self.offtrack_counter += 1
            reward -= 0.60
        else:
            self.offtrack_counter = max(0, self.offtrack_counter - 2)

        if self.offtrack_counter >= 25:
            reward -= 25.0
            terminated = True
            info["off_track_terminated"] = True

        # Centering reward: not too strict, but encourages stable driving.
        if lateral < 1.20:
            reward += 0.04
        elif lateral < 2.00:
            reward += 0.015
        elif lateral > 2.40:
            reward -= 0.15

        # Heading shaping.
        if heading_error < 0.25:
            reward += 0.035
        elif heading_error > 0.95:
            reward -= 0.20

        # Speed shaping. Softer than before because high speed was over-punished.
        if 8.0 <= speed <= 30.0:
            reward += 0.025
        elif 30.0 < speed <= 42.0:
            reward -= 0.04
        elif speed > 42.0:
            reward -= 0.12
        elif speed < 2.5:
            reward -= 0.03

        # Slow down near obstacles, but do not make the agent scared to move.
        near_ob = self.nearest_forward_obstacle(lookahead=22)
        if near_ob is not None and speed > 32.0:
            reward -= 0.15

        if progress >= 0.90:
            reward += 1.0

        if progress >= 0.95:
            reward += 100.0
            info["lap_complete"] = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def _spawn_fixed_world_obstacles(self):
        base = self.env.unwrapped
        track = getattr(base, "track", None)

        if track is None or len(track) < 100:
            return

        n = len(track)

        # Avoid very beginning and very end.
        start = int(0.16 * n)
        end = int(0.88 * n)

        obstacle_indices = np.linspace(start, end, self.num_obstacles, dtype=int)

        # Kept exactly the same obstacle offsets from your version.
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

            # Avoid spawning obstacles too close to each other.
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

        # Main wire.
        segments.append((
            [(x1, y1), (x2, y2)],
            (40, 40, 40)
        ))

        # Small sharp barbs.
        for s in [-0.65, -0.25, 0.20, 0.60]:
            bx = cx + s * length * tx
            by = cy + s * length * ty

            p1 = (bx - width * nx, by - width * ny)
            p2 = (bx + width * nx, by + width * ny)

            q1 = (bx - width * 0.75 * (tx + nx), by - width * 0.75 * (ty + ny))
            q2 = (bx + width * 0.75 * (tx + nx), by + width * 0.75 * (ty + ny))

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

            # Small warning base under the wire.
            warning = self._make_circle_polygon(ob["x"], ob["y"], ob["r"] * 1.45)
            base.road_poly.append((warning, (180, 40, 40)))

            # Draw barbed wire as multiple thin dark polygons.
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

    def _track_lateral_position(self):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return 0.0

        idx = self._nearest_track_index()

        if idx is None:
            return 0.0

        car_x = car.hull.position[0]
        car_y = car.hull.position[1]

        _, beta, tx, ty = track[idx]

        nx = -np.sin(beta)
        ny = np.cos(beta)

        return float((car_x - tx) * nx + (car_y - ty) * ny)

    def _heading_error_to_track(self):
        base = self.env.unwrapped
        car = getattr(base, "car", None)
        track = getattr(base, "track", None)

        if car is None or track is None or len(track) == 0:
            return 0.0

        idx = self._nearest_track_index()

        if idx is None:
            return 0.0

        _, beta, _, _ = track[idx]
        car_angle = car.hull.angle

        diff = beta - car_angle
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        return float(diff)

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

    def _check_collision(self):
        base = self.env.unwrapped
        car = getattr(base, "car", None)

        if car is None:
            return False

        car_x = car.hull.position[0]
        car_y = car.hull.position[1]

        car_radius = 0.75

        for ob in self.obstacles:
            d = np.sqrt((car_x - ob["x"]) ** 2 + (car_y - ob["y"]) ** 2)

            if d < car_radius + ob["r"]:
                return True

        return False

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
        # Small clipping prevents very large target spikes.
        reward = float(np.clip(reward, -50.0, 50.0))
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
            # Double DQN target: policy chooses, target evaluates.
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * (1.0 - dones) * next_q
            target_q = torch.clamp(target_q, -80.0, 120.0)

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        return loss.item()

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
        obstacle_radius=0.32,
        collision_penalty=-35.0,
        obstacle_step_penalty=-0.0005,
        debug=debug,
    )

    return env


def closest_action_index(action):
    distances = [np.linalg.norm(action - a) for a in ACTIONS]
    return int(np.argmin(distances))


def angle_normalize(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def simple_centerline_action(env):
    base = env.unwrapped
    car = getattr(base, "car", None)
    track = getattr(base, "track", None)

    if car is None or track is None or len(track) == 0:
        return np.array([0.0, 0.30, 0.0], dtype=np.float32)

    car_x = car.hull.position[0]
    car_y = car.hull.position[1]
    car_angle = car.hull.angle

    dists = []

    for _, _, x, y in track:
        dists.append((x - car_x) ** 2 + (y - car_y) ** 2)

    nearest = int(np.argmin(dists))

    target = (nearest + 8) % len(track)
    _, _, tx, ty = track[target]

    desired_angle = np.arctan2(ty - car_y, tx - car_x)
    angle_diff = angle_normalize(desired_angle - car_angle)

    steering = float(np.clip(angle_diff * 0.75, -0.52, 0.52))

    lateral_signed = 0.0

    if hasattr(env, "_track_lateral_position"):
        lateral_signed = env._track_lateral_position()

    # Strong center recovery before grass.
    if lateral_signed > 1.55:
        steering -= 0.35

    elif lateral_signed < -1.55:
        steering += 0.35

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

            # Avoid obstacle but do not over-dodge into grass.
            if ob_offset > 0:
                steering -= 0.25
            else:
                steering += 0.25

            # If already near edge, recover center first.
            if lateral_signed > 1.70:
                steering -= 0.45
            elif lateral_signed < -1.70:
                steering += 0.45

    steering = float(np.clip(steering, -0.55, 0.55))

    if obstacle_close:
        gas = 0.14
        brake = 0.10

    elif abs(steering) < 0.12:
        gas = 0.34
        brake = 0.00

    elif abs(steering) < 0.32:
        gas = 0.24
        brake = 0.02

    else:
        gas = 0.12
        brake = 0.08

    return np.array([steering, gas, brake], dtype=np.float32)


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

        if done:
            episode_seed += 1
            obs, info = env.reset(seed=episode_seed)
            state = stacker.reset(obs)

    print(f"[WARMUP] Done. Buffer size: {len(replay_buffer)}")


def train():
    env = make_env(debug=False)
    stacker = FrameStack(FRAME_STACK)

    agent = DQNAgent()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    warmup_replay_buffer(env, stacker, replay_buffer)

    best_score = -float("inf")
    global_step = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        episode_seed = SEED + episode

        obs, info = env.reset(seed=episode_seed)
        state = stacker.reset(obs)

        episode_reward = 0.0
        hit_obstacle = False
        off_track = False
        last_loss = None
        lap_complete = False
        max_progress = 0.0

        lateral_sum = 0.0
        speed_sum = 0.0

        for step in range(MAX_STEPS):
            global_step += 1

            action_index = agent.select_action(state, training=True)
            action = ACTIONS[action_index]

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = stacker.step(next_obs)

            done = terminated or truncated

            replay_buffer.push(state, action_index, reward, next_state, done)

            state = next_state
            episode_reward += reward

            lateral_sum += info.get("lateral", 0.0)
            speed_sum += info.get("speed", 0.0)
            max_progress = max(max_progress, info.get("progress", 0.0))

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

            if done:
                break

        avg_lateral = lateral_sum / max(1, step + 1)
        avg_speed = speed_sum / max(1, step + 1)

        # Save based mainly on progress, but still include reward.
        # This avoids saving a car that survives but barely moves.
        score_for_saving = episode_reward + 700.0 * max_progress

        if score_for_saving > best_score and not off_track:
            best_score = score_for_saving
            agent.save(MODEL_PATH)

        if lap_complete and not off_track:
            print("[SUCCESS] Lap completed. Saving successful model.")
            agent.save(MODEL_PATH)

        print(
            f"Episode {episode:04d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"BestScore: {best_score:8.2f} | "
            f"Steps: {step + 1:4d} | "
            f"Progress: {max_progress * 100:6.2f}% | "
            f"LapDone: {lap_complete} | "
            f"Epsilon: {agent.epsilon():.3f} | "
            f"Hit: {hit_obstacle} | "
            f"OffTrack: {off_track} | "
            f"AvgLat: {avg_lateral:.2f} | "
            f"AvgSpeed: {avg_speed:.2f} | "
            f"Buffer: {len(replay_buffer)} | "
            f"Loss: {last_loss}"
        )

    env.close()

    if best_score == -float("inf"):
        print("[WARNING] No best model was saved. Saving final model instead.")
        agent.save(MODEL_PATH)

    print(f"Training complete. Best model saved to {MODEL_PATH}")


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


def record_debug_video(video_path="debug_visible_obstacles_v2.mp4"):
    env = make_env(debug=True)
    frames = []

    obs, info = env.reset(seed=SEED)

    total_reward = 0.0
    hit_obstacle = False
    off_track = False
    max_progress = 0.0

    for step in range(900):
        action = simple_centerline_action(env)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        total_reward += reward
        max_progress = max(max_progress, info.get("progress", 0.0))

        if info.get("hit_obstacle", False):
            hit_obstacle = True

        if info.get("off_track_terminated", False):
            off_track = True

        if terminated or truncated:
            break

    env.close()

    frames = upscale_frames(frames, scale=4)
    imageio.mimsave(video_path, frames, fps=30, quality=9, macro_block_size=16)

    print(
        f"[DEBUG VIDEO] reward={total_reward:.2f}, "
        f"steps={step + 1}, "
        f"progress={max_progress * 100:.2f}%, "
        f"hit_obstacle={hit_obstacle}, "
        f"off_track={off_track}"
    )
    print(f"Saved debug video to {video_path}")


def record_trained_video(video_path=VIDEO_PATH):
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
        print("[WARNING] Skipping trained video.")
        return

    env = make_env(debug=True)
    stacker = FrameStack(FRAME_STACK)

    agent = DQNAgent()
    agent.load(MODEL_PATH)

    frames = []

    obs, info = env.reset(seed=SEED)
    state = stacker.reset(obs)

    total_reward = 0.0
    hit_obstacle = False
    off_track = False
    max_progress = 0.0
    lap_complete = False

    for step in range(MAX_STEPS):
        action_index = agent.select_action(state, training=False)
        action = ACTIONS[action_index]

        obs, reward, terminated, truncated, info = env.step(action)
        state = stacker.step(obs)

        frame = env.render()
        frames.append(frame)

        total_reward += reward
        max_progress = max(max_progress, info.get("progress", 0.0))

        if info.get("lap_complete", False):
            lap_complete = True

        if info.get("hit_obstacle", False):
            hit_obstacle = True

        if info.get("off_track_terminated", False):
            off_track = True

        if terminated or truncated:
            break

    env.close()

    frames = upscale_frames(frames, scale=4)
    imageio.mimsave(video_path, frames, fps=30, quality=9, macro_block_size=16)

    print(
        f"[VIDEO] reward={total_reward:.2f}, "
        f"steps={step + 1}, "
        f"progress={max_progress * 100:.2f}%, "
        f"lap_complete={lap_complete}, "
        f"hit_obstacle={hit_obstacle}, "
        f"off_track={off_track}"
    )
    print(f"Saved trained video to {video_path}")


if __name__ == "__main__":
    set_seeds(SEED)

    print("Using device:", DEVICE)

    record_debug_video()
    train()
    record_trained_video()
