# car_racing_obstacles_td3_visible_fixed.py
#
# Fixed TD3 version.
#
# Important fixes compared to previous TD3 file:
# 1. Keeps your obstacle rendering/spawn logic basically the same.
# 2. Adds a stronger expert controller.
# 3. Adds behavior cloning pretraining before TD3 training.
# 4. Starts with very low exploration noise.
# 5. Makes curriculum slower: 0 -> 2 -> 6 -> 12 obstacles.
# 6. Keeps reward simpler and gives off-track recovery chance.
# 7. Uses TD3 continuous control instead of DQN discrete actions.

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
ACTION_DIM = 3

TOTAL_EPISODES = 600
MAX_STEPS = 2000

BATCH_SIZE = 64
BUFFER_SIZE = 120_000
GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 5e-5
CRITIC_LR = 5e-4

WARMUP_STEPS = 25000
BC_STEPS = 3000
START_LEARNING_AFTER = 8000
TRAIN_EVERY = 2

POLICY_DELAY = 2
TARGET_NOISE_STD = 0.08
TARGET_NOISE_CLIP = 0.18

EXPLORATION_NOISE_START = 0.08
EXPLORATION_NOISE_END = 0.025
EXPLORATION_DECAY_STEPS = 180_000

MODEL_PATH = "td3_carracing_visible_obstacles_fixed.pt"
VIDEO_PATH = "td3_visible_obstacles_fixed_run.mp4"


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def curriculum_obstacle_count(episode):
    # Slower curriculum.
    # Your previous TD3 was failing at 0 obstacles, so do not rush obstacles.
    if episode <= 180:
        return 0
    if episode <= 300:
        return 2
    if episode <= 450:
        return 6
    return 12


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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


def clip_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[0] = np.clip(action[0], -1.0, 1.0)
    action[1] = np.clip(action[1], 0.0, 1.0)
    action[2] = np.clip(action[2], 0.0, 1.0)
    return action


class VisibleTrackObstacleCarRacing(gym.Wrapper):
    def __init__(
        self,
        env,
        num_obstacles=12,
        obstacle_radius=0.32,
        collision_penalty=-20.0,
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
        action = clip_action(action)
        obs, base_reward, terminated, truncated, info = self.env.step(action)
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
        info["progress_delta"] = progress_delta
        info["lap_complete"] = progress >= 0.95
        info["hit_obstacle"] = False
        info["off_track_terminated"] = False

        reward = float(base_reward)

        # Keep reward simple.
        reward += 0.40 * progress_delta * 100.0

        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1

        if hit and self.collision_cooldown == 0:
            reward += self.collision_penalty
            self.collision_cooldown = 20
            info["hit_obstacle"] = True
        else:
            reward += self.obstacle_step_penalty

        # More forgiving off-track logic.
        if lateral > 4.20:
            self.offtrack_counter += 1
            reward -= 0.04
        elif lateral > 3.20:
            self.offtrack_counter += 1
            reward -= 0.02
        else:
            self.offtrack_counter = max(0, self.offtrack_counter - 2)

        if self.offtrack_counter > 35:
            reward -= 5.0
            terminated = True
            info["off_track_terminated"] = True

        if lateral < 1.35:
            reward += 0.015
        elif lateral > 2.80:
            reward -= 0.02

        if 12.0 < speed < 35.0:
            reward += 0.01
        elif speed > 45.0:
            reward -= 0.03
        elif speed < 2.0:
            reward -= 0.02

        if heading_error < 0.35:
            reward += 0.01
        elif heading_error > 1.25:
            reward -= 0.03

        if progress >= 0.95:
            reward += 100.0
            info["lap_complete"] = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def _spawn_fixed_world_obstacles(self):
        base = self.env.unwrapped
        track = getattr(base, "track", None)

        if self.num_obstacles <= 0:
            return

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

        segments.append((
            [(x1, y1), (x2, y2)],
            (40, 40, 40)
        ))

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

            warning = self._make_circle_polygon(ob["x"], ob["y"], ob["r"] * 1.45)
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


def make_env(num_obstacles=12, debug=False):
    base_env = gym.make(
        ENV_ID,
        render_mode="rgb_array",
        continuous=True,
        domain_randomize=False,
        lap_complete_percent=0.95,
    )

    env = VisibleTrackObstacleCarRacing(
        base_env,
        num_obstacles=num_obstacles,
        obstacle_radius=0.32,
        collision_penalty=-20.0,
        obstacle_step_penalty=-0.0005,
        debug=debug,
    )

    return env


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        reward = float(np.clip(reward, -50.0, 50.0))
        action = clip_action(action)
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        return states, actions, rewards, next_states, dones

    def sample_states_actions(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, _, _, _ = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)

        return states, actions

    def __len__(self):
        return len(self.buffer)


class ConvEncoder(nn.Module):
    def __init__(self):
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
            self.out_dim = self.conv(dummy).shape[1]

    def forward(self, x):
        return self.conv(x)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvEncoder()

        self.fc = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.steer_head = nn.Linear(256, 1)
        self.gas_head = nn.Linear(256, 1)
        self.brake_head = nn.Linear(256, 1)

        # Better initial behavior:
        # low brake, moderate gas.
        nn.init.constant_(self.gas_head.bias, -0.6)
        nn.init.constant_(self.brake_head.bias, -4.0)
        nn.init.constant_(self.steer_head.bias, 0.0)

    def forward(self, state):
        z = self.encoder(state)
        z = self.fc(z)

        steer = torch.tanh(self.steer_head(z))

        # Limit gas range slightly so early policy does not go insane.
        gas = 0.05 + 0.55 * torch.sigmoid(self.gas_head(z))

        # Brake allowed, but starts near zero.
        brake = 0.45 * torch.sigmoid(self.brake_head(z))

        return torch.cat([steer, gas, brake], dim=1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvEncoder()

        self.q1 = nn.Sequential(
            nn.Linear(self.encoder.out_dim + ACTION_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(self.encoder.out_dim + ACTION_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        z = self.encoder(state)
        za = torch.cat([z, action], dim=1)
        return self.q1(za), self.q2(za)

    def q1_only(self, state, action):
        z = self.encoder(state)
        za = torch.cat([z, action], dim=1)
        return self.q1(za)


class TD3Agent:
    def __init__(self):
        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.total_steps = 0
        self.train_steps = 0

    def exploration_noise(self):
        frac = max(0.0, (EXPLORATION_DECAY_STEPS - self.total_steps) / EXPLORATION_DECAY_STEPS)
        return EXPLORATION_NOISE_END + (EXPLORATION_NOISE_START - EXPLORATION_NOISE_END) * frac

    def select_action(self, state, training=True):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if training:
            self.total_steps += 1
            noise_std = self.exploration_noise()

            noise = np.array([
                np.random.normal(0.0, noise_std),
                np.random.normal(0.0, noise_std * 0.35),
                np.random.normal(0.0, noise_std * 0.20),
            ], dtype=np.float32)

            action = action + noise

        return clip_action(action)

    def behavior_clone_step(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return None

        states, expert_actions = replay_buffer.sample_states_actions(BATCH_SIZE)
        pred_actions = self.actor(states)

        loss = F.mse_loss(pred_actions, expert_actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        self.actor_target.load_state_dict(self.actor.state_dict())

        return loss.item()

    def train_step(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return None, None

        self.train_steps += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            noise = torch.randn_like(actions) * TARGET_NOISE_STD
            noise[:, 0] = noise[:, 0].clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            noise[:, 1] = noise[:, 1].clamp(-TARGET_NOISE_CLIP * 0.5, TARGET_NOISE_CLIP * 0.5)
            noise[:, 2] = noise[:, 2].clamp(-TARGET_NOISE_CLIP * 0.5, TARGET_NOISE_CLIP * 0.5)

            next_actions = self.actor_target(next_states) + noise
            next_actions[:, 0] = next_actions[:, 0].clamp(-1.0, 1.0)
            next_actions[:, 1] = next_actions[:, 1].clamp(0.0, 1.0)
            next_actions[:, 2] = next_actions[:, 2].clamp(0.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + GAMMA * (1.0 - dones) * target_q
            target_q = torch.clamp(target_q, -80.0, 120.0)

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        actor_loss_value = None

        if self.train_steps % POLICY_DELAY == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_only(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

            actor_loss_value = actor_loss.item()

        return critic_loss.item(), actor_loss_value

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                TAU * source_param.data + (1.0 - TAU) * target_param.data
            )

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.train_steps = ckpt.get("train_steps", 0)


def angle_normalize(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def get_track_info(env):
    base = env.unwrapped
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

    return base, car, track, best_idx


def simple_centerline_action(env):
    info = get_track_info(env)

    if info is None:
        return np.array([0.0, 0.28, 0.0], dtype=np.float32)

    base, car, track, nearest = info

    car_x = car.hull.position[0]
    car_y = car.hull.position[1]
    car_angle = car.hull.angle

    # Look ahead more when moving faster.
    v = car.hull.linearVelocity
    speed = float(np.sqrt(v[0] ** 2 + v[1] ** 2))

    lookahead = int(np.clip(6 + speed * 0.20, 6, 16))
    target = (nearest + lookahead) % len(track)

    _, _, tx, ty = track[target]

    desired_angle = np.arctan2(ty - car_y, tx - car_x)
    angle_diff = angle_normalize(desired_angle - car_angle)

    # Main steering.
    steering = angle_diff * 0.95

    lateral_signed = 0.0

    if hasattr(env, "_track_lateral_position"):
        lateral_signed = env._track_lateral_position()

    # Use track normal to recover center, but not too aggressively.
    steering -= 0.18 * lateral_signed

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

            # If obstacle is on right side, steer slightly left. Opposite for left.
            if ob_offset > 0:
                steering -= 0.20
            else:
                steering += 0.20

            # But center recovery wins near edge.
            if lateral_signed > 1.70:
                steering -= 0.35
            elif lateral_signed < -1.70:
                steering += 0.35

    steering = float(np.clip(steering, -0.75, 0.75))

    abs_steer = abs(steering)

    # Better speed controller.
    if obstacle_close:
        target_speed = 17.0
    elif abs_steer < 0.12:
        target_speed = 31.0
    elif abs_steer < 0.32:
        target_speed = 24.0
    else:
        target_speed = 16.0

    if speed < target_speed - 3.0:
        gas = 0.42
        brake = 0.00
    elif speed < target_speed:
        gas = 0.28
        brake = 0.00
    elif speed > target_speed + 7.0:
        gas = 0.04
        brake = 0.18
    elif speed > target_speed + 3.0:
        gas = 0.08
        brake = 0.06
    else:
        gas = 0.18
        brake = 0.00

    return clip_action(np.array([steering, gas, brake], dtype=np.float32))


def evaluate_expert(num_obstacles=0, seed=SEED, max_steps=1000, video_path=None):
    env = make_env(num_obstacles=num_obstacles, debug=(num_obstacles == 12 and video_path is not None))
    frames = []

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    max_progress = 0.0
    hit_obstacle = False
    off_track = False

    for step in range(max_steps):
        action = simple_centerline_action(env)
        obs, reward, terminated, truncated, info = env.step(action)

        if video_path is not None:
            frames.append(env.render())

        total_reward += reward
        max_progress = max(max_progress, info.get("progress", 0.0))

        if info.get("hit_obstacle", False):
            hit_obstacle = True

        if info.get("off_track_terminated", False):
            off_track = True

        if terminated or truncated:
            break

    env.close()

    if video_path is not None and len(frames) > 0:
        frames = upscale_frames(frames, scale=4)
        imageio.mimsave(video_path, frames, fps=30, quality=9, macro_block_size=16)

    print(
        f"[EXPERT TEST] obstacles={num_obstacles}, "
        f"reward={total_reward:.2f}, "
        f"steps={step + 1}, "
        f"progress={max_progress * 100:.2f}%, "
        f"hit_obstacle={hit_obstacle}, "
        f"off_track={off_track}"
    )

    return max_progress, off_track, hit_obstacle


def warmup_replay_buffer(stacker, replay_buffer, steps=WARMUP_STEPS):
    print(f"[WARMUP] Collecting {steps} expert steps...")

    collected = 0
    episode_seed = SEED

    # Mostly clean driving first.
    warmup_choices = [0, 0, 0, 0, 2, 2, 6]

    env = make_env(num_obstacles=random.choice(warmup_choices), debug=False)
    obs, info = env.reset(seed=episode_seed)
    state = stacker.reset(obs)

    while collected < steps:
        action = simple_centerline_action(env)

        # Very small expert noise only.
        action = action + np.array([
            np.random.normal(0.0, 0.015),
            np.random.normal(0.0, 0.01),
            np.random.normal(0.0, 0.005),
        ], dtype=np.float32)

        action = clip_action(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = stacker.step(next_obs)

        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        collected += 1

        if done:
            env.close()
            episode_seed += 1
            env = make_env(num_obstacles=random.choice(warmup_choices), debug=False)
            obs, info = env.reset(seed=episode_seed)
            state = stacker.reset(obs)

    env.close()
    print(f"[WARMUP] Done. Buffer size: {len(replay_buffer)}")


def pretrain_actor_bc(agent, replay_buffer, steps=BC_STEPS):
    print(f"[BC] Pretraining actor for {steps} steps from expert replay...")

    last_loss = None

    for i in range(1, steps + 1):
        last_loss = agent.behavior_clone_step(replay_buffer)

        if i % 500 == 0:
            print(f"[BC] step={i}, loss={last_loss}")

    print(f"[BC] Done. Final loss={last_loss}")


def train():
    stacker = FrameStack(FRAME_STACK)
    agent = TD3Agent()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # First test expert. If this is bad, RL will be bad too.
    evaluate_expert(num_obstacles=0, seed=SEED, max_steps=1000, video_path="debug_expert_no_obstacles.mp4")
    evaluate_expert(num_obstacles=12, seed=SEED, max_steps=1000, video_path="debug_expert_12_obstacles.mp4")

    warmup_replay_buffer(stacker, replay_buffer)
    pretrain_actor_bc(agent, replay_buffer)

    best_score = -float("inf")
    global_step = 0

    for episode in range(1, TOTAL_EPISODES + 1):
        obstacle_count = curriculum_obstacle_count(episode)
        env = make_env(num_obstacles=obstacle_count, debug=False)

        episode_seed = SEED + episode

        obs, info = env.reset(seed=episode_seed)
        state = stacker.reset(obs)

        episode_reward = 0.0
        hit_obstacle = False
        off_track = False
        last_critic_loss = None
        last_actor_loss = None
        lap_complete = False
        max_progress = 0.0

        lateral_sum = 0.0
        speed_sum = 0.0

        for step in range(MAX_STEPS):
            global_step += 1

            # Early phase: mostly cloned policy, small expert fallback chance.
            if global_step < START_LEARNING_AFTER:
                if random.random() < 0.70:
                    action = simple_centerline_action(env)
                else:
                    action = agent.select_action(state, training=True)
            else:
                action = agent.select_action(state, training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = stacker.step(next_obs)

            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            lateral_sum += info.get("lateral", 0.0)
            speed_sum += info.get("speed", 0.0)
            max_progress = max(max_progress, info.get("progress", 0.0))

            if info.get("lap_complete", False):
                lap_complete = True

            if global_step > 1000 and global_step % TRAIN_EVERY == 0:
                last_critic_loss, maybe_actor_loss = agent.train_step(replay_buffer)

                if maybe_actor_loss is not None:
                    last_actor_loss = maybe_actor_loss

            if info.get("hit_obstacle", False):
                hit_obstacle = True

            if info.get("off_track_terminated", False):
                off_track = True

            if done:
                break

        env.close()

        avg_lateral = lateral_sum / max(1, step + 1)
        avg_speed = speed_sum / max(1, step + 1)

        score_for_saving = episode_reward + 1000.0 * max_progress

        if score_for_saving > best_score and not off_track:
            best_score = score_for_saving
            agent.save(MODEL_PATH)

        if lap_complete and not off_track:
            print("[SUCCESS] Lap completed. Saving successful TD3 model.")
            agent.save(MODEL_PATH)

        print(
            f"Episode {episode:04d} | "
            f"Obs: {obstacle_count:2d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"BestScore: {best_score:8.2f} | "
            f"Steps: {step + 1:4d} | "
            f"Progress: {max_progress * 100:6.2f}% | "
            f"LapDone: {lap_complete} | "
            f"Noise: {agent.exploration_noise():.3f} | "
            f"Hit: {hit_obstacle} | "
            f"OffTrack: {off_track} | "
            f"AvgLat: {avg_lateral:.2f} | "
            f"AvgSpeed: {avg_speed:.2f} | "
            f"Buffer: {len(replay_buffer)} | "
            f"CriticLoss: {last_critic_loss} | "
            f"ActorLoss: {last_actor_loss}"
        )

    if best_score == -float("inf"):
        print("[WARNING] No best model was saved. Saving final model instead.")
        agent.save(MODEL_PATH)

    print(f"Training complete. Best TD3 model saved to {MODEL_PATH}")


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


def record_trained_video(video_path=VIDEO_PATH, num_obstacles=12):
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
        print("[WARNING] Skipping trained video.")
        return

    env = make_env(num_obstacles=num_obstacles, debug=True)
    stacker = FrameStack(FRAME_STACK)

    agent = TD3Agent()
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
        action = agent.select_action(state, training=False)

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

    train()
    record_trained_video()