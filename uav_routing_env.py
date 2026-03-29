import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import os

class UAVRoutingEnv(gym.Env):
    def __init__(self, csv_file=None):
        super(UAVRoutingEnv, self).__init__()
        
        self.csv_file = csv_file
        self.max_targets = 50 
        self.map_bounds = 50.0 
        
        self.max_battery_time = 120.0 
        self.uav_airspeed = 8.0 
        self.wind_speed_mag = 5.55 # 20 km/h
        
        self.action_space = spaces.Discrete(self.max_targets)
        
        obs_size = 5 + (self.max_targets * 4)
        self.observation_space = spaces.Box(
            low=-200.0, high=200.0, shape=(obs_size,), dtype=np.float32
        )

    def _load_or_generate_targets(self):
        targets = np.zeros((self.max_targets, 2), dtype=np.float32)
        scores = np.zeros(self.max_targets, dtype=np.float32)
        active_mask = np.zeros(self.max_targets, dtype=np.float32)
        
        if self.csv_file is not None and os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            num_real = min(len(df), self.max_targets)
            targets[:num_real] = df[['Gazebo_X', 'Gazebo_Y']].values[:num_real]
            scores[:num_real] = df['Score'].values[:num_real]
            active_mask[:num_real] = 1.0
        else:
            num_real = random.randint(20, self.max_targets)
            for i in range(num_real):
                targets[i] = [
                    random.uniform(-self.map_bounds, self.map_bounds),
                    random.uniform(-self.map_bounds, self.map_bounds)
                ]
                scores[i] = round(random.uniform(0.1, 1.0), 2)
                active_mask[i] = 1.0
                
        for i in range(num_real, self.max_targets):
            targets[i] = [999.0, 999.0] 
            scores[i] = 0.0
            
        sort_indices = np.argsort(scores)[::-1]
        self.targets = targets[sort_indices]
        self.scores = scores[sort_indices]
        self.active_mask = active_mask[sort_indices]
        
        return num_real

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([0.0, 0.0]) 
        self.battery_time_left = self.max_battery_time
        self.current_step = 0
        self.is_first_step = True 
        
        self.num_real_targets = self._load_or_generate_targets()
        
        self.visited = np.zeros(self.max_targets, dtype=np.float32)
        for i in range(self.max_targets):
            if self.active_mask[i] == 0.0:
                self.visited[i] = 1.0 
        
        theta = random.uniform(0, 2 * np.pi)
        self.wind_vector = np.array([
            self.wind_speed_mag * np.cos(theta),
            self.wind_speed_mag * np.sin(theta)
        ], dtype=np.float32)
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0:2] = self.drone_pos
        obs[2] = self.battery_time_left
        obs[3:5] = self.wind_vector 
        
        for i in range(self.max_targets):
            idx = 5 + (i * 4)
            if self.visited[i] == 1.0 or self.active_mask[i] == 0.0:
                # EGO-CENTRIC VISION: Push dead/visited cubes 200m away
                obs[idx:idx+2] = [200.0, 200.0]
                obs[idx+2] = -1.0 
                obs[idx+3] = 1.0
            else:
                # EGO-CENTRIC VISION: Provide coordinates relative to the drone
                obs[idx:idx+2] = self.targets[i] - self.drone_pos
                obs[idx+2] = self.scores[i]
                obs[idx+3] = 0.0
        return obs

    def calculate_travel_time(self, start_pos, end_pos):
        delta = end_pos - start_pos
        dist = np.linalg.norm(delta)
        if dist < 0.01: return 0.0
        
        u = delta / dist 
        w_dot_u = np.dot(self.wind_vector, u)
        wind_mag_sq = np.dot(self.wind_vector, self.wind_vector)
        
        discriminant = self.uav_airspeed**2 - wind_mag_sq + w_dot_u**2
        if discriminant < 0:
            vg = 0.5 
        else:
            vg = w_dot_u + np.sqrt(discriminant)
            
        vg = max(vg, 0.5) 
        return dist / vg

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        self.current_step += 1

        # THE WRIST SLAP (Unchanged)
        if self.visited[action] == 1.0 or self.active_mask[action] == 0.0:
            reward = -50.0 
            self.battery_time_left -= 10.0 
            if self.battery_time_left <= 0:
                terminated = True
        else:
            target_pos = self.targets[action]
            
            # --- THE NEW WIND & VOLUME ECONOMY ---
            movement_bonus = 250.0          # INCREASED: Makes it greedy for ANY cube
            time_penalty_multiplier = 5.0   # INCREASED: Makes it heavily prefer downwind paths
            
            if self.is_first_step:
                self.drone_pos = target_pos
                self.visited[action] = 1.0
                # DECREASED score multiplier to 300.0 so distance/wind matters more than point value
                reward = (self.scores[action] * 300.0) + movement_bonus 
                self.is_first_step = False
            else:
                time_required = self.calculate_travel_time(self.drone_pos, target_pos)
                
                if time_required > self.battery_time_left:
                    reward = -50.0 
                    terminated = True
                else:
                    self.drone_pos = target_pos
                    self.battery_time_left -= time_required
                    self.visited[action] = 1.0
                    
                    # The Final Formula
                    reward = (self.scores[action] * 300.0) + movement_bonus - (time_required * time_penalty_multiplier)

        if np.all(self.visited[self.active_mask == 1.0] == 1.0):
            terminated = True

        if self.current_step >= self.max_targets * 4:
            truncated = True 

        return self._get_obs(), reward, terminated, truncated, {}
