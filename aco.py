#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# 1. ENVIRONMENT DEFINITION
# ==========================================
class UAVRoutingEnv(gym.Env):
    def __init__(self, csv_file=None):
        super(UAVRoutingEnv, self).__init__()
        self.csv_file = csv_file
        self.max_targets = 50 
        self.map_bounds = 50.0 
        self.max_battery_time = 80.0 
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
                obs[idx:idx+2] = [200.0, 200.0]
                obs[idx+2] = -1.0 
                obs[idx+3] = 1.0
            else:
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

        if self.visited[action] == 1.0 or self.active_mask[action] == 0.0:
            reward = -50.0 
            self.battery_time_left -= 10.0 
            if self.battery_time_left <= 0:
                terminated = True
        else:
            target_pos = self.targets[action]
            movement_bonus = 250.0 
            time_penalty_multiplier = 5.0 
            
            if self.is_first_step:
                self.drone_pos = target_pos
                self.visited[action] = 1.0
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
                    reward = (self.scores[action] * 300.0) + movement_bonus - (time_required * time_penalty_multiplier)

        if np.all(self.visited[self.active_mask == 1.0] == 1.0):
            terminated = True
        if self.current_step >= self.max_targets * 4:
            truncated = True 

        return self._get_obs(), reward, terminated, truncated, {}

# ==========================================
# 2. EEPC-ACO ALGORITHM IMPLEMENTATION
# ==========================================
def eepc_aco(targets, scores, calc_cost, E, nt=50, na=30, alpha=1.0, beta=2.0, gamma=2.0, rho=0.1):
    """Algorithm 1: EEPC-ACO algorithm implementation"""
    n_p = len(targets)
    
    # Require: E, Λ, r, v_w (implicitly handled by calc_cost)
    # 1: Set c_hat <- inf, r_hat <- 0
    best_c = float('inf')
    best_r = 0.0
    best_path = []
    
    # 2: Calculate cost C_ij of all edges
    C = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(n_p):
            if i != j:
                C[i, j] = calc_cost(targets[i], targets[j])
            else:
                C[i, j] = float('inf')
                
    # Initialize Pheromones (\delta_ij)
    tau = np.ones((n_p, n_p)) 
    
    history_r = []
    history_c = []

    # 3: for l <- 1 to n_t do (Loop A)
    for l in range(nt):
        paths, path_rewards, path_costs = [], [], []
        
        # 4: for k <- 1 to n_a do (Loop B)
        for k in range(na):
            # 5: Initialize path as an empty set
            path = []
            # 6: Randomly choosing starting node from Λ
            curr_node = random.randint(0, n_p - 1)
            path.append(curr_node)
            
            # 7: c_tilde <- 0, r_tilde <- 0
            c_tilde = 0.0
            r_tilde = scores[curr_node]
            
            unvisited = set(range(n_p))
            unvisited.remove(curr_node)
            
            # 8: for i <- 1 to n_p do (Loop C)
            for step in range(n_p - 1):
                if not unvisited: break
                
                candidates = list(unvisited)
                probs = []
                
                # 9: for j <- 1 to n_p do (Loop D)
                for j in candidates:
                    # 10: Compute (\delta_ij)^\alpha (\zeta_ij)^\beta (r_j)^\gamma
                    delta = tau[curr_node, j] ** alpha
                    zeta = (1.0 / max(C[curr_node, j], 0.01)) ** beta
                    r_val = max(scores[j], 0.01) ** gamma
                    probs.append(delta * zeta * r_val)
                    
                # 12: Calculate probability p_ij
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                    # 13: Choose next node
                    next_node = np.random.choice(candidates, p=probs)
                else:
                    next_node = random.choice(candidates)
                
                step_cost = C[curr_node, next_node]
                
                # 15: if c_tilde >= E then break
                if c_tilde + step_cost >= E:
                    break 
                
                # 14: Compute cost of path
                c_tilde += step_cost
                r_tilde += scores[next_node]
                
                # 18: Update path R^k
                path.append(next_node)
                unvisited.remove(next_node)
                curr_node = next_node

            paths.append(path)
            path_rewards.append(r_tilde)
            path_costs.append(c_tilde)
            
            # 21: if r_tilde > r_hat or (r_tilde >= r_hat and c_tilde < c_hat)
            if r_tilde > best_r or (r_tilde == best_r and c_tilde < best_c):
                # 22-24: Update best path parameters
                best_path = path
                best_r = r_tilde
                best_c = c_tilde

        # 27: Update pheromone level on all edges
        tau = (1 - rho) * tau
        for p, r, c in zip(paths, path_rewards, path_costs):
            delta_tau = r / max(c, 1.0)
            for i in range(len(p) - 1):
                tau[p[i], p[i+1]] += delta_tau
                
        history_r.append(best_r)
        history_c.append(best_c if best_c != float('inf') else 0)

    # 29: return R_hat, c_hat, r_hat
    return best_path, best_c, best_r, history_r, history_c

# ==========================================
# 3. EVALUATION & DASHBOARD PLOTTING
# ==========================================
def main():
    # Initialize environment
    env = UAVRoutingEnv(csv_file=None)
    
    # --- BUDGET OVERRIDE ---
    env.max_battery_time = 40.0 

    print("\n" + "="*50)
    print("🐜 EEPC-ACO DOMAIN GENERALIZATION TEST (3 Scenarios)")
    print("="*50)
    
    scenarios_data = []
    all_conv_r = []
    all_conv_c = []

    # Run 3 completely different episodes
    for scenario_idx in range(3):
        print(f"Simulating Scenario {scenario_idx + 1} with ACO...")
        obs, _ = env.reset()
        
        active_idx = np.where(env.active_mask == 1.0)[0]
        real_targets = env.targets[active_idx].copy()
        real_scores = env.scores[active_idx].copy()

        # Execute EEPC-ACO
        best_path_indices, best_c, best_r, conv_r, conv_c = eepc_aco(
            targets=real_targets,
            scores=real_scores,
            calc_cost=env.calculate_travel_time,
            E=env.max_battery_time,
            nt=40, na=30, alpha=1.0, beta=2.0, gamma=2.0, rho=0.1
        )
        
        all_conv_r.append(conv_r)
        all_conv_c.append(conv_c)
        
        # Replay the chosen path in the Gym Environment to get Exact RL Rewards
        route_x, route_y = [], []
        total_env_reward = 0
        wind_x, wind_y = env.wind_vector
        
        for node_idx in best_path_indices:
            action = active_idx[node_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward > 0.0:
                route_x.append(env.drone_pos[0])
                route_y.append(env.drone_pos[1])
                
            total_env_reward += reward
            if terminated or truncated:
                break

        scenarios_data.append({
            'route_x': route_x,
            'route_y': route_y,
            'wind_x': wind_x,
            'wind_y': wind_y,
            'targets': real_targets,
            'scores': real_scores,
            'reward': total_env_reward
        })

    print("Simulations Complete! Generating Multi-Scenario Dashboard...")

    # Set up a large 2-row grid for the dashboard
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
    
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_conv = fig.add_subplot(gs[1, :2])
    ax_table = fig.add_subplot(gs[1, 2])

    # --- PLOT 1, 2, 3: The Three Random Scenarios ---
    for i, ax in enumerate(axes_top):
        data = scenarios_data[i]
        
        ax.scatter(data['targets'][:, 0], data['targets'][:, 1], color='green', s=data['scores']*200, label='Target', alpha=0.6)
        
        for j in range(len(data['targets'])):
            ax.text(data['targets'][j, 0] + 1.5, data['targets'][j, 1] + 1.5, f"R:{data['scores'][j]:.2f}", 
                     fontsize=8, fontweight='bold', color='darkgreen')

        ax.plot(data['route_x'], data['route_y'], color='blue', linestyle='-', linewidth=2, alpha=0.8, label='ACO Flight Trajectory')
        
        if len(data['route_x']) > 0:
            ax.scatter(data['route_x'][0], data['route_y'][0], color='cyan', marker='*', s=400, edgecolors='black', zorder=5, label='Start Node')
            ax.scatter(data['route_x'][-1], data['route_y'][-1], color='red', marker='X', s=200, edgecolors='black', zorder=5, label='End Node')
        
        wind_mag_kmh = np.linalg.norm([data['wind_x'], data['wind_y']]) * 3.6 
        
        ax.set_xlim([-50.0, 50.0])
        ax.set_ylim([-50.0, 50.0])
        
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        indicator_x = xmin + 0.05 * (xmax - xmin) 
        indicator_y = ymin + 0.95 * (ymax - ymin) 
        
        ax.quiver(indicator_x, indicator_y, data['wind_x'], data['wind_y'], color='purple', width=0.015, scale=50, 
                   alpha=0.8, zorder=5, label=f'Wind Direction ({wind_mag_kmh:.0f} km/h)')
        
        ax.set_title(f'Scenario {i+1} (Score: {data["reward"]:.2f})', fontsize=14, weight='bold')
        ax.set_xlabel('Gazebo X (meters)')
        ax.set_ylabel('Gazebo Y (meters)')
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT 4: Convergence (Adapted for ACO) ---
    avg_conv_r = np.mean(all_conv_r, axis=0)
    avg_conv_c = np.mean(all_conv_c, axis=0)
    iterations = np.arange(1, len(avg_conv_r) + 1)
    
    ax_conv_twin = ax_conv.twinx()
    
    ln1 = ax_conv.plot(iterations, avg_conv_c, color='red', linewidth=2, label='Energy Used (Time)')
    ln2 = ax_conv_twin.plot(iterations, avg_conv_r, color='green', linewidth=2, label='Collected Path Score')
    
    ax_conv.set_title('EEPC-ACO Convergence Profile (Average over 3 Scenarios)', fontsize=14, weight='bold')
    ax_conv.set_xlabel('ACO Iterations ($n_t$)', fontsize=12)
    ax_conv.set_ylabel('Avg Best Cost / Time (Red)', fontsize=12, color='red')
    ax_conv_twin.set_ylabel('Avg Best Total Score (Green)', fontsize=12, color='green')
    
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_conv.legend(lns, labs, loc='center right')
    ax_conv.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT 5: Table (Adapted for ACO) ---
    ax_table.axis('tight')
    ax_table.axis('off')
    ax_table.set_title("Solver & Environment Configuration", fontsize=14, weight='bold', pad=20)

    avg_score = sum([d['reward'] for d in scenarios_data]) / 3.0

    # -> FIX IS HERE: Added 'r' prefixes to strings containing LaTeX backslashes <-
    table_data = [
        ["Algorithm", "EEPC-ACO (Algorithm 1)"],
        ["Heuristics", r"$\alpha=1.0, \beta=2.0, \gamma=2.0$"],
        [r"Evaporation ($\rho$)", "0.1"],
        [r"Ants ($n_a$) / Iters ($n_t$)", "30 / 40"],
        ["Battery Constraint", f"{env.max_battery_time} seconds"],
        ["Wind Magnitude", f"{env.wind_speed_mag * 3.6:.1f} km/h"],
        ["Map Size", "100m x 100m"],
        ["Avg Env Score (3 Runs)", f"{avg_score:.2f}"]
    ]
    
    table = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    plt.tight_layout()
    plot_path = "aco_thesis_analysis_dashboard.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight') 
    print(f"Dashboard saved to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()
