#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from uav_routing_env import UAVRoutingEnv

def main():
    base_path = os.path.expanduser("~/px4_ws/src/uav_sim_env/scripts")
    model_file = os.path.join(base_path, "drl_agent/best_model.zip")
    log_file = os.path.join(base_path, "drl_agent/logs/progress.csv")

    if not os.path.exists(model_file):
        print("Model not found! Run train_agent.py first.")
        return

    # Initialize environment for dynamic random testing
    # Initialize environment for dynamic random testing
    env = UAVRoutingEnv(csv_file=None)
    
    # --- BUDGET OVERRIDE ---
    # Force the environment to use a strict 60-second battery limit for this test
    env.max_battery_time = 60.0 
    
    model = PPO.load(model_file, env=env)

    print("\n" + "="*50)
    print("🧠 DYNAMIC DOMAIN GENERALIZATION TEST (3 Scenarios)")
    print("="*50)
    
    scenarios_data = []

    # Run 3 completely different episodes
    for scenario_idx in range(3):
        print(f"Simulating Scenario {scenario_idx + 1}...")
        obs, _ = env.reset()
        
        route_x, route_y = [], []
        total_reward = 0
        done = False
        wind_x, wind_y = env.wind_vector
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            # PLOTTER FIX: Only draw a line if the reward was strictly positive (a valid move)
            if reward > 0.0:
                route_x.append(env.drone_pos[0])
                route_y.append(env.drone_pos[1])
                
            total_reward += reward
            done = terminated or truncated

        # Extract only the real targets for plotting
        active_idx = np.where(env.active_mask == 1.0)[0]
        real_targets = env.targets[active_idx].copy()
        real_scores = env.scores[active_idx].copy()

        scenarios_data.append({
            'route_x': route_x,
            'route_y': route_y,
            'wind_x': wind_x,
            'wind_y': wind_y,
            'targets': real_targets,
            'scores': real_scores,
            'reward': total_reward
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

        ax.plot(data['route_x'], data['route_y'], color='blue', linestyle='-', linewidth=2, alpha=0.8, label='DRL Flight Trajectory')
        
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

    # --- PLOT 4: Convergence ---
    if os.path.exists(log_file):
        df_logs = pd.read_csv(log_file)
        df_logs.columns = df_logs.columns.str.strip()
        
        if 'train/loss' in df_logs.columns and 'rollout/ep_rew_mean' in df_logs.columns:
            df_clean = df_logs.dropna(subset=['train/loss', 'rollout/ep_rew_mean'])
            ax_conv_twin = ax_conv.twinx()
            
            ln1 = ax_conv.plot(df_clean['time/total_timesteps'].to_numpy(), df_clean['train/loss'].to_numpy(), color='red', linewidth=2, label='Training Loss')
            ln2 = ax_conv_twin.plot(df_clean['time/total_timesteps'].to_numpy(), df_clean['rollout/ep_rew_mean'].to_numpy(), color='green', linewidth=2, label='Average Reward')
            
            ax_conv.set_title('Dynamic Convergence: Loss vs. Reward', fontsize=14, weight='bold')
            ax_conv.set_xlabel('Training Timesteps', fontsize=12)
            ax_conv.set_ylabel('Loss (Red)', fontsize=12, color='red')
            ax_conv_twin.set_ylabel('Average Episode Reward (Green)', fontsize=12, color='green')
            
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            ax_conv.legend(lns, labs, loc='upper left')
            ax_conv.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT 5: Table ---
    ax_table.axis('tight')
    ax_table.axis('off')
    ax_table.set_title("Agent & Environment Configuration", fontsize=14, weight='bold', pad=20)

    avg_score = sum([d['reward'] for d in scenarios_data]) / 3.0

    table_data = [
        ["Framework", "Stable Baselines3 (PyTorch)"],
        ["Algorithm", "PPO (Optuna NAS Optimized)"],
        ["Input Dimensions", str(env.observation_space.shape[0])],
        ["Action Dimensions", str(env.action_space.n)],
        ["Battery Constraint", f"{env.max_battery_time} seconds"],
        ["Wind Magnitude", f"{env.wind_speed_mag * 3.6:.1f} km/h"],
        ["Map Size", "100m x 100m"],
        ["Avg Score (3 Runs)", f"{avg_score:.2f}"]
    ]
    
    table = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    plt.tight_layout()
    plot_path = os.path.join(base_path, "drl_agent/thesis_analysis_dashboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight') 
    print(f"Dashboard saved to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()
