#!/usr/bin/env python3
import os
import shutil
import optuna
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
from uav_routing_env import UAVRoutingEnv

# File Paths
BASE_PATH = os.path.expanduser("~/px4_ws/src/uav_sim_env/scripts")
CSV_FILE = os.path.join(BASE_PATH, "target_coordinates.csv")
LOG_DIR = os.path.join(BASE_PATH, "drl_agent/logs/")
MODEL_DIR = os.path.join(BASE_PATH, "drl_agent/")

def optimize_agent(trial):
    """
    This function creates a unique neural network for each trial.
    Optuna will guess parameters, and this function tests them.
    """
    # 1. Optuna suggests a Learning Rate
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
    
    # 2. Optuna suggests a Neural Network Architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large", "deep"])
    net_arch_mapping = {
        "small": dict(pi=[64, 64], vf=[64, 64]),          # Default SB3
        "medium": dict(pi=[128, 128], vf=[128, 128]),     # Wider
        "large": dict(pi=[256, 256], vf=[256, 256]),      # Massive memory
        "deep": dict(pi=[128, 128, 128], vf=[128, 128, 128]) # 3 Layers deep
    }
    net_arch = net_arch_mapping[net_arch_type]
    
    # 3. Optuna suggests an Activation Function
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = nn.Tanh if activation_fn_name == "tanh" else nn.ReLU

    # Format the policy arguments
    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=net_arch
    )

    # Initialize a clean environment for this specific trial
    env = UAVRoutingEnv(csv_file=None)
    env = Monitor(env)

    # Build the PPO model with the trial's unique DNA
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, policy_kwargs=policy_kwargs, verbose=0)
    
    # Run a "Mini-Training" session (100,000 steps) to see if this brain is smart
    print(f"\n--- Trial {trial.number}: Testing {net_arch_type} network with LR {learning_rate:.5f} ({activation_fn_name}) ---")
    try:
        model.learn(total_timesteps=100000)
    except Exception as e:
        print(f"Trial failed (Likely unstable parameters): {e}")
        return -10000.0 # Return terrible score if it crashes

    # Evaluate the brain's performance on 5 random maps
    eval_env = UAVRoutingEnv(csv_file=None)
    eval_env = Monitor(eval_env)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
    
    print(f"Trial {trial.number} Result: {mean_reward:.2f} points")
    return mean_reward

def main():
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("="*50)
    print("🧬 COMMENCING NEURAL ARCHITECTURE SEARCH (NAS)")
    print("="*50)
    
    # Create the Optuna study (Goal is to MAXIMIZE the reward)
    study = optuna.create_study(direction="maximize")
    
    # Run 15 different trial combinations
    study.optimize(optimize_agent, n_trials=15)

    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE - WINNING DNA FOUND!")
    print("="*50)
    
    best_params = study.best_params
    print(f"Best Score: {study.best_value}")
    print(f"Best Learning Rate: {best_params['learning_rate']}")
    print(f"Best Architecture: {best_params['net_arch']}")
    print(f"Best Activation: {best_params['activation_fn']}")

    # ==========================================
    # FINAL TRAINING PHASE (Using the Best DNA)
    # ==========================================
    print("\n🚀 LAUNCHING FINAL 1.5M TIMESTEP TRAINING WITH WINNING DNA...")
    
    # Reconstruct the winning architecture
    net_arch_mapping = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[128, 128], vf=[128, 128]),
        "large": dict(pi=[256, 256], vf=[256, 256]),
        "deep": dict(pi=[128, 128, 128], vf=[128, 128, 128])
    }
    final_arch = net_arch_mapping[best_params['net_arch']]
    final_activation = nn.Tanh if best_params['activation_fn'] == "tanh" else nn.ReLU
    
    final_policy_kwargs = dict(
        activation_fn=final_activation,
        net_arch=final_arch
    )

    env = UAVRoutingEnv(csv_file=None)
    env = Monitor(env, LOG_DIR)
    eval_env = UAVRoutingEnv(csv_file=CSV_FILE)
    eval_env = Monitor(eval_env)

    model = PPO("MlpPolicy", env, 
                learning_rate=best_params['learning_rate'], 
                policy_kwargs=final_policy_kwargs, 
                verbose=1)

    new_logger = configure(LOG_DIR, ["stdout", "csv"])
    model.set_logger(new_logger)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=15, min_evals=5, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=5,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        callback_after_eval=stop_train_callback,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Train the final super-model
    model.learn(total_timesteps=1500000, callback=eval_callback)
    
    final_model_path = os.path.join(MODEL_DIR, "final_uav_model")
    model.save(final_model_path)
    print(f"\n✅ SUCCESS: The optimal model was saved to {os.path.join(MODEL_DIR, 'best_model.zip')}")

if __name__ == '__main__':
    main()
