# trainrobot.py
# Safe macOS version: resets Taichi to avoid bus errors
import taichi as ti  # NEW: reset Taichi runtime to avoid bus error
# ti.reset()           # NEW: clears any leftover GPU/CPU context

import os, numpy as np
os.environ["ENABLE_TAICHI_HEADER_PRINT"] = "False"
from tqdm import tqdm

from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="robot_2.npy", help="Path to saved robot .npy file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load robot
    robot = np.load(args.input, allow_pickle=True).item()
    print(f"Loaded robot: {robot['n_masses']} masses, {robot['n_springs']} springs")

    # Load configuration
    config = load_config(args.config)

    # Force single simulation to avoid semaphore / multiprocessing issues
    config["simulator"]["n_sims"] = 1
    config["simulator"]["n_masses"] = int(robot.get("max_n_masses", robot["n_masses"]))
    config["simulator"]["n_springs"] = int(robot.get("max_n_springs", robot["n_springs"]))

    # Initialize simulator for training
    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True  # NEW: enable training gradients
    )

    # Initialize simulator with the robot geometry
    simulator.initialize([robot["masses"]], [robot["springs"]])

    # Train the robot
    print("Starting training...")
    fitness_history = simulator.train()  # shape (1, n_learning_steps)
    np.save("fitness_history.npy", fitness_history)
    print("Training complete.")

    # Save trained control parameters back to robot
    final_control = simulator.get_control_params([0])[0]
    robot["control_params"] = final_control

    # Ensure max dimensions are saved for visualizer
    robot["max_n_masses"] = config["simulator"]["n_masses"]
    robot["max_n_springs"] = config["simulator"]["n_springs"]

    np.save("robot_3.npy", robot)
    print("Saved trained robot to robot_test.npy")