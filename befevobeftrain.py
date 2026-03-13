from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Set the random seed for reproducibility
    np.random.seed(config["seed"])

    # Generate a single random robot
    robots = load_robots(num_robots=1)

    # Determine max masses/springs for memory allocation
    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

    # Initialize simulator
    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=False
    )

    # Extract masses and springs
    masses = [robot["masses"] for robot in robots]
    springs = [robot["springs"] for robot in robots]

    # Initialize the simulator state
    simulator.initialize(masses, springs)

    # Step the simulation without training
    sim_steps = config["simulator"]["sim_steps"]
    robot_idx = 0
    positions_over_time = []

    for t in range(sim_steps):
        simulator.compute_com(t)
        simulator.nn1(t)
        simulator.nn2(t)
        simulator.apply_spring_force(t)
        simulator.apply_goal_force(t)
        simulator.advance(t + 1)

        positions = simulator.x.to_numpy()[robot_idx, t + 1, :max_num_masses]
        positions_over_time.append(positions.copy())

    # Visualize
    plt.figure(figsize=(6,6))
    for pos in positions_over_time:
        plt.scatter(pos[:,0], pos[:,1], c='orange', alpha=0.6)
    plt.scatter(simulator.target_x, simulator.target_y, c='red', s=100, label="Target")
    plt.title("Random Robot Simulation (No Training)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()