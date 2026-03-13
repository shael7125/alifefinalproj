# generate_robot.py
from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    np.random.seed(config["seed"])

    # Generate one random robot
    robots = load_robots(num_robots=1)
    robot = robots[0]

    # Extract number of masses and springs
    n_masses = robot["n_masses"]
    n_springs = robot["n_springs"]

    # Save max dimensions to robot (needed for visualizer memory allocation)
    robot["max_n_masses"] = n_masses
    robot["max_n_springs"] = n_springs

    # No training, no simulator initialization needed
    np.save("robot_0.npy", robot)
    print(f"Saved random robot as robot_0.npy with {n_masses} masses and {n_springs} springs.")