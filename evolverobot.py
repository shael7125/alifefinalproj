# single parent hill climber scheme
from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots, mutate_robot
import numpy as np

# Default number of generations
DEFAULT_GENERATIONS = 10

def evaluate_robots(robots, config):
    """
    Runs simulator evaluation and returns final fitness for each robot.
    Expects a list of robots (even if length 1) and returns a numpy array of fitness.
    """
    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]

    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)

    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs
    config["simulator"]["n_sims"] = len(robots)

    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True
    )

    masses = [robot["masses"] for robot in robots]
    springs = [robot["springs"] for robot in robots]

    simulator.initialize(masses, springs)

    fitness_history = simulator.train()
    fitness = fitness_history[:, -1]

    return fitness

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="robot_0.npy", help="Path to parent robot .npy file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="Number of hill climbing generations")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    # Load parent robot
    parent = np.load(args.input, allow_pickle=True).item()
    print(f"Loaded parent robot: {parent['n_masses']} masses, {parent['n_springs']} springs")

    # Evaluate initial parent
    parent_fitness = evaluate_robots([parent], config)[0]
    print(f"Initial parent fitness: {parent_fitness:.6f}")

    # Hill climbing loop
    for generation in range(args.generations):
        print(f"\n=== Generation {generation + 1} ===")

        # Create child by mutating parent
        child = mutate_robot(parent)

        # Evaluate child
        child_fitness = evaluate_robots([child], config)[0]
        print(f"Child fitness: {child_fitness:.6f}")

        # Selection: keep the fitter robot
        if child_fitness > parent_fitness:
            parent = child
            parent_fitness = child_fitness
            print("Child accepted as new parent")
        else:
            print("Child rejected, parent remains")

        print(f"Best fitness so far: {parent_fitness:.6f}")

    # Save final evolved robot
    np.save("robot_test.npy", parent)
    print("\nHill climbing complete. Final robot saved as 'robot_2.npy'.")