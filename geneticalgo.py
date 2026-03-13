from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots, mutate_robot
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    best_fitness_history = []
    avg_fitness_history = []
    worst_fitness_history = []

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print(config)

    np.random.seed(config["seed"])

    n_iterations = config["simulator"].get("n_iterations", 3)   

    # INITIAL POPULATION
    robots = load_robots(num_robots=config["simulator"]["n_sims"])

    for iteration in range(n_iterations):

        print(f"\n=== GENERATION {iteration} ===")

        # Extract geometry
        num_masses = [robot["n_masses"] for robot in robots]
        num_springs = [robot["n_springs"] for robot in robots]

        max_num_masses = max(num_masses)
        max_num_springs = max(num_springs)

        config["simulator"]["n_masses"] = max_num_masses
        config["simulator"]["n_springs"] = max_num_springs

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

        print("Fitness:", fitness)

        # Rank robots
        ranking = np.argsort(fitness)[::-1]

        # Remove worst robot
        worst_index = ranking[-1]
        print(f"Removing robot {worst_index}")

        survivors = [robots[i] for i in ranking[:-1]]

        # Mutate survivors
        mutated_children = [mutate_robot(robot) for robot in survivors]

        # New generation = survivors + mutated children
        robots = survivors + mutated_children

        # Keep population size fixed
        robots = robots[:config["simulator"]["n_sims"]]

        # Save fitness
        best_fitness_history.append(np.max(fitness))
        avg_fitness_history.append(np.mean(fitness))
        worst_fitness_history.append(np.min(fitness))

    # ===== FINAL EVALUATION =====

    print("\n=== FINAL EVALUATION ===")

    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]

    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)

    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

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

    ranking = np.argsort(fitness)[::-1]

    top_3_idxs = ranking[:3]
    top_3_robots = [robots[i] for i in top_3_idxs]

    top_3_control_params = simulator.get_control_params(top_3_idxs)

    for i in range(3):
        robot = top_3_robots[i]
        control_params = top_3_control_params[i]

        robot["control_params"] = control_params
        robot["max_n_masses"] = max_num_masses
        robot["max_n_springs"] = max_num_springs

        np.save(f"robot_{i}.npy", robot)

iterations = np.arange(len(best_fitness_history))

plt.figure()
plt.plot(iterations, best_fitness_history, label="Best")
plt.plot(iterations, avg_fitness_history, label="Average")
plt.plot(iterations, worst_fitness_history, label="Worst")

plt.legend()

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Over Generations")

plt.xticks(iterations)  # ensures integers only
plt.grid(True)

plt.show()