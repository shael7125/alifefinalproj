# train_robot_safe.py
import numpy as np
import taichi as ti
from simulator import Simulator
from utils import load_config
from robot import load_robots
from tqdm import tqdm

# ---------------- CONFIG ----------------
CONFIG_FILE = "config.yaml"
CHUNK_SIZE = 50  # max sim steps per chunk to avoid semaphore issues

# ---------------- LOAD ----------------
config = load_config(CONFIG_FILE)
np.random.seed(config["seed"])

# Load robot_0.npy
robot = np.load("robot_0.npy", allow_pickle=True).item()
n_masses = robot["masses"].shape[0]
n_springs = robot["springs"].shape[0]

# ---------------- CHUNKING ----------------
total_steps = config["simulator"]["sim_steps"]
num_chunks = (total_steps + CHUNK_SIZE - 1) // CHUNK_SIZE  # ceil division
chunk_sizes = [(i*CHUNK_SIZE, min((i+1)*CHUNK_SIZE, total_steps)) for i in range(num_chunks)]

# ---------------- SIMULATOR ----------------
# We create one simulator per chunk to avoid Taichi resource leaks
fitness_history_all = []

for start, end in chunk_sizes:
    chunk_steps = end - start
    sim_config = {
        "n_sims": 1,
        "sim_steps": chunk_steps,
        "learning_steps": config["simulator"]["learning_steps"],
        "n_masses": n_masses,
        "n_springs": n_springs,
        "ground_height": config["simulator"]["ground_height"],
        "dt": config["simulator"]["dt"],
        "springA": config["simulator"]["springA"],
        "springK": config["simulator"]["springK"],
        "gravity": config["simulator"]["gravity"],
        "friction": config["simulator"]["friction"],
        "restitution": config["simulator"]["restitution"],
        "drag_damping": config["simulator"]["drag_damping"],
        "eps": config["simulator"]["eps"],
        "nn_hidden_size": config["simulator"]["nn_hidden_size"],
        "nn_cpg_count": config["simulator"]["nn_cpg_count"],
        "cpg_omega": config["simulator"]["cpg_omega"],
        "adam_beta1": config["simulator"]["adam_beta1"],
        "adam_beta2": config["simulator"]["adam_beta2"],
        "learning_rate": config["simulator"]["learning_rate"],
    }

    # Initialize simulator for this chunk
    sim = Simulator(sim_config=sim_config, taichi_config=config["taichi"], seed=config["seed"], needs_grad=True)
    sim.initialize([robot["masses"]], [robot["springs"]])

    # Train in this chunk
    chunk_fitness = []
    for step in range(sim_config["learning_steps"]):
        chunk_fitness.append(sim.learning_step())
    chunk_fitness.append(sim.evaluation_step())
    fitness_history_all.append(np.array(chunk_fitness).T)

# ---------------- MERGE FITNESS ----------------
fitness_history_all = np.hstack(fitness_history_all)  # shape: (1, total_steps)
np.save("fitness_history_robot_0.npy", fitness_history_all)

# ---------------- SAVE CONTROL PARAMS ----------------
top_control_params = sim.get_control_params([0])[0]
robot["control_params"] = top_control_params
np.save("robot_0_trained.npy", robot)

print(f"Training complete. Fitness history saved to 'fitness_history_robot_0.npy'.")
print(f"Trained robot saved to 'robot_0_trained.npy'.")