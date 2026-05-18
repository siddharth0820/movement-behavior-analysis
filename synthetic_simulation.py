import numpy as np
import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict

GRID_SIZE = 4
START = (4, 1)
GOAL = (1, 4)
MAX_STEPS = 200

SIMULATION_RUNS = 500
PROGRESSIVE_GUIDED_RUNS = 500

random.seed(42)

GUIDANCE_SCHEDULE = [50, 45, 39, 34, 28, 23, 17, 12, 6, 1]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos, grid_size):
    r, c = pos

    moves = [
        (r + 1, c),
        (r - 1, c),
        (r, c + 1),
        (r, c - 1)
    ]

    valid = []

    for m in moves:
        if 1 <= m[0] <= grid_size and 1 <= m[1] <= grid_size:
            valid.append(m)

    return valid


def get_direction(a, b):
    r1, c1 = a
    r2, c2 = b

    if r2 < r1:
        return "up"
    elif r2 > r1:
        return "down"
    elif c2 > c1:
        return "right"
    elif c2 < c1:
        return "left"
    else:
        return None


def direction_counts(path):
    counts = {
        "up": 0,
        "down": 0,
        "left": 0,
        "right": 0
    }

    for i in range(len(path) - 1):
        direction = get_direction(path[i], path[i + 1])

        if direction is not None:
            counts[direction] += 1

    return counts


def move_entropy(path):
    cell_direction_counts = defaultdict(lambda: {
        "up": 0,
        "down": 0,
        "left": 0,
        "right": 0
    })

    for i in range(len(path) - 1):
        cell = path[i]
        direction = get_direction(path[i], path[i + 1])

        if direction is not None:
            cell_direction_counts[cell][direction] += 1

    entropies = []

    for cell, counts in cell_direction_counts.items():
        cell_total = sum(counts.values())

        if cell_total == 0:
            continue

        entropy = 0

        for count in counts.values():
            if count > 0:
                p = count / cell_total
                entropy -= p * np.log(p)

        entropies.append(entropy)

    if len(entropies) == 0:
        return 0

    return np.mean(entropies)


def generate_random_path(start, goal, grid_size, max_steps):
    pos = start
    path = [pos]

    for _ in range(max_steps):
        if pos == goal:
            break

        neighbors = get_neighbors(pos, grid_size)
        pos = random.choice(neighbors)
        path.append(pos)

    return path


def choose_goal_oriented_move(neighbors, goal):
    best_distance = min(manhattan(n, goal) for n in neighbors)
    best_neighbors = [n for n in neighbors if manhattan(n, goal) == best_distance]
    return random.choice(best_neighbors)


def generate_guided_path(start, goal, grid_size, max_steps, nth_move):
    pos = start
    path = [pos]

    for step_index in range(1, max_steps + 1):
        if pos == goal:
            break

        neighbors = get_neighbors(pos, grid_size)

        if nth_move > 0 and step_index % nth_move == 0:
            pos = choose_goal_oriented_move(neighbors, goal)
        else:
            pos = random.choice(neighbors)

        path.append(pos)

    return path


def analyze_moves(path, goal):
    good = 0
    bad = 0

    for i in range(len(path) - 1):
        d1 = manhattan(path[i], goal)
        d2 = manhattan(path[i + 1], goal)

        if d2 < d1:
            good += 1
        else:
            bad += 1

    return good, bad


def compute_metrics(path, start, goal):
    optimal_steps = manhattan(start, goal)
    steps = len(path) - 1
    efficiency = optimal_steps / steps if steps > 0 else 0

    good, bad = analyze_moves(path, goal)
    good_ratio = good / steps if steps > 0 else 0
    skill_score = (efficiency + good_ratio) / 2
    entropy = move_entropy(path)

    return {
        "steps": steps,
        "optimal_steps": optimal_steps,
        "efficiency": efficiency,
        "good_moves": good,
        "bad_moves": bad,
        "good_ratio": good_ratio,
        "skill_estimate": skill_score,
        "entropy": entropy,
        "direction_counts": direction_counts(path)
    }


def run_single_random_simulation(start, goal, grid_size, max_steps):
    path = generate_random_path(start, goal, grid_size, max_steps)
    metrics = compute_metrics(path, start, goal)

    return {
        "start": start,
        "goal": goal,
        "path": path,
        **metrics
    }


def run_batch_random_simulation(runs, start, goal, grid_size, max_steps):
    data = []

    for i in range(runs):
        result = run_single_random_simulation(start, goal, grid_size, max_steps)
        result["run"] = i + 1
        data.append(result)

    return data


def get_progressive_nth_move(run_index, total_runs, schedule):
    block_size = total_runs / len(schedule)
    schedule_index = int(run_index // block_size)

    if schedule_index >= len(schedule):
        schedule_index = len(schedule) - 1

    return schedule[schedule_index]


def run_progressive_guided_simulation(runs, start, goal, grid_size, max_steps, schedule):
    data = []

    for i in range(runs):
        nth_move = get_progressive_nth_move(i, runs, schedule)

        path = generate_guided_path(start, goal, grid_size, max_steps, nth_move)
        metrics = compute_metrics(path, start, goal)

        result = {
            "run": i + 1,
            "start": start,
            "goal": goal,
            "path": path,
            "guided_every_nth_move": nth_move,
            "guidance_frequency": 1 / nth_move,
            **metrics
        }

        data.append(result)

    return data


def summarize(data, title):
    avg_steps = np.mean([d["steps"] for d in data])
    avg_efficiency = np.mean([d["efficiency"] for d in data])
    avg_skill = np.mean([d["skill_estimate"] for d in data])
    avg_entropy = np.mean([d["entropy"] for d in data])

    print(f"\n{title}")
    print("Runs:", len(data))
    print("Avg steps:", round(avg_steps, 2))
    print("Avg efficiency:", round(avg_efficiency, 3))
    print("Avg skill:", round(avg_skill, 3))
    print("Avg entropy:", round(avg_entropy, 3))


def aggregate_direction_counts(data):
    total_counts = {
        "up": 0,
        "down": 0,
        "left": 0,
        "right": 0
    }

    for d in data:
        counts = d["direction_counts"]

        for direction in total_counts:
            total_counts[direction] += counts[direction]

    return total_counts


random_data = run_batch_random_simulation(
    SIMULATION_RUNS,
    START,
    GOAL,
    GRID_SIZE,
    MAX_STEPS
)

summarize(random_data, "Random simulation summary")

with open("mouse_paths_random.json", "w") as f:
    json.dump(random_data, f, indent=2)


progressive_data = run_progressive_guided_simulation(
    PROGRESSIVE_GUIDED_RUNS,
    START,
    GOAL,
    GRID_SIZE,
    MAX_STEPS,
    GUIDANCE_SCHEDULE
)

summarize(progressive_data, "Guided simulation summary")

with open("mouse_paths_progressive_guided.json", "w") as f:
    json.dump(progressive_data, f, indent=2)


# ------------------------------------------------------------
# Plot 1: Random steps with regression
# ------------------------------------------------------------

random_runs = [d["run"] for d in random_data]
random_steps = [d["steps"] for d in random_data]

random_slope, random_intercept = np.polyfit(random_runs, random_steps, 1)
random_regression_line = [
    random_slope * x + random_intercept for x in random_runs
]

print("\nLinear regression on random steps")
print("Random slope:", round(random_slope, 4))
print("Random intercept:", round(random_intercept, 4))

plt.figure()
plt.plot(random_runs, random_steps, linewidth=1.5, alpha=0.7, label="Random steps")
plt.plot(random_runs, random_regression_line, linestyle="--", linewidth=2, label="Regression")
plt.xlabel("Run")
plt.ylabel("Steps")
plt.title("Random Movement Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("random_steps.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 2: Guided steps with regression
# ------------------------------------------------------------

progressive_runs = [d["run"] for d in progressive_data]
progressive_steps = [d["steps"] for d in progressive_data]

guided_slope, guided_intercept = np.polyfit(progressive_runs, progressive_steps, 1)
guided_regression_line = [
    guided_slope * x + guided_intercept for x in progressive_runs
]

print("\nLinear regression on guided steps")
print("Guided slope:", round(guided_slope, 4))
print("Guided intercept:", round(guided_intercept, 4))

plt.figure()
plt.plot(progressive_runs, progressive_steps, linewidth=1.5, alpha=0.7, label="Guided steps")
plt.plot(progressive_runs, guided_regression_line, linestyle="--", linewidth=2, label="Regression")
plt.xlabel("Run")
plt.ylabel("Steps")
plt.title("Guided Movement Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("guided_steps.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 3: Step trend comparison
# ------------------------------------------------------------

plt.figure()
plt.plot(random_runs, random_regression_line, linestyle="--", linewidth=2, label="Random regression")
plt.plot(progressive_runs, guided_regression_line, linestyle="--", linewidth=2, label="Guided regression")
plt.xlabel("Run")
plt.ylabel("Steps")
plt.title("Step Trend Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("steps_regression_comparison.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 4: Random entropy (RAW + regression, no smoothing)
# ------------------------------------------------------------

random_entropy = [d["entropy"] for d in random_data]

# Regression on raw data
entropy_slope, entropy_intercept = np.polyfit(random_runs, random_entropy, 1)
entropy_regression = [
    entropy_slope * x + entropy_intercept for x in random_runs
]

print("\nRandom entropy (raw)")
print("Entropy slope:", round(entropy_slope, 6))
print("Entropy intercept:", round(entropy_intercept, 4))

plt.figure()
plt.plot(random_runs, random_entropy, linewidth=1.2, alpha=0.6, label="Entropy per run")
plt.plot(random_runs, entropy_regression, linestyle="--", linewidth=2, label="Regression")
plt.xlabel("Run")
plt.ylabel("Entropy")
plt.title("Random Entropy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("random_entropy.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 5: Guided entropy (RAW + regression, no smoothing)
# ------------------------------------------------------------

guided_entropy = [d["entropy"] for d in progressive_data]

guided_entropy_slope, guided_entropy_intercept = np.polyfit(progressive_runs, guided_entropy, 1)
guided_entropy_regression = [
    guided_entropy_slope * x + guided_entropy_intercept for x in progressive_runs
]

print("\nGuided entropy (raw)")
print("Guided entropy slope:", round(guided_entropy_slope, 6))
print("Guided entropy intercept:", round(guided_entropy_intercept, 4))

plt.figure()
plt.plot(progressive_runs, guided_entropy, linewidth=1.2, alpha=0.6, label="Entropy per run")
plt.plot(progressive_runs, guided_entropy_regression, linestyle="--", linewidth=2, label="Regression")
plt.xlabel("Run")
plt.ylabel("Entropy")
plt.title("Guided Entropy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("guided_entropy.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 6: Direction counts
# ------------------------------------------------------------

directions = ["up", "down", "left", "right"]
batch_size = 100

batch_labels = []
batch_values = []

for start_index in range(0, len(progressive_data), batch_size):
    end_index = start_index + batch_size
    batch = progressive_data[start_index:end_index]

    counts = aggregate_direction_counts(batch)
    values = [counts[d] for d in directions]

    batch_labels.append(f"Runs {start_index + 1} to {end_index}")
    batch_values.append(values)

x = np.arange(len(directions))
width = 0.15

plt.figure()

for i, values in enumerate(batch_values):
    plt.bar(x + (i - 2) * width, values, width, label=batch_labels[i])

plt.xlabel("Direction")
plt.ylabel("Count")
plt.title("Direction Counts")
plt.xticks(x, directions)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("direction_counts.png", dpi=300)
plt.show()

