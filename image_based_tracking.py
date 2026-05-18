import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

GOLDEN_FOLDER = "golden_run"
RANDOM_FOLDER = "random_run_1"
RANDOM_FOLDER_2 = "random_run_2"

START = (4, 4)
GOAL = (1, 1)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def detect_qr(img):
    detector = cv2.QRCodeDetector()
    retval, decoded, points, _ = detector.detectAndDecodeMulti(img)

    if not retval or points is None:
        return []

    centers = []
    for pts in points:
        center = np.mean(pts, axis=0)
        centers.append(center)

    return centers


def estimate_grid(qr_centers, grid_size=4):
    qr_centers = np.array(qr_centers)

    min_x, min_y = qr_centers.min(axis=0)
    max_x, max_y = qr_centers.max(axis=0)

    xs = np.linspace(min_x, max_x, grid_size)
    ys = np.linspace(min_y, max_y, grid_size)

    grid = []
    for y in ys:
        for x in xs:
            grid.append((x, y))

    return np.array(grid)


def assign_to_grid(grid, qr_centers, threshold=50):
    occupied = {}

    for i, g in enumerate(grid):
        for c in qr_centers:
            if np.linalg.norm(g - c) < threshold:
                occupied[i] = c

    return occupied


def find_hole(grid, occupied):
    for i in range(len(grid)):
        if i not in occupied:
            return i, grid[i]
    return None, None


def process_image(img):
    qr_centers = detect_qr(img)

    if len(qr_centers) < 5:
        print("Too few QR codes detected")
        return None, None, None

    grid = estimate_grid(qr_centers)
    occupied = assign_to_grid(grid, qr_centers)
    hole_id, hole_pos = find_hole(grid, occupied)

    return grid, occupied, (hole_id, hole_pos)


def pos_to_rc(pos, grid):
    xs = sorted(set([int(round(p[0])) for p in grid]))
    ys = sorted(set([int(round(p[1])) for p in grid]))

    col = np.argmin([abs(pos[0] - x) for x in xs]) + 1
    row = np.argmin([abs(pos[1] - y) for y in ys]) + 1

    return int(row), int(col)


def draw(img, grid, occupied, hole_pos, color=(0, 255, 0)):
    vis = img.copy()

    for p in grid:
        r, c = pos_to_rc(p, grid)

        cv2.circle(vis, tuple(p.astype(int)), 6, (255, 0, 0), -1)
        cv2.putText(
            vis,
            f"({r},{c})",
            tuple((p + np.array([10, -10])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    for c_pos in occupied.values():
        cv2.circle(vis, tuple(c_pos.astype(int)), 8, color, -1)

    hole_rc = pos_to_rc(hole_pos, grid)

    cv2.circle(vis, tuple(hole_pos.astype(int)), 18, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"HOLE {hole_rc}",
        tuple((hole_pos + np.array([15, 15])).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    return vis


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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

    return {
        "steps": steps,
        "optimal_steps": optimal_steps,
        "efficiency": efficiency,
        "good_moves": good,
        "bad_moves": bad,
        "good_ratio": good_ratio,
        "skill_estimate": skill_score
    }


def list_images(folder):
    files = []

    if not os.path.isdir(folder):
        return files

    for name in os.listdir(folder):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(folder, name))

    return sorted(files)


def extract_holes_from_folder(folder):
    image_files = list_images(folder)

    if not image_files:
        print(f"No images found in folder: {folder}")
        return []

    holes = []

    print(f"\nReading folder: {folder}")

    for file in image_files:
        print("Processing:", os.path.basename(file))

        img = cv2.imread(file)

        if img is None:
            print("Could not read image:", file)
            holes.append((None, None, None, os.path.basename(file), None))
            continue

        grid, occ, hole = process_image(img)

        if grid is None or occ is None or hole is None or hole[0] is None or hole[1] is None:
            print("Hole could not be determined for:", os.path.basename(file))
            holes.append((None, None, None, os.path.basename(file), img))
            continue

        r, c = pos_to_rc(hole[1], grid)
        print(f"Detected hole: ({r},{c})")

        holes.append((hole, grid, occ, os.path.basename(file), img))

    return holes


def build_path_from_holes(holes, move_threshold=None):
    valid_entries = [h for h in holes if h[0] is not None]

    if len(valid_entries) == 0:
        return [], []

    if len(valid_entries) == 1:
        hole, grid, occ, name, img = valid_entries[0]
        _, pos = hole
        rc = pos_to_rc(pos, grid)
        return [rc], []

    diagnostics = []
    hole_sequence = []

    first_hole, first_grid, _, _, _ = valid_entries[0]
    _, first_pos = first_hole
    r0, c0 = pos_to_rc(first_pos, first_grid)
    hole_sequence.append((r0, c0))

    tile_distance = np.linalg.norm(first_grid[0] - first_grid[1])
    diagonal_distance = tile_distance * np.sqrt(2)

    if move_threshold is None:
        move_threshold = (diagonal_distance - tile_distance) / 2

    for i in range(len(valid_entries) - 1):
        hole_a, grid_a, _, name_a, _ = valid_entries[i]
        hole_b, grid_b, _, name_b, _ = valid_entries[i + 1]

        _, pos1 = hole_a
        _, pos2 = hole_b

        r1, c1 = pos_to_rc(pos1, grid_a)
        r2, c2 = pos_to_rc(pos2, grid_b)

        movement_dist = np.linalg.norm(pos2 - pos1)
        is_valid = abs(movement_dist - tile_distance) <= move_threshold

        diagnostics.append({
            "from_file": name_a,
            "to_file": name_b,
            "from_rc": (r1, c1),
            "to_rc": (r2, c2),
            "movement_distance": float(movement_dist),
            "tile_distance": float(tile_distance),
            "valid_single_tile_move": bool(is_valid)
        })

        hole_sequence.append((r2, c2))

    return hole_sequence, diagnostics


def evaluate_run(folder, run_name, run_type, start, goal):
    holes = extract_holes_from_folder(folder)
    path, diagnostics = build_path_from_holes(holes)

    if not path:
        return {
            "name": run_name,
            "type": run_type,
            "start": start,
            "goal": goal,
            "error": "No valid path extracted"
        }

    metrics = compute_metrics(path, start, goal)

    return {
        "name": run_name,
        "type": run_type,
        "start": start,
        "goal": goal,
        "path": path,
        "positions_count": len(path),
        "frame_to_frame_checks": diagnostics,
        "raw_holes": holes,
        **metrics
    }


def print_run_result(result):
    print("\n---------------------------")
    print("Run:", result["name"])
    print("Type:", result["type"])

    if "error" in result:
        print("Error:", result["error"])
        return

    print("Steps:", result["steps"])
    print("Efficiency:", round(result["efficiency"], 3))
    print("Skill:", round(result["skill_estimate"], 3))


# RUNS

golden_result = evaluate_run(GOLDEN_FOLDER, "golden_run", "golden", START, GOAL)
random_result = evaluate_run(RANDOM_FOLDER, "random_run_1", "random", START, GOAL)
random_result_2 = evaluate_run(RANDOM_FOLDER_2, "random_run_2", "random", START, GOAL)

print_run_result(golden_result)
print_run_result(random_result)
print_run_result(random_result_2)

# SAVE

results = {
    "golden_run": {k: v for k, v in golden_result.items() if k != "raw_holes"},
    "random_run_1": {k: v for k, v in random_result.items() if k != "raw_holes"},
    "random_run_2": {k: v for k, v in random_result_2.items() if k != "raw_holes"},
}

with open("real_run_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# PLOT

labels = ["golden", "random 1", "random 2"]
skills = [
    golden_result["skill_estimate"],
    random_result["skill_estimate"],
    random_result_2["skill_estimate"]
]

plt.bar(labels, skills)
plt.title("Golden vs Random Runs")
plt.ylabel("Skill")
plt.show()