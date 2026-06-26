import ast
import yaml
from pathlib import Path
from pytorch_implementation import model
import re


# ── percorso esperimento ────────────────────────────────────────────────────
experiment = Path('./results_gesture_reshaped_16/26_04_13_10_78538_roigesture_matrix_depth_4_4_BASELINE')
print(f"[Experiment] {experiment.name}")

# ── carica config.yaml ──────────────────────────────────────────────────────
with open(experiment / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_name = config['dataset']
mode    = config['mode']
print(f"[Config] dataset={dataset_name}, mode={mode}")

# ── trova l'indice dell'iterazione migliore ─────────────────────────────────
algo_logs = experiment / 'algorithm_logs'

score_path = algo_logs / 'score_report.txt'
acc_path   = algo_logs / 'acc_report.txt'

if score_path.exists():
    scores = [float(l.strip()) for l in score_path.read_text().splitlines() if l.strip()]
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    print(f"[Selection] Using score_report.txt — best iteration: {best_idx} (score={scores[best_idx]:.4f})")
else:
    accs = [float(l.strip()) for l in acc_path.read_text().splitlines() if l.strip()]
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    print(f"[Selection] Using acc_report.txt — best iteration: {best_idx} (acc={accs[best_idx]:.4f})")

# ── carica l'iperparametro corrispondente da hyper-neural.txt ───────────────
hyper_path = algo_logs / 'hyper-neural.txt'
with open(hyper_path, 'r') as f:
    lines = [l.strip() for l in f if l.strip()]

best_params = ast.literal_eval(lines[best_idx])
print(f"[Hyperparams] {best_params}")


# ── carica il file .out ─────────────────────────────────────────────────────
out_files = list(experiment.glob('*.out'))
if not out_files:
    raise FileNotFoundError(f"No .out file found in {experiment}")
out_path = out_files[0]
print(f"[Out file] {out_path.name}")

out_text = out_path.read_text(errors='replace')

# ── spezza il testo per iterazione ─────────────────────────────────────────
# ogni blocco inizia con "--- ITERATION N ---"
iteration_blocks = re.split(r'---\s*ITERATION\s+\d+\s*---', out_text)
# index 0 è l'header prima della prima iterazione, quindi iteration N → blocks[N+1]
print(f"[Out file] Found {len(iteration_blocks) - 1} iterations")

target_block = iteration_blocks[best_idx + 1]

# ── estrai layer_x_block e input_shape dal blocco corretto ─────────────────
match = re.search(
    r'Building model with input_shape=\(([^)]+)\).*?layer_x_block=(\d+)',
    target_block
)
if not match:
    raise ValueError(f"Could not find 'Building model' line for iteration {best_idx}")


# ── estrai i parametri rilevanti ────────────────────────────────────────────
input_shape   = tuple(int(x) for x in match.group(1).split(','))
input_shape   = (input_shape[2], input_shape[0], input_shape[1])
layer_x_block = int(match.group(2))

print(f"[Model] iteration={best_idx}, input_shape={input_shape}, layer_x_block={layer_x_block}")
n_classes = 11

# ── istanzia la rete ────────────────────────────────────────────────────────
print("[Network] Initializing neural network...")
model = model.TorchModel(
            params=best_params,
            input_shape=input_shape,
            n_classes=n_classes,
            layer_x_block=layer_x_block,
            batch=True
        )
print("[Network] Ready.")

model.summary()