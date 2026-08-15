import os
import runpy

# ============================================================

# NEWSGUARD AI — TRAINING ENTRY POINT

# ============================================================

BASE_DIR = os.path.dirname(
os.path.abspath(__file__)
)

TRAINING_SCRIPT = os.path.join(
BASE_DIR,
"src",
"train_final.py"
)

if not os.path.exists(TRAINING_SCRIPT):


    raise FileNotFoundError(
        f"Training script not found at: {TRAINING_SCRIPT}"
)


print("=" * 60)
print("        NEWSGUARD AI — MODEL TRAINING")
print("=" * 60)

print(
"\nStarting final model training...\n"
)

runpy.run_path(
TRAINING_SCRIPT,
run_name="**main**"
)
