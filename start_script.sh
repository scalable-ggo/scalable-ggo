module load 2025
module load cuda/12.9

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
