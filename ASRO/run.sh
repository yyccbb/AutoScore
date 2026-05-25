CONFIG_PATH="${1:-../config.example.yaml}"

python train.py --config "$CONFIG_PATH" > result.txt
