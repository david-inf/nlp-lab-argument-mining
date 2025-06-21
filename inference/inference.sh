python inference/inference.py --ftdata mixed
# python inference/inference.py --ftdata ibm

python inference/train.py --ftdata mixed --dataset molecular
# python inference/train.py --ftdata ibm --dataset molecular

python inference/train.py --ftdata mixed --dataset thoracic
# python inference/train.py --ftdata ibm --dataset thoracic

python inference/train.py --ftdata mixed --dataset merge
# python inference/train.py --ftdata ibm --dataset merge