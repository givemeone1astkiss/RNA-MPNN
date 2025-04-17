# Path for data.
DATA_PATH='./data/'
# Path for saving logs.
LOG_PATH='./logs/'
# Path for saving generated SMILES strings.
OUTPUT_PATH='./out/'

FEAT_DIMS = {
    'node': {
        'angle': 12,
        'distance': 80,
        'direction': 9,
    },
    'edge': {
        'orientation': 4,
        'distance': 96,
        'direction': 15,
    }
}

SPLIT_RATIO = [0.9, 0.1, 0.0]
