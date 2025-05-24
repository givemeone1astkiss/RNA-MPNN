# Path for data.
DATA_PATH: str='./data/'
# Path for saving logs.
LOG_PATH: str='./logs/'
# Path for saving generated SMILES strings.
OUTPUT_PATH: str='./out/'

FEAT_DIMS: dict = {
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

SPLIT_RATIO: list = [0.9, 0.1, 0.0]

COMPETITION_DATA: str = '/saisdata/'
COMPETITION_OUT: str = '/saisresult/'

BEST_CKPT: str = 'out/checkpoints/RDesign-X/Final-V1.ckpt'

NUM_RES_TYPES: int = 4
DEFAULT_SEED: int = 42