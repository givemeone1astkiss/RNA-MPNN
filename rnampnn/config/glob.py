# Path for data.
DATA_PATH='./data/'
# Path for saving logs.
LOG_PATH='./logs/'
# Path for saving generated SMILES strings.
OUTPUT_PATH='./out/'

COMPETITION_DATA = '/saisdata/'
COMPETITION_OUT = '/saisresult/'

MIN_LEN = 1000
NUM_MAIN_SEQ_ATOMS = 7
NUM_RES_TYPES = 4
DEFAULT_HIDDEN_DIM = 128

LEPS = 1e6
SEPS = 1e-6
DEFAULT_SEED = 42

VOCAB = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
REVERSE_VOCAB = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}

BEST_CKPT: str = 'out/checkpoints/RNAMPNN-X/Final-V0.ckpt'