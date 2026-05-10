from __future__ import annotations
import os
from pathlib import Path

USE_LM = os.environ.get('USE_LM', '1') == '1'
USE_BERT = os.environ.get('USE_BERT', '1') == '1'
USE_TWO_STAGE = os.environ.get('USE_TWO_STAGE', '1') == '1'
USE_ROLLING = os.environ.get('USE_ROLLING', '1') == '1'

LM_CANDIDATES = os.environ.get(
    'LM_CANDIDATES',
    'readerbench/RoGPT2-large,readerbench/RoGPT2-medium,readerbench/RoGPT2-base,'
    'dumitrescustefan/gpt-neo-romanian-780m,dumitrescustefan/gpt-neo-romanian-125m',
).split(',')
LM_MAX_LEN = int(os.environ.get('LM_MAX_LEN', '1024'))

BERT_NAME = os.environ.get('BERT_NAME', 'dumitrescustefan/bert-base-romanian-cased-v1')
BERT_MAX_LEN = int(os.environ.get('BERT_MAX_LEN', '512'))
BERT_BATCH = int(os.environ.get('BERT_BATCH',   '32'))
BERT_PCA_DIM = int(os.environ.get('BERT_PCA_DIM', '24'))

N_FOLDS = int(os.environ.get('N_FOLDS',  '5'))
N_SEEDS = int(os.environ.get('N_SEEDS',  '3'))
SEED = 42

ROLLING_WINDOWS = [int(x) for x in os.environ.get('ROLLING_WINDOWS', '10,20,50').split(',')]
ROLLING_SKIP_THRESHOLD = float(os.environ.get('ROLLING_SKIP_THRESHOLD', '50.0'))

_PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = Path(os.environ.get('INPUT_DIR',  str(_PROJECT_ROOT)))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', str(_PROJECT_ROOT)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(
    f'[cfg] INPUT_DIR={INPUT_DIR}  USE_LM={USE_LM}  USE_BERT={USE_BERT}  '
    f'USE_TWO_STAGE={USE_TWO_STAGE}  USE_ROLLING={USE_ROLLING}  N_SEEDS={N_SEEDS}'
)
