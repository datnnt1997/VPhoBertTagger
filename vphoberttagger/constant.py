from .helper import init_logger
from .model import PhoBertSoftmax, PhoBertLstmCrf
from datetime import datetime

import re


LOGGER = init_logger(datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))

LABEL2ID_VLSP2016 = ['O', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC']
LABEL2ID_VLSP2018 = ['O', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-LOCATION', 'I-LOCATION', 'B-PERSON', 'I-PERSON', 'B-MISCELLANEOUS', 'I-MISCELLANEOUS']


MODEL_MAPPING = {
    'softmax': PhoBertSoftmax,
    'lstm_crf': PhoBertLstmCrf
}

OPENTAG_PATTERN = re.compile('<ENAMEX TYPE="(.+?)">')
CLOSETAG_PATTERN = re.compile('</ENAMEX>')

LABEL_MAPPING = {
    'vlsp2016': {
        'label2id': LABEL2ID_VLSP2016,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_VLSP2016)},
        'header': ['token', 'pos', 'chunk', 'ner']
    },
    'vlsp2018': {
        'label2id': LABEL2ID_VLSP2018,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_VLSP2018)},
        'header': ['token', 'ner']
    }
}