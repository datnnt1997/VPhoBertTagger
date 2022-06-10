from helper import init_logger
from datetime import datetime


LOGGER = init_logger(datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))

LABEL2ID = ['O', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC']
ID2LABEL = {idx: label for idx, label in enumerate(LABEL2ID)}
