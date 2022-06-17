import os
import re
import random
import logging

import torch
import numpy


def set_ramdom_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if not os.path.isdir('./logs'):
            os.makedirs('./logs')
        log_file = os.path.join('./logs/', log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def normalize_text(txt: str) -> str:
    # Remove special character
    txt = re.sub("\xad|\u200b|\ufeff", "", txt)
    # Normalize vietnamese accents
    txt = re.sub(r"òa", "oà", txt)
    txt = re.sub(r"óa", "oá", txt)
    txt = re.sub(r"ỏa", "oả", txt)
    txt = re.sub(r"õa", "oã", txt)
    txt = re.sub(r"ọa", "oạ", txt)
    txt = re.sub(r"òe", "oè", txt)
    txt = re.sub(r"óe", "oé", txt)
    txt = re.sub(r"ỏe", "oẻ", txt)
    txt = re.sub(r"õe", "oẽ", txt)
    txt = re.sub(r"ọe", "oẹ", txt)
    txt = re.sub(r"ùy", "uỳ", txt)
    txt = re.sub(r"úy", "uý", txt)
    txt = re.sub(r"ủy", "uỷ", txt)
    txt = re.sub(r"ũy", "uỹ", txt)
    txt = re.sub(r"ụy", "uỵ", txt)
    txt = re.sub(r"Ủy", "Uỷ", txt)
    # Remove multi-space
    txt = re.sub(" +", " ", txt)
    return txt.strip()
