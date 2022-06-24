import os
import re
import random
import logging

import torch
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def set_ramdom_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
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


def get_total_model_parameters(model):
    total_params, trainable_params = 0, 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            trainable_params += params
        total_params += params
    return total_params,


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

    txt = re.sub(r'"', '”', txt)

    # Remove multi-space
    txt = re.sub(" +", " ", txt)
    return txt.strip()


def _get_tags(sents):
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}
        for i, tag in enumerate(iob_tags):
            if tag == 'O' and curr_tag['type']:
                tags.append(tuple(curr_tag.values()))
                curr_tag = {'type': None, 'start_idx': None,
                            'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B'):
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I'):
                curr_tag['end_idx'] = i
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))
    tags = set(tags)
    return tags


def plot_confusion_matrix(y_true, y_pred, classes, labels,
                          normalize=False,
                          title=None,
                          output_dir='./',
                          cmap=plt.cm.Blues):
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
         print('Confusion matrix, without normalization')

    # plt.rcParams['savefig.dpi'] = 200
    # plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # --- bar 크기 조절 --- #
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # --- bar 크기 조절 --- #
    # ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    return ax
