import re


def normalize_text(txt):
    txt = re.sub("\xad|\u200b", "", txt)
    return txt.strip()