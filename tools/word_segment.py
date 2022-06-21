"""
Word Segmentation by RDRsegmenter which minhpqn modified the vocabulary from the original version of datquocnguyen
https://github.com/datquocnguyen/RDRsegmenter
"""

import re
import os
import pathlib
from argparse import ArgumentParser
from collections import OrderedDict


class Syllable(object):

    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end

    def __repr__(self):
        return str((self.text, self.start, self.end))


class Token(object):

    def __init__(self, _syllables):
        self.start_syl_id = None
        self.end_syl_id = None
        self.syllables = _syllables
        self.text = "_".join([syllable.text for syllable in _syllables])
        self.text2 = " ".join([syllable.text for syllable in _syllables])
        if len(self.syllables) > 0:
            self.start = self.syllables[0].start
            self.end = self.syllables[-1].end
        else:
            self.start = None
            self.end = None

    def set_syl_indexes(self, start_syl_id, end_syl_id):
        self.start_syl_id = start_syl_id
        self.end_syl_id = end_syl_id

    def __repr__(self):
        return str((self.text, self.start_syl_id, self.end_syl_id, self.start, self.end))


def find_syl_index(start, end, syllables):
    """Find start and end indexes of syllables
    """
    start_syl_id = None
    end_syl_id = None
    for i, syl in enumerate(syllables):
        if syl.start == start:
            start_syl_id = i
        if syl.end == end:
            end_syl_id = i + 1

        if i > 0 and syl.start >= start >= syllables[i - 1].end:
            start_syl_id = i
        if i == 0 and syl.start > start:
            start_syl_id = i

        if i < len(syllables) - 1 and syl.end < end < syllables[i + 1].start:
            end_syl_id = i + 1

        if syl.end >= end > syl.start:
            end_syl_id = i + 1
        if i == len(syllables) - 1 and syl.end <= end:
            end_syl_id = i + 1

        if i > 0 and syl.start < start and syllables[i - 1].end < start:
            start_syl_id = i

        if syl.start < start and syl.end >= end:
            start_syl_id = i
            end_syl_id = i + 1

        if i == 0 and len(syllables) > 0 and syl.start < start and syl.end < end:
            start_syl_id = i

    if start_syl_id is None:
        print("Cannot find start_syl_id '{}' (end={}) in '{}'".format(start, end, syllables))
    if end_syl_id is None:
        print("Cannot find end_syl_id '{}' (start={}) in '{}'".format(end, start, syllables))

    return start_syl_id, end_syl_id


def find_tok_index(start_syl_id, end_syl_id, token_list):
    start_tok_id = None
    end_tok_id = None

    for i, tk in enumerate(token_list):
        if tk.start_syl_id == start_syl_id:
            start_tok_id = i
        if tk.end_syl_id == end_syl_id:
            end_tok_id = i + 1
    return start_tok_id, end_tok_id


def remove_xml_tags(entity):
    entity = re.sub(r"<ENAMEX TYPE=”(.+?)”>", "", entity)
    entity = re.sub(r"</ENAMEX>", "", entity)
    return entity


def depth_level(astring):
    """
    E.g.,
    Tôi là sinh viên -> 0
    ĐHQG <ENAMEX>Hà Nội</ENAMEX> -> 1
    Khoa thanh nhạc <ENAMEX>Học viên âm nhạc <ENAMEX>HCM</ENAMEX></ENAMEX> -> 2
    Args:
        astring: input string with XML tags
    Returns:
        The depth level of a string
    """
    level = 0
    first = True
    first_add_child = True
    OPEN_TAG = 1
    stack = []
    i = 0
    while i < len(astring):
        if astring[i:].startswith("<ENAMEX TYPE="):
            if first:
                level += 1
                first = False
            if len(stack) > 0:
                if first_add_child:
                    level += 1
                    first_add_child = False
            stack.append(OPEN_TAG)
            i += len("<ENAMEX TYPE=")
        elif astring[i:].startswith("</ENAMEX>"):
            stack.pop()
            i += len("</ENAMEX>")
        else:
            i += 1
    return level


def get_entities(line):
    """

    Args:
        line (string): Input sentence (single sentence) with XML tags
        E.g., Đây là lý do khiến <ENAMEX TYPE=”PERSON”>Yoon Ah</ENAMEX> quyết định cắt mái tóc dài 'nữ thần'

    Returns:
        raw (string): raw sentence
        entities (list): list of entities (json object) (Wit.ai)
    """
    debug = False
    raw = ""
    entities = []

    regex_opentag = re.compile(r"<ENAMEX TYPE=”(.+?)”>")
    regex_closetag = re.compile(r"</ENAMEX>")
    next_start_pos = 0
    match1 = regex_opentag.search(line, next_start_pos)
    stack = []
    if match1:
        raw += line[0:match1.start()]
        next_start_pos = match1.end()
        stack.append(match1)
    else:
        raw = line

    while len(stack) != 0:
        if debug:
            print("#Current stack", stack)
        match1 = stack.pop()
        if debug:
            print("#From next_start_pos {}: {}".format(next_start_pos, line[next_start_pos:]))
        next_closetag1 = regex_closetag.search(line, next_start_pos)
        if not next_closetag1:
            print(line)
            raise ValueError("Close tag not found")
        next_end_pos1 = next_closetag1.start()
        match2 = regex_opentag.search(line, next_start_pos, next_end_pos1)
        if match2:
            raw += line[next_start_pos:match2.start()]
            next_start_pos1 = match2.end()
            next_closetag2 = regex_closetag.search(line, next_start_pos1)
            if not next_closetag2:
                raise ValueError("Close tag not found")
            next_end_pos2 = next_closetag2.start()
            match3 = regex_opentag.search(line, next_start_pos1, next_end_pos2)
            if match3:
                level = 1
                raw += line[next_start_pos1:match3.start()]
                next_start_pos2 = match3.end()
                value = line[next_start_pos2:next_end_pos2]
                _type = match3.group(1)

                entity = OrderedDict()
                entity["type"] = _type
                entity["value"] = value
                entity["start"] = len(raw)
                entity["end"] = entity["start"] + len(value)
                entity["level"] = level
                entities.append(entity)

                if debug:
                    print("#Entity:", value, _type, level)
                raw += value
                next_start_pos = next_closetag2.end()
                stack.append(match1)
                stack.append(match2)
            else:
                # Get entity between match2 and next_closetag2
                value = remove_xml_tags(line[match2.end():next_end_pos2])
                _type = match2.group(1)
                # abc <ENAMEX> xyz <ENAMEX>dhg</ENAMEX> mpq</ENAMEX> r
                level = 1 + depth_level(line[match2.end():next_end_pos2])
                if debug:
                    print("#current: ", raw)
                raw += line[next_start_pos1:next_closetag2.start()]
                if debug:
                    print("->", raw)
                entity = OrderedDict()
                entity["type"] = _type
                entity["value"] = value
                entity["start"] = len(raw) - len(value)
                entity["end"] = len(raw)
                entity["level"] = level
                entities.append(entity)

                if debug:
                    print("#Entity:", value, _type, level)
                next_start_pos = next_closetag2.end()
                stack.append(match1)
                next_match2 = regex_opentag.search(line, next_start_pos)
                next_closetag3 = regex_closetag.search(line, next_start_pos)

                if next_match2:
                    if next_closetag3 and next_match2.start() < next_closetag3.start():
                        if debug:
                            print("Next match2:", line[next_match2.start():])
                        if debug:
                            print("#current: ", raw)
                        raw += line[next_start_pos:next_match2.start()]
                        if debug:
                            print("->", raw)
                        next_start_pos = next_match2.end()
                        stack.append(next_match2)
        else:
            # Get entity between match1 and next_closetag1
            value = remove_xml_tags(line[match1.end():next_closetag1.start()])
            _type = match1.group(1)
            level = 1 + depth_level(line[match1.end():next_closetag1.start()])
            if debug:
                print("#current: ", raw)
            raw += line[next_start_pos:next_closetag1.start()]
            if debug:
                print("->", raw)
            entity = OrderedDict()
            entity["type"] = _type
            entity["value"] = value
            entity["start"] = len(raw) - len(value)
            entity["end"] = len(raw)
            entity["level"] = level
            entities.append(entity)
            if debug:
                print("#Entity:", value, _type, level)
            next_start_pos = next_closetag1.end()

            next_match1 = regex_opentag.search(line, next_start_pos)
            next_closetag3 = regex_closetag.search(line, next_start_pos)
            if next_match1:
                if next_closetag3 and next_match1.start() < next_closetag3.start():
                    if debug:
                        print("#Next match1:", line[next_match1.start():])
                    if debug:
                        print("#current: ", raw)
                    raw += line[next_start_pos:next_match1.start()]
                    if debug:
                        print("->", raw)
                    next_start_pos = next_match1.end()
                    stack.append(next_match1)
                else:
                    continue
            else:
                if debug:
                    print("#current: ", raw)
                if debug:
                    print("{} {}".format(next_closetag1.end(), line[next_closetag1.end():]))
                if not re.search(r"</ENAMEX>", line[next_closetag1.end():]):
                    raw += line[next_closetag1.end():]
                    if debug:
                        print("->", raw)

    return raw, entities


def text_normalize(text):
    """
    Chuẩn hóa dấu tiếng Việt
    """

    text = re.sub(r"òa", "oà", text)
    text = re.sub(r"óa", "oá", text)
    text = re.sub(r"ỏa", "oả", text)
    text = re.sub(r"õa", "oã", text)
    text = re.sub(r"ọa", "oạ", text)
    text = re.sub(r"òe", "oè", text)
    text = re.sub(r"óe", "oé", text)
    text = re.sub(r"ỏe", "oẻ", text)
    text = re.sub(r"õe", "oẽ", text)
    text = re.sub(r"ọe", "oẹ", text)
    text = re.sub(r"ùy", "uỳ", text)
    text = re.sub(r"úy", "uý", text)
    text = re.sub(r"ủy", "uỷ", text)
    text = re.sub(r"ũy", "uỹ", text)
    text = re.sub(r"ụy", "uỵ", text)
    text = re.sub(r"Ủy", "Uỷ", text)
    text = re.sub(r'"', '”', text)

    return text


def preprocess(text):
    text = text.strip()
    text = text.strip(u"\ufeff")
    text = text.strip(u"\u200b\u200b\u200b\u200b\u200b\u200b\u200b")
    text = text_normalize(text)
    return text


def get_raw(line):
    regex_opentag = re.compile(r"<ENAMEX TYPE=”(.+?)”>")
    regex_closetag = re.compile(r"</ENAMEX>")
    text = regex_opentag.sub("", line)
    text = regex_closetag.sub("", text)
    return text


def is_end_of_sentence(i, line):
    exception_list = [
        "Mr.",
        "MR.",
        "GS.",
        "Gs.",
        "PGS.",
        "Pgs.",
        "pgs.",
        "TS.",
        "Ts.",
        "ts.",
        "MRS.",
        "Mrs.",
        "mrs.",
        "Tp.",
        "tp.",
        "Kts.",
        "kts.",
        "BS.",
        "Bs.",
        "Co.",
        "Ths.",
        "MS.",
        "Ms.",
        "TT.",
        "TP.",
        "tp.",
        "ĐH.",
        "Corp.",
        "Dr.",
        "Prof.",
        "BT.",
        "Ltd.",
        "P.",
        "MISS.",
        "miss.",
        "TBT.",
        "Q.",
    ]
    if i == len(line) - 1:
        return True

    if line[i + 1] != " ":
        return False

    if i < len(line) - 2 and line[i + 2].islower():
        return False

    if re.search(r"^(\d+|[A-Za-z])\.", line[:i + 1]):
        return False

    for w in exception_list:
        pattern = re.compile("%s$" % w)
        if pattern.search(line[:i + 1]):
            return False

    return True


def is_valid_xml(astring):
    """Check well-formed XML"""
    if not re.search(r"<ENAMEX TYPE=”(.+?)”>", astring):
        return True

    OPEN_TAG = 1
    stack = []
    i = 0
    while i < len(astring):
        # print(astring[i:], stack, level)
        if astring[i:].startswith("<ENAMEX TYPE="):
            stack.append(OPEN_TAG)
            i += len("<ENAMEX TYPE=")
        elif astring[i:].startswith("</ENAMEX>"):
            if len(stack) > 0:
                stack.pop()
            else:
                # raise ValueError(astring)
                print("Invalid XML format: %s" % astring)
                return False
            i += len("</ENAMEX>")
        else:
            i += 1
    if len(stack) > 0:
        return False
    else:
        return True


def sent_tokenize(line):
    """Do sentence tokenization by using regular expression"""
    sentences = []
    cur_pos = 0
    if not re.search(r"\.", line):
        return [line]

    for match in re.finditer(r"\.", line):
        _pos = match.start()
        end_pos = match.end()
        if is_end_of_sentence(_pos, line):
            tmpsent = line[cur_pos:end_pos]
            tmpsent = tmpsent.strip()
            if is_valid_xml(tmpsent):
                cur_pos = end_pos
                sentences.append(tmpsent)

    if len(sentences) == 0:
        sentences.append(line)
    elif cur_pos < len(line) - 1:
        sentences.append(line[cur_pos + 1:])
    return sentences


def create_syl_index(tokens):
    i = 0
    for tk in tokens:
        start_syl_id = i
        end_syl_id = i + len(tk.syllables)
        tk.set_syl_indexes(start_syl_id, end_syl_id)
        i = end_syl_id


def word_tokenize(text, raw_text):
    tokens = []
    syllables = []

    words = text.split()

    _pos = 0
    for w in words:
        syls = []
        _syls = w.split("_")
        _syls = [s for s in _syls if s != ""]
        for s in _syls:
            start = raw_text.find(s, _pos)
            end = start + len(s)
            _pos = end
            syl = Syllable(s, start, end)
            syls.append(syl)
            syllables.append(syl)
        token = Token(syls)
        tokens.append(token)
    create_syl_index(tokens)
    return tokens, syllables


def read(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines]


def xml2tokens(xml_tagged_sent, tokenized_sent, raw_sent):
    """Convert XML-based tagged sentence into CoNLL format based on syllables
    Args:
        xml_tagged_sent (string): Input sentence (single sentence) with XML tags

    Returns:
        tokens (list): list of tuples (tk, level1_tag, level2_tag)
          level1_tag is entity tag (BIO scheme) at the level 1
          level2_tag is entity tag at the level 2 (nested entity)
    """
    raw, entities = get_entities(xml_tagged_sent)
    if re.search(r"ENAMEX", raw):
        print(xml_tagged_sent)
        print(raw)
        # count += 1

    tokens, syllables = word_tokenize(tokenized_sent, raw_sent)
    level1_syl_tags = ["O"] * len(syllables)
    level2_syl_tags = ["O"] * len(syllables)
    level3_syl_tags = ["O"] * len(syllables)

    level1_token_tags = ["O"] * len(tokens)
    level2_token_tags = ["O"] * len(tokens)
    level3_token_tags = ["O"] * len(tokens)

    flag = False
    for entity in entities:
        value = entity["value"]
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]
        start_syl_id, end_syl_id = find_syl_index(start, end, syllables)
        start_tok_id, end_tok_id = find_tok_index(start_syl_id, end_syl_id, tokens)

        if start_syl_id is not None and end_syl_id is not None:
            if entity["level"] == 1:
                level1_syl_tags[start_syl_id] = "B-" + entity_type
                for i in range(start_syl_id + 1, end_syl_id):
                    level1_syl_tags[i] = "I-" + entity_type
            elif entity["level"] == 2:
                level2_syl_tags[start_syl_id] = "B-" + entity_type
                for i in range(start_syl_id + 1, end_syl_id):
                    level2_syl_tags[i] = "I-" + entity_type
            else:
                level3_syl_tags[start_syl_id] = "B-" + entity_type
                for i in range(start_syl_id + 1, end_syl_id):
                    level3_syl_tags[i] = "I-" + entity_type
        else:
            print("{},{},”{}” in '{}' ({})".format(start, end, value, raw, xml_tagged_sent))
            flag = True

        if start_tok_id is not None and end_tok_id is not None:
            if entity["level"] == 1:
                level1_token_tags[start_tok_id] = "B-" + entity_type
                for i in range(start_tok_id + 1, end_tok_id):
                    level1_token_tags[i] = "I-" + entity_type
            elif entity["level"] == 2:
                level2_token_tags[start_tok_id] = "B-" + entity_type
                for i in range(start_tok_id + 1, end_tok_id):
                    level2_token_tags[i] = "I-" + entity_type
            else:
                level3_token_tags[start_tok_id] = "B-" + entity_type
                for i in range(start_tok_id + 1, end_tok_id):
                    level3_token_tags[i] = "I-" + entity_type
        else:
            pass
            # print("{},{},”{}” in '{}' ({})".format(start_syl_id, end_syl_id, value, raw, xml_tagged_sent))

    ret_syllables = list(zip([s.text for s in syllables], level1_syl_tags, level2_syl_tags, level3_syl_tags))
    ret_tokens = list(zip([tk.text for tk in tokens], level1_token_tags, level2_token_tags, level3_token_tags))
    return ret_syllables, ret_tokens, raw, flag


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sent_tokenize", action="store_true")
    parser.add_argument("--tmpdir", default="./work_dir")
    parser.add_argument("data_dir")
    parser.add_argument("syllable_out")
    parser.add_argument("ws_out")
    args = parser.parse_args()

    print("# Data directory: {}".format(args.data_dir))
    print("# Temporary directory: {}".format(args.tmpdir))
    tmpdir = args.tmpdir
    pathlib.Path(tmpdir).mkdir(exist_ok=True, parents=True)
    dirname = os.path.basename(args.data_dir).split("/")[-1]
    raw_text = os.path.join(tmpdir, dirname + "-raw.txt")
    xml_text = os.path.join(tmpdir, dirname + "-xml.txt")
    subdirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    fo1 = open(raw_text, "w", encoding="utf-8")
    fo2 = open(xml_text, "w", encoding="utf-8")
    for d in subdirs:
        dd = os.path.join(args.data_dir, d)
        files = [f for f in os.listdir(dd) if os.path.isfile(os.path.join(dd, f))]
        for filename in files:
            ff = os.path.join(dd, filename)
            with open(ff, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    line = line.strip(u"\ufeff")
                    line = line.strip(u"\u200b\u200b\u200b\u200b\u200b\u200b\u200b")
                    line = text_normalize(line)
                    if line == "":
                        continue
                    if args.sent_tokenize:
                        sentences = sent_tokenize(line)
                        for s in sentences:
                            raw = get_raw(s)
                            fo1.write("%s\n" % raw)
                            fo2.write("%s\n" % s)
                    else:
                        raw = get_raw(line)
                        fo1.write("%s\n" % raw)
                        fo2.write("%s\n" % line)
    fo1.close()
    fo2.close()

    # Do word-segmentation
    comd = "./word_segment.sh %s" % raw_text
    print(comd)
    os.system(comd)

    tokenized_file = raw_text + ".WS"
    raw_lines = read(raw_text)
    xml_lines = read(xml_text)
    tokenized_lines = read(tokenized_file)
    assert len(xml_lines) == len(tokenized_lines)
    assert len(xml_lines) == len(raw_lines)

    fo1 = open(args.syllable_out, "w", encoding="utf-8")
    fo2 = open(args.ws_out, "w", encoding="utf-8")
    for xml_sent, tokenized_sent, raw_sent in zip(xml_lines, tokenized_lines, raw_lines):
        syllables, tokens, _, _ = xml2tokens(xml_sent, tokenized_sent, raw_sent)
        raw_ = "".join([s[0] for s in syllables])
        if raw_ == "":
            print(xml_sent)
            continue
        for tp in syllables:
            fo1.write("{}\n".format("\t".join(tp)))
        fo1.write("\n")
        for tp in tokens:
            fo2.write("{}\n".format("\t".join(tp)))
        fo2.write("\n")
    fo1.close()
    fo2.close()
