from annotated_text import annotated_text

from vphoberttagger.arguments import get_predict_argument
from vphoberttagger.predictor import ViTagger

import streamlit as st

args = get_predict_argument()
st.markdown("<h1 style='text-align: center;'>ViTagger Demo</h1>", unsafe_allow_html=True)
@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def load_models(model_path, no_cuda):
    tagger = ViTagger(model_path, no_cuda=no_cuda)
    return tagger
def process_text(doc, expected_tag):
    tokens = []
    for token in doc:
        if token[-1] in expected_tag and token[-1] in ["PER", "PERSON"]:
            tokens.append((token[0], "PERSON", "#fef6ce"))
        elif token[-1] in expected_tag and token[-1] in ["ORG", "ORGANIZATION"]:
            tokens.append((token[0], "ORGANIZATION", "#d8f2fe"))
        elif token[-1] in expected_tag and token[-1] in ["LOC", "LOCATION"]:
            tokens.append((token[0], "LOCATION", "#f4cfdb"))
        elif token[-1] in expected_tag and token[-1] in ["MISC", "MISCELLANEOUS"]:
            tokens.append((token[0], "MISCELLANEOUS", "#edfcd9"))
        else:
            tokens.append(" " + token[0] + " ")
    return tokens
tagger = load_models(args.model_path, args.no_cuda)
labels = []
for l in tagger.label2id:
    if l == 'O':
        continue
    else:
        _, tag = l.split("-")
        if tag not in labels:
            labels.append(tag)
tags = st.multiselect("Which label do you extract?", labels, default=labels)
text_input = st.text_area("Enter your text to extract")
if st.button('Extract'):
    doc = tagger.extract_entity_doc(text_input)
    tokens = process_text(doc, tags)
    annotated_text(*tokens)