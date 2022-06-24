from annotated_text import annotated_text

from .predictor import ViTagger

import streamlit as st


@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def load_models(model_path):
    tagger = ViTagger(model_path, no_cuda=True)
    return tagger

def process_text(doc, selected_entities):
    tokens = []
    for token in doc:
        if (token[-1] == "PERSON") & ("PER" in selected_entities):
            tokens.append((token.text, "Person", "#faa"))
        elif (token.ent_type_ in ["GPE", "LOC"]) & ("LOC" in selected_entities):
            tokens.append((token.text, "Location", "#fda"))
        elif (token.ent_type_ == "ORG") & ("ORG" in selected_entities):
            tokens.append((token.text, "Organization", "#afa"))
        else:
            tokens.append(" " + token.text + " ")

    if anonymize:
        anonmized_tokens = []
        for token in tokens:
            if type(token) == tuple:
                anonmized_tokens.append(("X" * len(token[0]), token[1], token[2]))
            else:
                anonmized_tokens.append(token)
        return anonmized_tokens

    return tokens

tagger = load_models()

text_input = st.text_area("Type a text to anonymize")

anonymize = st.checkbox("Anonymize")
doc = tagger.extract_entity_doc(text_input)
tokens = process_text(doc, selected_entities)

annotated_text(*tokens)