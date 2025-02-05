import streamlit as st
import os# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import re

MODELS_DIR = "./models" 

# Language mapping
LANGUAGE_MAPPING = {
    "dzo": "dzo_Tibt",
    "eng": "eng_Latn"
}

#for preprocessing text files assuming they have paragraphs
def process_txt(text):
    paragraphs = text.strip().split("\n\n")
    sentences = []

    for paragraph in paragraphs:
        split_sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        sentences.extend(split_sentences)
    return [s for s in sentences if s]

#for listing all the models in model folder
def list_models():
    if os.path.exists(MODELS_DIR):
        return [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]
    return []

@st.cache_resource
def load_model(path,tgt_lang,src_lang):
    st.write(path, tgt_lang, src_lang)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    return model, tokenizer

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").input_ids
    translated_tokens = model.generate(
        inputs, max_new_tokens=512
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

st.title("Translation App")

with st.expander("Load Model:"):
    st.write("The models should be placed in the models directory!!!")
    st.write("Select a model from the list below:")
    
    models = list_models()

    if models:
        # Select model from available options
        selected_model = st.selectbox("Available Models:", models)
        model_path = os.path.join(MODELS_DIR, selected_model)
        st.write(f"**Selected Model:** {selected_model}")

        src,tgt = st.columns((2))

        # Dropdowns for target and source language
        with src:
            src_lang_key = st.selectbox("Select Source Language:", ["dzo", "eng"])

        with tgt:
            tgt_lang_key = st.selectbox("Select Target Language:", ["dzo", "eng"])

        tgt_lang = LANGUAGE_MAPPING[tgt_lang_key]
        src_lang = LANGUAGE_MAPPING[src_lang_key]

        if st.button("Load Model"):
            model, tokenizer = load_model(model_path, tgt_lang, src_lang)
            st.session_state["model"] = model
            st.session_state["tokenizer"] = tokenizer
            st.success(f"✅ Model '{selected_model}' loaded successfully!")
    
    else:
        st.write("No models found in the `/models/` directory.")

with st.expander("Translate:"):

    text_to_translate = st.text_input(src_lang_key)
    if st.button("Translate"):
        if text_to_translate:
            if "model" in st.session_state and "tokenizer" in st.session_state:
                translated_text = translate(text_to_translate, st.session_state["model"], st.session_state["tokenizer"])
                st.text_input(tgt_lang_key, value=translated_text)
            else:
                st.error("Please load a model first.")
        else:
            st.warning("Please enter text to translate.")

with st.expander("Batch Translation:"):
    st.write("Only csv and text files are allowed.")
    st.write("csv file is expected to have rows of sentences and text file contains paragraphs or rows of sentence")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None, names=['Source'])  # Ensure single-column input
            st.write(df)

            if st.button("Batch Translate"):
                if "model" in st.session_state and "tokenizer" in st.session_state:
                    df["Translation"] = df["Source"].apply(
                        lambda x: translate(str(x), st.session_state["model"], st.session_state["tokenizer"])
                    )
                    st.write("Translated Data:", df.head())

                    # Ensure correct column names before exporting
                    csv = df.to_csv(index=False, header=True).encode('utf-8')
                    st.download_button("Download Translated CSV", csv, "translated.csv", "text/csv")
                else:
                    st.write("please upload model first")
        elif uploaded_file.name.endswith(".txt"):
            text_data =  uploaded_file.getvalue().decode("utf-8")
            sentences = process_txt(text_data)

            df = pd.DataFrame(sentences, columns=['Source'])
            st.write(df)

            if st.button("Batch Translate"):
                if "model" in st.session_state and "tokenizer" in st.session_state:
                    df["Translation"] = df["Source"].apply(
                        lambda x: translate(str(x), st.session_state["model"], st.session_state["tokenizer"])
                    )
                    st.write("Translated Data:", df.head())

                    # Provide a CSV download option
                    csv = df.to_csv(index=False, header=True).encode('utf-8')
                    st.download_button("Download Translated CSV", csv, "translated.csv", "text/csv")
                else:
                    st.error("Please load a model first.")


    
