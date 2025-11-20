import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

st.set_page_config(page_title='Lab6 — Poetry LSTM Demo', layout='wide')
st.title('Lab 6 — Poetry LSTM Demo')
st.markdown('Interactive demo for preprocessing, model training, and poetry generation from `PoetryFoundationData.csv`.')

DATA_PATH = 'PoetryFoundationData.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lab6_lstm.h5')
HISTORY_PATH = 'training_history.json'


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


@st.cache_data
def prepare_corpus(df, sample_frac=1.0):
    poems = df['Poem'].astype(str).map(lambda s: s.replace(
        '\r\n', '\n').replace('\r', '\n').strip())
    if sample_frac < 1.0:
        poems = poems.sample(
            frac=sample_frac, random_state=42).reset_index(drop=True)
    corpus_list = poems.tolist()
    cleaned = [clean_text(p) for p in corpus_list if str(p).strip() != '']
    return cleaned


@st.cache_data
def build_tokenizer(cleaned_poems, oov_tok='<OOV>'):
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(cleaned_poems)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


@st.cache_data
def create_sequences(cleaned_poems, _tokenizer, n):
    sequences = []
    for text in cleaned_poems:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) <= n:
            continue
        for i in range(n, len(token_list)):
            seq = token_list[i - n: i + 1]
            sequences.append(seq)
    if len(sequences) == 0:
        return None, None
    sequences = np.array(sequences, dtype=np.int32)
    X = sequences[:, :-1]
    y = sequences[:, -1]
    X = pad_sequences(X, maxlen=n, padding='pre')
    return X, y


@st.cache_resource
def build_model(vocab_size, n, embedding_dim=100, lstm_units=(100, 100), dropout=0.2):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
              output_dim=embedding_dim, input_length=n))
    for i, u in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        model.add(LSTM(u, return_sequences=return_seq))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Sidebar controls
st.sidebar.header('Controls')
sample_frac = st.sidebar.slider(
    'Corpus sample fraction', 0.1, 1.0, 0.25, step=0.05)
context_len = st.sidebar.selectbox('Context length (n)', [3, 5, 8], index=1)
embed_dim = st.sidebar.selectbox('Embedding dim', [50, 100, 200], index=1)
lstm_cfg = st.sidebar.selectbox(
    'LSTM units config', ['(100,)', '(100,100)'], index=1)
if lstm_cfg == '(100,)':
    lstm_units = (100,)
else:
    lstm_units = (100, 100)
dropout = st.sidebar.slider('Dropout rate', 0.0, 0.5, 0.2, step=0.05)
epochs = st.sidebar.slider('Train epochs', 1, 20, 6)
batch_size = st.sidebar.selectbox('Batch size', [128, 256, 512], index=2)

load_button = st.sidebar.button('Load saved model')
train_button = st.sidebar.button('Train model')
save_button = st.sidebar.button('Save trained model')

# Load CSV
if not os.path.exists(DATA_PATH):
    st.warning(
        f'Please place `PoetryFoundationData.csv` in the folder: {os.getcwd()} or upload below.')
    uploaded = st.file_uploader(
        'Upload PoetryFoundationData.csv', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.stop()
else:
    df = load_csv(DATA_PATH)

st.subheader('Dataset preview')
st.write(df.head())

# Prepare corpus
cleaned_poems = prepare_corpus(df, sample_frac=sample_frac)
st.write(f'Number of poems used: {len(cleaned_poems)}')

# Tokenizer
tokenizer, vocab_size = build_tokenizer(cleaned_poems)
st.write(f'Vocab size (approx): {vocab_size}')

# Create sequences
X, y = create_sequences(cleaned_poems, tokenizer, context_len)
if X is None:
    st.error(
        'No sequences created. Try reducing context length or increasing sample fraction.')
    st.stop()

st.write('Sequence shapes:')
st.write({'X': X.shape, 'y': y.shape})

# Quick decode helper
idx_to_word = {i: w for w, i in tokenizer.word_index.items()}


def decode_tokens(tokens):
    return ' '.join(idx_to_word.get(int(t), '<OOV>') for t in tokens if t != 0)


st.subheader('Example sequence')
st.write('Context:', decode_tokens(X[0]))
st.write('Target:', idx_to_word.get(int(y[0]), '<OOV>'))

# Buttons
model = None
if load_button:
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.success(f'Loaded model from {MODEL_PATH}')
        except Exception as e:
            st.error(f'Error loading model: {e}')
    else:
        st.info('No saved model found. Train and save a model first.')

if train_button:
    st.info('Preparing dataset and training model — this may take a few minutes depending on settings')
    # One-hot encode y per-batch during tf.data pipeline to avoid large memory
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(10000).map(lambda x, y: (
        x, tf.one_hot(y, depth=vocab_size, dtype=tf.float32)))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(vocab_size=vocab_size, n=context_len,
                        embedding_dim=embed_dim, lstm_units=lstm_units, dropout=dropout)
    history = model.fit(dataset, epochs=epochs, verbose=1)

    # Save history
    with open(HISTORY_PATH, 'w', encoding='utf-8') as fh:
        json.dump(history.history, fh)

    # Plot loss
    fig, ax = plt.subplots()
    ax.plot(history.history.get('loss', []), label='loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

if model is None:
    st.info(
        'No model loaded or trained yet — use the sidebar to train or load a saved model.')

# Generation
st.subheader('Generate poetry')
seed_text = st.text_input('Seed text', value='the moon')
num_words = st.number_input(
    'Words to generate', min_value=5, max_value=100, value=20)
temp = st.slider('Temperature', 0.1, 1.5, 0.8)
use_sampling = st.checkbox('Use sampling', value=True)


def sample_from_preds(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds[0] = 0.0
    if temperature <= 0 or np.isclose(temperature, 0.0):
        return int(np.argmax(preds))
    preds = preds / np.sum(preds)
    log_preds = np.log(preds + 1e-12) / temperature
    exp_preds = np.exp(log_preds - np.max(log_preds))
    probs = exp_preds / np.sum(exp_preds)
    return int(np.random.choice(len(probs), p=probs))


@st.cache_data
def generate_text(seed, num_words, temperature, sample_flag, tokenizer, model, n):
    cleaned = clean_text(seed)
    token_list = tokenizer.texts_to_sequences([cleaned])[0]
    generated = token_list.copy()
    for _ in range(num_words):
        input_seq = pad_sequences([generated[-n:]], maxlen=n, padding='pre')
        preds = model.predict(input_seq, verbose=0)[0]
        if sample_flag:
            next_idx = sample_from_preds(preds, temperature=temperature)
        else:
            preds[0] = 0.0
            next_idx = int(np.argmax(preds))
        generated.append(next_idx)
    words = [idx_to_word.get(t, '<OOV>') for t in generated if t != 0]
    return ' '.join(words)


if model is not None:
    if st.button('Generate'):
        with st.spinner('Generating...'):
            generated = generate_text(seed_text, int(num_words), float(
                temp), use_sampling, tokenizer, model, context_len)
            st.text_area('Generated text', value=generated, height=200)

# Save model
if save_button and model is not None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    st.success(f'Saved model to {MODEL_PATH}')

st.markdown('---')
st.markdown('Notes: This app reproduces the lab pipeline — cleaning, tokenization, creating sequences, building and training LSTM models, and generating text. For faster interactive use keep epochs small and sample fraction modest.')
