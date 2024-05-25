import io
import os
import torch
import streamlit as st
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from faster_whisper import WhisperModel
# from trans.IndicTransTokenizer import IndicTransTokenizer
from trans.IndicTransTokenizer.IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
# from IndicTransTokenizer.IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

# Set the port address
os.environ["STREAMLIT_SERVER_PORT"] = "8501"

# Initialize models
model_size = "small"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

def initialize_model_and_tokenizer(ckpt_dir, direction):
    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model

# def initialize_model_and_tokenizer(ckpt_dir, direction):
#     tokenizer = IndicTransTokenizer(direction=direction)
#     model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, trust_remote_code=True)
#     model = model.to(DEVICE).half() if DEVICE == "cuda" else model.to(DEVICE)
#     model.eval()
#     return tokenizer, model

en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic")
ip = IndicProcessor(inference=True)

# en_indic_model = AutoModelForSeq2SeqLM.from_pretrained("en-indic-mod", trust_remote_code=True)
# en_indic_tokenizer = IndicTransTokenizer(direction="en-indic")
# ip = IndicProcessor(inference=True)

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Session state
if 'english_text' not in st.session_state:
    st.session_state['english_text'] = ''
    st.session_state['translated_text'] = ''
    st.session_state['run'] = False

# Audio parameters
st.sidebar.header('Audio Parameters')
ENERGY_THRESHOLD = int(st.sidebar.text_input('Energy Threshold', 1000))
RECORD_TIMEOUT = float(st.sidebar.text_input('Record Timeout (seconds)', 2))
PHRASE_TIMEOUT = float(st.sidebar.text_input('Phrase Timeout (seconds)', 3))

args = {
    'energy_threshold': ENERGY_THRESHOLD,
    'record_timeout': RECORD_TIMEOUT,
    'phrase_timeout': PHRASE_TIMEOUT,
}

# The last time a recording was retrieved from the queue.
phrase_time = None
# Current raw audio bytes.
last_sample = bytes()
# Thread-safe Queue for passing data from the threaded recording callback.
data_queue = Queue()
# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = args['energy_threshold']
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

# Try to initialize the microphone
try:
    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)
except Exception as e:
    st.error(f"Error initializing microphone: {e}")
    source = None

def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push them into the thread-safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(input_sentences[i: i + BATCH_SIZE], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, src=True, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
        generated_tokens = tokenizer.batch_decode(generated_tokens.cpu().tolist(), src=False)
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    torch.cuda.empty_cache()
    return translations

if source:
    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually, but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=args['record_timeout'])

def transcribe(audio_content):
    segments, _ = model.transcribe(audio_content, beam_size=5, language="en")
    return segments

def translate(subtitle, src_lang, tgt_lang):
    translated_subtitle = batch_translate(subtitle, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)
    return translated_subtitle

def send_receive():
    global phrase_time, last_sample

    while st.session_state['run']:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=args['phrase_timeout']):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to WAV data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Transcribe the audio
                segments = transcribe(wav_data)
                english_text = [segment.text for segment in segments]

                # Translate the text
                src_lang = "eng_Latn"
                tgt_lang = st.session_state.get('tgt_lang', "hin_Deva")
                translated_text = translate(english_text, src_lang, tgt_lang)

                # Update session state text
                st.session_state['english_text'] += '\n'.join(english_text) + '\n'
                st.session_state['translated_text'] += '\n'.join(translated_text) + '\n'

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name == 'nt' else 'clear')
                st.write('\n'.join(english_text))
                st.write('\n'.join(translated_text))
                # Flush stdout.
                st.write('', end='', flush=True)

                # Infinite loops are bad for processors; must sleep.
                sleep(0.25)

        except KeyboardInterrupt:
            break

# Web user interface
st.title('🎙️ Real-Time Subtitle Translator')

with st.expander('Instructions'):
    st.markdown('''
    - Click on Start, Wait for 10 second for once to model load.
    - Now start Speaking and see what you spoke in next 2-3 second.
    ''')

col1, col2 = st.columns(2)

col1.button('Start', on_click=lambda: st.session_state.update({'run': True}))
col2.button('Stop', on_click=lambda: st.session_state.update({'run': False}))

tgt_lang = st.selectbox("Select target language", ["hin_Deva", "guj_Gujr", "tam_Taml", "tel_Telu"], key="tgt_lang")

if st.session_state['run'] and source:
    send_receive()
else:
    st.write("Click Start to begin real-time transcription and translation.")

col1, col2 = st.columns(2)
with col1:
    st.text_area("English Text", value=st.session_state['english_text'], height=400)
with col2:
    st.text_area("Translated Text", value=st.session_state['translated_text'], height=400)
