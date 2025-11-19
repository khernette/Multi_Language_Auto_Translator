import tempfile
from datetime import datetime

import streamlit as st
import sounddevice as sd
import wavio
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(
    page_title="Smart Multi-Language Voice Translator",
    page_icon="🌏",
    layout="centered"
)

# ============================
# LOAD WHISPER MODEL
# ============================
@st.cache_resource
def load_whisper_model():
    # You can change to "small" for better accuracy if your machine can handle it
    return whisper.load_model("base")

model = load_whisper_model()

# ============================
# LANGUAGE CONFIG
# ============================

# Language pairs (two-letter codes used by GoogleTranslate + gTTS)
LANGUAGE_PAIRS = {
    "English ↔ Arabic": ("en", "ar"),
    "English ↔ Sinhala": ("en", "si"),
    "English ↔ Filipino (Tagalog)": ("en", "tl"),
    "English ↔ Hindi": ("en", "hi"),
}

# Friendly display names
LANG_FRIENDLY = {
    "en": "English",
    "ar": "Arabic",
    "si": "Sinhala",
    "tl": "Filipino",
    "hi": "Hindi",
}

# gTTS compatible codes
GTTS_MAP = {
    "en": "en",
    "ar": "ar",
    "si": "si",
    "tl": "tl",
    "hi": "hi",
}

# Whisper language code normalization
WHISPER_LANG_NORMALIZE = {
    "en": "en",
    "ar": "ar",
    "si": "si",
    "tl": "tl",
    "fil": "tl",  # sometimes Tagalog/Filipino may appear as 'fil'
    "hi": "hi",
}


# ============================
# HELPER FUNCTIONS
# ============================

def record_audio(duration=5, fs=16000):
    """Record audio from the microphone for 'duration' seconds."""
    st.info(f"Recording for {duration} seconds… Please speak.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_wav.name, recording, fs, sampwidth=2)
    return temp_wav.name


def transcribe_and_detect(audio_path):
    """
    Use Whisper to both transcribe and auto-detect language.
    Returns (recognized_text, detected_lang_code_raw, normalized_lang_code)
    """
    result = model.transcribe(audio_path)  # auto detect language
    text = result.get("text", "").strip()
    lang_raw = result.get("language", "")
    lang_norm = WHISPER_LANG_NORMALIZE.get(lang_raw, lang_raw)
    return text, lang_raw, lang_norm


def translate_text(text, src_lang, tgt_lang):
    """Translate text using GoogleTranslator via deep-translator."""
    translator = GoogleTranslator(source=src_lang, target=tgt_lang)
    return translator.translate(text)


def text_to_speech(text, lang_code):
    """Convert text to speech using gTTS and return path to MP3."""
    tts = gTTS(text=text, lang=lang_code)
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_mp3.name)
    return temp_mp3.name


def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []


def add_turn(src_lang, tgt_lang, src_text, tgt_text, detected_raw):
    """
    Save a full conversation turn to history.
    """
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_text": src_text,
        "tgt_text": tgt_text,
        "detected_raw": detected_raw,
    })


def render_history():
    """
    Show multi-turn conversation memory.
    """
    if "history" not in st.session_state or not st.session_state.history:
        return

    st.markdown("### 🕑 Conversation History")
    for turn in st.session_state.history[::-1]:  # newest first
        time = turn["time"]
        src = turn["src_lang"]
        tgt = turn["tgt_lang"]
        src_text = turn["src_text"]
        tgt_text = turn["tgt_text"]
        detected_raw = turn["detected_raw"]

        st.markdown(f"**🧑 [{time}] ({LANG_FRIENDLY.get(src, src)})**")
        st.write(src_text)
        st.markdown(
            f"*Detected by Whisper as:* `{detected_raw}` → normalized as `{src}`"
        )

        st.markdown(f"**🤖 Translation ({LANG_FRIENDLY.get(tgt, tgt)})**")
        st.write(tgt_text)
        st.markdown("---")


# ============================
# UI
# ============================

init_history()

st.title("🌏 Smart Multi-Language Voice Translator")
st.write(
    "Auto-detects whether you speak English, Arabic, Sinhala, Filipino, or Hindi, "
    "and translates to the other language in the selected pair. "
    "Perfect for back-and-forth conversations."
)

st.markdown("---")

# 1) Choose language pair (auto conversation mode inside the pair)
pair_label = st.selectbox(
    "Choose Translation Pair:",
    list(LANGUAGE_PAIRS.keys())
)

lang_a, lang_b = LANGUAGE_PAIRS[pair_label]
lang_a_name = LANG_FRIENDLY[lang_a]
lang_b_name = LANG_FRIENDLY[lang_b]

st.info(
    f"Auto Conversation Mode: Speak either **{lang_a_name}** or **{lang_b_name}**.\n\n"
    f"If you speak {lang_a_name}, it translates to {lang_b_name}.\n"
    f"If you speak {lang_b_name}, it translates to {lang_a_name}."
)

duration = st.slider("🎙 Recording Duration (seconds)", 3, 15, 5)

col_buttons = st.columns([1, 1])
with col_buttons[0]:
    start = st.button("🎙 Start Recording")
with col_buttons[1]:
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.success("Conversation history cleared.")

st.markdown("---")

if start:
    try:
        # Record
        audio_path = record_audio(duration=duration, fs=16000)
        st.audio(audio_path, format="audio/wav")

        # Transcribe + auto-detect
        st.info("Transcribing and detecting language...")
        recognized_text, detected_raw, detected_norm = transcribe_and_detect(audio_path)

        if not recognized_text:
            st.error("Could not detect any speech. Please try again.")
        else:
            st.success("Recognized Speech:")
            st.write(recognized_text)
            st.markdown(
                f"*Whisper detected language:* `{detected_raw}` → normalized: `{detected_norm}`"
            )

            # Decide direction based on detected language
            if detected_norm == lang_a:
                src_lang = lang_a
                tgt_lang = lang_b
            elif detected_norm == lang_b:
                src_lang = lang_b
                tgt_lang = lang_a
            else:
                st.error(
                    f"Detected language `{detected_norm}` is not part of the selected pair "
                    f"({lang_a_name} ↔ {lang_b_name}). Please speak in one of those languages."
                )
                src_lang = None
                tgt_lang = None

            if src_lang and tgt_lang:
                # Translate
                st.info(
                    f"Translating from {LANG_FRIENDLY[src_lang]} to {LANG_FRIENDLY[tgt_lang]}..."
                )
                translated = translate_text(recognized_text, src_lang, tgt_lang)

                st.success(f"Translation ({LANG_FRIENDLY[tgt_lang]}):")
                st.write(translated)

                # Save to multi-turn memory
                add_turn(src_lang, tgt_lang, recognized_text, translated, detected_raw)

                # Voice output
                st.info("Generating voice output...")
                tts_lang_code = GTTS_MAP.get(tgt_lang, "en")
                mp3_path = text_to_speech(translated, tts_lang_code)
                st.audio(mp3_path, format="audio/mp3")

    except Exception as e:
        st.error(f"Error: {e}")

render_history()
