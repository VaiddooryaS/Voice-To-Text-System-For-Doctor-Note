import streamlit as st
import whisper
import torch
import os
import re
import noisereduce as nr
import scipy.io.wavfile as wav
import spacy
import shutil
from pydub import AudioSegment
import webrtcvad
import collections
import numpy as np
from tempfile import NamedTemporaryFile

# --- Load Models ---
asr_model = whisper.load_model("base")
nlp = spacy.load("en_core_sci_lg")

# --- Simple VAD ---
def simple_vad(audio_segment, aggressiveness=3):
    vad = webrtcvad.Vad(aggressiveness)
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio_segment.get_array_of_samples())
    sample_rate = 16000
    frame_duration = 30  # ms
    bytes_per_sample = 2
    frame_size = int(sample_rate * frame_duration / 1000) * bytes_per_sample
    frame_step = frame_size // bytes_per_sample

    speech = AudioSegment.empty()
    for i in range(0, len(samples), frame_step):
        frame = samples[i:i + frame_step]
        if len(frame) < frame_step:
            break
        raw = frame.astype(np.int16).tobytes()
        if vad.is_speech(raw, sample_rate):
            speech += AudioSegment(data=raw, sample_width=2, frame_rate=16000, channels=1)
    return speech

# --- Preprocess Audio ---
def preprocess_audio(mp3_path):
    wav_path = "converted.wav"
    audio = AudioSegment.from_mp3(mp3_path).set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")

    # Apply VAD
    speech_only = simple_vad(audio)
    speech_only.export("cleaned.wav", format="wav")

    # Denoising
    rate, data = wav.read("cleaned.wav")
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wav.write("denoised.wav", rate, reduced_noise)

    return "denoised.wav"

# --- Transcription ---
def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path)
    return result["text"]

# --- NER Extraction ---
def extract_medical_entities_pure(text):
    import re
    from scispacy.abbreviation import AbbreviationDetector
    nlp.add_pipe("abbreviation_detector")
    doc = nlp(text)

    sections = {
        "Patient Demographics": set(),
        "Medications": set(),
        "Allergies": set(),
        "Vitals": set(),
        "Diagnosis": set(),
        "Treatment Plan": set()
    }

    demographics_prefixes = [r"\b(name is|born on|age|years? old|female|male|girl|boy|address|contact number|occupation|job|Miss|Mr|Mrs|lives at|phone number)\b"]
    diagnosis_prefixes = [r"\b(diagnosed with|diagnosis|assessment|clinical impression|confirmed as|likely secondary to|indicative of|suggests)\b"]
    vitals_patterns = [r"\b(blood pressure|heart rate|pulse|temperature|oxygen saturation|vitals)\b", r"\b\d{2,3}/\d{2,3}\b", r"\b\d{2,3} ?bpm\b", r"\b\d{2,3}%\b", r"\b(normal|elevated|within range)\b"]
    treatment_patterns = [r"\b(treatment plan|start(ed)? on|continue|management|advised|monitor|follow[- ]?up|labs|tests|evaluation|referred to|dietary advice|education)\b"]
    allergy_patterns = [r"\b(allergic to|allergies|no known allergies|nkda|no drug allergies|no food allergies|hypersensitivity)\b"]
    med_patterns = [r"\b(prescribed|medication|drug|taking|takes|administered|therapy|supplements|iron|oral|iv)\b"]

    for sentence in re.split(r'(?<=[.!?])\s+', text):
        lowered = sentence.lower().strip()
        if any(re.search(p, lowered) for p in demographics_prefixes):
            sections["Patient Demographics"].add(sentence.strip())
        if any(re.search(p, lowered) for p in diagnosis_prefixes):
            sections["Diagnosis"].add(sentence.strip())
        if any(re.search(p, lowered) for p in vitals_patterns):
            sections["Vitals"].add(sentence.strip())
        if any(re.search(p, lowered) for p in treatment_patterns):
            sections["Treatment Plan"].add(sentence.strip())
        if any(re.search(p, lowered) for p in allergy_patterns):
            sections["Allergies"].add(sentence.strip())
        if any(re.search(p, lowered) for p in med_patterns):
            if not re.search(r"\b(no (long[- ]?term )?medications?|not on any|none)\b", lowered):
                sections["Medications"].add(sentence.strip())

    for ent in doc.ents:
        ent_text = ent.text.strip()
        label = ent.label_.lower()
        if not ent_text or len(ent_text.split()) > 6:
            continue
        if label == 'person':
            sections["Patient Demographics"].add(ent_text)
        elif label in ['drug', 'chemical', 'treatment']:
            sections["Medications"].add(ent_text)
        elif label in ['disease', 'diagnosis', 'condition']:
            sections["Diagnosis"].add(ent_text)
        elif label in ['procedure']:
            sections["Treatment Plan"].add(ent_text)

    return {k: sorted(list(v)) for k, v in sections.items()}

# --- Streamlit UI ---
st.title("🎙️ Medical Audio Transcription & Entity Extraction")

uploaded_file = st.file_uploader("Upload an MP3 recording", type=["mp3"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.info("Preprocessing audio...")
    cleaned_audio = preprocess_audio(temp_path)

    st.info("Transcribing audio...")
    transcript = transcribe_audio(cleaned_audio)
    st.subheader("Transcript")
    st.write(transcript)

    st.info("Extracting medical entities...")
    entities = extract_medical_entities_pure(transcript)
    st.subheader("Extracted Medical Information")
    for section, items in entities.items():
        st.markdown(f"**{section}**")
        for item in items:
            st.write(f"- {item}")

    os.remove(temp_path)
