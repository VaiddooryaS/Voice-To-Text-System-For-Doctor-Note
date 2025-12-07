# Voice-To-Text-System-For-Doctor-Note

1. Implemented an end-to-end speech-processing pipeline using Whisper ASR to transcribe noisy clinical audio with high accuracy across accents and real-world medical environments.

2. Designed a robust audio preprocessing module with noise reduction, voice activity detection (VAD), and signal normalization to enhance transcription reliability in complex acoustic settings.

3. Integrated SciSpaCy’s en_core_sci_lg biomedical NER model to extract symptoms, diagnoses, medications, vitals, allergies, and patient demographics from unstructured medical speech.

4. Developed a hybrid entity-extraction approach combining NER with custom regex rules to improve precision and recall for clinical concepts not covered by standard biomedical models.

5. Engineered a modular processing pipeline that orchestrates audio upload, preprocessing, transcription, entity extraction, and structured output generation through a single callable function.

6. Built a Streamlit-ready interactive interface to display transcripts and extracted medical information, enabling clinicians to review, validate, and integrate structured notes into EMR workflows.
