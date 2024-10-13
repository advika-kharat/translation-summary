import os
import time
import pygame
from gtts import gTTS
import streamlit as st
import speech_recognition as sr
from googletrans import LANGUAGES, Translator
import fasttext

isTranslateOn = False

translator = Translator()  # Initialize the translator module.
pygame.mixer.init()  # Initialize the mixer module.

# Create a mapping between language names and language codes
language_mapping = {name: code for code, name in LANGUAGES.items()}

# Load FastText pre-trained language identification model
model = fasttext.load_model('C:/Users/Advika/Desktop/real-time-language-translator/fastText_model/lid.176.bin')


def get_language_code(language_name):
    return language_mapping.get(language_name, language_name)


def detect_language(spoken_text):
    predictions = model.predict(spoken_text, k=1)  # Predict top language
    lang_code = predictions[0][0].split("__label__")[1]  # Extract language code
    return lang_code


def translator_function(spoken_text, from_language, to_language):
    return translator.translate(spoken_text, src='{}'.format(from_language), dest='{}'.format(to_language))


def text_to_voice(text_data, to_language):
    myobj = gTTS(text=text_data, lang='{}'.format(to_language), slow=False)
    myobj.save("cache_file.mp3")
    audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
    audio.play()
    os.remove("cache_file.mp3")


def main_process(output_placeholder, from_language, to_language):
    global isTranslateOn

    while isTranslateOn:

        rec = sr.Recognizer()
        with sr.Microphone() as source:
            output_placeholder.text("Listening...")
            rec.pause_threshold = 1
            audio = rec.listen(source, phrase_time_limit=10)

        try:
            output_placeholder.text("Processing...")
            spoken_text = rec.recognize_google(audio)  # Default language set to English

            # Automatically detect the language of the spoken text
            detected_language = detect_language(spoken_text)
            st.text(detected_language)

            # Display the input text (spoken text) on the screen
            output_placeholder.text(f"Input (Detected {detected_language}): {spoken_text}")

            # Use detected language as the source for translation
            translated_text = translator_function(spoken_text, detected_language, to_language)

            # Display both the input and output (translated text) on the screen
            output_placeholder.text(f"Input (Detected {detected_language}): {spoken_text}\n\n"
                                    f"Output ({to_language}): {translated_text.text}")

            # Convert translated text to speech
            text_to_voice(translated_text.text, to_language)

        except Exception as e:
            print(e)


# UI layout
st.title("Language Translator")

# Dropdowns for selecting languages
to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))

# Convert language names to language codes
to_language = get_language_code(to_language_name)

# Button to trigger translation
start_button = st.button("Start")
stop_button = st.button("Stop")

# Check if "Start" button is clicked
if start_button:
    if not isTranslateOn:
        isTranslateOn = True
        output_placeholder = st.empty()
        main_process(output_placeholder, 'en', to_language)  # Default source language 'en'

# Check if "Stop" button is clicked
if stop_button:
    isTranslateOn = False
