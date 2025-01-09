import speech_recognition as sr
import streamlit as st

def recognize_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("Listening for your question...")
        audio = recognizer.listen(source)
    
    try:
        question = recognizer.recognize_google(audio)
        st.success(f"Question recognized: {question}")
        return question
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio. Please try again.")
        return None
    except sr.RequestError:
        st.error("Could not request results from the speech recognition service.")
        return None