from gtts import gTTS
import speech_recognition as sr

# Text to speech
def text_to_speech(text, output_file="response.mp3"):
    tts = gTTS(text)
    tts.save(output_file)
    return output_file

# Audio to text (using Google Speech Recognition)
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the recognition service: {e}"
