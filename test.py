import io
import os
from google.cloud import speech

GOOGLE_APPLICATION_CREDENTIALS = "/home/snowholt/coding/python/google_capstone/gen-lang-client-0303460212-13d5c2280ba7.json"
# Path to your tiny test audio file
file_path = "voices/test.wav"

# Replace with the desired language code
language_code = "en-US"

def test_speech_connection(file_path):
    """Tests the Google Cloud Speech-to-Text API connection with a small audio file."""
    
    # Set the environment variable for Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Adjust if needed
        sample_rate_hertz=48000,                               # Adjust if needed
        language_code=language_code,
    )

    try:
        response = client.recognize(request={"config": config, "audio": audio})
        if response.results:
            print("🎉 Connection to Google Cloud Speech-to-Text API successful!")
            print(f"First result: {response.results[0].alternatives[0].transcript} (Confidence: {response.results[0].alternatives[0].confidence:.2f})")
            return True
        else:
            print("🤔 Connection successful, but no speech detected (as expected for a silent/non-speech audio). API key likely OK.")
            return True
    except Exception as e:
        print(f"🚨 Error connecting to Google Cloud Speech-to-Text API: {e}")
        return False

if __name__ == "__main__":
    test_speech_connection(file_path)