import os
import io
import requests
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------
# 1. Constants and Configuration
# ---------------------------------------------------------

PROVIDERS = {
    "OpenAI": {
        "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "label": "Select Voice",
    },
    "Azure": {
        "voices": ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural"],
        "label": "Select Voice",
    },
    "Naver Clova": {
        "voices": ["nara", "jinho", "mijin", "kyuri"],
        "label": "Select Speaker",
    }
}

# ---------------------------------------------------------
# 2. TTS Provider Logic
# ---------------------------------------------------------

def get_openai_tts(text: str, voice: str) -> io.BytesIO:
    """Generate TTS using OpenAI's Audio API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    
    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    return io.BytesIO(response.content)

def get_azure_tts(text: str, voice: str) -> io.BytesIO:
    """Generate TTS using Azure Cognitive Services."""
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise ValueError("Azure credentials (KEY/REGION) are missing.")

    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
    }
    ssml = f"<speak version='1.0' xml:lang='en-US'><voice xml:lang='en-US' name='{voice}'>{text}</voice></speak>"
    
    response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
    response.raise_for_status()
    return io.BytesIO(response.content)

def get_naver_tts(text: str, speaker: str) -> io.BytesIO:
    """Generate TTS using Naver Clova Premium Voice."""
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Naver credentials (ID/SECRET) are missing.")

    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "speaker": speaker,
        "volume": "0", "speed": "0", "pitch": "0",
        "format": "mp3", "text": text
    }
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return io.BytesIO(response.content)

def generate_speech(provider: str, text: str, voice: str) -> io.BytesIO:
    """Dispatcher function to call the correct TTS provider."""
    dispatch_map = {
        "OpenAI": get_openai_tts,
        "Azure": get_azure_tts,
        "Naver Clova": get_naver_tts,
    }
    func = dispatch_map.get(provider)
    if not func:
        raise ValueError(f"Unsupported provider: {provider}")
    return func(text, voice)

# ---------------------------------------------------------
# 3. Streamlit UI Components
# ---------------------------------------------------------

def render_sidebar():
    """Renders the sidebar for provider and voice selection."""
    st.sidebar.header("⚙️ Configuration")
    provider = st.sidebar.radio("Select TTS Provider", list(PROVIDERS.keys()))
    
    provider_config = PROVIDERS[provider]
    voice = st.sidebar.selectbox(provider_config["label"], provider_config["voices"])
    
    return provider, voice

def render_main_area():
    """Renders the main content area for text input and generation."""
    st.title("🎤 TTS Model Comparison App")
    st.info("Input your text and select a provider on the left to generate speech.")
    
    text_input = st.text_area(
        "Enter text to convert to speech",
        placeholder="Type something here...",
        height=200
    )
    return text_input

def main():
    st.set_page_config(page_title="TTS Comparison", page_icon="🎤")
    
    # UI Layout
    provider, voice = render_sidebar()
    text_input = render_main_area()
    
    if st.button("Generate Speech", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text.")
            return
            
        with st.spinner(f"Generating audio using **{provider}** ({voice})..."):
            try:
                audio_bytes = generate_speech(provider, text_input, voice)
                
                st.success("✨ Generation Complete!")
                st.audio(audio_bytes, format="audio/mp3")
                
                st.download_button(
                    label="📥 Download MP3",
                    data=audio_bytes,
                    file_name=f"{provider.lower().replace(' ', '_')}_{voice}.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Failed to generate speech: {str(e)}")

if __name__ == "__main__":
    main()
