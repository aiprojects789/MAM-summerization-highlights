import os
import sys
import time
import tempfile
import json
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioClip

# Google Cloud libraries
from google.cloud import videointelligence_v1 as vi
from google.cloud import speech, vision
from google.oauth2 import service_account

import google.generativeai as genai
import whisper
from pytube import YouTube

# FFmpeg configuration (adjust path as needed)
if sys.platform == "win32":
    ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"
else:
    ffmpeg_path = "ffmpeg"
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
os.environ["FFMPEG_BINARY"] = ffmpeg_path

# Load credentials
with open('project-alpha-456519-5c9ae36437f1.json') as f:
    credentials_info = json.load(f)
credentials = service_account.Credentials.from_service_account_info(credentials_info)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "project-alpha-456519-5c9ae36437f1.json"

# Initialize clients
vision_client = vision.ImageAnnotatorClient(credentials=credentials)
video_client = vi.VideoIntelligenceServiceClient(credentials=credentials)
speech_client = speech.SpeechClient(credentials=credentials)

# Configure Gemini (Generative AI)
api_key = st.secrets["api"]["key"]
genai.configure(api_key=api_key)

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Format seconds into HH:MM:SS
def format_time(seconds):
    return str(timedelta(seconds=seconds)).split(".")[0]

# ---------------------------
# Analysis with Video Intelligence API
# ---------------------------
def analyze_video(video_path):
    client = vi.VideoIntelligenceServiceClient(credentials=credentials)
    with open(video_path, "rb") as file:
        input_content = file.read()
    features = [
        vi.Feature.LABEL_DETECTION,
        vi.Feature.SHOT_CHANGE_DETECTION,
        vi.Feature.SPEECH_TRANSCRIPTION
    ]
    # Configure speech transcription for Arabic (change as needed)
    speech_config = vi.SpeechTranscriptionConfig(
        language_code="ar-AE",
        enable_automatic_punctuation=True
    )
    video_context = vi.VideoContext(speech_transcription_config=speech_config)
    
    operation = client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
            "video_context": video_context
        }
    )
    st.info("Google Cloud is processing the video. Please wait...")
    result = operation.result(timeout=300)
    annotation_result = result.annotation_results[0]
    
    # --- Scene detection using shot annotations ---
    scenes = []
    if annotation_result.shot_annotations:
        for shot in annotation_result.shot_annotations:
            start_time = shot.start_time_offset.total_seconds()
            end_time = shot.end_time_offset.total_seconds()
            scenes.append({
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "labels": []
            })
    # Fallback: if no shot annotations, use whole video duration.
    if not scenes:
        with VideoFileClip(video_path) as clip:
            total_duration = clip.duration
        scenes = [{
            "start": 0,
            "end": total_duration,
            "duration": total_duration,
            "labels": []
        }]
        st.warning("No distinct scenes detected; using full video as one scene.")
    
    # --- Label detection ---
    overall_labels = set()
    if annotation_result.segment_label_annotations:
        for label in annotation_result.segment_label_annotations:
            for segment in label.segments:
                seg_start = segment.segment.start_time_offset.total_seconds()
                seg_end = segment.segment.end_time_offset.total_seconds()
                for scene in scenes:
                    # If the detected segment lies entirely within a scene, add the label.
                    if seg_start >= scene["start"] and seg_end <= scene["end"]:
                        scene["labels"].append(label.entity.description)
                        overall_labels.add(label.entity.description)
    
    # --- Speech transcription ---
    transcript_overall = ""
    if annotation_result.speech_transcriptions:
        for transcription in annotation_result.speech_transcriptions:
            if transcription.alternatives:
                transcript_overall += transcription.alternatives[0].transcript + " "
    
    return {
        "transcript": transcript_overall.strip(),
        "scenes": scenes,
        "labels": list(overall_labels)
    }

# ---------------------------
# Automatic Highlight Creation
# ---------------------------
def generate_highlight(video_path, scenes, duration=30):
    """
    Automatically generate a highlight clip from the video.
    The selection heuristic here is simply choosing the longest scene.
    The clip length is trimmed to at most `duration` seconds.
    """
    if not scenes:
        st.error("No scenes detected to generate highlight.")
        return None

    # Select the scene with maximum duration.
    selected_scene = max(scenes, key=lambda s: s["duration"])
    start, end = selected_scene["start"], selected_scene["end"]
    if (end - start) > duration:
        end = start + duration

    with VideoFileClip(video_path) as clip:
        subclip = clip.subclip(start, end)
        # Ensure clip has audio; if missing, provide silent audio.
        if not subclip.audio:
            audio = AudioClip(lambda t: [0, 0], duration=subclip.duration)
            subclip = subclip.set_audio(audio)
        highlight_path = os.path.join(tempfile.gettempdir(), "highlight.mp4")
        subclip.write_videofile(
            highlight_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-ar", "16000"],
            logger=None  # reduce verbosity
        )
    return highlight_path

# ---------------------------
# Banner & Thumbnail Generation
# ---------------------------
def generate_banner(video_path, summary_text):
    """
    Generate a banner using a key frame (from the midpoint of the video)
    with the summary text overlaid.
    """
    try:
        with VideoFileClip(video_path) as clip:
            middle = clip.duration / 2
            frame = clip.get_frame(middle)
        
        # Convert frame from BGR (OpenCV) to RGB.
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        try:
            # Use a TTF font if available.
            font = ImageFont.truetype("arial.ttf", 30)
        except Exception as fe:
            font = ImageFont.load_default()
        
        # Wrap text every 40 characters for better readability.
        wrapped_text = "\n".join([summary_text[i:i+40] for i in range(0, len(summary_text), 40)])
        draw.text((10, 10), wrapped_text, fill="white", font=font, stroke_width=2, stroke_fill="black")
        banner_path = os.path.join(tempfile.gettempdir(), "banner.jpg")
        img.save(banner_path)
        return banner_path
    except Exception as e:
        st.error(f"Banner generation failed: {str(e)}")
        return None

# ---------------------------
# Whisper Transcription
# ---------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def whisper_transcribe(video_path):
    model = load_whisper_model()
    result = model.transcribe(video_path, language="ar")
    return result["text"]

# ---------------------------
# Translation & Summarization with Gemini
# ---------------------------
def translate_text(text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Translate to English: {text}")
    return response.text

def summarize_with_gemini(translated_text, labels):
    model = genai.GenerativeModel('gemini-1.5-flash')
    label_description = ", ".join(labels)
    prompt = f"""
    Given the following video description and transcript, generate a detailed and coherent summary of what the video is about.

    Video Description (based on detected content): {label_description}

    Transcript: {translated_text}
    """
    response = model.generate_content(prompt)
    return response.text

# ---------------------------
# Main Application
# ---------------------------
def main():
    st.set_page_config(layout="wide", page_title="Arabic Video Analyzer with Cloud Vision")
    st.title("📹 Arabic Video Analysis Platform")

    video_file = st.file_uploader("Upload Arabic Video", type=["mp4", "mov"])
    youtube_url = st.text_input("Or Enter YouTube URL")

    video_path = None
    if youtube_url:
        try:
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            video_path = stream.download(output_path=tempfile.gettempdir())
            st.success("YouTube video downloaded!")
        except Exception as e:
            st.error(f"YouTube Error: {str(e)}")
    
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

    if video_path:
        st.video(video_path)
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
            st.caption(f"Duration: {format_time(duration)}")
        except:
            st.caption("Unable to determine video duration.")

        if st.button("Analyze Video"):
            with st.spinner("Analyzing with Google Cloud, Whisper, and Gemini..."):
                try:
                    # 1. Google Cloud Video Intelligence Analysis (labels, scenes, transcripts)
                    analysis = analyze_video(video_path)
                    
                    # 2. Whisper-based transcription
                    whisper_text = whisper_transcribe(video_path)
                    
                    # 3. Combine transcripts (from Cloud and Whisper)
                    full_transcript = analysis["transcript"] + "\n\n" + whisper_text
                    
                    # 4. Translate combined transcript to English using Gemini.
                    translated = translate_text(full_transcript)
                    
                    # 5. Summarize based on translated transcript and detected labels.
                    summary = summarize_with_gemini(translated, analysis["labels"])
                    
                    scenes = analysis["scenes"]
                    
                    # Create tabs for different results.
                    tab1, tab2, tab3, tab4 = st.tabs(["Transcript", "Scenes", "Highlights", "Report"])
                    
                    with tab1:
                        st.subheader("Arabic Transcript")
                        st.write(full_transcript)
                        st.subheader("English Translation")
                        st.write(translated)
                        st.subheader("Summary")
                        st.write(summary)
                    
                    with tab2:
                        st.subheader("Detected Scenes")
                        for idx, scene in enumerate(scenes):
                            with st.expander(f"Scene {idx+1}: {format_time(scene['start'])} - {format_time(scene['end'])}"):
                                st.write("**Duration:**", f"{scene['duration']:.1f} seconds")
                                st.write("**Labels:**", ", ".join(set(scene["labels"])))
                    
                    with tab3:
                        st.subheader("Automatic Highlight Clip")
                        highlight_path = generate_highlight(video_path, scenes, duration=30)
                        if highlight_path:
                            st.video(highlight_path)
                    
                    with tab4:
                        st.subheader("Banner & Technical Report")
                        banner_path = generate_banner(video_path, summary)
                        if banner_path:
                            st.image(banner_path)
                    
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                finally:
                    if os.path.exists(video_path):
                        os.remove(video_path)

if __name__ == "__main__":
    main()
