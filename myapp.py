import streamlit as st
import tempfile, os, shutil, time, subprocess, base64, requests
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import videointelligence_v1 as vi, vision
import boto3
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip



# Configuration & Clients Setup
# Google Cloud Setup
google_credentials = st.secrets["google_cloud"]
credentials_dict = {
    "type": "service_account",
    "project_id": google_credentials["project_id"],
    "private_key_id": google_credentials["private_key_id"],
    "private_key": google_credentials["private_key"].replace('\\n', '\n'),
    "client_email": google_credentials["client_email"],
    "client_id": google_credentials["client_id"],
    "auth_uri": google_credentials["auth_uri"],
    "token_uri": google_credentials["token_uri"],
    "auth_provider_x509_cert_url": google_credentials["auth_provider_x509_cert_url"],
    "client_x509_cert_url": google_credentials["client_x509_cert_url"]
}

with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp_file:
    json.dump(credentials_dict, temp_file)
    temp_file_path = temp_file.name
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path




# Initializing Clients
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
speech_client = speech.SpeechClient()
vi_client = vi.VideoIntelligenceServiceClient()
vision_client = vision.ImageAnnotatorClient()
rekognition = boto3.client(
    'rekognition',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
    region_name="us-east-1"
)



# Setting up pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if "video-highlights" not in pc.list_indexes().names():
    pc.create_index(
        name="video-highlights",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index("video-highlights")





# Setting up Templated.io 
TEMPLATED_API_URL = "https://api.templated.io/v1/renders"
TEMPLATED_API_KEY = st.secrets["TEMPLATED_API_KEY"]
TEMPLATED_TEMPLATE_ID = st.secrets["TEMPLATED_TEMPLATE_ID"]

#Webapp title
st.title("🎬 AI Video Analyzer")

# Defining Helper Functions
# Function for extracting video duration 
def get_video_duration(path):
    clip = VideoFileClip(path)
    duration = clip.duration
    clip.close()
    return duration

# Saving video in temp dir
def save_temp_video(uploaded):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path, temp_dir


# Function for transcribing the video in chunks (spiliting video under 60sec)
def transcribe(path, chunk_length=59, overlap=0.5):
    """
    Splits audio into overlapping chunks, transcribes each in multiple languages,
    and returns both full text and per-chunk transcripts with timestamps.
    """
    base, _ = os.path.splitext(path)
    clip = AudioFileClip(path)
    duration = clip.duration
    transcripts = []

    def _transcribe_segment(start, end):
        fn = f"{base}_chunk_{int(start*1000)}.wav"
        clip.subclipped(start, end).write_audiofile(
            fn, fps=16000, ffmpeg_params=["-ac", "1"], logger=None
        )
        with open(fn, "rb") as f:
            audio = speech.RecognitionAudio(content=f.read())
        os.remove(fn)

        best = ""
        for lang in ("en-US", "ar", "ur"):
            cfg = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=lang,
                enable_automatic_punctuation=True
            )
            try:
                resp = speech_client.recognize(config=cfg, audio=audio)
                text = " ".join(r.alternatives[0].transcript for r in resp.results)
                if len(text) > len(best):
                    best = text
            except Exception:
                continue
        return best.strip()

    if duration <= chunk_length:
        text = _transcribe_segment(0, duration)
        transcripts.append({"start": 0, "end": duration, "text": text})
    else:
        start = 0.0
        while start < duration:
            end = min(start + chunk_length, duration)
            text = _transcribe_segment(start, end)
            transcripts.append({"start": start, "end": end, "text": text})
            start += chunk_length - overlap
            time.sleep(0.2)

    # filter out empty chunks
    segments = [s for s in transcripts if s["text"]]
    full_text = " ".join(s["text"] for s in segments)
    return full_text, segments


# Breaking vide in frames and analysing frames
def analyze_frame(image_bytes):
    labels = []
    try:
        # AWS Rekognition
        aws_labels = rekognition.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=15)
        labels += [l["Name"] for l in aws_labels.get("Labels", [])]
        
        # Celebrity Recognition
        celebs = rekognition.recognize_celebrities(Image={"Bytes": image_bytes})
        labels += [c["Name"] for c in celebs.get("CelebrityFaces", [])]
        
        # Google Vision
        gimg = vision.Image(content=image_bytes)
        gv_labels = vision_client.label_detection(image=gimg)
        labels += [l.description for l in gv_labels.label_annotations]
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    
    return list(dict.fromkeys(labels))[:20]

# For scene detection and  generating summaries of scenes 
def enrich_shots(path, transcript_segments):
    """
    Detects shot changes, analyzes frames, and generates LLM summaries including
    both labels and the transcript for that scene.
    """
    with open(path, "rb") as f:
        content = f.read()
    req = vi.AnnotateVideoRequest(
        input_content=content,
        features=[vi.Feature.SHOT_CHANGE_DETECTION]
    )
    op = vi_client.annotate_video(request=req)
    shots = op.result(timeout=300).annotation_results[0].shot_annotations

    records = []
    for shot in shots:
        start = shot.start_time_offset.total_seconds()
        end = shot.end_time_offset.total_seconds()
        labels = set()
        # analyze frames
        for t in [start + (end - start) * i / 3 for i in (0, 1, 2)]:
            frame_path = save_frame(path, t, f"frame_{t}.jpg")
            with open(frame_path, "rb") as f:
                labels.update(analyze_frame(f.read()))
            os.remove(frame_path)

        # extract transcript for this scene
        scene_text = " ".join(
            seg["text"] for seg in transcript_segments
            if seg["start"] < end and seg["end"] > start
        )

        # Generate scene summary with transcript context
        try:
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": (
                        # "Summarize this video scene considering visual elements, "
                        # "possible activities, emotions, and context. Use the transcript to enrich accuracy."
                        """You are a professional video‐scene summarization engine. For each shot, generate a clear, 120–150-word paragraph that:

                        Names any people, characters, or figures you can identify (and their roles, if known).

                        Describes the key visual elements and setting.

                        Summarizes any actions or interactions.

                        Captures the emotional tone or atmosphere.

                        References any spoken text when relevant.

                        Write in a neutral, reportorial style—concise but vivid. Avoid filler or speculation beyond what’s visible."""
                    )},
                    {"role": "user", "content": (
                        f"Shot start–end: {start}s–{end}s\n"
                        f"Detected elements: {', '.join(labels)}\n"
                        f"Transcript excerpt: {scene_text if scene_text else '[No speech]'}"
                    )}
                ],
                temperature=0.7,
                # max_tokens=150
                max_tokens=300
            )
            summary = resp.choices[0].message.content.strip()
        except Exception as e:
            summary = "Scene analysis unavailable"
            st.error(f"GPT-4 error: {str(e)}")

        records.append({
            "start": start,
            "end": end,
            "labels": list(labels),
            "summary": summary
        })

    return records


# Saving each frame
def save_frame(path, t, outp):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(outp, frame)
    cap.release()
    return outp

# Indexing scenes on pinecone
def index_scenes(scenes):
    vectors = []
    for i, scene in enumerate(scenes):
        text = f"{scene['summary']} {' '.join(scene['labels'])}"
        emb = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append((str(i), emb.data[0].embedding, scene))
    index.upsert(vectors=vectors)

# generating Scene summaries in chunks
def summarize_chunks_with_gpt4(client, summaries, chunk_size=5):
    chunk_summaries = []

    system_msg = (
        "You are an expert video summarization assistant. Your task is to summarize a group of related video scenes "
        "into a single, fluent paragraph of 80–100 words. Capture:\n"
        "- The main idea or purpose of the scenes\n"
        "- Key events and actions\n"
        "- Important people or characters\n"
        "- Mentioned locations (if any)\n"
        "- Emotional tone or insights when relevant\n"
        "The summary should feel natural, informative, and capture the essence of what occurred in the scenes."
    )

    for i in range(0, len(summaries), chunk_size):
        chunk = summaries[i:i + chunk_size]
        user_msg = "\n".join(f"- {s}" for s in chunk)

        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=800,
            temperature=0.7
        )

        summary = resp.choices[0].message.content.strip()
        chunk_summaries.append(summary)
        time.sleep(1)  # helps avoid hitting rate limits

    return chunk_summaries


# Generating final full summary 
def final_summary_with_openai(chunk_summaries , full_text):
    system_msg = "You are a professional video summarization expert. "
    user_msg = (

        "Combine these chunk summaries into a cohesive and comprehensive summary of the entire video in 150–200 words. "
        "Focus on the core idea, key events, people involved, locations, and any significant developments. "
        "Ensure the summary flows logically, highlights the central message, and captures the essence of the video without unnecessary detail.\n\n"
        + "\n".join(f"- {s}" for s in chunk_summaries)
        + "\n\nTranscript Excerpt (for context if needed):\n"
        + (full_text[:5000] + "…")



    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=800,
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

# for extracting the main agenda for banner title
def extract_key_line(transcript: str, summary: str, client) -> str:
    """
    Ask the LLM to return the single most important line (agenda) in the video,
    by looking at both the full transcript and the full summary.
    Output must be exactly one line.
    """
    combined = (
        "Transcript:\n" + transcript + "\n\n" +
        "Summary:\n" + summary
    )
    prompt = (
        "Given the full video transcript and a concise summary, identify the single most important headline or agenda line that captures the video's core message. "
        "Output exactly one line.\n\n" + combined
    )
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a video summarization expert that provides a single headline line."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
    # ensure single line
    return resp.choices[0].message.content.strip().splitlines()[0]



# for searching the best match for prompt
def semantic_search(prompt, top_k=3):
    emb = client.embeddings.create(input=[prompt], model="text-embedding-ada-002")
    return index.query(
        vector=emb.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )

# function for better scene index management
def index_scenes(scenes):
    # delete any old vectors so IDs always line up 0…len(scenes)-1
    index.delete(delete_all=True)
    vectors = []
    for i, scene in enumerate(scenes):
        text = f"{scene['summary']} {' '.join(scene['labels'])}"
        emb = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append((str(i), emb.data[0].embedding, scene))
    index.upsert(vectors=vectors)

# Image processing utilities
def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# validating image path
def is_valid_image(file_path):
    try:
        Image.open(file_path)
        return True
    except Exception:
        return False

# Resizing image
def resize_image(input_path, output_path, size=(846, 541)):
    with Image.open(input_path) as img:
        img = img.resize(size)
        img.save(output_path)



#  Configuration for thumbnail
DEFAULT_FONT = "arial.ttf"  
TITLE_FONT_SIZE = 60
SUBTITLE_FONT_SIZE = 40
LOGO_SIZE = (80, 80)
PADDING = 30
BAR_HEIGHT = 120
BAR_COLOR = (0, 123, 255, 200)  
TEXT_COLOR = (255, 255, 255, 255)
SHADOW_COLOR = (0, 0, 0, 150)



# Webapp UI Flow 
st.session_state.setdefault("uploaded", None)
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("chunk_transcripts", [])
st.session_state.setdefault("scenes", [])
st.session_state.setdefault("video_title", "")


if 'scene_records' not in st.session_state:
    st.session_state['scene_records'] = []

uploaded = st.file_uploader("Upload video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
if uploaded:
    video_path, temp_dir = save_temp_video(uploaded)
    duration = get_video_duration(video_path)
    st.session_state.uploaded = video_path
    

    if st.button("1️⃣ Transcribe Video"):
        full, segments = transcribe(st.session_state.uploaded)
        st.session_state.transcript = full
        st.session_state.chunk_transcripts = segments

    



    if duration > 240:  # 4 minutes
        st.warning("Note: Full analysis limited to first 4 minutes for free tier services")

    # Processing Buttons 
    col1, col2 = st.columns(2)
    with col1:
        
        if st.button("✨ Analyze Scenes"):
            with st.spinner():
                # transcript already in state
                shots = enrich_shots(video_path, st.session_state.chunk_transcripts)
                st.session_state.scenes = shots
                index_scenes(shots)
                st.session_state['scene_records'] = shots
                st.success(f"Generated {len(shots)} scene records.")


    # Defining tab setup
    tab0, tab1, tab6, tab2, tab5, tab3, tab4,  = st.tabs([
        "Transcription" , "Scene Breakdown", "Full video summary", "Semantic Search", 
        "Promo Clip", "Banner Creator",  "Download metadata" 
    ])

    st.session_state.setdefault("selected_scenes", [])

    # tab0 for displaying transcript
    with tab0:
        st.header("Full Transcript")
        if st.session_state.transcript:
        # if st.button("1️⃣ Transcribe Video"):
            full_text, segments = transcribe(video_path)
            st.session_state['full_text'] = full_text
            st.session_state['chunk_transcripts'] = segments

        if st.session_state.get('full_text'):
            st.text_area(
                "Combined Transcript",
                value=st.session_state['full_text'],
                height=300
            )
        else:
            st.info("Click **1️⃣ Transcribe Video** first.")
    


    # tab1 for scene summaries 
    with tab1:  # Scene Breakdown
        if st.session_state.scenes:
            st.subheader("📽️ Scene Analysis")
            for i, scene in enumerate(st.session_state.scenes):
                with st.expander(f"Scene {i+1} ({scene['start']:.1f}s - {scene['end']:.1f}s)"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        frame_path = save_frame(video_path, scene["start"], f"scene_{i}.jpg")
                        st.image(frame_path, width=200)
                        
                        st.write(f"**Detected Elements:** {', '.join(scene['labels'])}")
                    with col2:
                        st.write(f"**AI Summary:** {scene['summary']}")
        else:
            st.info("Click 'Analyze Scenes' to process video content")


    with tab2:
        st.subheader("🔍 Find Moments by Description")
        query = st.text_input("Describe the moment you want to find:", 
                            placeholder="e.g., 'emotional speech', 'action sequence', 'beautiful landscape'")
        
        if query and st.session_state.scenes:
            results = semantic_search(query)
            st.subheader("Top Matching Scenes")
            
            # Reset selection when new search
            if query != st.session_state.get("last_query", ""):
                st.session_state.selected_scenes = []
            st.session_state.last_query = query
            
            for i, match in enumerate(results["matches"]):
                scene_idx = int(match["id"])
                if 0 <= scene_idx < len(st.session_state.scenes):
                    scene = st.session_state.scenes[scene_idx]
                else:
                    continue 

                scene = st.session_state.scenes[scene_idx]
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    frame_path = save_frame(video_path, scene["start"], f"match_{i}.jpg")
                    st.image(frame_path, caption=f"{scene['end']-scene['start']:.1f}s scene")
                with col2:
                    # Add checkbox
                    selected = st.checkbox(
                        f"Select this scene (Score: {match['score']:.2f})",
                        value=scene_idx in st.session_state.selected_scenes,
                        key=f"select_{i}"
                    )
                    
                    if selected and scene_idx not in st.session_state.selected_scenes:
                        st.session_state.selected_scenes.append(scene_idx)
                    elif not selected and scene_idx in st.session_state.selected_scenes:
                        st.session_state.selected_scenes.remove(scene_idx)
                    
                    st.write(f"**Time:** {scene['start']:.1f}s - {scene['end']:.1f}s")
                    st.write(f"**Summary:** {scene['summary']}")
                    st.write(f"**Labels:** {', '.join(scene['labels'][:10])}")


    # tab 3 for banner generation
    with tab3:
        if st.button("🤖 Generate Modern Banner"):
            # Preconditions: scenes + text
            scenes = st.session_state.get('scenes', [])
            if not scenes:
                st.error("⚠️ Run Scene Analysis first!")
                st.stop()

            transcript = st.session_state.get('full_text', '')
            full_sum = st.session_state.get('full_summary')
            if not full_sum:
                sums = [r['summary'] for r in st.session_state.get('scene_records', [])]
                full_sum = final_summary_with_openai(sums, transcript)
                st.session_state['full_summary'] = full_sum
            banner_title = extract_key_line(transcript, full_sum, client)
            #st.write(banner_title)
            title_font_size = str(max(24, 60 - len(banner_title)//3)) + "px"

            # Select best video frame
            res = semantic_search(banner_title, top_k=3)
            valid = [m for m in res.get('matches', []) if m.get('id', '').isdigit()]
            if not valid:
                st.error("No valid scene—re-run analysis.")
                st.stop()
            scene = scenes[int(valid[0]['id'])]

            # Extract, resize and validate frame image
            frame_path = save_frame(st.session_state.uploaded, scene['start'], 'auto_frame.jpg')
            #st.write(frame_path)
            resized_path = 'resized_frame.jpg'
            resize_image(frame_path, resized_path, size=(846, 541))
            if not (is_valid_image(resized_path) and os.path.getsize(resized_path) <= 5 * 1024 * 1024):
                st.error("Invalid or too-large resized frame image")
                st.stop()
            bg_data = image_to_base64(resized_path)

            # Prepare API credentials
            api_key = st.secrets['TEMPLATED_API_KEY']
            template_id = st.secrets['TEMPLATED_TEMPLATE_ID']
            url = 'https://api.templated.io/v1/render'
            headers = {'Authorization': f'Bearer {api_key}'}

            data = {
                'template': template_id,
                'layers': {
                    'shape-2': {},
                    'image-2': {
                        'image_url': f'data:image/jpeg;base64,{bg_data}',
                        'width': 846,
                        'height': 541,
                    },
                    'shape-1': {},
                    'text-3': {
                        'text': banner_title,
                        'color': '#2f4f4f',
                        'text_align': 'center',
                        'vertical_align': 'middle'
                    }
                }
            }

            # Send render request
            resp = requests.post(url, json=data, headers=headers)
            if resp.status_code == 200:
                out = resp.json()
                if 'url' in out:
                    st.image(out['url'], caption=banner_title)
                    st.markdown(f"[⬇ Download Banner]({out['url']})", unsafe_allow_html=True)
                else:
                    st.error("Missing image URL in response")
                    st.json(out)
            else:
                st.error(f"Render failed ({resp.status_code})")
                st.code(resp.text)





    # For downloading json file
    with tab4:  
        # Always read from persisted scene_records to avoid rerun clearing scenes
        if st.session_state.get('scene_records'):
            json_data = json.dumps(st.session_state['scene_records'], indent=2)
            st.download_button(
                "📥 Download Scene Analysis (JSON)",
                data=json_data,
                file_name="video_analysis.json",
                mime="application/json",
                key="download_scene_records"
            )
        else:
            st.info("No scene records available. Analyze scenes first.")


    # For promo generation
    with tab5:
        if st.session_state.selected_scenes:
            st.subheader("🎥 Compile Selected Scenes into Promo")
            if st.button("Generate Promo"):
                with st.spinner("Rendering promo clip..."):
                    clips = []
                    for idx in st.session_state.selected_scenes:
                        scene = st.session_state.scenes[idx]
                        clip = VideoFileClip(video_path).subclipped(scene['start'], scene['end'])
                        clips.append(clip)
                    promo = concatenate_videoclips(clips)
                    out_path = os.path.join(temp_dir, "promo_clip.mp4")
                    promo.write_videofile(out_path, codec="libx264", audio_codec="aac")
                    st.video(out_path)
                    with open(out_path, "rb") as f:
                        btn_data = f.read()
                    st.download_button("Download Promo Clip", data=btn_data, file_name="promo_clip.mp4", mime="video/mp4")
        else:
            st.info("Select scenes in the 'Semantic Search' tab to enable promo creation.")


    # displaying full scene summary
    with tab6:
        st.header("Generate Full Video Summary")
        if not st.session_state.get('scene_records'):
            st.info("Please process scenes first (" +
                    "click '1️⃣ Process Scenes' in the sidebar).")
        else:
            if st.button("🔄 Summarize Full Video"):
                # Extract scene summaries
                scene_summaries = [r['summary'] for r in st.session_state['scene_records']]
                # Step 1: Gemini chunk summarization
                chunk_summaries = summarize_chunks_with_gpt4(client , scene_summaries)
                # Step 2: OpenAI final summary
                full_summary = final_summary_with_openai(chunk_summaries , full_text)
                st.subheader("Full Video Summary")
                st.write(full_summary)
    

    # Cleanup
    st.button("🧹 Clear All", on_click=lambda: [
        shutil.rmtree(temp_dir, ignore_errors=True),
        st.session_state.clear()
    ])
