import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.youtube import YouTubeTools

# --- Environment Variable Setup ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="YouTube Video Assistant",
    layout="wide",
    page_icon="🎥"
)
st.title("🎥 YouTube Transcript Assistant")
st.markdown(
    """
    Enter a YouTube video URL, and then choose a tab below to generate a summary, study notes, or practice questions.
    """
)

# --- Helper Function ---
def extract_video_id(url: str) -> str | None:
    """Extracts the YouTube video ID from various URL formats."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host == "youtu.be":
        return parsed.path.lstrip("/")
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        for seg in ("/embed/", "/shorts/", "/live/"):
            if parsed.path.startswith(seg):
                return parsed.path.split("/")[2]
    return None

# --- Tool Initialization ---
yt_tools = YouTubeTools(get_video_captions=True, get_video_data=False)

# --- AGENT DEFINITIONS ---
summary_agent = Agent(
    name="Summary Agent",
    model=Gemini(id="gemini-1.5-flash", api_key=API_KEY),
    description="""
        You are an expert summarizer. Your task is to analyze the provided YouTube video transcript and generate a comprehensive summary.
        The summary should be well-structured:
        1.  **Main Takeaway**: Start with a single, concise paragraph that captures the core message of the video.
        2.  **Key Points**: Follow with a series of bullet points highlighting the main topics, arguments, and conclusions discussed.
        The final output must be informative, easy to understand, and written in a formal tone.
    """,
    tools=[yt_tools],
    markdown=True,
)

notes_agent = Agent(
    name="Study Notes Agent",
    model=Gemini(id="gemini-1.5-flash", api_key=API_KEY),
    description="""
        You are a meticulous academic assistant creating detailed study notes from video transcripts.
        Your task is to transform the transcript into a hierarchical set of notes using markdown.
        - Use main headings (`##`) for major topics.
        - Use nested bullet points for key details, definitions, and examples under each topic.
        - Emphasize important terms by making them **bold**.
        The goal is to create notes that are perfectly structured for revision and quick learning.
    """,
    tools=[yt_tools],
    markdown=True,
)

# --- CORRECTED QA AGENT WITH PRECISE FORMATTING ---
qa_agent = Agent(
    name="Question Generation Agent",
    model=Gemini(id="gemini-1.5-flash", api_key=API_KEY),
    description="""
        You are an expert quiz designer. Your task is to generate questions from the video transcript.

        Your output MUST follow this exact structure:

        1.  **## Multiple Choice Questions**
            - Generate exactly 5 MCQs.
            - For each question, you MUST follow this precise format:
              - The question number and text on a single line.
              - Each of the four options (A, B, C, D) must be on its own separate line.
              - The correct answer must be on a new line immediately after the options, formatted exactly as '**Ans:**' followed by the correct option letter.

            - **Example of the required format for one question:**
              `3. What is the capital of France?`
              `A. Berlin`
              `B. Madrid`
              `C. Paris`
              `D. Rome`
              `**Ans:** C`

        2.  **## Descriptive Questions**
            - After the MCQs, generate 3-4 thought-provoking descriptive questions.
    """,
    tools=[yt_tools],
    markdown=True,
)


# --- Streamlit UI and Logic ---
video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("nvalid YouTube URL. Please enter a valid URL.")
    else:
        st.success(f"✅ Video ID `{video_id}` extracted successfully! Select a tab below to proceed.")

        tab1, tab2, tab3 = st.tabs(["📄 Summary", "📘 Study Notes", "❓ Questions"])

        with tab1:
            if st.button("Generate Summary", key="summary_btn"):
                with st.spinner("Generating summary..."):
                    try:
                        prompt = f"Please summarize the key points from this YouTube video: {video_url}"
                        response = summary_agent.run(prompt)
                        if "unable to retrieve" in response.content.lower() or "could not find captions" in response.content.lower():
                             st.error("Failed to get captions. The video might not have captions enabled or could be private.")
                        else:
                             st.markdown(response.content)
                    except Exception as e:
                        st.error(f"An error occurred: {e}. This often means the video has no captions available.")

        with tab2:
            if st.button("Generate Study Notes", key="notes_btn"):
                with st.spinner("Generating study notes..."):
                    try:
                        prompt = f"Create detailed, hierarchical study notes from this video: {video_url}"
                        response = notes_agent.run(prompt)
                        if "unable to retrieve" in response.content.lower() or "could not find captions" in response.content.lower():
                             st.error("Failed to get captions. The video might not have captions enabled or could be private.")
                        else:
                             st.markdown(response.content)
                    except Exception as e:
                        st.error(f"An error occurred: {e}. This often means the video has no captions available.")

        with tab3:
            if st.button("Generate Questions", key="questions_btn"):
                with st.spinner("Generating questions..."):
                    try:
                        prompt = f"Generate MCQs and descriptive questions from this video: {video_url}"
                        response = qa_agent.run(prompt)
                        if "unable to retrieve" in response.content.lower() or "could not find captions" in response.content.lower():
                             st.error("Failed to get captions. The video might not have captions enabled or could be private.")
                        else:
                             st.markdown(response.content)
                    except Exception as e:
                        st.error(f"An error occurred: {e}. This often means the video has no captions available.")