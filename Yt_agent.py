import streamlit as st
from dotenv import load_dotenv
import os
from urllib.parse import urlparse, parse_qs

from agno.tools.youtube import YouTubeTools
from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found in environment variables")
    st.stop()

def extract_video_id(url: str) -> str | None:
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

yt_tools = YouTubeTools(get_video_captions=True, get_video_data=False)

# Agents
summary_agent = Agent(
    name="Summary Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    role="Summarize YouTube transcripts",
    description="""You summarize the video transcript in a clear, simple, and structured way.""",
    tools=[yt_tools],
    markdown=True,
)

notes_agent = Agent(
    name="Study Notes Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    role="Create study notes from transcripts",
    description="""You generate clear, structured, bullet-point style study notes.""",
    tools=[yt_tools],
    markdown=True,
)

qa_agent = Agent(
    name="Question Generation Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    role="Generate MCQs and descriptive questions from transcripts",
    description="""You generate 5-6 MCQs and 3-4 descriptive questions based on the transcript.""",
    tools=[yt_tools],
    markdown=True,
)

# Streamlit UI
st.title("🎥 YouTube Transcript Assistant")
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        st.success("✅ Video ID extracted successfully!")

        if st.button("📄 Generate Summary"):
            with st.spinner("Generating summary..."):
                summary_prompt = f"Summarize this YouTube video: {video_url}"
                response = summary_agent.run(summary_prompt)
                st.markdown(response.content)

        if st.button("📘 Generate Study Notes"):
            with st.spinner("Generating study notes..."):
                notes_prompt = f"Create study notes from this video: {video_url}"
                response = notes_agent.run(notes_prompt)
                st.markdown(response.content)

        if st.button("❓ Generate Questions"):
            with st.spinner("Generating questions..."):
                question_prompt = f"Generate questions from this video: {video_url}"
                response = qa_agent.run(question_prompt)
                st.markdown(response.content)
