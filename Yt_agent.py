import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from agno.tools.youtube import YouTubeTools
from agno.agent import Agent
from agno.models.google import Gemini
from agno.team.team import Team

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables")

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

summary_agent = Agent(
    name="Summary Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    role="Summarize YouTube transcripts",
    description="""
        You are a helpful and intelligent YouTube assistant.

        Your task is to read the transcript (captions) of a YouTube video and generate a clear, concise, and meaningful summary. 
        The summary should capture the main ideas, key points, and important messages from the video, making it easy for someone to understand 
        the content without watching the full video.

        ## Your Responsibilities:
        - Analyze the full caption text carefully.
        - Identify core topics, themes, and important facts or arguments.
        - Present the summary in simple and easy-to-understand language.
        - Maintain the original intent and tone of the speaker, where appropriate.

        ## Output Guidelines:
        - Use markdown formatting for clean readability.
        - Keep the summary informative but not overly detailed.
        - Do not invent or assume content that isn't in the transcript.
    """,
    tools=[yt_tools],
    show_tool_calls=True,
    markdown=True,
)

notes_agent = Agent(
    name="Study Notes Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    role="Create study notes from transcripts",
    description="""
        You are an expert academic assistant specializing in creating study notes from video transcripts.

        Your task is to process the given YouTube video transcript and generate comprehensive, well-structured study notes. 
        These notes should help a student learn and revise the material effectively.

        ## Your Responsibilities:
        - Identify and extract key concepts, definitions, and important terminology.
        - List out main arguments, examples, and supporting details.
        - Note any step-by-step processes, formulas, or methodologies discussed.
        - Extract any questions posed or answered that are crucial for understanding.
        - Identify actionable advice, tips, or key takeaways.

        ## Output Guidelines:
        - Use markdown formatting extensively for structure and readability (e.g., headings, subheadings, bullet points, bold text for key terms).
        - Organize the notes logically, perhaps by topic or theme.
        - Ensure the notes are detailed enough for study but avoid verbatim transcription of long passages.
        - Focus on clarity and conciseness.
        - If the video content is conversational, try to distill the academic or informational core.
    """,
    tools=[yt_tools],
    show_tool_calls=True,
    markdown=True,
)

qa_agent = Agent(
    name="Question Generation Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    role="Generate MCQs and descriptive questions from transcripts",
    description="""
        You are an expert quiz and question generation assistant.

        Your task is to process the given YouTube video transcript and generate:
        1.  5-6 Multiple Choice Questions (MCQs). Each MCQ should have a question, 4 distinct options (A, B, C, D), and a clearly indicated correct answer.
        2.  3-4 Descriptive Questions. Each descriptive question should have a clear question and a comprehensive answer derived from the transcript.

        ## Your Responsibilities:
        - Analyze the transcript thoroughly to identify key information suitable for questions.
        - For MCQs, ensure options are plausible but only one is correct. Distractors should be relevant to the topic.
        - For descriptive questions, ensure answers are accurate, comprehensive, and cover the main points related to the question from the transcript.
        - Base all questions and answers strictly on the provided transcript content. Do not invent information.

        ## Output Guidelines:
        - Use markdown formatting for all questions and answers.
        - Start with a heading "## Multiple Choice Questions (MCQs)".
        - For each MCQ, format as:
            **Question X:** [Question text]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            **Correct Answer:** [Letter of correct option] (e.g., Correct Answer: B)
        - After MCQs, add a heading "## Descriptive Questions".
        - For each Descriptive Question, format as:
            **Question X:** [Question text]
            **Answer:** [Answer text]
    """,
    tools=[yt_tools],
    show_tool_calls=True,
    markdown=True,
)

youtube_assistant = Team(
    name="YouTube Assistant Team",
    model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
    tools=[yt_tools],
    instructions=[
        "Route to Summary Agent if prompt contains 'summary' or 'summarize'",
        "Route to Study Notes Agent if prompt contains 'notes' or 'study notes'",
        "Route to Question Generation Agent if prompt contains 'questions' or 'quiz'",
    ],
    members=[summary_agent, notes_agent, qa_agent],
    mode="route",
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ").strip()
    video_id = extract_video_id(video_url)
    if not video_id:
        print("❌ Could not extract video ID; please check your URL.")
        exit(1)

    print("\n📄 Generating Summary...\n")
    summary_prompt = f"Summarize this YouTube video: {video_url}"
    summary_agent.print_response(summary_prompt, markdown=True)

    ask_notes = input("\n📘 Would you like to generate study notes? (yes/no): ").strip().lower()
    if ask_notes in ["yes", "y"]:
        print("\n📘 Generating Study Notes...\n")
        notes_prompt = f"Create study notes from this video: {video_url}"
        notes_agent.print_response(notes_prompt, markdown=True)

    ask_questions = input("\n❓ Would you like to generate questions? (yes/no): ").strip().lower()
    if ask_questions in ["yes", "y"]:
        print("\n❓ Generating Questions...\n")
        question_prompt = f"Generate questions from this video: {video_url}"
        qa_agent.print_response(question_prompt, markdown=True)
