import traceback
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import requests as http_requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import joblib
import re
import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle
from dotenv import load_dotenv

# RAG Chatbot imports
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

app = Flask(__name__)
CORS(app)

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "lgbm_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
model = None
vectorizer = None

# ==========================================================
# CHATBOT CONFIG — In-memory session store
# ==========================================================
# Stores { video_id: { "qa_chain": ..., "title": ... } }
chat_sessions = {}

# Load embeddings model once at startup (reused across sessions)
# Using multilingual model to support Hindi, Hinglish, and other languages
print("Loading HuggingFace multilingual embeddings model...")
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    print("✔ Multilingual embeddings model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading embeddings model: {e}")
    embeddings_model = None


# ==========================================================
# PREPROCESSING
# ==========================================================
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {
            'not', 'no', 'but', 'however', 'yet'
        }

        comment = ' '.join([w for w in comment.split() if w not in stop_words])

        lem = WordNetLemmatizer()
        comment = ' '.join([lem.lemmatize(w) for w in comment.split()])

        return comment
    except Exception:
        return comment


# ==========================================================
# LOAD MODEL + VECTORIZER SAFELY
# ==========================================================
def load_model():
    global model, vectorizer
    try:
        print("Loading model/vectorizer...")

        with open("lgbm_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        print("✔ Model and vectorizer loaded successfully.")

    except Exception as e:
        print(f"❌ Error loading model/vectorizer: {e}")
        traceback.print_exc()



# Load at startup
load_model()


# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def home():
    return "Welcome to our ML API"


# ==========================================================
# PREDICT
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer failed to load"}), 500

    data = request.json
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        X = vectorizer.transform(preprocessed)
        preds = model.predict(X)
        preds = [str(p) for p in preds]

        response = [
            {"comment": c, "sentiment": s}
            for c, s in zip(comments, preds)
        ]

        return jsonify(response)

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ==========================================================
# PREDICT WITH TIMESTAMP
# ==========================================================
@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer failed to load"}), 500

    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [x["text"] for x in comments_data]
        timestamps = [x["timestamp"] for x in comments_data]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)

        preds = model.predict(transformed).tolist()
        preds = [str(p) for p in preds]

        response = [
            {"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(comments, preds, timestamps)
        ]

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ==========================================================
# WORDCLOUD
# ==========================================================
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# ==========================================================
# PIE CHART
# ==========================================================
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# ==========================================================
# TREND GRAPH
# ==========================================================
@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        totals = monthly.sum(axis=1)
        pct = (monthly.T / totals).T * 100

        for s in [-1, 0, 1]:
            if s not in pct.columns:
                pct[s] = 0

        pct = pct[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        for s in [-1, 0, 1]:
            plt.plot(pct.index, pct[s], marker="o", label=labels[s], color=colors[s])

        plt.title("Monthly Sentiment Trend")
        plt.xlabel("Month")
        plt.ylabel("Percentage (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# ==========================================================
# HELPER — Fetch YouTube video metadata (title, description)
# ==========================================================
def fetch_video_metadata(video_id):
    """
    Fetch video title and description from YouTube using the
    oEmbed API (no API key required).
    Falls back to scraping the page title if oEmbed fails.
    """
    video_title = f"Video {video_id}"
    video_description = ""

    # Method 1: oEmbed API (free, no key needed)
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        resp = http_requests.get(oembed_url, timeout=5)
        if resp.status_code == 200:
            oembed_data = resp.json()
            video_title = oembed_data.get("title", video_title)
            print(f"✔ Got video title via oEmbed: {video_title}")
    except Exception as e:
        print(f"⚠ oEmbed failed: {e}")

    # Method 2: Try to get description from page meta tags
    try:
        page_resp = http_requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
            timeout=5
        )
        if page_resp.status_code == 200:
            # Extract description from meta tag
            desc_match = re.search(
                r'<meta\s+name="description"\s+content="([^"]+)"',
                page_resp.text
            )
            if desc_match:
                video_description = desc_match.group(1)
                print(f"✔ Got video description ({len(video_description)} chars)")
    except Exception as e:
        print(f"⚠ Page scrape failed: {e}")

    return video_title, video_description


# ==========================================================
# CHATBOT — Initialize session for a YouTube video
# ==========================================================
@app.route("/chat/init", methods=["POST"])
def chat_init():
    """
    Initialize a RAG chatbot session for a YouTube video.
    Fetches the video transcript, splits it, creates embeddings,
    and stores the QA chain in memory keyed by video_id.

    Request body: { "video_url": "https://www.youtube.com/watch?v=..." }
    Response:     { "status": "ready", "video_id": "...", "title": "...", "chunks": N }
    """
    if embeddings_model is None:
        return jsonify({"error": "Embeddings model failed to load. Check server logs."}), 500

    data = request.get_json()
    video_url = data.get("video_url")

    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400

    try:
        # Extract video_id from URL
        video_id_match = re.search(r'(?:v=|/)([A-Za-z0-9_-]{11})', video_url)
        if not video_id_match:
            return jsonify({"error": "Invalid YouTube URL"}), 400
        video_id = video_id_match.group(1)

        # Return early if already initialized
        if video_id in chat_sessions:
            session = chat_sessions[video_id]
            return jsonify({
                "status": "ready",
                "video_id": video_id,
                "title": session.get("title", "Unknown"),
                "chunks": session.get("chunks", 0),
                "message": "Session already active"
            })

        # 1. Fetch video metadata (title, description)
        print(f"📥 Fetching metadata for video: {video_id}")
        video_title, video_description = fetch_video_metadata(video_id)

        # 2. Fetch transcript (try multiple languages: Hindi, English, auto-generated)
        print(f"📥 Fetching transcript for video: {video_id}")
        transcript_text = ""
        transcript_lang = "unknown"
        try:
            api = YouTubeTranscriptApi()

            # Try fetching in order: Hindi, English, then any available
            lang_attempts = [
                (["hi"], "Hindi"),
                (["en"], "English"),
                (["hi-Latn"], "Hindi (Latin script)"),
            ]

            for langs, lang_name in lang_attempts:
                try:
                    transcript = api.fetch(video_id, languages=langs)
                    transcript_text = " ".join([snippet.text for snippet in transcript])
                    transcript_lang = lang_name
                    print(f"✔ Found {lang_name} transcript")
                    break
                except Exception:
                    continue

            # Fallback: fetch default (any language / auto-generated)
            if not transcript_text.strip():
                try:
                    transcript = api.fetch(video_id)
                    transcript_text = " ".join([snippet.text for snippet in transcript])
                    transcript_lang = "auto-detected"
                    print(f"✔ Found auto-detected transcript")
                except Exception:
                    pass

        except Exception as transcript_err:
            print(f"❌ Transcript error: {transcript_err}")

        if not transcript_text.strip():
            return jsonify({"error": "No transcript found for this video. The video may not have captions in any supported language."}), 404

        print(f"✔ Loaded transcript ({len(transcript_text)} chars, language: {transcript_lang})")

        # 3. Build enriched document with metadata prepended
        metadata_header = f"VIDEO TITLE: {video_title}\n"
        if video_description:
            metadata_header += f"VIDEO DESCRIPTION: {video_description}\n"
        metadata_header += f"VIDEO URL: {video_url}\n\nTRANSCRIPT:\n"

        enriched_text = metadata_header + transcript_text

        transcript_docs = [Document(
            page_content=enriched_text,
            metadata={
                "source": video_url,
                "video_id": video_id,
                "title": video_title,
            }
        )]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(transcript_docs)
        print(f"✔ Split into {len(chunks)} chunks")

        # 4. Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        print("✔ Vector store created")

        # 5. Build QA chain with custom prompt
        hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        if not hf_token:
            return jsonify({"error": "HUGGINGFACEHUB_ACCESS_TOKEN not set in .env file"}), 500

        llm = ChatOpenAI(
            model="meta-llama/Llama-3.1-8B-Instruct",
            openai_api_key=hf_token,
            openai_api_base="https://router.huggingface.co/v1",
            temperature=0.3,
        )

        # Custom prompt that gives the model full context
        # Includes multilingual understanding instructions
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant that answers questions about a YouTube video.
You can understand multiple languages including Hindi, Hinglish (Hindi written in English/Latin script), and English.

The video title is: "{video_title}"
{desc_line}
The transcript language is: {transcript_lang}

IMPORTANT INSTRUCTIONS:
- The transcript may be in Hindi, Hinglish (Hindi written in Roman/Latin script), English, or a mix of these languages.
- You MUST understand and process the transcript regardless of the language.
- If the transcript is in Hindi or Hinglish, understand the meaning and answer the question accurately.
- Always respond in English, even if the transcript is in Hindi/Hinglish.
- If the user asks in Hindi or Hinglish, still respond in English.
- Always mention the video title when relevant.
- Provide detailed and accurate responses based on the transcript content.

Transcript context:
{{context}}

Question: {{question}}

Answer (in English):""".format(
                video_title=video_title,
                desc_line=f'Video description: "{video_description}"' if video_description else "",
                transcript_lang=transcript_lang
            )
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=False,
        )

        # Store session
        chat_sessions[video_id] = {
            "qa_chain": qa_chain,
            "title": video_title,
            "chunks": len(chunks),
        }

        print(f"✔ Chat session ready for: {video_title}")

        return jsonify({
            "status": "ready",
            "video_id": video_id,
            "title": video_title,
            "chunks": len(chunks),
        })

    except Exception as e:
        print(f"❌ Chat init error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to initialize chat: {str(e)}"}), 500


# ==========================================================
# CHATBOT — Ask a question about the video
# ==========================================================
@app.route("/chat/ask", methods=["POST"])
def chat_ask():
    """
    Ask a question about a video's content using the RAG chain.

    Request body: { "video_id": "...", "question": "..." }
    Response:     { "answer": "...", "video_id": "..." }
    """
    data = request.get_json()
    video_id = data.get("video_id")
    question = data.get("question")

    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400
    if not question:
        return jsonify({"error": "No question provided"}), 400

    session = chat_sessions.get(video_id)
    if not session:
        return jsonify({"error": "Chat session not found. Call /chat/init first."}), 404

    try:
        qa_chain = session["qa_chain"]
        result = qa_chain.invoke(question)
        answer = result.get("result", "Sorry, I couldn't find an answer.")

        return jsonify({
            "answer": answer,
            "video_id": video_id,
        })

    except Exception as e:
        print(f"❌ Chat ask error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get answer: {str(e)}"}), 500


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
