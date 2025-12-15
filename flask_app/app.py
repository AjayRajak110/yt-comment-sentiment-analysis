import traceback
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)
CORS(app)

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "lgbm_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"  
 # <--- FINAL CORRECT FILE
model = None
vectorizer = None


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
# RUN
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
