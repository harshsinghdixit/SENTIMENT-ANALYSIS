import os
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import pipeline
from googleapiclient.discovery import build
import torch
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
load_dotenv()

app = Flask(__name__)
CORS(app)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

device = 0 if torch.cuda.is_available() else -1
emotion_pipeline = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=device,
    truncation=True,
    max_length=512
)

def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    if len(url_or_id) == 11 and not ("/" in url_or_id or "." in url_or_id):
        return url_or_id
    parsed_url = urlparse(url_or_id)
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
        if "/shorts/" in parsed_url.path:
            return parsed_url.path.split("/shorts/")[1].split("?")[0]
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/").split("?")[0]
    return None

def calculate_impact_tier(likes, replies):
    raw_score = (likes * 0.7) + (replies * 0.3)
    if raw_score >= 50:
        return {"level": "VERY HIGH IMPACT", "color": "#6f42c1", "bg": "#f3e8ff"}
    elif raw_score >= 15:
        return {"level": "HIGH IMPACT", "color": "#0969da", "bg": "#ddf4ff"}
    elif raw_score >= 5:
        return {"level": "MODERATE IMPACT", "color": "#1b7c83", "bg": "#e6f7f8"}
    elif raw_score > 0:
        return {"level": "LOW IMPACT", "color": "#57606a", "bg": "#f6f8fa"}
    else:
        return {"level": "VERY LOW IMPACT", "color": "#8c959f", "bg": "#f3f4f6"}

def fetch_enriched_comments(video_id, order='relevance', max_results=20):
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is missing from .env")

    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    api_order = 'time' if order in ['newest', 'oldest'] else 'relevance'
    
    raw_items = []
    next_page_token = None
    fetch_all = (max_results == -1)
    target_count = max_results if not fetch_all else 5000

    while True:
        fetch_limit = 100 if fetch_all else min(100, target_count - len(raw_items))
        if fetch_limit <= 0:
            break

        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            order=api_order,
            maxResults=fetch_limit,
            pageToken=next_page_token,
            textFormat='plainText'
        ).execute()

        items = response.get('items', [])
        if not items:
            break

        raw_items.extend(items)
        next_page_token = response.get('nextPageToken')

        if not next_page_token or (not fetch_all and len(raw_items) >= target_count):
            break

    if order == 'oldest':
        raw_items.reverse()

    packets = []
    for item in raw_items:
        snippet = item['snippet']['topLevelComment']['snippet']
        text = snippet['textDisplay'].strip()
        if not text:
            continue

        likes = snippet.get('likeCount', 0)
        replies = item['snippet'].get('totalReplyCount', 0)

        try:
            lang_code = detect(text)
        except:
            lang_code = "unknown"

        truncated_text = text[:500]
        pred = emotion_pipeline(truncated_text)[0][0]
        label = pred['label']
        conf = round(float(pred['score']), 4)
        
        impact = calculate_impact_tier(likes, replies)

        packets.append({
            "comment_id": item['id'],
            "author_name": snippet.get('authorDisplayName', '@anonymous'),
            "comment_text": text,
            "published_at": snippet.get('publishedAt', ''),
            "like_count": likes,
            "reply_count": replies,
            "emotion": label,
            "confidence": conf,
            "impact": impact,
            "demographics": {
                "language": lang_code.upper()
            }
        })

    return packets

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze-youtube', methods=['POST'])
def analyze_youtube():
    data = request.json or {}
    raw_input = data.get('video_id')
    max_results = data.get('max_results', 20)
    sort_order = data.get('sort_order', 'top')

    if not raw_input:
        return jsonify({"error": "No URL or Video ID provided"}), 400

    video_id = extract_video_id(raw_input)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL or Video ID"}), 400

    try:
        packets = fetch_enriched_comments(video_id, order=sort_order, max_results=max_results)

        emotion_counts = {}
        for p in packets:
            e = p['emotion']
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        total = len(packets) or 1
        chart_percentages = {
            e: round((c / total) * 100, 1)
            for e, c in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        }

        return jsonify({
            "status": "success",
            "video_id_analyzed": video_id,
            "total_comments_analyzed": len(packets),
            "sort_order_used": sort_order,
            "chart_percentages": chart_percentages,
            "packets": packets
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)