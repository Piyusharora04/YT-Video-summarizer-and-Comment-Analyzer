import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_comment_downloader import *
import requests

st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")

@st.cache_resource
def load_model_and_tokenizer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model_path = "D:/Desktop/MP1/myTrial3.h5"
model, tokenizer = load_model_and_tokenizer(model_path)

def get_video_thumbnail(video_id):
    api_key = "AIzaSyC4JXlj6bXJPRJHxhn8kZXvMhPVNOdQUbc"
    base_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet", "id": video_id, "key": api_key}
    
    response = requests.get(base_url, params=params)
    data = response.json()
    items = data.get("items", [])
    if items:
        return items[0]["snippet"]["thumbnails"]["maxres"]["url"]
    return None

def get_id(video_link):
    if "feature=shared" in video_link:
        video_id = video_link.split("?")[0]
        video_id = video_id.split("/")[-1]
        return video_id
    else:
        video_id = video_link.split("=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]
        return video_id 

def yvs(video_url):
    transcript = YouTubeTranscriptApi.get_transcript(get_id(video_url))
    result = ' '.join([i['text'] for i in transcript])
    
    cohere_api_key = "2wHSGlZJesqUP5A6L5LmumjMEjH2P8sVVGFSA84a"
    
    def summarize_with_cohere(text):
        url = "https://api.cohere.ai/summarize"
        headers = {'Authorization': f'Bearer {cohere_api_key}'}
        data = {
            "model": "command",  
            "text": text
        }
        response = requests.post(url, headers=headers, json=data)

        if 'summary' in response.json():
            return response.json()['summary']
        else:
            return "Summary not available from the API for this segment."

    num_iters = int(len(result)/200)  
    summarized_text = []


    for i in range(0, num_iters + 1):
        start = i * 1000
        end = (i + 1) * 1000
        # print("Input text \n" + result[start:end])
        out = summarize_with_cohere(result[start:end])
        # print("Summarized text\n" + out)
        summarized_text.append(out)
        

    # print(summarized_text)

    output = []
    for item in summarized_text:
        if(item != 'Summary not available from the API for this segment.'):
            output.append(str(item))

    new_string = ''
    for item in output:
        new_string += item + '\n'
    summary = new_string 
    return summary

def tokenize_comments(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

@st.cache_data(show_spinner=False)
def analyze_comments(video_url):
    downloader = YoutubeCommentDownloader()
    comments = list(downloader.get_comments_from_url(video_url, sort_by=0))
    df = pd.DataFrame(comments[:1000])
    
    if "text" in df:
        tokenized_inputs = df['text'].apply(tokenize_comments)
        input_ids = torch.cat([x['input_ids'] for x in tokenized_inputs])
        attention_mask = torch.cat([x['attention_mask'] for x in tokenized_inputs])

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        df['sentiment'] = [id2label[pred] for pred in predictions]
        return df
    return pd.DataFrame()

def basics():
    with st.sidebar:
        st.title("The Ultimate YouTube Video Analyzer cum Summarizer")
        video_link = st.text_input("Enter YouTube Video Link")
        if st.button("Let's Go...."):
            st.session_state.video_link = video_link
            st.session_state.comments_data = None  # Reset comments data for a new video

    if "video_link" in st.session_state:
        video_link = st.session_state.video_link
        col1, col2 = st.columns(2)

        with col1:
            thumbnail_url = get_video_thumbnail(get_id(video_link))
            st.image(thumbnail_url if thumbnail_url else st.write("No thumbnail available"))

        with col2:
            summary = yvs(video_link)
            st.success("Summary:")
            st.write(summary)

        with st.spinner("Analyzing Comments... Please Wait!"):
            if st.session_state.comments_data is None:  # Perform analysis only if not already done
                st.session_state.comments_data = analyze_comments(video_link)

            data1 = st.session_state.comments_data
            if not data1.empty:                
                total_pos = data1['sentiment'].value_counts().get('positive', 0)
                total_neu = data1['sentiment'].value_counts().get('neutral', 0)
                total_neg = data1['sentiment'].value_counts().get('negative', 0)

                pi_labels = 'Positive', 'Neutral', 'Negative'
                sizes = [total_pos, total_neu, total_neg]

                fig1, ax1 = plt.subplots(figsize=(5, 5))  # Decrease the size using figsize=(width, height)
                ax1.pie(sizes, labels=pi_labels, autopct='%1.1f%%', shadow=True, startangle=90)
                ax1.axis('equal')

                col_center = st.columns([1, 1, 1])  
                with col_center[1]:  
                    st.pyplot(fig1)


            if total_pos > (2 * total_neg):
                st.success("This Video has Great Reach and Liked by a lot...")
            else:
                st.warning("Bad Video")

basics()

if "comments_data" in st.session_state:
    data1 = st.session_state.comments_data
    add_selectbox = st.selectbox("What type of comments would you like to see...?", ["Positive", "Negative", "Neutral"])

    if add_selectbox == "Positive":
        st.write(data1[data1["sentiment"] == "positive"]["text"])
    elif add_selectbox == "Negative":
        st.write(data1[data1["sentiment"] == "negative"]["text"])
    elif add_selectbox == "Neutral":
        st.write(data1[data1["sentiment"] == "neutral"]["text"])