# YouTube Video Summarizer and Comment Sentiment Analyzer 📝🔍

##  🌟 Overview
This project is a YouTube Video Summarizer and Comment Sentiment Analyzer built using Streamlit. It retrieves video transcripts and comments from YouTube, provides a summarized version of the video content, and analyzes the sentiment of user comments to gauge the public’s opinion about the video. The project integrates NLP models and the Cohere API for summarization, leveraging the powerful BERT transformer model for sentiment analysis.

![Image 1](https://drive.google.com/uc?export=view&id=1Hlfq15gNgn2M_srqxew-ZD4w4g4WvDQR)
![Image 2](https://drive.google.com/uc?export=view&id=1W8REBWAwyqKUtmmPa7XX-zgXcFXT7npT)

---

## 🚀Features
- **Video Summarization**: Summarizes the transcript of a YouTube video to give a quick overview of its content.
- **Sentiment Analysis of Comments**: Analyzes YouTube comments to classify them as positive, neutral, or negative.
- **Thumbnail Display**: Retrieves and displays the video’s thumbnail for visual context.
- **Sentiment Distribution**: Shows sentiment analysis results in a pie chart to give a visual breakdown.
- **Filtered Comment View**: Allows users to view comments filtered by sentiment type (positive, negative, neutral).

---

## Project Structure
- 📂 YouTube-Video-Summarizer
- 📄 app.py # Main application file 
- 📄 requirements.txt # Project dependencies 
- 📄 README.md # Project documentation

---

## ⚙️Installation and Setup

### 📋Prerequisites
- Python 3.8 or higher
- Cohere API Key: [Sign up here](https://cohere.ai/) to get an API key.
- YouTube API Key: [Get API Key](https://console.developers.google.com/) by creating a project and enabling the YouTube Data API v3.

### 🛠Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Piyusharora04/YT-Video-summarizer-and-Comment-Analyzer.git
2. **Model Download**:
   - Due to file size constraints, the model file (model.h5) is hosted on Google Drive. You can download it [here](https://drive.google.com/drive/folders/1ywtPTyhhMRKBXxIVJNL_p8BFbk8D3czT?usp=sharing).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Add API Keys:
  In the Streamlit secrets configuration, add your API keys
   - [secrets]
     - YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
     - COHERE_API_KEY = "YOUR_COHERE_API_KEY"
5. Run the Application:
   ```bash
   streamlit run app.py

---

## 🖥 Usage
- Open the application in your browser, and enter a YouTube video link.
- The app will retrieve the transcript, generate a summary, and display it alongside the video’s thumbnail.
- Comments are downloaded, analyzed, and classified into positive, neutral, or negative sentiments, displayed in both a table and a pie chart.

---

## 📦 Dependencies
- To run this project, the following dependencies are needed:

1. numpy
2. pandas
3. matplotlib
4. streamlit
5. transformers
6. torch
7. youtube-transcript-api
8. youtube-comment-downloader
9. requests
10. Pillow
- All dependencies are listed in requirements.txt and can be installed in one command.

---

## 📊 Model Information
- Summarization: Utilizes Cohere’s NLP model via API for accurate summarization.
- Sentiment Analysis: Uses a pre-trained BERT model to classify comments into sentiment categories.
---

## 🔮 Future Enhancements
- Advanced Summarization Techniques: Incorporate other NLP models to improve the summarization quality.
- More Granular Sentiment Analysis: Include additional categories and subcategories for comment sentiment.
- Historical Analysis: Allow sentiment tracking over time for video comments.

---

## 🤝Contributing
- If you wish to contribute, please fork the repository and submit a pull request. For major changes, open an issue to discuss the changes.



