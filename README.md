---
title: Movie Recommender System
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🎬 Movie Recommender System

This project is a Flask-based movie recommendation system powered by
**Machine Learning** and **NLP**.  

It uses preprocessed TMDB data and cosine similarity to recommend
similar movies. Posters and details are dynamically fetched from
the **TMDB API**.

- **Backend**: Flask
- **Frontend**: Jinja, HTML, CSS, JS
- **Data**: TMDB (movies, credits)
- **Deployment**: Hugging Face Spaces (Docker SDK)




# 🎬 Movie Recommender System

## 📌 Project Motivation
Movie recommendation systems are widely used by platforms like Netflix, Amazon Prime, and Hulu to enhance user experience by suggesting movies based on their preferences.  
This project demonstrates how machine learning and NLP can be applied to build a personalized **Movie Recommender System**.

---

## 📝 Project Overview
The Movie Recommender System suggests movies to users based on **content similarity** using features like cast, crew, genres, keywords, and movie descriptions.  
It uses **content-based filtering** powered by machine learning models to recommend the most relevant movies.

---

## ✨ Features
- 🎥 Personalized movie recommendations  
- 🖼️ Movie posters displayed for results  
- 🔄 Trending movie slider on homepage  
- ⚡ Fast search powered by preprocessed movie features  
- 🌐 Integration with **TMDB API** for fetching posters  

---

## 🛠️ Technology Used
- **Programming Language**: Python  
- **Framework**: Flask  
- **Libraries**: Pandas, NumPy, scikit-learn, Requests  
- **Frontend**: HTML, CSS, JavaScript (Vanilla)  
- **Deployment**: Hugging Face Spaces (Docker)  

---

## 📂 Project Structure
```
├── app.py                # Main Flask app
├── assets/               # Static assets like icons/images
├── data/                 # Dataset (raw & processed)
├── models/               # Trained model (pkl)
├── src/Model_Building/   # Scripts for preprocessing & feature engineering
├── static/               # CSS, JS, icons
├── templates/            # HTML templates (base, index, about)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Build instructions for Hugging Face
└── README.md             # Project documentation
```

---

## ⚙️ How it Works
1. **Data Collection**: Dataset from TMDB (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`).  
2. **Data Preprocessing**: Cleaning, merging, and extracting key movie features.  
3. **Feature Engineering**: NLP techniques to build feature vectors (Bag of Words/TF-IDF).  
4. **Model Training**: Cosine similarity to measure movie similarity.  
5. **Prediction**: Top 5 similar movies returned as recommendations.  
6. **Integration**: Posters fetched via TMDB API.  

---

## 📌 Key Files
- `app.py` → Runs Flask server & routes (homepage, recommend, about).  
- `src/Model_Building/01_data_preprocessing.py` → Prepares raw data.  
- `src/Model_Building/02_data_featuring.py` → Feature engineering.  
- `models/movie_recommender.pkl` → Trained model (stored with Git LFS).  
- `templates/index.html` → Homepage with search & recommendation UI.  
- `templates/about.html` → About project page.  
- `static/css/styles.css` → Styling.  
- `requirements.txt` → Dependencies for deployment.  

---

## 📬 Contact
- 👤 **Author**: Abhijeet Thombare  
- 🔗 [GitHub](https://github.com/abhithombare45/Movie-Recommender-System)  
- 🔗 [Hugging Face]([https://huggingface.co/spaces/abhithombare45/Movie-Recommender-System](https://huggingface.co/spaces/abhithombare45/Movie-Recommendation-System))  
