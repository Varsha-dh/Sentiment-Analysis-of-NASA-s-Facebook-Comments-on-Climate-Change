# 🚀 Sentiment Analysis of NASA's Facebook Comments on Climate Change

This project performs sentiment analysis on public comments from NASA's Facebook posts related to climate change using a multilingual BERT model. The aim is to understand public perception, emotional tone, and engagement trends over time.

## 📌 Objective
Analyze user sentiment on NASA’s climate-related Facebook posts using advanced NLP techniques, identify key discussion topics, and explore how sentiment correlates with user engagement (likes and comments).


This dataset encompasses over 500 user comments collected from high-performing
posts on NASA's Facebook page dedicated to climate change
(https://web.facebook.com/NASAClimateChange/). The comments, gathered from
various posts between 2020 and 2023, offer a diverse range of public opinions and
sentiments about climate change and NASA's related activities.

Data Science Applications
Despite not being a large dataset, it offers valuable opportunities for analysis and
Natural Language Processing (NLP). Potential applications include:

● Sentiment Analysis: Gauge public opinion on climate change and NASA's
communication strategies.
● Trend Analysis: Identify shifts in public sentiment over the specified period.
● Engagement Analysis: Understand the correlation between the content of a
post and user engagement.
● Topic Modeling: Discover prevalent themes in public discourse about climate
change.

## 🧰 Tools & Technologies

- **Language**: Python  
- **Libraries**:
  - `transformers`, `torch` – for sentiment analysis using BERT
  - `pandas`, `matplotlib`, `seaborn` – data handling and visualization
  - `gensim`, `pyLDAvis` – topic modeling
  - `nltk`, `re` – text preprocessing
## 🧠 Model Used

**Model 1: `nlptown/bert-base-multilingual-uncased-sentiment`**  
- Fine-tuned for multilingual sentiment classification  
- Outputs sentiment from **1 (very negative) to 5 (very positive) stars**  
- Chosen for its **multilingual support**, **fine-grained output**, and **alignment with engagement trends**

Also compared with:
- **VADER** (lexicon-based)
- **TextBlob** (rule-based)
- **cardiffnlp/twitter-roberta-base-sentiment & distilbert-base-uncased-finetuned-sst-2-english** (transformer-based custom models)

## 📊 Key Insights

- **Sentiment Distribution**: Majority of comments were neutral to positive  
- **Engagement Patterns**:
  - Neutral/Negative sentiment received more **comments**
  - Positive sentiment attracted more **likes** on average
- **Topic Modeling** (via LDA): Revealed 5 major themes including:
  - Global warming
  - Climate skepticism
  - NASA’s scientific contributions
  - Urgency in action
  - Carbon emissions

📦 sentiment-nasa-climate/
├── data/          # Contains: climate_nasa.csv
├── notebooks/     # Contains: sentiment_analysis.ipynb (Jupyter Notebook)
├── README.md      # You want to edit this

## 📌 How to Run

1. Install dependencies:

```bash
pip install transformers torch pandas matplotlib seaborn gensim nltk

2.Run sentiment model:

!pip install torch
!pip install transformers
!pip install transformers torch -q
pip install nltk gensim pyLDAvis



model1_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
model1 = AutoModelForSequenceClassification.from_pretrained(model1_name)
model1.eval()

3.Use notebooks in /notebooks for full analysis.

✅ Conclusion
This project reveals public sentiment trends and thematic concerns around climate change as discussed on NASA's Facebook page. The multilingual BERT model helped capture nuanced sentiment in a diverse, global dataset.

📬 Contact
Author: Varsha Dudhat
Email: dhameliyavarsha@gmail.com
LinkedIn: [https://www.linkedin.com/in/varsha-dudhat-25429613a/]
GitHub: [https://github.com/Varsha-dh]



