# Olist E-Commerce Intelligence

End-to-end data science project on the Olist Brazilian E-Commerce public dataset. I built this to bring together everything I've learned so far — SQL, EDA, machine learning, clustering, NLP, deep learning, dashboards, and deployment — in one project instead of scattered exercises.

## What this project does

Olist is a Brazilian marketplace that connects small businesses to big retail channels. The dataset has ~100k orders from 2016-2018, split across 9 CSV files (customers, orders, items, payments, reviews, products, sellers, geolocation, category translations).

I used it to answer a few real business questions:
- What drives revenue, and where are the customers?
- Does late delivery actually hurt customer satisfaction?
- Can we predict whether a customer will be satisfied with an order, before they leave a review?
- Can we group customers by value (RFM) to know who's worth investing in?
- Can we read the sentiment in customer reviews (originally in Portuguese) automatically?

## Project structure

```
olist-ecommerce-data-science/
├── notebooks/          # main analysis notebook (cleaning → EDA → ML → NLP)
├── models/             # trained model (satisfaction_model.pkl)
├── api/                # FastAPI app + Dockerfile
├── dashboard/          # Power BI file / screenshots
└── README.md
```

## How I approached it

1. **Data cleaning** — loaded the 9 tables, handled missing values, and found a real gotcha: `customer_id` changes with every order in Olist, even for the same person. The stable ID is `customer_unique_id`. Took me a while to catch that one.
2. **SQL** — pushed the cleaned tables into PostgreSQL and wrote the join queries there instead of doing everything in pandas, mainly so the project actually shows SQL skills, not just Python.
3. **EDA** — sales, customers, delivery, and reviews. The biggest finding: only 3% of customers ever buy twice, and late orders drop the average review score from 4.29 to 2.27.
4. **Machine Learning** — predicting customer satisfaction (Satisfied/Unsatisfied) from order data. Tried Logistic Regression, Random Forest, and XGBoost. Random Forest (tuned) came out on top at ~78% accuracy. Made sure not to leak `review_score` into the features since that's literally the target.
5. **Customer Segmentation** — RFM + K-Means, 4 clusters. Found that 88% of customers are either inactive or low-value, which is a real business problem worth flagging.
6. **NLP** — sentiment analysis on review text. Reviews are in Portuguese, so I translated a 2,000-review sample locally using MarianMT (Google Translate's API kept rate-limiting me after a few thousand requests). TF-IDF + Logistic Regression hit 81% accuracy — better than the ML model above, because the customer's own words turned out to be a stronger signal than delivery/payment metadata.
7. **Deep Learning** — tried an LSTM on the same review text. It didn't learn anything (val accuracy stuck at 60.9% the whole time). Root cause: 1,600 training examples just isn't enough to train an embedding layer from scratch. Kept this in the writeup instead of hiding it — it's a legitimate finding about when deep learning isn't the right tool.
8. **Power BI Dashboard** — 5 pages (Overview, Sales, Customers, Delivery, ML Results) built on the final cleaned dataset.
9. **FastAPI** — wrapped the satisfaction model in an API with a `/predict` endpoint.
10. **Docker** — containerized the API so it runs the same way anywhere.

## Known limitations

- The NLP/DL sections use a 2,000-review sample, not the full ~41k reviews with text — translation was the bottleneck (local hardware + no reliable translation API access at scale).
- `models/satisfaction_model.pkl` is ~85MB — GitHub flags this as large but it still works fine, just not ideal long-term (Git LFS would be the proper fix).
- The raw CSV files aren't included in this repo (see `.gitignore`) — download them from Kaggle using the link below.

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle

## Running the API locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

Or with Docker:

```bash
docker build -t olist-satisfaction-api .
docker run -d -p 8000:8000 olist-satisfaction-api
```

Then check `http://127.0.0.1:8000/docs` for the interactive API docs.

## Stack

Python, pandas, PostgreSQL, scikit-learn, XGBoost, TensorFlow/Keras, Hugging Face Transformers (MarianMT), Power BI, FastAPI, Docker, Git,N8N

## Author

Ahmed — Data Science & AI diploma student. This is my first end-to-end portfolio project, part of a broader plan to build a few more like it.
