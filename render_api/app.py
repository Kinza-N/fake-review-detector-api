
import os, json, hashlib
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import *

HERE = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(HERE, "xgboost_hybrid.pkl"))

THRESHOLD = json.load(
    open(os.path.join(HERE, "xgboost_threshold.json"))
)["threshold"]

REFERENCE = pd.read_csv(
    os.path.join(HERE, "reference_train.csv.gz")
)

REFERENCE["review_normalized"] = (
    REFERENCE["review_normalized"].astype(str)
)

MAX_REVIEWS = 50

def hid(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


LABELS = {
    "char_count": ("Review length", "{:.0f} characters"),
    "word_count": ("Word count", "{:.0f} words"),
    "sentence_count": ("Number of sentences", "{:.0f}"),
    "avg_word_length": ("Average word length", "{:.1f} letters"),
    "exclamation_count": ("Exclamation marks", "{:.0f}"),
    "question_count": ("Question marks", "{:.0f}"),
    "comma_count": ("Commas", "{:.0f}"),
    "uppercase_count": ("Capital letters", "{:.0f}"),
    "uppercase_ratio": ("Share of capital letters", "{:.0%}"),
    "digit_count": ("Digits", "{:.0f}"),
    "digit_ratio": ("Share of digits", "{:.0%}"),
    "unique_word_ratio": ("Vocabulary variety", "{:.0%} unique words"),
    "repeated_char_count": ("Stretched letters (like 'sooo')", "{:.0f}"),
    "url_count": ("Links in text", "{:.0f}"),
    "emoji_count": ("Emojis", "{:.0f}"),
    "positive_word_count": ("Praise words (good, best, perfect...)", "{:.0f}"),
    "negative_word_count": ("Complaint words (bad, worst, waste...)", "{:.0f}"),
    "rating_numeric": ("Star rating", "{:.0f} stars"),
    "is_high_rating": ("High rating (4-5 stars)", "yesno"),
    "is_low_rating": ("Low rating (1-2 stars)", "yesno"),
    "is_extreme_rating": ("Extreme rating (1 or 5 stars)", "yesno"),
    "customer_review_count": ("Reviews written by this reviewer", "{:.0f}"),
    "customer_product_count": ("Different products this reviewer rated", "{:.0f}"),
    "customer_rating_mean": ("Reviewer's average rating", "{:.1f}"),
    "customer_rating_std": ("Variation in reviewer's ratings", "{:.1f}"),
    "customer_active_days": ("Days this reviewer has been active", "{:.0f}"),
    "reviews_per_active_day": ("Reviews per active day", "{:.2f}"),
    "product_review_count": ("Reviews on this product", "{:.0f}"),
    "product_customer_count": ("Different reviewers of this product", "{:.0f}"),
    "product_rating_mean": ("Product's average rating", "{:.1f}"),
    "product_rating_std": ("Spread of this product's ratings", "{:.1f}"),
    "rating_deviation": ("Distance from product's average rating", "{:.1f} stars"),
    "rating_product_deviation": ("Rating vs product average", "{:+.1f} stars"),
}

SKIP = {
    "review_year", "review_month", "review_day", "review_weekday"
}

CUSTOMER_FEATS = {
    "customer_review_count", "customer_product_count",
    "customer_rating_mean", "customer_rating_std",
    "customer_active_days", "reviews_per_active_day"
}

PRODUCT_FEATS = {
    "product_review_count", "product_customer_count",
    "product_rating_mean", "product_rating_std",
    "rating_deviation", "rating_product_deviation"
}


def fmt_value(name, v):
    label, f = LABELS[name]

    if f == "yesno":
        return label, "yes" if v >= 0.5 else "no"

    return label, f.format(v)


def strength_word(w):
    return "strong" if w >= 1.0 else (
        "moderate" if w >= 0.4 else "slight"
    )


def explain(x_row, contribs, is_fake, has_cust, has_prod, k=3):

    items = []

    for name, c, v in zip(XGB_FEATURES, contribs[:-1], x_row):

        if name in SKIP:
            continue

        if name in CUSTOMER_FEATS and not has_cust:
            continue

        if name in PRODUCT_FEATS and not has_prod:
            continue

        label, val = fmt_value(name, float(v))

        items.append({
            "text": f"{label}: {val}",
            "pushes": "fake" if c > 0 else "real",
            "weight": float(abs(c))
        })

    items.sort(key=lambda d: -d["weight"])

    verdict = "fake" if is_fake else "real"

    pick = lambda lst: [
        {"text": i["text"], "strength": strength_word(i["weight"])}
        for i in lst
    ]

    return {
        "supporting": pick([
            i for i in items if i["pushes"] == verdict
        ][:k]),

        "opposing": pick([
            i for i in items if i["pushes"] != verdict
        ][:1])
    }


class ReviewIn(BaseModel):
    text: str
    rating: Optional[float] = None
    author: Optional[str] = ""
    date: Optional[str] = ""


class Payload(BaseModel):
    product_id: Optional[str] = "unknown"
    reviews: List[ReviewIn]


app = FastAPI(title="Fake Review Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(p: Payload):

    reviews = p.reviews[:MAX_REVIEWS]

    if not reviews:
        return {"results": []}

    rows = [{
        "Product_ID": p.product_id,
        "Customer_ID": hid(r.author or "anonymous"),
        "Customer_Name": "",
        "Date": r.date,
        "Rating": r.rating,
        "review": r.text,
        "review_normalized": normalize(r.text)
    } for r in reviews]

    df = create_features(pd.DataFrame(rows))
    df = create_behavior_features(df, REFERENCE)

    X = df[XGB_FEATURES].fillna(0).astype(float)

    probs = model.predict_proba(X)[:, 1]

    contribs = model.get_booster().predict(
        xgb.DMatrix(X),
        pred_contribs=True
    )

    results = []

    for i, pr in enumerate(probs):

        is_fake = bool(pr >= THRESHOLD)

        has_cust = (
            float(df["customer_review_count"].iloc[i]) > 0
        )

        has_prod = (
            float(df["product_review_count"].iloc[i]) > 0
        )

        results.append({
            "index": i,
            "probability": float(pr),
            "is_fake": is_fake,
            "reasons": explain(
                X.iloc[i].values,
                contribs[i],
                is_fake,
                has_cust,
                has_prod
            )
        })

    return {
        "threshold": THRESHOLD,
        "results": results
    }
