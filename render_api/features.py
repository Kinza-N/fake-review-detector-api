
import re
import numpy as np
import pandas as pd

TEXT_FEATURES = [
    "char_count","word_count","sentence_count","avg_word_length",
    "exclamation_count","question_count","comma_count",
    "uppercase_count","uppercase_ratio","digit_count","digit_ratio",
    "unique_word_ratio","repeated_char_count","url_count",
    "emoji_count","positive_word_count","negative_word_count","rating_numeric"]

BEHAVIOR_FEATURES = [
    "customer_review_count","customer_product_count","customer_rating_mean","customer_rating_std",
    "product_review_count","product_customer_count","product_rating_mean","product_rating_std",
    "review_year","review_month","review_day","review_weekday",
    "customer_active_days","reviews_per_active_day",
    "is_high_rating","is_low_rating","is_extreme_rating",
    "rating_deviation","rating_product_deviation"]

XGB_FEATURES = TEXT_FEATURES + BEHAVIOR_FEATURES

POS = {"good","great","excellent","amazing","best","love","nice","perfect","acha","achi","zabardast","behtareen","pasand"}
NEG = {"bad","worst","poor","fake","broken","waste","bakwas","ghatiya","bekar","bura"}

def normalize(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()

def create_features(df):
    data = df.copy()
    text = data["review"].fillna("").astype(str)
    data["char_count"] = text.str.len()
    data["word_count"] = text.apply(lambda x: len(x.split()))
    data["sentence_count"] = text.apply(lambda x: max(1, len(re.findall(r"[.!?]+", x))))
    data["avg_word_length"] = data["char_count"] / data["word_count"].replace(0, 1)
    data["exclamation_count"] = text.str.count("!")
    data["question_count"] = text.str.count(r"\?")
    data["comma_count"] = text.str.count(",")
    data["uppercase_count"] = text.apply(lambda x: sum(c.isupper() for c in x))
    data["uppercase_ratio"] = data["uppercase_count"] / data["char_count"].replace(0, 1)
    data["digit_count"] = text.apply(lambda x: sum(c.isdigit() for c in x))
    data["digit_ratio"] = data["digit_count"] / data["char_count"].replace(0, 1)
    data["unique_word_ratio"] = text.apply(lambda x: len(set(x.lower().split())) / max(1, len(x.split())))
    data["repeated_char_count"] = text.apply(lambda x: len(re.findall(r"(.)\1{2,}", x.lower())))
    data["url_count"] = text.apply(lambda x: len(re.findall(r"https?://|www\.", x.lower())))
    data["emoji_count"] = text.apply(lambda x: len(re.findall(r"[\U00010000-\U0010ffff]", x)))
    data["positive_word_count"] = text.apply(lambda x: sum(w in POS for w in x.lower().split()))
    data["negative_word_count"] = text.apply(lambda x: sum(w in NEG for w in x.lower().split()))
    data["rating_numeric"] = pd.to_numeric(data["Rating"], errors="coerce").fillna(0)
    return data

def create_behavior_features(data, reference_data):
    data = data.copy().reset_index(drop=True)
    reference_data = reference_data.copy()
    for d in (data, reference_data):
        d["Customer_ID"] = d["Customer_ID"].astype(str)
        d["Product_ID"] = d["Product_ID"].astype(str)
        d["Rating"] = pd.to_numeric(d["Rating"], errors="coerce")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")

    cust = (reference_data.groupby("Customer_ID").agg(
        customer_review_count=("review","count"),
        customer_product_count=("Product_ID","nunique"),
        customer_rating_mean=("Rating","mean"),
        customer_rating_std=("Rating","std")).reset_index())
    prod = (reference_data.groupby("Product_ID").agg(
        product_review_count=("review","count"),
        product_customer_count=("Customer_ID","nunique"),
        product_rating_mean=("Rating","mean"),
        product_rating_std=("Rating","std")).reset_index())
    data = data.merge(cust, on="Customer_ID", how="left").merge(prod, on="Product_ID", how="left")

    num = ["customer_review_count","customer_product_count","customer_rating_mean","customer_rating_std",
           "product_review_count","product_customer_count","product_rating_mean","product_rating_std"]
    data[num] = data[num].fillna(0)

    data["review_year"] = data["Date"].dt.year
    data["review_month"] = data["Date"].dt.month
    data["review_day"] = data["Date"].dt.day
    data["review_weekday"] = data["Date"].dt.weekday

    cd = (reference_data.groupby("Customer_ID")["Date"]
          .agg(customer_first_date="min", customer_last_date="max").reset_index())
    cd["customer_active_days"] = (cd["customer_last_date"] - cd["customer_first_date"]).dt.days
    data = data.merge(cd[["Customer_ID","customer_active_days"]], on="Customer_ID", how="left")
    data["customer_active_days"] = data["customer_active_days"].fillna(0)

    data["is_high_rating"] = (data["Rating"] >= 4).astype(int)
    data["is_low_rating"] = (data["Rating"] <= 2).astype(int)
    data["is_extreme_rating"] = data["Rating"].isin([1, 5]).astype(int)
    data["rating_deviation"] = (data["Rating"] - data["product_rating_mean"]).abs()
    data["rating_product_deviation"] = data["Rating"] - data["product_rating_mean"]
    data["reviews_per_active_day"] = data["customer_review_count"] / data["customer_active_days"].replace(0, 1)
    return data
