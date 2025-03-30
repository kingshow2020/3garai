import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"
METRICS_PATH = "metrics.json"

latest_score = None

def preprocess(df):
    # إعادة تسمية الأعمدة تلقائيًا إذا كانت من واثق أو مشابهة
    col_mapping = {
        "سعر الصفقة (ريال)": "السعر",
        "المساحة (متر مربع)": "المساحة"
    }
    df = df.rename(columns=col_mapping)

    # حذف الأعمدة غير المهمة
    keep_cols = ["السعر", "المساحة", "النوع", "المدينة", "الحي", "الواجهة", "الاستخدام"]
    for col in df.columns:
        if col not in keep_cols:
            df.drop(columns=col, inplace=True)

    df["المساحة"] = pd.to_numeric(df["المساحة"].astype(str).str.replace("٫", "."), errors="coerce")
    df["السعر"] = pd.to_numeric(df["السعر"].astype(str).str.replace("٫", "."), errors="coerce")
    df.dropna(subset=["المساحة", "السعر", "النوع", "المدينة"], inplace=True)
    df = df[df["السعر"] > 0]

    for col in ["النوع", "المدينة", "الحي", "الواجهة", "الاستخدام"]:
        if col in df.columns:
            df[col] = df[col].fillna("غير محدد").astype("category")
        else:
            df[col] = "غير محدد"
            df[col] = df[col].astype("category")

    return df

def update_metrics(mae):
    if not os.path.exists(METRICS_PATH):
        metrics = {
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "weekly_progress": []
        }
    else:
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)

    previous = metrics["weekly_progress"][-1]["mean_absolute_error"] if metrics["weekly_progress"] else None
    week = len(metrics["weekly_progress"]) + 1
    improvement = None
    if previous and previous > 0:
        improvement = f"{round((previous - mae) / previous * 100)}%"

    metrics["weekly_progress"].append({
        "week": week,
        "mean_absolute_error": round(mae),
        "accuracy_improvement": improvement
    })

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def train_model_if_needed():
    global latest_score
    if not os.path.exists(DATA_PATH):
        return

    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    if df.empty:
        return

    for col in ["النوع", "المدينة", "الحي", "الواجهة", "الاستخدام"]:
        df[col] = df[col].astype("category")

    X = df[["المساحة", "النوع", "المدينة", "الحي", "الواجهة", "الاستخدام"]].copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].astype("category").cat.codes

    y = df["السعر"]

    model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    latest_score = mean_absolute_error(y, model.predict(X))
    update_metrics(latest_score)

def predict_price(data):
    if not os.path.exists(MODEL_PATH):
        raise Exception("النموذج غير متوفر")

    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([{
        "المساحة": data.المساحة,
        "النوع": data.النوع,
        "المدينة": data.المدينة,
        "الحي": getattr(data, "الحي", "غير محدد"),
        "الواجهة": getattr(data, "الواجهة", "غير محدد"),
        "الاستخدام": getattr(data, "الاستخدام", "غير محدد")
    }])

    for col in ["النوع", "المدينة", "الحي", "الواجهة", "الاستخدام"]:
        input_df[col] = pd.Series([input_df[col][0]]).astype("category").cat.codes[0]

    prediction = model.predict(input_df)[0]
    safe_zone = prediction * 0.85

    return {
        "السعر المتوقع": f"{round(prediction):,} ريال",
        "الحد الآمن للمزايدة": f"{round(safe_zone):,} ريال"
    }
