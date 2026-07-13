"""
NLP Module — AI Symptom Checker.

Pipeline: free-text symptom description -> TF-IDF vectorization (word
1-2 grams) -> Logistic Regression multi-class classifier -> ranked list
of likely conditions with calibrated probabilities. A lightweight,
practical "TF-IDF + ML hybrid" NLP approach (transformer models like
BERT need pretrained weights downloaded from the internet, which is not
available in this offline sandbox).
"""
import os, sys, pickle, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "nlp"))
from disease_symptoms import DISEASE_SYMPTOMS

TEMPLATES = [
    "I have {s}",
    "I have been experiencing {s}",
    "I'm suffering from {s}",
    "Feeling {s} for the past few days",
    "My symptoms are {s}",
    "I've had {s} since yesterday",
    "Experiencing {s} and it's getting worse",
    "I feel {s}",
    "Patient reports {s}",
    "I've noticed {s} recently",
]

def join_symptoms(symptoms):
    if len(symptoms) == 1:
        return symptoms[0]
    return ", ".join(symptoms[:-1]) + " and " + symptoms[-1]

def generate_training_data(n_per_disease=60, seed=42):
    rng = random.Random(seed)
    texts, labels = [], []
    diseases = list(DISEASE_SYMPTOMS.keys())
    for disease in diseases:
        symptoms = DISEASE_SYMPTOMS[disease]
        for _ in range(n_per_disease):
            k = rng.randint(2, min(4, len(symptoms)))
            chosen = rng.sample(symptoms, k)
            rng.shuffle(chosen)
            template = rng.choice(TEMPLATES)
            text = template.format(s=join_symptoms(chosen))
            # occasionally mix in one symptom from a different disease (real-world noise)
            if rng.random() < 0.12:
                other = rng.choice(diseases)
                if other != disease:
                    noisy_symptom = rng.choice(DISEASE_SYMPTOMS[other])
                    text += f", also some {noisy_symptom}"
            texts.append(text)
            labels.append(disease)
    return texts, labels

def main():
    print("Generating symptom-description training sentences...")
    texts, labels = generate_training_data()
    print(f"  {len(texts)} training sentences across {len(DISEASE_SYMPTOMS)} conditions")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=5.0)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    proba = clf.predict_proba(Xte)
    top3 = top_k_accuracy_score(y_test, proba, k=3, labels=clf.classes_)
    print(f"\n  🗣️ AI Symptom Checker   Top-1 Accuracy: {acc*100:.2f}%   Top-3 Accuracy: {top3*100:.2f}%")

    bundle = {
        "vectorizer": vectorizer, "model": clf,
        "diseases": list(DISEASE_SYMPTOMS.keys()),
        "disease_symptoms": DISEASE_SYMPTOMS,
        "accuracy": round(acc*100, 2), "top3_accuracy": round(top3*100, 2),
        "name": "AI Symptom Checker", "icon": "🗣️", "color": "#0d6efd",
        "description": "Chat-based NLP assistant that analyzes symptom descriptions "
                        "and suggests likely conditions with confidence scores.",
    }
    out_path = os.path.join(os.path.dirname(__file__), "nlp_symptom_checker.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print("✅ Saved model to", out_path)

if __name__ == "__main__":
    main()
