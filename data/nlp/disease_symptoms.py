"""
Curated Disease → Symptom knowledge base for the NLP Symptom Checker.

NOTE: The Kaggle "Disease Prediction Based on Symptoms" dataset could not
be downloaded in this environment (no internet access). This module uses
a hand-curated knowledge base of 20 common conditions and their typical
symptoms (standard clinical knowledge), from which realistic natural-
language training sentences are procedurally generated for the TF-IDF +
ML classifier. Swapping in the real Kaggle dataset later requires no
pipeline changes -- just replace DISEASE_SYMPTOMS with the real mapping.
"""

DISEASE_SYMPTOMS = {
    "Common Cold": ["runny nose", "sneezing", "sore throat", "mild fever", "cough", "nasal congestion"],
    "Influenza (Flu)": ["high fever", "body ache", "chills", "fatigue", "headache", "dry cough"],
    "COVID-19": ["fever", "dry cough", "loss of taste", "loss of smell", "shortness of breath", "fatigue"],
    "Migraine": ["severe headache", "nausea", "sensitivity to light", "sensitivity to sound", "visual aura"],
    "Diabetes": ["frequent urination", "excessive thirst", "fatigue", "blurred vision", "slow healing wounds", "unexplained weight loss"],
    "Hypertension": ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath", "nosebleed"],
    "Gastroenteritis": ["diarrhea", "vomiting", "stomach cramps", "nausea", "fever", "dehydration"],
    "Asthma": ["wheezing", "shortness of breath", "chest tightness", "coughing", "difficulty breathing"],
    "Pneumonia": ["fever", "chills", "cough with phlegm", "chest pain", "shortness of breath", "fatigue"],
    "Urinary Tract Infection": ["burning urination", "frequent urination", "cloudy urine", "pelvic pain", "fever"],
    "Food Poisoning": ["nausea", "vomiting", "diarrhea", "stomach pain", "fever", "weakness"],
    "Anemia": ["fatigue", "pale skin", "shortness of breath", "dizziness", "cold hands and feet", "irregular heartbeat"],
    "Allergic Rhinitis": ["sneezing", "runny nose", "itchy eyes", "nasal congestion", "watery eyes"],
    "Chickenpox": ["itchy rash", "fever", "fatigue", "loss of appetite", "blister like lesions"],
    "Typhoid": ["high fever", "weakness", "stomach pain", "headache", "loss of appetite", "constipation"],
    "Malaria": ["high fever", "chills", "sweating", "headache", "nausea", "muscle pain"],
    "Dengue Fever": ["high fever", "severe headache", "joint pain", "muscle pain", "skin rash", "pain behind eyes"],
    "Appendicitis": ["abdominal pain", "nausea", "vomiting", "fever", "loss of appetite", "pain near navel"],
    "Kidney Stones": ["severe back pain", "blood in urine", "nausea", "vomiting", "frequent urination", "fever"],
    "Anxiety Disorder": ["excessive worry", "restlessness", "rapid heartbeat", "sweating", "difficulty concentrating", "fatigue"],
}

SEVERITY_TAGS = {
    "Pneumonia": "moderate-high", "COVID-19": "moderate-high", "Dengue Fever": "high",
    "Typhoid": "high", "Malaria": "high", "Appendicitis": "urgent", "Kidney Stones": "moderate-high",
    "Diabetes": "moderate", "Hypertension": "moderate", "Asthma": "moderate",
}

def urgency_of(disease):
    return SEVERITY_TAGS.get(disease, "mild-moderate")
