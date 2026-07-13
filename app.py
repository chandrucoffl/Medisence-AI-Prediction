from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pickle, os, random, sys, io, base64
import numpy as np
from datetime import date
from PIL import Image

app = Flask(__name__)
app.secret_key = "medisense_ai_secret_key_change_in_production"

# ── DOCTOR STORE ────────────────────────────────────────────────
DOCTORS = {
    "admin": {
        "password":       "Chandru@123",
        "full_name":      "Admin Doctor",
        "age":            "35",
        "dob":            "1990-01-01",
        "mobile":         "9876543210",
        "specialization": "General Physician",
        "hospital":       "MediSense Hospital",
        "created_at":     str(date.today()),
    }
}

# ── MODEL LOADING ───────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "model"))
sys.path.insert(0, os.path.join(BASE, "data", "xray"))
from feature_utils import conv_feature_extract, preprocess_upload
from generate_xray_images import generate_dataset
from explainability import occlusion_heatmap, overlay_heatmap

with open(os.path.join(BASE, "model", "diseases.pkl"), "rb") as f:
    DISEASES = pickle.load(f)

with open(os.path.join(BASE, "model", "xray_dl.pkl"), "rb") as f:
    XRAY_MODEL = pickle.load(f)

with open(os.path.join(BASE, "model", "nlp_symptom_checker.pkl"), "rb") as f:
    NLP_MODEL = pickle.load(f)

sys.path.insert(0, os.path.join(BASE, "data", "nlp"))
from disease_symptoms import urgency_of
from db import init_db, create_patient, list_patients, get_patient, add_assessment, get_assessments

init_db()

# Friendly display labels for feature names (fallback: title-cased raw name)
FEATURE_LABELS = {
    # Diabetes
    "Pregnancies": "Pregnancies", "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)", "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (mu U/mL)", "BMI": "BMI", "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
    # Heart
    "age": "Age (years)", "sex": "Sex (1=Male, 0=Female)", "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)", "chol": "Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar > 120 (1=Yes, 0=No)", "restecg": "Resting ECG (0-2)",
    "thalach": "Max Heart Rate Achieved", "exang": "Exercise-Induced Angina (1=Yes, 0=No)",
    "oldpeak": "ST Depression (Oldpeak)", "slope": "Slope of Peak Exercise ST (0-2)",
    "ca": "Major Vessels Colored (0-3)", "thal": "Thalassemia (1=Normal,2=Fixed,3=Reversible)",
    # Kidney
    "blood_pressure": "Blood Pressure (mm Hg)", "specific_gravity": "Urine Specific Gravity",
    "albumin": "Albumin (0-5)", "sugar": "Sugar (0-5)",
    "blood_glucose_random": "Random Blood Glucose (mg/dL)", "blood_urea": "Blood Urea (mg/dL)",
    "serum_creatinine": "Serum Creatinine (mg/dL)", "sodium": "Sodium (mEq/L)",
    "potassium": "Potassium (mEq/L)", "haemoglobin": "Haemoglobin (g/dL)",
    "packed_cell_volume": "Packed Cell Volume (%)", "white_blood_cell_count": "White Blood Cell Count (/cumm)",
    "red_blood_cell_count": "Red Blood Cell Count (millions/cumm)",
}

def label_for(feat):
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

# ═══════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pwd  = request.form.get("password", "")
        if user in DOCTORS and DOCTORS[user]["password"] == pwd:
            otp = str(random.randint(1000, 9999))
            session["pending_otp"]   = otp
            session["temp_user"]     = user
            session["doctor_name"]   = DOCTORS[user]["full_name"]
            session["doctor_mobile"] = DOCTORS[user]["mobile"]
            return redirect(url_for("verify_otp"))
        return render_template("login.html", error="Invalid username or password!")
    return render_template("login.html")

@app.route("/verify", methods=["GET", "POST"])
def verify_otp():
    mobile = session.get("doctor_mobile", "XXXXXXXXXX")
    otp    = session.get("pending_otp", "")
    if request.method == "POST":
        if request.form.get("otp") == otp:
            session["logged_in"] = True
            session["username"]  = session.get("temp_user")
            session.pop("pending_otp", None)
            return redirect(url_for("home"))
        return render_template("verify.html", error="Wrong OTP! Try again.", mobile=mobile, otp=otp)
    return render_template("verify.html", mobile=mobile, otp=otp)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    today = str(date.today())
    if request.method == "POST":
        username         = request.form.get("username", "").strip()
        password         = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        full_name        = request.form.get("full_name", "").strip()
        age              = request.form.get("age", "").strip()
        dob              = request.form.get("dob", "").strip()
        mobile           = request.form.get("mobile", "").strip()
        specialization   = request.form.get("specialization", "").strip()
        hospital         = request.form.get("hospital", "").strip()

        form_data = {"username": username, "full_name": full_name, "age": age,
                     "dob": dob, "mobile": mobile, "specialization": specialization,
                     "hospital": hospital}

        if not username or len(username) < 4:
            return render_template("signup.html", error="Username must be at least 4 characters.", form_data=form_data, today=today)
        if username in DOCTORS:
            return render_template("signup.html", error=f"Username '{username}' already exists!", form_data=form_data, today=today)
        if len(password) < 8:
            return render_template("signup.html", error="Password must be at least 8 characters.", form_data=form_data, today=today)
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match!", form_data=form_data, today=today)
        if not mobile.isdigit() or len(mobile) != 10:
            return render_template("signup.html", error="Mobile number must be exactly 10 digits.", form_data=form_data, today=today)
        if not age.isdigit() or not (22 <= int(age) <= 80):
            return render_template("signup.html", error="Age must be between 22 and 80.", form_data=form_data, today=today)
        if not full_name:
            return render_template("signup.html", error="Full name is required.", form_data=form_data, today=today)
        if not specialization:
            return render_template("signup.html", error="Please select your specialization.", form_data=form_data, today=today)

        DOCTORS[username] = {"password": password, "full_name": full_name, "age": age,
                              "dob": dob, "mobile": mobile, "specialization": specialization,
                              "hospital": hospital, "created_at": today}
        return redirect(url_for("login", registered="yes"))

    return render_template("signup.html", today=today)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ═══════════════════════════════════════════════════════════════
#  MAIN ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    diseases_info = {k: {"name": v["name"], "description": v["description"],
        "icon": v["icon"], "color": v["color"], "accuracy": v["accuracy"], "auc": v["auc"]}
        for k, v in DISEASES.items()}
    xray_info = {"name": XRAY_MODEL["name"], "description": XRAY_MODEL["description"],
        "icon": XRAY_MODEL["icon"], "color": XRAY_MODEL["color"],
        "accuracy": XRAY_MODEL["accuracy"], "auc": XRAY_MODEL["auc"]}
    nlp_info = {"name": NLP_MODEL["name"], "description": NLP_MODEL["description"],
        "icon": NLP_MODEL["icon"], "color": NLP_MODEL["color"],
        "accuracy": NLP_MODEL["accuracy"], "auc": NLP_MODEL["top3_accuracy"]}
    active_patient = None
    if session.get("active_patient_id"):
        active_patient = get_patient(session["active_patient_id"])
    return render_template("home.html", diseases=diseases_info, xray=xray_info, nlp=nlp_info,
                            active_patient=active_patient)

def predict_row(disease_key, feature_values):
    """Run one disease model on a dict of {feature_name: value}. Returns (result, confidence_str)."""
    d = DISEASES[disease_key]
    values = [float(feature_values.get(n, 0) or 0) for n in d["feature_names"]]
    arr = np.array(values).reshape(1, -1)
    X = d["scaler"].transform(arr)
    pred = d["model"].predict(X)[0]
    prob = d["model"].predict_proba(X)[0]
    positive_idx = 1
    is_high_risk = (pred == positive_idx)
    if d.get("invert_positive"):
        is_high_risk = not is_high_risk
    result = "High Risk" if is_high_risk else "Low Risk"
    conf = f"{max(prob)*100:.1f}%"
    return result, conf

def nav_modules(current=None):
    """List of all 6 diagnostic modules for the quick model-switcher bar."""
    items = []
    for key, d in DISEASES.items():
        items.append({"key": key, "url": f"/predict/{key}", "name": d["name"], "icon": d["icon"], "current": key == current})
    items.append({"key": "xray", "url": "/predict/xray", "name": XRAY_MODEL["name"], "icon": XRAY_MODEL["icon"], "current": current == "xray"})
    items.append({"key": "nlp", "url": "/symptom-checker", "name": NLP_MODEL["name"], "icon": NLP_MODEL["icon"], "current": current == "nlp"})
    return items

@app.route("/predict/<disease_key>", methods=["GET"])
def predict_page(disease_key):
    if not session.get("logged_in"): return redirect(url_for("login"))
    if disease_key not in DISEASES: return redirect(url_for("home"))
    d = DISEASES[disease_key]
    features = [{"name": f, "label": label_for(f)} for f in d["feature_names"]]
    return render_template("predict.html",
        disease_key=disease_key, disease_name=d["name"], disease_icon=d["icon"],
        disease_color=d["color"], disease_accuracy=d["accuracy"], disease_auc=d["auc"],
        disease_description=d["description"], features=features,
        nav_modules=nav_modules(current=disease_key))

@app.route("/predict/<disease_key>", methods=["POST"])
def run_predict(disease_key):
    if disease_key not in DISEASES: return jsonify({"error": "Invalid disease model"}), 400
    try:
        d = DISEASES[disease_key]
        feature_values = {n: request.form.get(n, 0) for n in d["feature_names"]}
        result, conf = predict_row(disease_key, feature_values)
        badge = "danger" if result == "High Risk" else "success"

        saved_note = None
        if session.get("active_patient_id"):
            add_assessment(session["active_patient_id"], "ML", d["name"], result, conf, result)
            saved_note = session.get("active_patient_name")

        all_diseases_dict = {k: {"name": v["name"], "icon": v["icon"], "color": v["color"]}
                              for k, v in DISEASES.items()}
        all_diseases_dict["xray"] = {"name": XRAY_MODEL["name"], "icon": XRAY_MODEL["icon"], "color": XRAY_MODEL["color"]}
        return render_template("result.html",
            result=result, badge=badge, confidence=conf,
            disease_name=d["name"], disease_icon=d["icon"], disease_color=d["color"],
            disease_key=disease_key, saved_note=saved_note,
            all_diseases=all_diseases_dict)
    except Exception as e:
        return render_template("result.html", result="Error", badge="warning", confidence="N/A",
            disease_name="Error", disease_icon="❌", disease_color="#ccc",
            disease_key=disease_key, all_diseases={})

def _norm_col(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

@app.route("/predict/<disease_key>/csv", methods=["POST"])
def run_predict_csv(disease_key):
    if not session.get("logged_in"): return redirect(url_for("login"))
    if disease_key not in DISEASES: return redirect(url_for("home"))
    d = DISEASES[disease_key]
    file = request.files.get("csv_file")
    if not file or file.filename == "":
        return redirect(url_for("predict_page", disease_key=disease_key))

    import pandas as pd
    try:
        df = pd.read_csv(file)
    except Exception:
        return render_template("csv_result.html", disease_name=d["name"], disease_icon=d["icon"],
            disease_color=d["color"], disease_key=disease_key, results=[],
            matched_count=0, total_features=len(d["feature_names"]), unmatched=[],
            error="Couldn't read that file as a CSV. Please check the format and try again.")

    norm_cols = {_norm_col(c): c for c in df.columns}
    col_map = {feat: norm_cols[_norm_col(feat)] for feat in d["feature_names"] if _norm_col(feat) in norm_cols}

    id_col = next((norm_cols[c] for c in ["name", "patient", "patientname", "patientid", "id"] if c in norm_cols), None)

    results = []
    high_count = 0
    for idx, row in df.iterrows():
        feature_values = {}
        for feat in d["feature_names"]:
            if feat in col_map:
                try:
                    feature_values[feat] = float(row[col_map[feat]])
                except (ValueError, TypeError):
                    feature_values[feat] = 0.0
            else:
                feature_values[feat] = 0.0
        result, conf = predict_row(disease_key, feature_values)
        if result == "High Risk":
            high_count += 1
        patient_label = str(row[id_col]) if id_col is not None else f"Patient Row {idx + 1}"
        results.append({"row": idx + 1, "patient": patient_label, "result": result, "confidence": conf})

        if session.get("active_patient_id") and len(df) == 1:
            add_assessment(session["active_patient_id"], "ML", f"{d['name']} (CSV)", result, conf, result)

    unmatched = [f for f in d["feature_names"] if f not in col_map]
    return render_template("csv_result.html", disease_name=d["name"], disease_icon=d["icon"],
        disease_color=d["color"], disease_key=disease_key, results=results,
        matched_count=len(col_map), total_features=len(d["feature_names"]),
        unmatched=[label_for(f) for f in unmatched], high_count=high_count, error=None)

@app.route("/sample/<disease_key>/<risk_type>")
def sample(disease_key, risk_type):
    """Load a realistic sample (low_risk / high_risk) from the training data medians."""
    if disease_key not in DISEASES: return jsonify({"error": "invalid"}), 400
    import pandas as pd
    d = DISEASES[disease_key]
    data_files = {"diabetes": "diabetes.csv", "heart_disease": "heart.csv", "kidney_disease": "kidney.csv"}
    if disease_key == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer()
        idx = list(bc.target).index(1 if risk_type == "low_risk" else 0)
        return jsonify(dict(zip(d["feature_names"], [float(v) for v in bc.data[idx]])))
    df = pd.read_csv(os.path.join(BASE, "data", data_files[disease_key]))
    target_col = "Outcome" if disease_key == "diabetes" else ("target" if disease_key == "heart_disease" else "classification")
    want = 0 if risk_type == "low_risk" else 1
    subset = df[df[target_col] == want].copy()
    # Pick the example the model itself is MOST confident about, so the
    # sample buttons always demo a clean, unambiguous Low/High Risk case.
    Xs = d["scaler"].transform(subset[d["feature_names"]].values)
    proba_positive = d["model"].predict_proba(Xs)[:, 1]
    subset["_conf"] = proba_positive if want == 1 else (1 - proba_positive)
    row = subset.sort_values("_conf", ascending=False).iloc[0]
    return jsonify({f: float(row[f]) for f in d["feature_names"]})

@app.route("/predict/xray", methods=["GET"])
def xray_page():
    if not session.get("logged_in"): return redirect(url_for("login"))
    return render_template("predict_xray.html",
        accuracy=XRAY_MODEL["accuracy"], auc=XRAY_MODEL["auc"],
        is_synthetic=XRAY_MODEL.get("is_synthetic", True),
        nav_modules=nav_modules(current="xray"))

def _img_to_data_url(arr_uint8):
    im = Image.fromarray(arr_uint8).convert("L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@app.route("/sample_xray/<label>")
def sample_xray(label):
    """Generate a fresh synthetic Normal/Pneumonia sample X-ray for the demo buttons."""
    if not session.get("logged_in"): return jsonify({"error": "unauthorized"}), 401
    X, y = generate_dataset(n_per_class=1, seed=random.randint(0, 100000))
    want = 0 if label == "normal" else 1
    idx = int(np.where(y == want)[0][0])
    return jsonify({"image": _img_to_data_url(X[idx])})

def _template_corr(arr, flat_ref, norm_ref):
    v = arr.astype(np.float64).flatten() - arr.astype(np.float64).mean()
    vn = np.linalg.norm(v) + 1e-8
    return float(np.dot(v, flat_ref) / (vn * norm_ref))

def _run_xray_on_pil(pil_img):
    """Shared prediction logic for one image. Returns a result dict."""
    img_size = XRAY_MODEL["img_size"]
    clean_img = preprocess_upload(pil_img, size=img_size)
    arr = np.array(clean_img, dtype=np.uint8)

    feat = conv_feature_extract(arr).reshape(1, -1)
    Xs = XRAY_MODEL["scaler"].transform(feat)
    pred = XRAY_MODEL["model"].predict(Xs)[0]
    proba = XRAY_MODEL["model"].predict_proba(Xs)[0]

    label = "Pneumonia" if pred == 1 else "Normal"
    conf = f"{max(proba)*100:.1f}%"

    # Out-of-distribution check: does this image's pixel pattern actually
    # resemble EITHER of the chest-X-ray templates the model was trained on?
    # Neural/MLP classifiers are notoriously OVERCONFIDENT on out-of-
    # distribution inputs, so the raw softmax score alone can't be trusted.
    # When this is flagged, we do NOT show a confident High/Low Risk
    # verdict — an unreliable "High Risk — Pneumonia" claim on an unrelated
    # image is actively misleading. Instead the result is honestly reported
    # as inconclusive.
    out_of_distribution = False
    normal_t, pneu_t, thresh = (XRAY_MODEL.get("normal_template"),
                                 XRAY_MODEL.get("pneumonia_template"),
                                 XRAY_MODEL.get("ood_corr_threshold"))
    if normal_t is not None and pneu_t is not None and thresh is not None:
        nt_flat = normal_t.flatten() - normal_t.mean(); nt_norm = np.linalg.norm(nt_flat) + 1e-8
        pt_flat = pneu_t.flatten() - pneu_t.mean(); pt_norm = np.linalg.norm(pt_flat) + 1e-8
        best_corr = max(_template_corr(arr, nt_flat, nt_norm), _template_corr(arr, pt_flat, pt_norm))
        out_of_distribution = best_corr < thresh

    if out_of_distribution:
        return {"arr": arr, "label": "Inconclusive", "confidence": conf,
                "result": "Inconclusive", "badge": "warning",
                "raw_label": label, "out_of_distribution": True}

    return {"arr": arr, "label": label, "confidence": conf,
            "result": "High Risk" if label == "Pneumonia" else "Low Risk",
            "badge": "danger" if label == "Pneumonia" else "success",
            "raw_label": label, "out_of_distribution": False}

@app.route("/predict/xray", methods=["POST"])
def run_xray_predict():
    if not session.get("logged_in"): return redirect(url_for("login"))
    try:
        data_url = request.form.get("image_data")
        if data_url:
            header, encoded = data_url.split(",", 1)
            pil_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        else:
            file = request.files.get("xray_image")
            if not file or file.filename == "":
                return redirect(url_for("xray_page"))
            pil_img = Image.open(file.stream)

        r = _run_xray_on_pil(pil_img)
        arr = r["arr"]

        def to_data_url(pil_img_out):
            buf = io.BytesIO(); pil_img_out.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        original_img = Image.fromarray(arr).convert("RGB").resize((256, 256), Image.NEAREST)
        heatmap_url = None
        show_heatmap = False
        if not r["out_of_distribution"] and r["label"] == "Pneumonia":
            # ── CV Module: Occlusion-sensitivity explainability heatmap ──
            # (only computed when the verdict is actually trustworthy)
            heat, _ = occlusion_heatmap(arr, XRAY_MODEL["model"], XRAY_MODEL["scaler"],
                                         conv_feature_extract, target_class=1, patch=8, stride=8)
            overlay_img = overlay_heatmap(arr, heat, alpha=0.5).resize((256, 256), Image.NEAREST)
            heatmap_url = to_data_url(overlay_img)
            show_heatmap = True

        saved_note = None
        if session.get("active_patient_id") and not r["out_of_distribution"]:
            add_assessment(session["active_patient_id"], "DL", "Chest X-ray Pneumonia Detector",
                            r["label"], r["confidence"], r["label"])
            saved_note = session.get("active_patient_name")

        return render_template("xray_result.html",
            result=r["result"], badge=r["badge"], confidence=r["confidence"], label=r["label"],
            out_of_distribution=r["out_of_distribution"],
            original_data_url=to_data_url(original_img),
            heatmap_data_url=heatmap_url,
            show_heatmap=show_heatmap, saved_note=saved_note,
            all_diseases={k: {"name": v["name"], "icon": v["icon"], "color": v["color"]}
                          for k, v in DISEASES.items()})
    except Exception:
        return render_template("result.html", result="Error", badge="warning", confidence="N/A",
            disease_name="Error", disease_icon="❌", disease_color="#ccc",
            disease_key="xray", all_diseases={})

@app.route("/predict/xray/batch", methods=["POST"])
def run_xray_batch_predict():
    """Batch-predict multiple uploaded X-ray images at once (the image
    equivalent of the CSV batch-upload feature on the ML pages)."""
    if not session.get("logged_in"): return redirect(url_for("login"))
    files = request.files.getlist("xray_images")
    files = [f for f in files if f and f.filename]
    if not files:
        return redirect(url_for("xray_page"))

    results, high_count = [], 0
    for i, file in enumerate(files, start=1):
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext in ("csv", "txt", "xlsx", "xls", "pdf", "docx"):
            results.append({"row": i, "patient": file.filename, "result": "Not an image",
                             "confidence": f"'.{ext}' files aren't supported here — upload JPG/PNG X-ray images. "
                                           f"(For CSV patient data, use the ML disease pages instead.)", "flag": False})
            continue
        try:
            pil_img = Image.open(file.stream)
            r = _run_xray_on_pil(pil_img)
            if r["result"] == "High Risk":
                high_count += 1
            results.append({"row": i, "patient": file.filename, "result": r["result"],
                             "confidence": r["confidence"], "flag": r["out_of_distribution"]})
            if session.get("active_patient_id") and len(files) == 1 and not r["out_of_distribution"]:
                add_assessment(session["active_patient_id"], "DL", "Chest X-ray Pneumonia Detector",
                                r["label"], r["confidence"], r["label"])
        except Exception:
            results.append({"row": i, "patient": file.filename, "result": "Not an image",
                             "confidence": "Couldn't open this as an image file.", "flag": False})

    return render_template("xray_batch_result.html", results=results, high_count=high_count)

@app.route("/symptom-checker")
def symptom_checker_page():
    if not session.get("logged_in"): return redirect(url_for("login"))
    session["symptom_chat_text"] = ""
    session["symptom_chat_msgs"] = []
    return render_template("symptom_checker.html",
        accuracy=NLP_MODEL["accuracy"], top3_accuracy=NLP_MODEL["top3_accuracy"],
        nav_modules=nav_modules(current="nlp"))

GREETINGS = {"hi", "hello", "hey", "hii", "hiii", "yo", "sup", "thanks", "thank you",
             "ok", "okay", "yes", "no", "bye", "good morning", "good evening"}

@app.route("/symptom-checker/ask", methods=["POST"])
def symptom_checker_ask():
    if not session.get("logged_in"): return jsonify({"error": "unauthorized"}), 401
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Please tell me what symptoms you're experiencing.", "predictions": []})

    # Greetings/filler ("hi", "ok", ...) shouldn't pollute the symptom context —
    # respond warmly and don't add them to the accumulated history.
    if user_msg.lower().strip(" !.,") in GREETINGS:
        return jsonify({"reply": "Hi! Go ahead and describe any symptoms you're feeling — "
                                  "for example \"I have a sore throat and mild fever\".",
                         "predictions": [], "urgent": False})

    # Rolling window of the last few messages, NOT the entire chat history —
    # unbounded accumulation dilutes the TF-IDF signal over a long session
    # (many unrelated test messages mixed together permanently drags every
    # future prediction's confidence down). Keeping only the most recent
    # exchanges keeps context relevant without that drift.
    msgs = session.get("symptom_chat_msgs", [])
    msgs.append(user_msg)
    msgs = msgs[-4:]
    session["symptom_chat_msgs"] = msgs
    accumulated = " ".join(msgs)
    session["symptom_chat_text"] = accumulated  # kept for backward compatibility

    Xv = NLP_MODEL["vectorizer"].transform([accumulated])
    proba = NLP_MODEL["model"].predict_proba(Xv)[0]
    classes = NLP_MODEL["model"].classes_
    order = np.argsort(proba)[::-1][:3]
    top_predictions = [{"disease": classes[i], "confidence": round(float(proba[i]) * 100, 1)} for i in order]

    top_disease = top_predictions[0]["disease"]
    top_conf = top_predictions[0]["confidence"]
    text_lower = accumulated.lower()

    lines = []
    if top_conf < 35:
        lines.append("I need a bit more information to narrow this down. Could you describe any other symptoms you're feeling?")
    else:
        lines.append(f"Based on what you've described, here's what it could be:")
        for p in top_predictions:
            lines.append(f"  • {p['disease']} — {p['confidence']}% match")
        urgency = urgency_of(top_disease)
        if urgency in ("urgent", "high"):
            lines.append(f"⚠️ {top_disease} can be serious — please consult a doctor promptly.")

        # follow-up: symptoms of top disease not yet mentioned
        known_symptoms = NLP_MODEL["disease_symptoms"][top_disease]
        missing = [s for s in known_symptoms if s.split()[-1] not in text_lower and s not in text_lower]
        if missing:
            ask = ", ".join(missing[:2])
            lines.append(f"Do you also have {ask}? That would help me confirm.")

    reply = "\n".join(lines)

    saved_note = None
    if top_conf >= 35 and session.get("active_patient_id"):
        risk = "High Risk" if urgency_of(top_disease) in ("urgent", "high") else "Low Risk"
        add_assessment(session["active_patient_id"], "NLP", top_disease,
                        f"{top_conf}% match", f"{top_conf}%", risk)
        saved_note = session.get("active_patient_name")

    return jsonify({"reply": reply, "predictions": top_predictions,
                     "urgent": urgency_of(top_disease) in ("urgent", "high"),
                     "saved_note": saved_note})

@app.route("/symptom-checker/reset", methods=["POST"])
def symptom_checker_reset():
    session["symptom_chat_text"] = ""
    session["symptom_chat_msgs"] = []
    return jsonify({"ok": True})

@app.route("/patients", methods=["GET", "POST"])
def patients_page():
    if not session.get("logged_in"): return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "")
        if name:
            pid = create_patient(name, age or None, gender, session.get("doctor_name", "Unknown"))
            session["active_patient_id"] = pid
            session["active_patient_name"] = name
            return redirect(url_for("home"))
    return render_template("patients.html", patients=list_patients(),
        active_patient_id=session.get("active_patient_id"))

@app.route("/patients/select/<int:patient_id>")
def select_patient(patient_id):
    if not session.get("logged_in"): return redirect(url_for("login"))
    p = get_patient(patient_id)
    if p:
        session["active_patient_id"] = p["id"]
        session["active_patient_name"] = p["name"]
    return redirect(url_for("home"))

@app.route("/patients/clear")
def clear_patient():
    session.pop("active_patient_id", None)
    session.pop("active_patient_name", None)
    return redirect(url_for("home"))

@app.route("/patients/<int:patient_id>/report")
def patient_report(patient_id):
    if not session.get("logged_in"): return redirect(url_for("login"))
    patient = get_patient(patient_id)
    if not patient: return redirect(url_for("patients_page"))
    assessments = get_assessments(patient_id)

    high_risk_items = [a for a in assessments if a["risk_level"] in ("High Risk", "Pneumonia")]
    module_labels = {"ML": "🧬 Disease Risk Models", "DL": "🫁 Chest X-ray (Deep Learning)",
                      "NLP": "🗣️ Symptom Checker (NLP)"}
    grouped = {}
    for a in assessments:
        grouped.setdefault(a["module"], []).append(a)

    overall = "🟢 No significant risk factors detected." if not high_risk_items else \
        f"🔴 {len(high_risk_items)} high-risk finding(s) detected across AI modules — needs clinical review."

    return render_template("report.html", patient=patient, grouped=grouped,
        module_labels=module_labels, overall=overall, high_risk_items=high_risk_items,
        total_assessments=len(assessments))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
