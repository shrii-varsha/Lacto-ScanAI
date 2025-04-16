import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Milk Adulteration Detector", layout="centered")
st.title("🥛LactoScan-AI")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("finaldb.xlsx")
    df = df[df["Fat(%)"] != "Fat(%)"]
    base_features = ["Fat(%)", "Density(g/m3)", "Lactose(%)", "SNF(%)", "Protein(%)", "Water(%)"]
    df[base_features] = df[base_features].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=base_features, inplace=True)

    # Feature engineering
    df["SNF_Fat_Ratio"] = df["SNF(%)"] / df["Fat(%)"]
    df["Water_Protein_Ratio"] = df["Water(%)"] / df["Protein(%)"]

    def detect_adulterant(row):
        adulterants = {
            "Starch": row["Starch(g)"],
            "Sucrose": row["Sucrose(g)"],
            "Glucose": row["Glucose(g)"],
            "Sodium chloride": row["Sodium chloride(g)"],
            "Sodium Nitrate": row["Sodium Nitrate(g)"],
            "Formaldehyde": row["Formaldehyde(g)"],
            "Urea": row["Urea(g)"]
        }
        present = [k for k, v in adulterants.items() if v > 0]
        return "+".join(sorted(present)) if present else "None"

    df["Adulterant_Label"] = df.apply(detect_adulterant, axis=1)
    return df

df = load_data()

# Prepare data
base_features = ["Fat(%)", "Density(g/m3)", "Lactose(%)", "SNF(%)", "Protein(%)", "Water(%)"]
df["Adulterant_Label_Encoded"] = LabelEncoder().fit_transform(df["Adulterant_Label"])
df["SNF_Fat_Ratio"] = df["SNF(%)"] / df["Fat(%)"]
df["Water_Protein_Ratio"] = df["Water(%)"] / df["Protein(%)"]
feature_cols = base_features + ["SNF_Fat_Ratio", "Water_Protein_Ratio"]
X = df[feature_cols]
y = df["Adulterant_Label_Encoded"]
le = LabelEncoder()
y_encoded = le.fit_transform(df["Adulterant_Label"])

# Balance and train
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model with tuning
@st.cache_resource
def train_model():
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

model = train_model()

# Input UI
st.header("🔍 Enter Milk Sample Details")
user_input = {}
for feature in base_features:
    user_input[feature] = st.number_input(feature, min_value=0.0, format="%.3f")

if st.button("Predict Adulterant"):
    try:
        # Feature engineering
        user_input["SNF_Fat_Ratio"] = user_input["SNF(%)"] / user_input["Fat(%)"]
        user_input["Water_Protein_Ratio"] = user_input["Water(%)"] / user_input["Protein(%)"]
        input_df = pd.DataFrame([user_input])

        # Prediction
        pred_encoded = model.predict(input_df)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        st.success(f"🚨 Predicted Adulterant: {pred_label}")

        # Nearest match for quantities
        distances = euclidean_distances(X, input_df[feature_cols])
        closest_index = distances.argmin()
        closest_sample = df.iloc[closest_index]

        st.subheader("📊 Approximate Quantity of Adulterants:")
        adulterant_cols = [
            "Starch(g)", "Sucrose(g)", "Glucose(g)", "Sodium chloride(g)",
            "Sodium Nitrate(g)", "Formaldehyde(g)", "Urea(g)"
        ]
        for a in adulterant_cols:
            val = closest_sample[a]
            if val > 0:
                st.write(f"**{a.replace('(g)', '')}**: {val:.2f} g")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
