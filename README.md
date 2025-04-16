# 🥛 LactoScan-AI: Milk Adulteration Detector
**LactoScan-AI** is a Streamlit-based machine learning application that detects milk adulterants using a Random Forest Classifier and SMOTE for class balancing. Users can input values like fat, protein, lactose, water content, and density to predict the type of adulteration present. The app also estimates the approximate quantity of adulterants based on the nearest matching record from real-world data.

---

##  Features

- 🔍 **Adulterant Detection** – Predicts adulterants like starch, sucrose, glucose, sodium nitrate, formaldehyde, urea, etc.
- ⚖️ **Balanced Learning** – Uses SMOTE for handling class imbalance in the dataset.
- 🌲 **Random Forest Classifier** – Trained using GridSearchCV for best hyperparameters.
- 📊 **Nearest Sample Match** – Shows estimated quantity of adulterants from the closest match using Euclidean distance.
- 💻 **User-Friendly Interface** – Clean UI using Streamlit for interactive predictions.

---

##  Input Parameters

Users need to enter the following milk sample characteristics:

- `Fat (%)`
- `Density (g/m³)`
- `Lactose (%)`
- `SNF (%)`
- `Protein (%)`
- `Water (%)`

---

## Tech Stack

- Python
- Streamlit
- Pandas
- scikit-learn
- imbalanced-learn (SMOTE)
- openpyxl (for Excel file handling)

---

## Dataset

The dataset `finaldb.xlsx` should contain the following:

- Base features: `Fat(%)`, `Density(g/m3)`, `Lactose(%)`, `SNF(%)`, `Protein(%)`, `Water(%)`
- Adulterants: `Starch(g)`, `Sucrose(g)`, `Glucose(g)`, `Sodium chloride(g)`, `Sodium Nitrate(g)`, `Formaldehyde(g)`, `Urea(g)`

The app performs feature engineering to add:
- `SNF_Fat_Ratio`
- `Water_Protein_Ratio`

---

