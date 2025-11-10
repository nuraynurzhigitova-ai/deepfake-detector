import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Автонесие болжау", layout="centered")

# -----------------------------
# 1. Деректер жиыны (мысал)
# -----------------------------
data = {
    'age': [25, 45, 35, 29, 52, 41, 38, 26, 55, 48, 30, 42, 36, 28, 50, 60, 23, 34, 39, 47],
    'income': [400, 1200, 800, 500, 1500, 1000, 900, 450, 1600, 1100, 600, 950, 820, 480, 1400, 1700, 350, 760, 880, 1020],
    'job_years': [2, 10, 5, 3, 15, 7, 6, 2, 20, 9, 4, 8, 5, 2, 12, 25, 1, 5, 6, 7],
    'loan_amount': [300, 800, 600, 400, 1000, 700, 650, 350, 1200, 900, 500, 750, 640, 380, 1100, 1300, 280, 620, 680, 790],
    'credit_score': [60, 90, 75, 65, 95, 80, 78, 55, 98, 85, 70, 82, 76, 58, 92, 99, 50, 74, 79, 83],
    'approved': [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['age', 'income', 'job_years', 'loan_amount', 'credit_score']]
y = df['approved']

# -----------------------------
# 2. Бөлу (train/test)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -----------------------------
# 3. Интерфейс
# -----------------------------
st.title("🚗 Банк автонесиесін болжау")
st.write("Клиенттің деректеріне сүйеніп, банк несие береді ме – соны болжайды.")

model_type = st.selectbox("Қай модельді қолданамыз?", ["Decision Tree", "Random Forest", "Logistic Regression"])

if model_type == "Decision Tree":
    model = DecisionTreeClassifier(random_state=0)
elif model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=0)
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.markdown(f"*Модель дәлдігі:* {acc*100:.2f}%")

# -----------------------------
# 4. Пайдаланушы енгізулері
# -----------------------------
st.markdown("---")
st.subheader("Клиент деректерін енгізіңіз:")

age = st.slider("Жасы", 18, 70, 30)
income = st.number_input("Айлық табыс (мың теңге)", 0, 10000, 800)
job_years = st.slider("Жұмыс өтілі (жыл)", 0, 30, 5)
loan_amount = st.number_input("Несие сомасы (мың теңге)", 100, 5000, 700)
credit_score = st.slider("Кредиттік рейтинг (0–100)", 0, 100, 70)

if st.button("🔍 Болжау"):
    prediction = model.predict([[age, income, job_years, loan_amount, credit_score]])[0]
    prob = model.predict_proba([[age, income, job_years, loan_amount, credit_score]])[0][1] if hasattr(model, 'predict_proba') else None

    if prediction == 1:
        st.success(f"✅ Несие мақұлдануы мүмкін! (сенімділік ≈ {prob*100:.1f}% )" if prob is not None else "✅ Несие мақұлданды!")
    else:
        st.error(f"❌ Несие мақұлданбауы мүмкін (сенімділік ≈ {(1-prob)*100:.1f}% )" if prob is not None else "❌ Несие мақұлданбады!")

# -----------------------------
# 5. Confusion Matrix
# -----------------------------
st.markdown("---")
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center', color='black')
ax.set_xlabel("Болжам")
ax.set_ylabel("Нақты мән")
st.pyplot(fig)
