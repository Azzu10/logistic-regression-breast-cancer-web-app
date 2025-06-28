from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import io
import base64
import warnings
import json
from io import StringIO

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = "super_secret_key_azmatulla_2025"

@app.route("/", methods=["GET", "POST"])
def index():
    session.clear()
    report = None
    image_roc = image_sigmoid = image_class_dist = image_corr = image_feat = image_cm = None
    table_html = error = result_text = None
    roc_auc_val = None

    input_id = request.form.get("patient_id")
    file = request.files.get("file")

    # Load from upload or session
    if file:
        try:
            df = pd.read_csv(file)
            session['df'] = df.to_json(orient='split')
        except Exception as e:
            error = f"❌ Failed to read uploaded CSV: {str(e)}"
            return render_template("index.html", error=error)

    if 'df' not in session:
        return render_template("index.html", error=" Please upload a CSV file to start.")

    try:
        df_original = pd.read_json(StringIO(session['df']), orient="split")
        session.permanent = True


        df = df_original.copy()

        # Drop optional columns
        for col in ['Unnamed: 32', 'id']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        if 'diagnosis' not in df.columns:
            error = "❌ 'diagnosis' column missing in dataset."
            return render_template("index.html", error=error)

        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=5000)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose().round(2)
        report = report_df.to_html(classes="table table-striped")

        # Graphs
        def plot_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        # ROC
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc_val = round(roc_auc_score(y, y_prob), 3)
        fig1 = plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        image_roc = plot_to_base64(fig1)
        plt.close(fig1)

        # Sigmoid
        fig2 = plt.figure(figsize=(6, 3))
        z = np.linspace(-10, 10, 100)
        plt.plot(z, 1 / (1 + np.exp(-z)))
        plt.title("Sigmoid Function")
        plt.grid()
        image_sigmoid = plot_to_base64(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(4, 3))
        sns.countplot(x=pd.Series(y).map({0: 'Benign', 1: 'Malignant'}))
        plt.title("Class Distribution")
        plt.xlabel("Diagnosis")
        plt.ylabel("Count")
        image_class_dist = plot_to_base64(fig3)
        plt.close(fig3)


        # Correlation Heatmap
        fig4 = plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap='coolwarm', cbar=False)
        plt.title("Correlation Heatmap")
        image_corr = plot_to_base64(fig4)
        plt.close(fig4)

        # Feature Importance
        fig5 = plt.figure(figsize=(6, 4))
        coef = pd.Series(model.coef_[0], index=X.columns).abs().sort_values(ascending=False).head(10)
        coef.sort_values().plot(kind='barh', color='teal')
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Coefficient Magnitude")
        image_feat = plot_to_base64(fig5)
        plt.close(fig5)

        # Confusion Matrix
        fig6 = plt.figure(figsize=(4, 3))
        conf_matrix = confusion_matrix(y, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title("Confusion Matrix")
        image_cm = plot_to_base64(fig6)
        plt.close(fig6)

        # Table Preview
        table_html = df_original.head().to_html(classes="table table-bordered")

        # Patient Prediction by ID
        if input_id and input_id.strip().isdigit() and 'id' in df_original.columns:
            selected_id = float(input_id.strip())
            patient_row = df_original[df_original['id'] == selected_id].copy()

            if not patient_row.empty:
                drop_cols = [col for col in ['id', 'diagnosis', 'Unnamed: 32'] if col in patient_row.columns]
                patient_row.drop(columns=drop_cols, inplace=True)
                patient_scaled = scaler.transform(patient_row)
                pred = model.predict(patient_scaled)[0]
                result_text = f"🧬 Prediction for ID {int(selected_id)}: " + (
                    "🔴 Malignant (Cancerous)" if pred == 1 else "🟢 Benign (Non-Cancerous)")
            else:
                result_text = f"⚠️ ID {input_id} not found in dataset."
        elif input_id:
            result_text = "⚠️ Invalid ID format. Must be numeric."

    except Exception as e:
        error = f"❌ Error processing file: {str(e)}"

    return render_template("index.html",
                           report=report,
                           image_roc=image_roc,
                           image_sigmoid=image_sigmoid,
                           image_class_dist=image_class_dist,
                           image_corr=image_corr,
                           image_feat=image_feat,
                           image_cm=image_cm,
                           table_html=table_html,
                           roc_auc_val=roc_auc_val,
                           error=error,
                           result_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
