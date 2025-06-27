from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import io
import base64
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    image_roc = None
    image_sigmoid = None
    image_class_dist = None
    image_corr = None
    image_feat = None
    image_cm = None
    table_html = None
    error = None
    roc_auc_val = None
    result_text = None

    if request.method == "POST":
        file = request.files["file"]
        input_id = request.form.get("patient_id")

        if file:
            try:
                df = pd.read_csv(file)

                # Optional cleanup
                for col in ['Unnamed: 32']:
                    if col in df.columns:
                        df.drop(col, axis=1, inplace=True)

                if 'diagnosis' not in df.columns:
                    error = "❌ 'diagnosis' column missing in uploaded file."
                    return render_template("index.html", error=error)

                # Encode diagnosis
                df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

                # Prepare features & labels
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

                # 1. ROC Curve
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc_val = round(roc_auc_score(y, y_prob), 3)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val}")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                buf1 = io.BytesIO()
                plt.savefig(buf1, format='png')
                buf1.seek(0)
                image_roc = base64.b64encode(buf1.getvalue()).decode('utf-8')
                plt.close()

                # 2. Sigmoid Curve
                def sigmoid(z):
                    return 1 / (1 + np.exp(-z))
                z = np.linspace(-10, 10, 100)
                sig = sigmoid(z)
                plt.figure(figsize=(6, 3))
                plt.plot(z, sig)
                plt.title("Sigmoid Function")
                plt.xlabel("z")
                plt.ylabel("Sigmoid(z)")
                plt.grid()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png')
                buf2.seek(0)
                image_sigmoid = base64.b64encode(buf2.getvalue()).decode('utf-8')
                plt.close()

                # 3. Class Distribution
                plt.figure(figsize=(4, 3))
                sns.countplot(x=y)
                plt.xticks([0, 1], ['Benign', 'Malignant'])
                plt.title("Class Distribution")
                buf3 = io.BytesIO()
                plt.savefig(buf3, format='png')
                buf3.seek(0)
                image_class_dist = base64.b64encode(buf3.getvalue()).decode('utf-8')
                plt.close()

                # 4. Correlation Heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(df.corr(), cmap='coolwarm', cbar=False)
                plt.title("Correlation Heatmap")
                buf4 = io.BytesIO()
                plt.savefig(buf4, format='png')
                buf4.seek(0)
                image_corr = base64.b64encode(buf4.getvalue()).decode('utf-8')
                plt.close()

                # 5. Feature Importance
                coef = pd.Series(model.coef_[0], index=X.columns).abs().sort_values(ascending=False).head(10)
                plt.figure(figsize=(6, 4))
                coef.sort_values().plot(kind='barh', color='teal')
                plt.title("Top 10 Feature Importances")
                plt.xlabel("Coefficient Magnitude")
                buf5 = io.BytesIO()
                plt.savefig(buf5, format='png')
                buf5.seek(0)
                image_feat = base64.b64encode(buf5.getvalue()).decode('utf-8')
                plt.close()

                # 6. Confusion Matrix
                conf_matrix = confusion_matrix(y, y_pred)
                plt.figure(figsize=(4, 3))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Benign', 'Malignant'],
                            yticklabels=['Benign', 'Malignant'])
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                buf6 = io.BytesIO()
                plt.savefig(buf6, format='png')
                buf6.seek(0)
                image_cm = base64.b64encode(buf6.getvalue()).decode('utf-8')
                plt.close()

                # Table preview
                table_html = df.head().to_html(classes="table table-bordered")

                # 7. Patient-Level Prediction
                if input_id and input_id.strip().isdigit() and 'id' in df.columns:
                    selected_id_float = float(input_id.strip())
                    selected_row = df[df['id'] == selected_id_float]
                    
                    if not selected_row.empty:
                        selected_features = selected_row.drop(['id', 'diagnosis'], axis=1)
                        selected_scaled = scaler.transform(selected_features)
                        pred = model.predict(selected_scaled)[0]
                        result_text = f"🧬 Prediction for ID {int(selected_id_float)}: " + (
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
