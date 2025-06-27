from fpdf import FPDF
import os

# Create a folder for PDF output if not exist
os.makedirs("report", exist_ok=True)

# Report Class
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Breast Cancer Classification Report', border=False, ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Create PDF
pdf = PDFReport()
pdf.add_page()

summary = (
    "This report summarizes a logistic regression model trained to predict breast cancer diagnosis "
    "(malignant or benign) using the Kaggle Breast Cancer Wisconsin dataset.\n\n"
    "Model Evaluation:\n"
    "- Accuracy: ~96%\n"
    "- Precision: 0.975\n"
    "- Recall: 0.9286 (default) / 0.976 (threshold=0.3)\n"
    "- ROC-AUC: 0.996\n\n"
    "The following plots illustrate the model's performance and insights."
)


pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, summary)
pdf.ln(5)

# List of images to include
images = [
    ("ROC Curve", "images/roc_curve.png"),
    ("Sigmoid Function", "images/sigmoid_curve.png"),
    ("Class Distribution", "images/class_distribution.png"),
    ("Correlation Heatmap", "images/correlation_heatmap.png"),
    ("Top Feature Importances", "images/feature_importance.png"),
    ("Confusion Matrix", "images/confusion_matrix.png"),
]

# Add each image with a title
for title, img_path in images:
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, title, ln=True)
    pdf.image(img_path, w=170)
    pdf.ln(10)

# Save the PDF
pdf.output("report/final_report.pdf")
print("✅ PDF generated at: report/final_report.pdf")
