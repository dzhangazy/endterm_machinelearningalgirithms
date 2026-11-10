"""Create the endterm presentation shell without inventing unavailable results."""
import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Astana_Housing_Endterm_Presentation.pptx"
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

NAVY, BLUE, TEAL, WHITE, MUTED = RGBColor(17, 40, 70), RGBColor(44, 105, 176), RGBColor(0, 140, 140), RGBColor(255, 255, 255), RGBColor(83, 96, 112)


def add_text(slide, text, x, y, w, h, size=22, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text; run.font.name = "Aptos"; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return box


def slide(title, bullets, speaker, note=None, chart=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    banner = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(.65)); banner.fill.solid(); banner.fill.fore_color.rgb = NAVY; banner.line.fill.background()
    add_text(s, title, .55, .12, 11.8, .35, 26, WHITE, True)
    y = 1.05
    text_width = 5.8 if chart else 11.6
    for item in bullets:
        add_text(s, "• " + item, .8, y, text_width, .55, 20, NAVY)
        y += .72
    if chart and chart.exists():
        s.shapes.add_picture(str(chart), Inches(6.7), Inches(1.1), width=Inches(5.7), height=Inches(4.6))
    if note:
        shape = s.shapes.add_shape(1, Inches(.8), Inches(5.8), Inches(11.7), Inches(.75))
        shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor(235, 244, 250); shape.line.color.rgb = BLUE
        add_text(s, note, 1.0, 5.98, 11.3, .35, 14, MUTED)
    add_text(s, f"Speaker: {speaker}", .8, 6.88, 3.8, .25, 12, TEAL, True)
    add_text(s, f"SE-2431  |  Slide {len(prs.slides)}", 10.2, 6.88, 2.3, .25, 11, MUTED, align=PP_ALIGN.RIGHT)


def metric_line():
    path = ROOT / "reports" / "final_test_metrics.csv"
    if not path.exists():
        return "Final metrics will be inserted after a clean notebook run."
    with path.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return (f"Final {row['Model']}: RMSE(log) {float(row['RMSE (log)']):.3f}; "
            f"R2 {float(row['R2']):.3f}; MedAPE {float(row['MedAPE (%)']):.2f}%.")


slide("Astana Apartment Price Prediction", ["Endterm Project — Machine Learning Algorithms", "Nurkhan Kuanyshov and Bakytzhan Jangazy", "Group SE-2431"], "Both")
slide("Problem and research objective", ["Estimate apartment asking prices from listing characteristics.", "Question: Can area, location, rooms, floor, building characteristics and amenities predict price?", "Regression target: price; models learn log1p(price).", "Use case: decision support, not an official valuation."], "Nurkhan")
slide("Dataset and responsible scope", ["Astana Real Estate Dataset by turarr (Kaggle); advertisements from Krisha.kz.", "Asking prices, not verified final transaction prices.", "Potential duplicates, missing fields and seller-entry errors are documented limitations.", "Dataset size and feature inventory are reported dynamically by the notebook."], "Nurkhan", "Replace this note with the executed notebook's dataset-size table.")
slide("Preparation, EDA and feature engineering", ["Numeric conversion, floor extraction, duplicate removal, range checks and leakage-field removal.", "EDA: raw/log price, area-price, district distributions, correlations and missingness.", "Engineered: house age, first/top floor, floor ratio, area per room.", "Distance to centre is added only when actual coordinates exist."], "Nurkhan", chart=ROOT / "visualizations" / "eda" / "area_vs_price.png")
slide("Evaluation design and leakage control", ["Fixed 70/15/15 train, validation and test split with log-price quantile bins.", "Imputation, scaling and one-hot encoding are fitted inside pipelines only on training data.", "Five-fold CV and tuning use training data; validation selects; test is final once.", "Metrics: log RMSE, log MAE, R², KZT MAE and MedAPE."], "Bakytzhan")
slide("Regression models and tuning", ["Midterm retained: median baseline, Linear Regression, Decision Tree and KNN.", "Endterm adds Random Forest and Gradient Boosting.", "Bounded randomized five-fold hyperparameter searches keep CPU execution practical.", "Feature importance is interpreted as association—not causation."], "Bakytzhan")
slide("K-Means market segmentation", ["Exploratory clustering uses meaningful scaled numeric listing characteristics.", "Price is allowed here for segmentation, but never used as a supervised feature.", "Elbow inertia and silhouette scores guide k selection.", "Cluster summaries compare listing count, price, area, rooms and age."], "Nurkhan", chart=ROOT / "visualizations" / "clustering" / "cluster_selection.png")
slide("PCA: dimensionality reduction", ["PCA is fitted to standardized numerical training features.", "Scree and cumulative-variance plots show information retention.", "The first two components visualize listings and K-Means labels.", "PCA improves exploration, but components are less interpretable than original features."], "Nurkhan", chart=ROOT / "visualizations" / "pca" / "pca_variance.png")
slide("ANN for tabular regression", ["Appropriate deep learning choice: feedforward ANN, not unrelated CNN/NLP.", "Architecture: 128 → 64 → 32 ReLU; dropout 0.2; linear output.", "Adam optimizer, MSE loss, EarlyStopping and ReduceLROnPlateau.", "Validation loss history checks convergence and potential overfitting."], "Bakytzhan", chart=ROOT / "visualizations" / "ann" / "training_history.png")
slide("Unified model comparison", ["All models use the same split and metric definitions.", "Validation log-RMSE selects the model family.", "The selected approach is refit on train + validation, then evaluated once on test.", "Gradient Boosting was best on validation; the ANN did not improve traditional results."], "Bakytzhan", chart=ROOT / "visualizations" / "evaluation" / "model_comparison.png")
slide("Final model and error analysis", [metric_line(), "Inspect actual versus predicted prices and residual distribution.", "Report errors by price segment, district and largest listing errors.", "Treat renovation/view/building-reputation explanations as hypotheses when unrecorded."], "Bakytzhan", chart=ROOT / "visualizations" / "evaluation" / "error_analysis.png")
slide("Limitations and ethics", ["Listings can differ from transaction prices and represent a particular market period.", "Missing and inaccurate seller details limit predictions, especially premium homes.", "Accuracy may be uneven across districts and price segments.", "Predictions should support—not replace—professional appraisal and human judgement."], "Nurkhan")
slide("Conclusions and practical value", ["State whether the original objective was met using the final test evidence.", "Name the selected model and whether ANN justified its added complexity.", "Summarize the market patterns from clusters and PCA.", "Future work: verified sales, richer property quality data, temporal validation."], "Both", "Complete only after results exist; every conclusion must cite the executed notebook output.")
slide("Contributions and Q&A", ["Nurkhan: preparation, EDA, features, clustering, PCA and documentation.", "Bakytzhan: supervised modelling, tuning, ANN, evaluation and error analysis.", "Defense preparation: reports/defense_questions.md", "Questions"], "Both")

prs.save(OUT)
print(f"Wrote {OUT.name} with {len(prs.slides)} slides")
