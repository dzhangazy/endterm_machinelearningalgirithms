"""Build a 14-slide endterm deck from executed project evidence."""
import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Astana_Housing_Endterm_Presentation.pptx"
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
NAVY, BLUE, TEAL, WHITE, MUTED = RGBColor(17, 40, 70), RGBColor(44, 105, 176), RGBColor(0, 140, 140), RGBColor(255, 255, 255), RGBColor(83, 96, 112)


def text(slide, value, x, y, w, h, size=20, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame; frame.clear(); frame.word_wrap = True
    para = frame.paragraphs[0]; para.alignment = align
    run = para.add_run(); run.text = value; run.font.name = "Aptos"; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color


def final_metrics():
    with (ROOT / "reports" / "final_test_metrics.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return f"Final {row['Model']}: RMSE(log) {float(row['RMSE (log)']):.3f}; R2 {float(row['R2']):.3f}; MedAPE {float(row['MedAPE (%)']):.2f}%."


def add_slide(title, bullets, speaker, chart=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill; fill.solid(); fill.fore_color.rgb = WHITE
    banner = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(.65)); banner.fill.solid(); banner.fill.fore_color.rgb = NAVY; banner.line.fill.background()
    text(slide, title, .55, .12, 12, .35, 26, WHITE, True)
    width = 5.75 if chart else 11.6
    y = 1.03
    for bullet in bullets:
        text(slide, "- " + bullet, .8, y, width, .62, 18, NAVY)
        y += .78
    if chart and chart.exists():
        slide.shapes.add_picture(str(chart), Inches(6.7), Inches(1.1), width=Inches(5.7), height=Inches(4.6))
    text(slide, f"Speaker: {speaker}", .8, 6.88, 4, .25, 12, TEAL, True)
    text(slide, f"SE-2431 | Slide {len(prs.slides)}", 10, 6.88, 2.5, .25, 11, MUTED, align=PP_ALIGN.RIGHT)


fig = ROOT / "visualizations"
add_slide("Astana Apartment Price Prediction", ["Endterm Project - Machine Learning Algorithms", "Nurkhan Kuanyshov and Bakytzhan Jangazy", "Group SE-2431"], "Both")
add_slide("Problem and research objective", ["Estimate apartment asking prices from listing characteristics.", "Question: can area, location, rooms, floor, building characteristics and amenities predict price?", "Regression target: price; models learn log1p(price).", "Use case: decision support, not an official valuation."], "Nurkhan")
add_slide("Dataset and responsible scope", ["Astana Real Estate Dataset by turarr (Kaggle); Krisha.kz advertisements.", "Checked CSV: 5,003 listings, 23 columns, and a non-missing price target.", "Asking prices are not verified transaction prices.", "Duplicates, missing fields and seller-entry errors are documented limitations."], "Nurkhan")
add_slide("Preparation, EDA and feature engineering", ["Numeric conversion, floor extraction, duplicate removal, range checks and leakage-field removal.", "EDA: raw/log price, area-price, district distributions, correlations and missingness.", "Engineered: house age, first/top floor, floor ratio, area per room.", "Distance to centre is added only when real coordinates exist."], "Nurkhan", fig / "eda" / "area_vs_price.png")
add_slide("Evaluation design and leakage control", ["Fixed 70/15/15 train, validation and test split with log-price quantile bins.", "Imputation, scaling and one-hot encoding are fitted inside training pipelines only.", "Five-fold CV and tuning use train data; validation selects; test is used once.", "Metrics: log RMSE, log MAE, R2, KZT MAE and MedAPE."], "Bakytzhan")
add_slide("Regression models and tuning", ["Midterm retained: median baseline, Linear Regression, Decision Tree and KNN.", "Endterm adds Random Forest and Gradient Boosting.", "Grid/Randomized search with five-fold CV uses train data only.", "Feature importance is predictive association, not causation."], "Bakytzhan")
add_slide("K-Means market segmentation", ["Exploratory features: scaled area, rooms, price, age, floor ratio and coordinates.", "Price is allowed for segmentation but never used as a supervised predictor.", "k = 5 has the highest tested silhouette score; elbow and silhouette evidence are shown.", "Cluster summaries compare count, price, area, rooms and age."], "Nurkhan", fig / "clustering" / "cluster_selection.png")
add_slide("PCA: dimensionality reduction", ["PCA is fitted to standardized numerical training features only.", "The first two components retain 33.8% of numerical-feature variance.", "9 components retain about 90%; 11 retain about 95%.", "PCA supports visualization but is less interpretable than original features."], "Nurkhan", fig / "pca" / "pca_variance.png")
add_slide("ANN for tabular regression", ["Appropriate deep learning: feedforward ANN, not unrelated CNN/NLP.", "Architecture: 128 -> 64 -> 32 ReLU; dropout 0.2; linear output.", "Input and log-price target scaling are fit on training data only.", "Validation RMSE(log) = 0.171: competitive, but above Gradient Boosting (0.162)."], "Bakytzhan", fig / "ann" / "training_history.png")
add_slide("Unified model comparison", ["All models use the same split and metric definitions.", "Validation log-RMSE selects the model family.", "Gradient Boosting is selected; the ANN does not improve the best traditional model.", "The selected model is refit on train + validation before one test evaluation."], "Bakytzhan", fig / "evaluation" / "model_comparison.png")
add_slide("Final model and error analysis", [final_metrics(), "Actual/predicted and residual plots show model behaviour.", "Errors are inspected by price segment, district and largest listing errors.", "Unrecorded renovation/view/building explanations remain hypotheses."], "Bakytzhan", fig / "evaluation" / "error_analysis.png")
add_slide("Limitations and ethics", ["Listings can differ from transaction prices and represent one market period.", "Incomplete seller details limit prediction, especially for premium homes.", "Accuracy may differ by price segment and district.", "Predictions support, not replace, professional appraisal and human judgment."], "Nurkhan")
add_slide("Conclusions and practical value", [final_metrics(), "Gradient Boosting won validation selection; ANN complexity was not justified by its result.", "K-Means finds limited structure; PCA needs 9 components for 90% variance.", "Future work: verified sales, quality data and time-aware validation."], "Both")
add_slide("Contributions and Q&A", ["Nurkhan: preparation, EDA, features, clustering, PCA and documentation.", "Bakytzhan: supervised modelling, tuning, ANN, evaluation and error analysis.", "Each member presents an identifiable technical section.", "Questions"], "Both")
prs.save(OUT)
print(f"Wrote {OUT.name} with {len(prs.slides)} slides")
