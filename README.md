# Executive Summary: Credit Card Fraud Detection
Core Objective: Develop a high-recall machine learning model to identify fraudulent transactions within a highly imbalanced dataset (0.17% fraud rate)

### The Challenge
Detecting fraud in this dataset is a incredibly hard problem. Standard models often achieve 99.9% accuracy by simply predicting that no transaction is fraud, which is useless for business security. The goal was to find a model that could "see" the subtle patterns hidden within the noise.

### Technical Approach
Model Selection: Leveraged XGBoost for its efficiency with imbalanced tabular data.

SHAP-Driven Engineering: Performed SHAP Interaction Analysis to identify hidden relationships, such as how feat1 only becomes a high-risk signal when paired with specific feat10 values.

Error Analysis: Looked through the data to see where the model was incorrectly marking cases as fraud, then used that to engineer new features to avoid it.

Feature Innovation: Engineered custom interaction terms and behavioral ratios (e.g., Amount / feat3) to simplify complex decision boundaries for the model.

Robust Validation: Validated performance across multiple random seeds (42, 123, 2025) to ensure the engineered signals were stable and not just noise.

## Future Work & Methodology Reflections
If I were to extend this project for production-readiness, I would focus on the following:

1. Robustness Testing (K-Fold Cross-Validation)
While I validated the model across multiple random seeds, a more rigorous approach would involve K-Fold Cross-Validation. This would ensure that the engineered features (like the feat1 interactions) are consistently providing signal across different slices of the data and aren't just over-fitting to a specific train/test split.

2. Feature Ablation Studies
With more time, I would perform a formal Ablation Study—removing one engineered feature at a time and measuring the delta in PR-AUC. This would confirm exactly which "tricks" provided the most lift and which could be pruned to keep the model lean.

3. Hyperparameter Tuning vs. Feature Engineering
While hyperparameter optimization (e.g., tuning scale_pos_weight or max_depth) would likely yield incremental gains, I intentionally prioritized Feature Engineering for this iteration. In fraud detection, a better "feature" almost always beats a better "parameter." My goal was to master the art of uncovering hidden interactions via SHAP, which provides a higher long-term ROI for model development than grid-searching parameters.

4. Automated Feature Selection
I would implement a recursive feature elimination (RFE) process or a "Permutation Importance" check to see if any of the original 28 features have become redundant now that the new interaction terms are doing the heavy lifting.
