# Credit Card Fraud Detection — Learning with Imbalanced Data

An academic machine-learning study comparing resampling, class weighting and ensemble models for **highly imbalanced credit-card fraud detection**. The dataset contains **284,807 transactions**, with around **0.17% fraud cases**.

## Key findings

- Compared RandomUnderSampler, NearMiss, ClusterCentroids, RandomOverSampler, SMOTE and ADASYN, plus an experimental GAN-based oversampler.
- Benchmarked Logistic Regression, SVM and tree/ensemble models, including XGBoost, LightGBM and Balanced Random Forest.
- Applied resampling **inside training folds** to reduce leakage risk and tuned decision thresholds using stratified cross-validation.
- In the final recorded comparison, **XGBoost** made around **9.4 false alarms per detected fraud case**, versus **18.7** for Balanced Random Forest; it sacrificed some recall in exchange for fewer false alarms and faster training.

The figures are reported by the original notebook; the repository packaging does not reproduce them independently.

## Run

1. Obtain the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and follow the dataset's terms.
2. Place `creditcard.csv` alongside `credit_card_fraud_ml.ipynb`. The `.gitignore` excludes the data file.
3. Install the dependencies in `requirements.txt`.
4. Open the notebook in Jupyter or Colab and execute the sections in order.

Some experiments are computationally expensive. The original notebook includes saved analysis and output cells; they were not rerun for publication.

## Repository contents

- `credit_card_fraud_ml.ipynb` — EDA, resampling, model comparison, optimization and findings.
- `requirements.txt` — key third-party dependencies.

Academic project; results are for a research dataset, not production fraud detection.
