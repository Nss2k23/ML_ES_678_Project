# Online Shoppers Purchase Prediction — ES 678 Course Project

**Author:** Nelson Singh Shagolsem, 26250027

**Research question:** How do feature importance and classification thresholds affect performance in identifying purchasing sessions while controlling false positives?

Each row of the data is one browsing session on an online shop. The target `Revenue` is 1 if the session ended in a purchase (15.5% of sessions). We compare a majority-class baseline, Logistic Regression and Random Forest, tune them with 5-fold cross-validation on a stratified 60/20/20 train/validation/test split, choose the classification threshold on validation, and use the test set once. The graduate depth component is a feature-group ablation (31 combinations), a PageValues sensitivity and leakage audit, bootstrap confidence intervals, and an error analysis.

## Main results (test set, used once)

| Model | Threshold | PR-AUC (95% bootstrap CI) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Random Forest, all 17 features** | 0.65 | **0.728 (0.682–0.770)** | 0.668 | 0.678 | 0.673 |
| Random Forest, without PageValues | 0.55 | 0.342 (0.299–0.393) | 0.322 | 0.453 | 0.376 |
| Baseline (always "no purchase") | – | 0.155 | – | – | – |

- **Features:** PageValues dominates, followed by exit/bounce rates and the month. About 5 of the 17 features are enough, and the technical codes add nothing.
- **Threshold:** moving from 0.5 to 0.65 removes about 40% of false alarms (215 → 129) and loses about 10% of caught buyers (287 → 259). The bootstrap confirms this.

## Repository contents

| File | Contents |
|---|---|
| `MLCourseProject.ipynb` | The complete analysis, with all code, outputs and written results: data checks, preprocessing pipeline, model selection, threshold choice, feature importance (PCA, permutation, coefficients, ablation), test evaluation, bootstrap CIs, error analysis, conclusions, and the figures used in the paper |
| `README.md` | This file |

The LaTeX source of the paper is submitted separately as a ZIP and is not in this repository.

## Data

**Online Shoppers Purchasing Intention Dataset**, UCI Machine Learning Repository (ID 468), licence CC BY 4.0.
12,330 sessions × 18 columns. It is public and contains no personal data.

- Download: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset (click **Download** to get the ZIP)

The data are **not stored in this repository**. The notebook reads the ZIP you upload (see below).

## How to reproduce

1. Download the dataset ZIP from the UCI link above. Keep its name, which starts with `online`.
2. Open `MLCourseProject.ipynb` in Google Colab, using the "Open in Colab" badge at the top of the notebook.
3. In Colab, open the **Files** panel (folder icon on the left) and upload the ZIP. The first code cell finds it, unzips it and loads `online_shoppers_intention.csv`.
4. Click **Runtime → Run all**. A full run takes about 5–10 minutes on a standard Colab CPU; the Random Forest grid search is the slowest step.

The notebook also writes intermediate CSV files (`train.csv`, `val.csv`, `test.csv` and their processed versions) and a `figures/` folder of paper figures. These are produced by the run and are not needed as input.

**Fixed choices**
- Random seed `RANDOM_STATE = 42` is used for the data split, cross-validation folds, Random Forest, permutation importance and the bootstrap (1,000 resamples of the test set).
- All preprocessing (log-transform, scaling, one-hot encoding) is fitted on the training set only, inside a scikit-learn Pipeline.
- The test set is used once, in Section 8.

**Software used for the reported results**

| Package | Version |
|---|---|
| Python | 3.13.18 |
| scikit-learn | 1.6.1 |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |

All of these come pre-installed in Google Colab. With other versions, numbers may differ slightly in the last digit.

## AI-use disclosure

An AI assistant (Claude, Anthropic) was used for brainstorming the analysis plan, explaining concepts, writing and debugging code, drafting explanatory text in the notebook, and drafting this README. All code was run in the notebook, every reported number was checked against the cell outputs, and all modelling choices were reviewed by the author, who is responsible for the work.

## References

1. Sakar, C. O., & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
2. Google Analytics Help. How Page Value is calculated. https://support.google.com/analytics/answer/2695658
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
4. Oshiro, T. M., Perez, P. S., & Baranauskas, J. A. (2012). How Many Trees in a Random Forest? *MLDM 2012*, LNCS 7376, 154–168.
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
