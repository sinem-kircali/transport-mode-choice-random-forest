# Car or Public Transport? Random Forest vs Logistic Regression for Commuter Mode Choice

This project predicts and explains whether commuters outside Sweden's metropolitan regions drive or take public transport. It compares a tuned random forest with a logistic regression and evaluates both with metrics that account for the class imbalance: 85% of the commuters drive.

Built for the *Seminar Data Science for Marketing Analytics* course (MSc Data Science & Marketing Analytics, Erasmus University Rotterdam).

**→ Full analysis with code, tables and figures: [transport_mode.md](transport_mode.md)**

## Key results
| Model (test set, 738 commuters) | Accuracy | Recall: public transport | Balanced accuracy | AUC |
|---|---|---|---|---|
| Always "car" | 0.847 | 0.00 | 0.50 | 0.50 |
| Logistic regression | 0.850 | 0.07 | 0.53 | 0.767 |
| Random forest (mtry = 3) | 0.871 | 0.27 | 0.62 | 0.776 |
| Random forest, threshold = car share | 0.722 | **0.67** | **0.70** | 0.776 |

- **Accuracy is misleading here.** Always predicting "car" already reaches 85%. At the default threshold both models miss most public-transport users. Moving the threshold to the base rate finds two-thirds of them.
- **The forest is not significantly better than the logit** (DeLong test p = 0.66). The logit is therefore a credible, interpretable alternative.
- **Travel-time competitiveness drives the choice.** The probability of driving rises from 0.61 to 0.78 to 0.91 as public transport takes 1.5×, 2× and 3× the car's travel time, then levels off.
- **Service design matters** (logit odds ratios for driving):
  - multiple transfers ×3.0;
  - one transfer ×1.9;
  - walk over 500 m ×1.4;
  - high-frequency service ×0.52.
- **Gender:** women have about half the odds of driving that men have.

## Methods
- Data cleaning: decimal-comma conversion; 17 missing values removed (complete cases).
- 70/30 train/test split.
- Random forest: 500 trees, `mtry` tuned over 1–13 on the out-of-bag error, permutation importance, and partial dependence computed on the probability scale.
- Logistic regression: odds ratios with 95% CIs.
- Evaluation: accuracy, class-wise recall, balanced accuracy, AUC, ROC curves, a DeLong test for the AUC difference, and a threshold adjusted for the imbalance.

## Notes on this version
Extended from the original assignment script, which tuned and evaluated only the random forest by accuracy. New here:
- the logistic regression benchmark;
- the imbalance-aware metrics and the threshold analysis;
- the AUC comparison;
- the full `mtry` range;
- the probability-scale partial dependence.

## Structure
```
transport_mode.Rmd      source (R Markdown)
transport_mode.md       rendered analysis, viewable on GitHub
transport_mode_files/   figures
data/                   place TransportModeSweden.RData here (not included)
```

## Tech stack
R · randomForest · pROC · tidyverse · broom · R Markdown
