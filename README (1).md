# Codveda Data Science Internship Projects

Hands-on data science projects completed during my **Data Science Internship at Codveda Technologies**. The work is organised into three levels (Basic, Intermediate, Advanced), each delivered as a Jupyter notebook with code, outputs, visualisations and written interpretation.

**Author:** Solomon Buba Tarfa · [github.com/tarfajr](https://github.com/tarfajr)

`#CodvedaJourney` `#CodvedaExperience` `#CodvedaProjects` `#CodvedaAchievements` `#FutureWithCodveda`

---

## Repository structure

```
.
├── Level1_Basic_DataCleaning_and_EDA.ipynb
├── Level2_Intermediate_Regression_and_Classification.ipynb
├── Level3_Advanced_NLP_and_NeuralNetwork.ipynb
├── data/
│   ├── 1__iris.csv
│   ├── 3__Sentiment_dataset.csv
│   ├── 4__house_Prediction_Data_Set.csv
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
└── README.md
```

## Tasks completed

| Level | Task | Dataset | Techniques |
|---|---|---|---|
| 1 (Basic) | Data Cleaning & Preprocessing | Boston Housing | Missing-value handling, IQR outlier detection, one-hot encoding, standardisation |
| 1 (Basic) | Exploratory Data Analysis | Iris | Summary statistics, histograms, scatter and box plots, correlation matrix |
| 2 (Intermediate) | Predictive Modeling (Regression) | Boston Housing | Linear Regression, Decision Tree, Random Forest, MSE / RMSE / R² |
| 2 (Intermediate) | Classification | Iris | Logistic Regression vs Random Forest vs SVM, accuracy / precision / recall, ROC curves |
| 2 (Intermediate) | Clustering *(bonus)* | BigML Telecom Churn | K-Means, elbow method, silhouette score, PCA visualisation |
| 3 (Advanced) | NLP Text Classification | Sentiment dataset | Text cleaning, TF-IDF, Naive Bayes, Logistic Regression, precision / recall / F1 |
| 3 (Advanced) | Neural Network | Iris | Feed-forward MLP, loss curve, grid-search hyperparameter tuning |

Web scraping (Level 1 Task 1) and time-series analysis (Level 3 Task 1) were not selected. Each level requires any two of its three tasks.

## Key results

**Level 1**
- Cleaning flagged outliers in 238 of 506 housing rows (about 47%) using the 1.5 × IQR rule; removing them leaves 268 rows. This is an aggressive filter, so treat the cleaned file as a demonstration of the technique rather than a recommended final dataset.
- EDA on Iris shows petal length and petal width are highly correlated and are the strongest species separators. Setosa is cleanly separable; versicolor and virginica overlap.

**Level 2**
- Regression (predict house price, `MEDV`):

  | Model | RMSE | R² |
  |---|---|---|
  | Random Forest | 2.91 | 0.884 |
  | Decision Tree | 2.99 | 0.878 |
  | Linear Regression | 4.93 | 0.669 |

  The non-linear models clearly win, so the price relationship is not purely linear.
- Classification (Iris, 30-sample test set): Logistic Regression 93.3%, Random Forest 90.0%, SVM 96.7% accuracy. The test set is small, so differences of one or two samples should not be over-read.
- Clustering (customer segmentation): the silhouette score picked k = 2. One segment (922 customers) churns at 8.7% and the other (2,411 customers) at 16.7%, against an overall rate of 14.5%. The higher-churn segment has lower evening usage and fewer voicemail-plan customers. Silhouette scores are low overall (peak 0.14), so the segments are a useful rough split, not sharply separated groups.

**Level 3**
- Sentiment classification (3 classes, 147 test posts): Logistic Regression reached 85.0% accuracy and 0.78 macro F1; Naive Bayes reached 75.5% accuracy but only 0.50 macro F1 because it mostly predicts the majority "Positive" class.
- Neural network on Iris: best configuration (hidden layers 16 and 8, learning rate 0.001) scored 96.7% test accuracy.

## Setup and running

```bash
git clone https://github.com/tarfajr/<repository-name>.git
cd <repository-name>
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

**Before running, update the file paths.** The notebooks were built in a sandbox, so data is loaded from `/mnt/user-data/uploads/...`. In each notebook, replace that prefix with `data/` (for example `pd.read_csv("data/1__iris.csv")`). In Level 1, also change the output path of the cleaned CSV to a folder that exists on your machine.

## Design decisions and limitations

- **Sentiment labels grouped into three classes.** The raw column has about 190 fine-grained emotion labels for only 732 rows, which is too sparse to learn from. They were mapped by hand to Positive / Negative / Neutral. Anything not in either list falls into Neutral, which is why Neutral is the weakest class (F1 0.59).
- **Clustering uses behaviour only.** `State` and `Area code` were excluded: one-hot encoding 51 states swamped the usage features and gave very poor cluster quality. The `Churn` label was never used for clustering and was only compared against the segments afterwards.
- **Scikit-learn MLP instead of Keras.** The task list names TensorFlow/Keras, but TensorFlow could not be installed in the build environment. Level 3 uses scikit-learn's `MLPClassifier` and includes the equivalent Keras code, commented line by line, as a reference cell. That Keras code has not been executed.
- **No NLTK.** Text cleaning uses regular expressions and scikit-learn's built-in English stop-word list; no stemming or lemmatisation was applied.
- **Iris is small and easy.** With 150 rows and near-linearly-separable classes, the Iris results mainly demonstrate the workflow, not model strength.

## Tools

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## Acknowledgements

Thanks to **Codveda Technologies** for the internship and task framework.
