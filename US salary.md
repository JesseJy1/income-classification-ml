# US salary
we have been tasked with developing a random forest model for US salary data set to predict whether a person will earn more than $50k dollars per year from the other features in the data set.
## 1. Analyse the dataset

This dataset contains various demographic and employment-related features, such as age, years of education, weekly working hours, occupation, marital status, ethnicity, gender, and country.
The target variable >50k indicates whether an individual's annual income exceeds $50,000.
Preliminary observations reveal:

* The data includes both numerical and categorical features, making it suitable for classification tasks.
* Target category imbalance: Individuals with income ≤$50k significantly outnumber those with income >$50k.
* Categorical features like occupational class, occupation, and marital status within the dataset feature multi-level classifications and require one-hot encoding.


## 2. Clean the data

After checking for missing values using `df.isna().sum()`, I performed the following cleaning steps:
* Removed the meaningless index column `Unnamed: 0`
* Removed rows containing missing values (missing in categorical features)
* Converted the target variable to binary format:


`df['target'] = (df['>50k'] == '>50k').astype(int)`

The final result is a clean dataset.

## 3. Examine each feature

Observing the statistical characteristics of each variable:
* **Marital status** shows a significant correlation with income (married individuals are more likely to earn higher incomes)
* **Educational attainment** significantly influences income distribution
* **Gender** exhibits a pronounced income gap (a higher proportion of males earn high incomes)
* **Occupation** exhibits marked uneven distribution, with substantial income disparities across industries
* **Hours worked** shows a positive correlation with income

All these variables possess clear predictive significance and are therefore retained for modeling.


## 4. Select a performance metric

Due to the low proportion of samples with income >50k, class imbalance exists, making accuracy alone misleading.

Therefore, we focus on:

* F1-score (combining precision and recall)
* Particular attention to recall for class 1 (>50k), as we aim to identify as many high-income individuals as possible
* Using a confusion matrix to evaluate specific misclassification patterns


## 5. Modelling assumptions

In constructing the model, I assume the following conditions are satisfied:

* The training and test sets maintain consistent distributions through stratified sampling
* One-hot encoding does not introduce erroneous order relationships
* Random forests permit nonlinear relationships between features without requiring additional distributional assumptions


## 6. Transformations

Then I used ColumnTransformer to process different variable types:

* Categorical variables → OneHotEncoder
* Numeric variables → Passed through directly

This step ensures the entire preprocessing workflow is reproducible and can be directly embedded into a Pipeline.


## 7. Train-test split

When designing the model, I used:

* 80% training set
* 20% test set
* stratify to ensure consistent target class proportions
* random_state=42 to maintain reproducible results


## 8. Baseline performance

After building a baseline model using the following code:

```
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf = Pipeline(steps=[('preprocess', preprocess), ('model', rf_clf)])

clf.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
```
output：
![](US%20salary/%E6%88%AA%E5%B1%8F2025-12-11%2010.35.00.png)
The overall accuracy is 0.780.
This indicates:
* The model performs well in predicting the majority class (<50k)
* The recall rate for the high-income group is low, at only 0.491
* This suggests the model is missing a significant number of high-income individuals
Therefore, it is necessary to perform hyperparameter tuning to enhance the model's recognition capability for category 1.


## 9. Hyperparameter tuning

I used RandomizedSearchCV to search for the following parameters:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf
* max_features
* class_weight

The evaluation metric is set to F1-score, and a total of n_iter=20 combinations are searched.

```
param_distributions = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
    'model__class_weight': [None, 'balanced']
}

search = RandomizedSearchCV(
    clf,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
```

## 10. Performance of best model
I retested the model with the optimized hyperparameters and obtained the following output:

![](US%20salary/%E6%88%AA%E5%B1%8F2025-12-11%2010.40.38.png)
total accuracy：0.789
Weighted F1: 0.801

This indicates:
* The recall rate for the high-income category has significantly improved
* Original: 0.491
* Adjusted: 0.813
* Although precision has slightly decreased, the model is now less likely to miss high-income individuals—a critical outcome for the task objective. This demonstrates a successful enhancement of model performance.

## 11. Feature Importance

The top 20 most important features are:
![](US%20salary/%E6%88%AA%E5%B1%8F2025-12-11%2010.43.06.png)

### **Interpretation**

* **Marital status is a significant predictor of income**, particularly being married (Married-civ-spouse).

* Age, years of education, and years of work experience are naturally strong predictors.

* Gender also influences income prediction (consistent with observed trends).

* Professional and managerial occupations significantly impact income prediction.

## 12. Final Model

The optimal model identified via RandomizedSearchCV includes:

* Complete preprocessing pipeline

* Random Forest classifier with optimal parameter combinations

Final performance:

* **Accuracy: 0.789**

* **Class 1 Recall: 0.813**

* **Weighted F1: 0.801**

The model effectively identifies high-income groups, showing significant improvement over baseline models.

## 13. Future Work

1\. **Addressing Class Imbalance**
* Experiment with SMOTE and ADASYN
* Further refine class_weight adjustments

2\. **Expanded Hyperparameter Search**
* Focus on max_depth and min_samples_leaf

3\. **Explore Alternative Models**

* Gradient Boosting

* XGBoost / LightGBM (typically outperforms RF)

4\. **Feature Engineering**

* Merge rare classes

* Create interaction features (e.g., age × education)

* Design occupation level features

5\. **Learning Curve Analysis**

* Determine if more data is needed to improve model performance

