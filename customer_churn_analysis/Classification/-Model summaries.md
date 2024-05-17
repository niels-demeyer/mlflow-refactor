# Model metrics summaries
All metrics were taken after hyperparameter tuning.

As the **target variable seemed imbalanced**, the assumption is made that**oversampling through SMOTE** will increase model performance.
To test this, performance metrics were logged before and after this implementation.


### Evaluation metrics
-  **Train & Test Scores**: The accuracy on the training and test sets.
-  **ROC AUC**: Area Under the ROC Curve.
-  **Classification Report**: This report gives you a breakdown of precision, recall, f1-score, and support for each class.
-  **Cross Validation**: An additional layer of validation, confirming that the parameters selected indeed perform well across different subsets of the training data.

## Discussion:
### Best Models
Based on overall test scores, ROC AUC, and F1-scores:

1. **Gradient Boost**: Before SMOTE, this model shows exceptional performance with a test accuracy of 0.968, ROC AUC of 0.992, and a weighted average F1-score of 0.97.
2. **Random Forest**: Also displays strong performance with a test accuracy of 0.951 and a weighted average F1-score of 0.95 before SMOTE. After SMOTE, its test accuracy is slightly better at 0.956.

### Models Grouped by SMOTE Impact
**Improved with SMOTE**
1. **Random Forest** :
- Test accuracy improved slightly from 0.951 to 0.956.
- F1-scores and ROC AUC remained stable, indicating robustness in handling the minority class.

**Worsened with SMOTE** 
1. **K-Nearest Neighbours (KNN)**:
- Notable decrease in performance after SMOTE: Test score dropped from 0.906 to 0.853.
- F1-score and recall for the majority class decreased, suggesting potential overfitting to the minority class created by SMOTE.
2. **Decision Tree**:
- Test accuracy decreased from 0.938 to 0.92 after SMOTE.
- F1-score for "Attrited Customer" improved, but overall model accuracy and F1-scores decreased.
3. **Gradient Boost**:
- Slight decrease in test accuracy from 0.968 to 0.96 after SMOTE.
- A decrease in overall performance metrics, although still strong, indicates a minor negative impact from SMOTE.
4. **Logistic Regression**:
- Decrease in test accuracy from 0.897 to 0.858 after SMOTE.
- Despite improvements in recall for "Attrited Customer", the overall accuracy and precision decreased.
5. **AdaBoost**:
- Test accuracy slightly decreased from 0.958 to 0.958 (stable, but slight numerical drop in mean accuracy scores).
- Minor drop in precision and overall ROC AUC, indicating a slight disadvantage with SMOTE.

### Discussion
- **SMOTE Impact**: Generally, SMOTE tends to improve recall for the minority class at the potential cost of precision and overall accuracy, as observed in several models. This trade-off can be beneficial if the business case prioritizes identifying more of the minority class (Attrited Customers), even at the risk of more false positives (Existing Customers misclassified as Attrited).
- **Best Model Selection**: Gradient Boost and Random Forest stand out with high overall metrics and robustness to class imbalance, both before and after the introduction of SMOTE. The choice between them would depend on specific performance requirements and computational resources, as Gradient Boost generally takes longer to train.
- **Highest minority group recall**: Gradient Boost and AdaBoost show the highest minority recall rates. This might be prioritized, since the company is mainly focused on identifying the minority class.
- **Model Suitability**: The decision on whether to use SMOTE should be based on specific needs for precision vs. recall balance and how sensitive the model is to the introduction of synthetic data. In environments where false positives have high costs, careful calibration of the model's threshold or preference for models less sensitive to SMOTE might be necessary.

    
# Metrics 

## Logistic regression
### No SMOTE
```
Train Score:  0.906
Test Score:  0.897
ROC_AUC:  0.916
                   precision    recall  f1-score   support

Attrited Customer       0.76      0.52      0.62       327
Existing Customer       0.91      0.97      0.94      1699

         accuracy                           0.90      2026
        macro avg       0.84      0.75      0.78      2026
     weighted avg       0.89      0.90      0.89      2026

Mean Test Accuracy: 0.906
Mean Train Accuracy: 0.906
Mean Fit Time: 0.096
Mean Score Time: 0.015

```

### SMOTE
```
Train Score:  0.853
Test Score:  0.858
ROC_AUC:  0.914
                   precision    recall  f1-score   support

Attrited Customer       0.54      0.83      0.65       327
Existing Customer       0.96      0.86      0.91      1699

         accuracy                           0.86      2026
        macro avg       0.75      0.84      0.78      2026
     weighted avg       0.89      0.86      0.87      2026

Mean Test Accuracy: 0.851
Mean Train Accuracy: 0.851
Mean Fit Time: 0.147
Mean Score Time: 0.011

```

## Random Forest
### no SMOTE
```                    
Train Score:  1.0
Test Score:  0.951
ROC_AUC:  0.989
                   precision    recall  f1-score   support

Attrited Customer       0.93      0.76      0.83       327
Existing Customer       0.95      0.99      0.97      1699

         accuracy                           0.95      2026
        macro avg       0.94      0.87      0.90      2026
     weighted avg       0.95      0.95      0.95      2026

Mean Test Accuracy: 0.957
Mean Train Accuracy: 1.0
Mean Fit Time: 4.469
Mean Score Time: 0.092
```

### SMOTE
```
Train Score:  1.0
Test Score:  0.956
ROC_AUC:  0.987
                   precision    recall  f1-score   support

Attrited Customer       0.88      0.84      0.86       327
Existing Customer       0.97      0.98      0.97      1699

         accuracy                           0.96      2026
        macro avg       0.92      0.91      0.92      2026
     weighted avg       0.96      0.96      0.96      2026

Mean Test Accuracy: 0.957
Mean Train Accuracy: 1.0
Mean Fit Time: 6.593
Mean Score Time: 0.076

```

## AdaBoost
### no SMOTE
```                    
Train Score:  0.971
Test Score:  0.958
ROC_AUC:  0.986
                   precision    recall  f1-score   support

Attrited Customer       0.90      0.83      0.86       327
Existing Customer       0.97      0.98      0.97      1699

         accuracy                           0.96      2026
        macro avg       0.93      0.91      0.92      2026
     weighted avg       0.96      0.96      0.96      2026
    
Mean Test Accuracy: 0.963
Mean Train Accuracy: 0.971
Mean Fit Time: 1.784
Mean Score Time: 0.047

```

### SMOTE
```
Train Score:  0.968
Test Score:  0.958
ROC_AUC:  0.985
                   precision    recall  f1-score   support

Attrited Customer       0.86      0.88      0.87       327
Existing Customer       0.98      0.97      0.97      1699

         accuracy                           0.96      2026
        macro avg       0.92      0.93      0.92      2026
     weighted avg       0.96      0.96      0.96      2026

Mean Test Accuracy: 0.958
Mean Train Accuracy: 0.97
Mean Fit Time: 8.957
Mean Score Time: 0.088

```

## K-Nearest Neighbours
### no SMOTE
```                    
Train Score:  1.0
Test Score:  0.906
ROC_AUC:  0.925
                   precision    recall  f1-score   support

Attrited Customer       0.83      0.52      0.64       327
Existing Customer       0.91      0.98      0.95      1699

         accuracy                           0.91      2026
        macro avg       0.87      0.75      0.79      2026
     weighted avg       0.90      0.91      0.90      2026

Mean Test Accuracy: 0.911
Mean Train Accuracy: 1.0
Mean Fit Time: 0.075
Mean Score Time: 0.09

```
### SMOTE
```
Train Score:  0.941
Test Score:  0.853
ROC_AUC:  0.882
                   precision    recall  f1-score   support

Attrited Customer       0.53      0.81      0.64       327
Existing Customer       0.96      0.86      0.91      1699

         accuracy                           0.85      2026
        macro avg       0.74      0.84      0.77      2026
     weighted avg       0.89      0.85      0.86      2026

Mean Test Accuracy: 0.863
Mean Train Accuracy: 0.939
Mean Fit Time: 0.126
Mean Score Time: 0.104

```
## Decision Tree
### no SMOTE
```                    
Train Score:  0.967
Test Score:  0.938
ROC_AUC:  0.93
                   precision    recall  f1-score   support

Attrited Customer       0.81      0.80      0.81       327
Existing Customer       0.96      0.96      0.96      1699

         accuracy                           0.94      2026
        macro avg       0.89      0.88      0.88      2026
     weighted avg       0.94      0.94      0.94      2026

Mean Test Accuracy: 0.939
Mean Train Accuracy: 0.969
Mean Fit Time: 0.09
Mean Score Time: 0.011

```
### SMOTE
```
Train Score:  0.97
Test Score:  0.92
ROC_AUC:  0.923
                   precision    recall  f1-score   support

Attrited Customer       0.71      0.84      0.77       327
Existing Customer       0.97      0.93      0.95      1699

         accuracy                           0.92      2026
        macro avg       0.84      0.89      0.86      2026
     weighted avg       0.93      0.92      0.92      2026

Mean Test Accuracy: 0.929
Mean Train Accuracy: 0.972
Mean Fit Time: 0.363
Mean Score Time: 0.012

```
## Gradient Boost
### no SMOTE
```                    
Train Score:  1.0
Test Score:  0.968
ROC_AUC:  0.992
                   precision    recall  f1-score   support

Attrited Customer       0.93      0.87      0.90       327
Existing Customer       0.98      0.99      0.98      1699

         accuracy                           0.97      2026
        macro avg       0.95      0.93      0.94      2026
     weighted avg       0.97      0.97      0.97      2026

Mean Test Accuracy: 0.974
Mean Train Accuracy: 1.0
Mean Fit Time: 13.788
Mean Score Time: 0.025

```
### SMOTE
```
train_score:  0.966
test_score:  0.96
ROC_AUC:  0.988
                   precision    recall  f1-score   support

Attrited Customer       0.86      0.89      0.88       327
Existing Customer       0.98      0.97      0.98      1699

         accuracy                           0.96      2026
        macro avg       0.92      0.93      0.93      2026
     weighted avg       0.96      0.96      0.96      2026

Mean Test Accuracy: 0.957
Mean Train Accuracy: 0.968
Mean Fit Time: 4.899
Mean Score Time: 0.015

```