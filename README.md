# Insurance Claim Prediction using Neural Network
This project is designed to predict insurance claim approvals based on various customer attributes, including age, BMI, daily steps, number of children, smoking status, and medical charges. 
The dataset contains 1,338 rows and 9 columns, covering multiple aspects of an individual’s health and lifestyle. 
Here’s a breakdown of the data exploration, preprocessing, and modeling steps to develop an accurate prediction model.

## Dataset Overview
The dataset contains the following columns:
- age: Age of the insured.
- sex: Gender of the insured (0 = female, 1 = male).
- bmi: Body Mass Index of the insured.
- steps: Average daily steps taken.
- children: Number of dependents.
- smoker: Smoking status (1 = smoker, 0 = non-smoker).
- region: Geographical region (0–3).
- charges: Annual medical charges.
- insuranceclaim: Target variable indicating if an insurance claim was approved (1 = approved, 0 = denied).

Data insights show substantial variations in column scales, necessitating feature scaling and further preprocessing.

## Data Preprocessing
### Feature Scaling 

Standardize column ranges and minimize model bias, MinMax scaling was applied to age, bmi, steps, and charges features, scaling values between 0 and 1.

### Exploratory Data Analysis (EDA)

- Distribution and Outliers: Analyzed data distribution using density and box plots. Found outliers in charges, bmi, and smoker, which were retained due to their relevance.
- Correlation Analysis: Observed highest correlation (0.79) between smoker and charges. The region column showed low correlation and was subsequently dropped.
- Balancing Data: To address class imbalance in insuranceclaim, SMOTE oversampling was applied.

## Data Splitting
The dataset was split as follows:
- Training set: 80% of the data, with balanced classes.
- Test set: 10% of the data for evaluating model performance.
- Validation set: 10% of the data for fine-tuning the model.

## Model Architecture
A baseline neural network model was developed with the following structure:
- Input Layer: Accepts 7 input features.
- Hidden Layers: Two dense layers, each with ReLU activation.
- Output Layer: A softmax layer for binary classification.

The model is designed to optimize performance using categorical cross-entropy loss and an Adam optimizer.

## Results
- Accuracy Score: 0.896 (or 89.6%) indicates that the model correctly predicts claim approvals in nearly 90% of cases, which is generally quite strong, especially if compared with other benchmarks or similar datasets.
- Precision and Recall: Precision (0.923) and Recall (0.900) suggest the model is well-balanced. A high precision of 92.3% means it is good at predicting true approvals (few false positives), and a 90% recall indicates it captures most true approvals with few false negatives.
- F1-Score: 0.911, balancing precision and recall, reinforces that the model is performing reliably across the board.

In summary, these results reflect a strong performance, with high precision and recall indicating effective classification of insurance claims. However, depending on your project's goals, you may consider further fine-tuning or cross-validating to optimize these metrics further if needed.
