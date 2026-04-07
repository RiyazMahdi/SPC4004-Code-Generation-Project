# 🫀Heart Disease Prediction using AI-Generated Machine Learning

## Overview
This project investigates the use of AI-generated code in developing a machine learning (ML) model for predicting heart disease. The study focuses on evaluating the quality of the initial AI-generated solution, identifying its limitations, and improving it through iterative development.

The project demonstrates how AI-generated code can act as a useful starting point, but requires human refinement to produce a more reliable and well-structured model.

---

## Objectives
- Responsible and transparent use of AI tools.
- An understanding of a specific machine‑learning problem.
- Critical analysis of initial code.
- Well‑justified iterative improvements.
- A functional and stable final version of your code.
- Proper referencing of datasets and external materials used.
- Correct use of GitHub for version control and submission.

---

## Dataset
The dataset used in this project is derived from the **UCI Heart Disease dataset**, a widely used benchmark dataset in machine learning.

It contains clinical features such as:
- Age  
- Sex  
- Chest pain type  
- Cholesterol levels  
- Maximum heart rate  

The target variable is binary:
- `0` → No heart disease  
- `1` → Presence of heart disease  

---

## Models Used
- **Logistic Regression** (baseline model)  
- **Random Forest Classifier**  
- **Tuned Random Forest** (using GridSearchCV)  

---

## Development Process
The project was developed using version control to track the progression from the initial AI-generated code to the final improved model.

Key stages included:
1. Initial AI-generated logistic regression model  
2. Dataset inspection and preprocessing checks  
3. Improved evaluation using classification metrics  
4. Model comparison using Random Forest  
5. Hyperparameter tuning using GridSearchCV  
6. Final refined model and evaluation  

---

## Results

| Model | Test Accuracy | CV Accuracy | Recall (Disease) | Precision (Disease) | F1 (Disease) |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.814 ± 0.101 | 0.91 | 0.77 | 0.83 |
| Random Forest (default) | 0.84 | 0.802 ± 0.070 | — | — | — |
| Tuned Random Forest | 0.82 | 0.835 | 0.97 | 0.76 | 0.85 |

Best parameters identified by GridSearchCV: `n_estimators: 200`, `max_depth: None`

### Key Observations
- Logistic regression provided a strong baseline performance  
- Random forest did not initially outperform logistic regression  
- Hyperparameter tuning improved random forest performance  
- The tuned random forest model achieved the best results  

---

## Evaluation Metrics
The models were evaluated using:
- Accuracy  
- Confusion Matrix  
- Precision  
- Recall  
- F1-score  

Particular attention was given to **recall**, as false negatives (missed heart disease cases) are critical in healthcare applications.

---

## Limitations
- The dataset is relatively small and simplified  
- It does not fully represent real-world clinical data  
- The model does not include temporal or patient history data  
- Results may not generalise to real-world medical settings  

---

## Key Insights
- AI-generated code provides a useful starting point but is often incomplete  
- Human intervention is necessary to ensure proper evaluation and optimisation  
- Model performance improvements are often incremental rather than dramatic  
- Evaluation metrics beyond accuracy are essential in sensitive domains such as healthcare  

---

## Technologies Used
- Python  
- pandas  
- scikit-learn
- matplotlib
- seaborn
---



## Repository Structure
```
├── code/
│ ├── final_model.py
│ ├── heart_disease_model.py
│ ├── initial_model.py
│
├── data/
│ └── heart-disease_iterations.csv
│
├── README.md
└── requirements.txt
```


---

## How to Run the Project

1. Install dependencies:
```pip install -r -requirements.txt```
2. Run the script:
```python heart_disease_model.py```

---

##  Conclusion
This project highlights the importance of critically evaluating AI-generated code. While AI tools can quickly produce functional solutions, they require careful review, improvement, and validation to ensure reliability and effectiveness.

The final model demonstrates improved performance through iterative refinement, reinforcing the role of human oversight in machine learning workflows.

