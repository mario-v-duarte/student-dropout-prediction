# Predicting Student Attrition: An Early Warning Classification Pipeline

## 1. Project Overview & Business Problem
* **Objective:** Build an end-to-end machine learning classification pipeline to predict first-term student dropout in higher education.
* **Stakeholder:** University Director of Student Success.
* **Business Impact:** Traditional retention tracking often identifies at-risk students too late (at the end of the academic year or second semester). This model utilizes pre-enrollment metrics and first-semester academic performance to trigger immediate, proactive counseling and financial advising interventions before students withdraw.
* **Future Extension:** This codebase establishes the foundational data architecture for subsequent social impact analysis in MSBA 507, allowing non-profits and granting agencies to monitor the efficacy of student retention programming.

## 2. Dataset Description
* **Source:** UCI Machine Learning Repository ("Predict Students' Dropout and Academic Success" dataset). Originally from the Polytechnic Institute of Portalegre, Portugal.
* **Dimensions:** 4,424 rows and 31 finalized features (after pre-processing and data leakage removal).
* **Target Variable:** `target_binary` (1 = Dropout, 0 = Retained [Enrolled or Graduated]).

## 3. Repository Structure
- `data/raw/`: Contains untouched uci_student_dropout_2021.csv (git-ignored)
- `data/processed/`: Cleaned data outputs
- `notebooks/msba_502_project.ipynb`: Complete, runnable Jupyter Notebook
- `.gitignore`: Protects raw data and system files
- `README.md`: Project report and user manual

## 4. Methodology & Engineering Workflow
1. **Data Ingestion & Cleaning:** Loaded tabular microdata via Pandas, standardized column names into `snake_case`, and removed special characters.
2. **Data Leakage Elimination:** Dropped all second-semester variables to ensure the model relies strictly on first-term signals.
3. **Feature Engineering:** Derived a binary `first_generation` indicator by parsing parental educational qualification codes, bypassing high-cardinality noise.
4. **Exploratory Data Analysis (EDA):** Visualized baseline attrition distributions and multi-dimensional intersections (financial distress vs. gender/first-gen status).
5. **Statistical Validation:** Deployed Welch’s two-sample independent t-tests to mathematically confirm significant differences in admission grades and first-semester GPA between retained students and dropouts.
6. **Dimensionality Reduction & Multicollinearity Check:** Calculated Variance Inflation Factors (VIF) and Pearson correlation matrices, successfully dropping redundant or zero-correlation features (`curricular_units_1st_sem_enrolled`, parental occupations, macroeconomic indicators).
7. **Machine Learning Model:** Deployed a `StandardScaler` pipeline and trained a `LogisticRegression` classifier via `scikit-learn`.

## 5. Key Findings & Model Performance
* **Model Accuracy:** Achieved an overall classification accuracy of 85% with strong precision (0.85) for identifying at-risk dropouts.
* **Retention Anchors:** Successfully passing courses (`curricular_units_1st_sem_approved`, coefficient -1.86) and maintaining up-to-date tuition payments (`tuition_fees_up_to_date`, coefficient -0.75) are the strongest protective factors against dropping out.
* **Risk Multipliers:** Carrying institutional debt (`debtor`), older enrollment age, and being male significantly amplify dropout probability.

## 6. User Manual: How to Run This Code
To replicate this analysis locally:
1. Clone this repository to your local machine:
   `git clone https://github.com/mario-v-duarte/student-dropout-prediction.git`
2. Ensure your Python environment has the required libraries installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`).
3. Download the raw dataset from the UCI Machine Learning Repository, rename it to `uci_student_dropout_2021.csv`, and place it in the `data/raw/` directory.
4. Launch Jupyter Notebook from your project root:
   `jupyter notebook`
5. Open `notebooks/msba_502_project.ipynb` and run all cells sequentially.
