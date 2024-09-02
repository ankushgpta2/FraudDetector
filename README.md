# Fraud Detection Model Documentation

 
1. [Overarching Goal and Approach](#overarching-goal-and-approach)
2. [Steps to Run Pipeline](#steps-to-run-pipeline)
3. [Data Assumptions](#data-assumptions)
    - [Assumed Relationships](#assumed-relationships)
4. [Data Exploration](#data-exploration)
    - [General](#general)
        - [Basic Summary Statistic](#basic-summary-statistics)
        - [Distribution of transactionAmount Column](#distribution-of-transactionamount-column)
        - [Analysis of Duplicate Transactions](#analysis_of_duplicate_transactions)
    - [Acquiring Null Count](#acquiring-null-count)
    - [Unique Value Counts](#unique-value-counts)
    - [Min/Max Values](#minmax-values)
    - [Duplicated Charges](#duplicated-charges)
    - [Reversal Charges](#reversal-charges)
5. [General Processing Choices](#general-processing-choices)
    - [Filtering Out Duplicates](#filtering-out-duplicates)
    - [Dropping Empty Columns](#dropping-empty-columns)
6. [Feature Choices](#feature-choices)
    - [Data Splitting](#data-splitting)
    - [Encoding Choices](#encoding-choices)
    - [Multicollinearity](#multicollinearity)
    - [Feature Scaling](#feature-scaling)
    - [Feature Creation Choices](#feature-creation-choices)
    - [Handling Class Imbalance](#handling-class-imbalance)
7. [Model Choice & Properties](#model-choice--properties)
    - [Deciding Between a Shallow vs Deep ML Model](#deciding-between-a-shallow-vs-deep-ml-model)
    - [Why a Gradient Boosting Model vs Other Types](#why-a-gradient-boosting-model-vs-other-types)
    - [Why XGBoost Specifically vs Other GBM Models](#why-xgboost-specifically-vs-other-gbm-models)
    - [Data and Features Found Useful](#data-and-features-found-useful)
    - [Tuning](#tuning)
    - [Model Performance](#model-performance)
8. [Future Directions](#future-directions)


## Overarching Goal and Approach

The primary objective was to develop a robust fraud detection model to identify fraudulent transactions based on a set of features extracted from transactional data. The model is designed to utilize historical transaction data to predict whether a given transaction is fraudulent or not, using advanced machine learning techniques, particularly XGBoost. Multiple intermediate steps, between loading the data and training the selected model, were carried out to ensure the model can reach optimal performance (data cleaning, general processing, feature engineering, etc.).

## Steps to Run Pipeline

**Step #1**: 

Create the conda environment using the following command:

 `conda env create -f <path to environment.yml file>`

**Step #2**:

Activate the conda environment using the following command:

`conda activate CapitalOneDSChallenge`

**Step #3**:

Run the main.py script using the following command:

`python3.9 <path to main.py file>`

***NOTE***: The initial run might take some time (around 5 minutes on my local machine). I have incorporated automatic saving of certain intermediate datasets in a local directory, therefore allowing for two things (1) View/Check intermediate operations on the data and (2) Pick up where you left off (if an error is encountered).

***NOTE 2***: The pipeline will automatically download the transactions.zip from the repository and unzip it. To get around this, you need to manually place transactions.txt inside of the empty "data" folder provided. 

***NOTE 3***: I did some general refactoring, but I did not have enough time for certain things (reorganizing imports at the top of scripts in proper manner, adding args descriptions within functions/methods docstrings, and a couple other things).


## Data Assumptions

The following assumptions were made about the provided dataset:

- **Transaction Date/Time:** Represents the timestamp when the transaction was processed.

- **Account Number:** Uniquely identifies a customer's account; multiple transactions could be linked to the same account.

- **Customer ID:** Identifies a customer, potentially linked to multiple accounts or transactions.

- **Credit Limit:** The maximum credit available to the account holder.

- **Available Money:** The remaining credit available after the transaction.

- **Transaction Amount:** The monetary value of the transaction.

- **Merchant Name:** The name of the merchant where the transaction took place.

- **Merchant Country Code:** The country code where the merchant is located.

- **POS Entry Mode:** The method used to enter the transaction (chip, swipe).

- **POS Condition Code:** The condition of the terminal during the transaction.

- **Transaction Type:** The nature of the transaction (purchase, refund).

- **Reversal Indicator:** Transactions labeled as "REVERSAL" are assumed to be corrections of prior transactions.

### Assumed Relationships:
- **Account Number and Customer ID:** Single customer may have multiple accounts, hence a one-to-many relationship.

- **Transaction Amount and Credit Limit:** The transaction amount is less than or equal to the available credit limit.

- **Merchant Information and Fraud Likelihood:** Certain merchants or locations might have higher or lower fraud rates, therefore potentially affecting model predictions.

## Data Exploration

### General

#### Basic Summary Statistics

- The number of records identified during multiple steps of the pipeline, for different datasets:

    - **Raw Count**: 786,363

    - **Multi-Swipe Transactions**: 15,755

    - **Reversal Transactions**: 12,613

- The number of fields identified for all records, or entries within the .txt file, was **29** fields. 

***NOTE***: All basic summary statistics (count of nulls, min/max, unique values and counts, etc.) can be found under the **/analysis** folder. 

#### Distribution of transactionAmount Column:

![Alt text](./analysis/transactionAmount_histogram.png)

- **Description of the Histogram:**
    - Utilized the Freedman-Diaconis rule to determine optimal bin width based on certain properties of the underlying distribution (IQR, etc.).
    - Histogram is extremely left-skewed, indicating that a large proportion of transactions have smaller amounts.
    - There is a steep drop-off in the number of transactions as the amount increases (appears exponential), with very few transactions in bins for larger amounts. 

- **Hypothesis Based on Structure:**
    - Involves a large number of low-value transactions (microtransactions), therefore such as in-app purchases, low-value subscriptions, etc. for the companies. 
    - Might be data entry issues where large transactions are not recorded properly or are misclassified.
    - Dataset could be biased towards smaller transactions if it is not representative of the overall transaction population.
    - Since the number of fradulent transactions is small, and most of the the data is concentrated around smaller amounts with a couple of high outliers, the fraud transactions might be these outliers (higher in nature).
    - Fraudulent transactions might manifest in small amounts to avoid detection. The smaller transactions might contain a higher percentage of fraud transactions vs outside of this skewed distribution (blend in with legitimate small transactions by bad actor).

#### Analysis of Duplicate Transactions 

- **Multi-Swipe Total Amount:** $2,235,352.24
    - The average transaction cost for a multi-swipe duplicate is roughly $142 (total / # of transactions)
- **Reversal Total Amount:** $1,899,209.62
    - The average transaction cost for a reversal  duplicate is roughly $150 (total / # of transactions)

- The average cost per duplicate charge seems to be extremely high (the distribution for charges under both duplicate types are very skewing towards large amounts vs. the skew towards small amounts in the underlying distribution).

- Further exploration would involve:
    1. Segment / parse the distribution for these charges to get higher-resolution understanding.

    2. Extract patterns / correlations with other pertinent features (temporal components associated with these charges, particular merchants, particular time of day, etc.).

### Acquiring Null Count
- **Why?** 
    - Counting null values across all fields is crucial to assess data quality and determine the need for data imputation or exclusion.
    - Fields with excessive null values may need special handling, such as removal or imputation.
- **Approach / Findings?**
    -  Found the following fields within the provided dataset containing all null values:
        - merchantCity 
        - merchantState
        - merchantZip
        - echoBuffer
        - recurringAuthInd
    - All fields above were dropped from the dataset 

### Unique Value Counts
- **Why?**
    - Understanding the number of unique values, especially for categorical fields, helps in deciding the type of encoding technique to use (one-hot encoding vs target encoding, etc.).
- **Approach / Findings?**
    - Calculated unique value counts for almost all categorical variables in the dataset. 
    - Through these calculations, it was found that the target variable (isFraud) is extremely imbalanced between True/False cases (1:50 ratio).

### Min/Max Values
- **Why?**
    - Checking the range of values for certain fields, such as transactionAmount, helps in detecting outliers or anomalies that may affect the model's performance.
- **Approach / Findings?**
    - transactionAmount had a lower bound of 0 (even for transactions labeled as reversal), differing from how reversed transactions might normally be represented (as negative).
    - availableMoney had a lower bound of -1000, despite currentBalance having 0. 

### Duplicated Charges
- **Assumption**: 
    - Transactions from the same vendor, account, and within a 5-minute window could represent duplicate transactions (multi-swipes).
    - 5 minutes was selected as the default threshold through intuitive means, although more robust measures informed by the data could have been utilized (potential future implementation).

### Reversal Charges
- **Assumption**: 
    - Relevant reversal charges should have the same vendor, account number, and other matching fields. 
    - Transactions labeled as "REVERSAL" indicate corrections of prior transactions, therefore the original transaction + following repeated ones should be treated accordingly.

## General Processing Choices

### Filtering Out Duplicates
- **Why?**
    - Removing duplicate transactions, such as multi-swipes or reversals, prevents the model from learning incorrect patterns and improves its ability to generalize to new data.

- **Approach / Findings?**
    - These transactions could potentially signify potential fraud, but due to the available sample size and presence of many other features, it was decided it is safer to remove instead. As a future direction, the effects of not removing this data can be explored as well. 

### Dropping Empty Columns
- **Why?**
    - Completely empty columns do not contribute any useful information to the model, therefore they can safely be dropped to reduce data dimensionality and computational overhead. 

- **Approach / Findings?**
    - Almost all completely empty columns were related to merchant location. 

## Feature Choices

### Data Splitting
- **Why?**
    - Splitting the data into training, validation, and test sets is generally important for accurately gauging model performance/capabilities.
- **Approach / Findings?**
    - Ensured that all records belonging to the same account are within the same split (to prevent data leakage). This approach ensures that the model does not have an unfair advantage by learning patterns from the same account across different splits.
    - Ensured that the positive cases for fraud (isFraud = True) were fairly distributed across the splits (due to the significantly small number of these transactions);

### Encoding Choices
- **One-Hot Encoding**: 
    - **Why?**
        - Encoding in general was utilized for the following reasons:
            - ML models requires numerical inputs (models cannot directly interpret strings/categories as numerical values on their own)
            - To handle ordinality of categorical variables (some categorical variables have an inherent order)
            - To handle nominal categorical variables (model treats each category as distinct without implying any order)
    - **Approach / Findings?**
        - One-hot encoding is used for categorical fields with a limited number of unique values. This method ensures that the model can handle categorical data without assuming any ordinal relationship.
        - Through the initial data exploration, the unique counts for multiple categorical fields was relatively small, therefore this approach was selected (merchantCountryCode, etc.).

        - Selecting One-Hot Encoding over Target Encoding or Embeddings:     
            - Target encoding might introduce data leakage, especially with low-frequency categories.
            - Embedding is more suitable for deep learning models and may be overkill for XGBoost.
            - Both might be future directions to explore.

### Multicollinearity
- **Why?**
    - Multicollinearity can distort the model’s understanding of feature importance and lead to unstable predictions (due to complex correlative relationships).
    - Identifying and addressing multicollinearity ensures more reliable model interpretations and performance.

- **Approach / Findings?**

    - **Why VIF Over Correlation Matrix?**: 
        - VIF (Variance Inflation Factor) quantifies how much the variance of an estimated regression coefficient is increased due to collinearity.
        - It is more informative than a simple correlation matrix, which only measures pairwise correlations.
        - The computational overhead for VIF (typically more than correlation matrix calculations) was not significant, especially considering the complex relationship that many features in the dataset have with one another. 

    - **Why Default Threshold of 10?**: 
        - A VIF value above 10 often indicates significant multicollinearity that may need to be addressed by removing or transforming features.
        - XGBoost is fairly robust to navigating a decent amount of multicollinearity, therefore 10 seemed to be the reasonable edge at "too complex" for this particular model.

    - Multiple rounds of VIF were carried out, where each time, the feature with the largest VIF score is removed and VIF is calculated again for all remaining features. 

### Feature Scaling
- **Why?**

    - Standardizing features helps the model converge faster during training and ensures that all features contribute equally, particularly important for distance-based algorithms.

- **Approach / Findings?**

    - Utilized standardization of transactionAmount, creditLimit, and availableMoney features. 
    - Particularly important in order to ensure that these features are on the same scale, especially after noticing that some of these features had unintuitive differences in minimum/maximum values during initial data exploration. 

### Feature Creation Choices

- **New Features Created and Why?**

  - `transaction_to_credit_ratio`: Measures the relative size of a transaction compared to the available credit, a potential indicator of risky behavior.

  - `credit_minus_available`: Indicates the amount of credit utilized, which can signal potential financial distress.

  - `time_since_account_open`: Older accounts may have different fraud risk profiles than newly opened accounts.

  - `time_since_last_address_change`: Frequent address changes might correlate with fraud risk.

  - `transaction_amount_percentage`: Reflects the proportion of the transaction amount to the total available money.

  - `time_since_last_transaction`: Short time intervals between transactions might indicate unusual patterns.

  - `transaction_deviation`: Flags transactions that deviate significantly from the customer’s typical behavior.

### Handling Class Imbalance
- **Why?**
    -  With only around 2% of transactions flagged as fraud, the model could become biased toward predicting non-fraudulent transactions.

- **Approach / Findings?**
    - The decided approach was to adjust the `scale_pos_weight` parameter in XGBoost, in order to account for class imbalance.
    - The main factor for this decision was simply a lack of time and knowledge that XGBoost in particular has built in functionality for handling class imbalance.

    - **Potential Future Methods**:
        - **Oversampling**: Techniques like SMOTE (Synthetic Minority Over-sampling Technique), or even a deep generative model (VAE), to create synthetic samples of the minority class to balance the dataset.

        - **Undersampling**: Reducing the number of majority class samples to balance the dataset, though this might lead to loss of information.

        - **Hybrid Methods**: Combining oversampling and undersampling, or employing ensemble techniques designed to handle imbalance (Balanced Random Forest, etc.).

## Model Choice & Properties

### Deciding Between a Shallow vs Deep ML Model

- **Feature Properties**

    - Prior knowledge / intuition of the different features and the information they encode conceptually + the relationship with each other. 
    - Ability to conduct feature engineering (draw complex hierarchical relationships between features) without the need of a Deep Learning model. 

- **Computational Overhead**

    - Shallow models are more efficient and can often be trained on standard CPUs
    - Opting for a shallow model reduces both the computational cost and the complexity of model development/maintenance. 

- **Explainability**

    - Feature importance can be directly extracted and used to understand the model’s decisions, which is important in sensitive domains like fraud detection.

- **General Overfitting Risk**

    - The dataset is highly imbalanced (2% fraud cases) and deep learning models are more prone to overfitting on the minority class.

These various aspects to consider/weigh when deciding general model direction ties into the following approach (when it is not obvious to use a more complex model): 

(1) Start with a simple and explainable model

(2) If performance is below desired threshold, then gauge performance weak points

(3) If the weak points contributing to performance are derived due to hard constraints associated with the simpler model itself (cannot be mitigated through various techniques upstream on the data itself), then select a more complex model. 

There is almost always a tradeoff between model complexity and explainability. In the financial space, especially with banks, many rules/regulations require that the approach leans more on the side of explainability. 

### Why a Gradient Boosting Model vs Other Types?

- Gradient boosting models are effective in handling imbalanced datasets.

- Gradient boosting models can handle various types of data (categorical, numerical) without too much preprocessing. Optimal for the diverse set of features present in the provided dataset. 

- Can handle noisy data or complex, non-linear relationships. 

- Ability to fine-tune model capacity via parameters like tree depth and learning rate allows for better control over model bias and variance trade-offs.

### Why XGBoost Specifically vs Other GBM Models?

- Highly optimized for speed and performance.

- Includes built-in regularization (both L1 and L2), which helps prevent overfitting (common in models trained on imbalanced datasets). 

- Naturally handle missing values in the dataset (mitigates need/risk of missing values effecting performance).

### Data and Features Found Useful

- Feature importance metrics provided by XGBoost can be analyzed to identify the most influential features.

- Additionally, assessing multicollinearity (via VIF) helps to ensure that the model is not over-relying on redundant features.

### Tuning
- **Why?**
    - Hyperparameter tuning is essential to optimize the model’s performance by finding the best set of parameters that control model complexity, learning rate, and other factors.

- **Approach / Findings?**

    - **Selected Tuning Params**: 

        - **Max Depth**: To control the complexity of the model (deeper trees might capture more patterns but risk overfitting).

        - **Learning Rate**: For balancing the speed of learning (too high might skip optimal solutions, too low might converge too slowly).

        - **Subsample**: To help prevent overfitting by training on a random subset of data.

        - **Gamma**: Introduces regularization by specifying a minimum loss reduction required for further partitioning.

    - **Selected Tuning Method**: 

         - Grid Search was chosen to systematically explore a wide range of parameter values.

         - Randomized Search or Bayesian Optimization could also be used for efficiency in larger hyperparameter spaces. Might be potential future areas of exploration to improve model performance.

### Model Performance

- **Main Eval Metric Used:**
    - AUC (Area Under the ROC Curve) is particularly suited for imbalanced datasets as it evaluates the model’s ability to rank positive instances higher than negative ones without being affected by the class distribution.

    - The ability to handle imbalanced datasets was an extremely important factor to consider, due to the massive imbalance found in the target variable during initial exploration. 

- **Other Metrics Used**: 

    - Incorporated multiple other metrics for model performance evaluation in order to cover a variety of downstream use-cases and risk thresholds. 

    - **Accuracy**: Measures the proportion of correct predictions but can be misleading with imbalanced data.

    - **Precision**: Focuses on the proportion of true positives (among predicted positives), therefore important when the cost of false positives is high.

    - **Recall**: Measures the proportion of true positives identified by the model, important when missing fraudulent transactions is costly. 

    - **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure when both false positives and false negatives are important.

- **Results**:

    - **AUC (Area Under the Curve)**: 
        - The AUC generated was 0.7414
        - Model has a moderate ability to discriminate between the two classes. A perfect classifier would have an AUC of 1.0, while a completely random classifier would have an AUC of 0.5.
    - **Accuracy**: 
        - The accuracy generated was 0.2325
        - Low accuracy might be due to the imbalance in the dataset (with very few fraudulent transactions compared to non-fraudulent ones), leading to a model that correctly identifies fraud but fails to classify non-fraudulent transactions correctly.
    - **Precision**: 
        - The precision generated was 0.0186
        - Precision of 0.0186 (1.86%) means that only 1.86% of the transactions predicted as fraud by the model are actually fraudulent. This low precision indicates that the model has a high number of false positives (transactions incorrectly classified as fraud).
    - **Recall**: 
        - The recall generated was 0.9728
        - Recall of 0.9728 (97.28%) is very high, meaning that the model successfully identifies most of the fraudulent transactions. However, this comes at the cost of low precision, as seen above.
    - **F1 Score**: 
        - The F1 score generated was 0.0366
        - F1 score of 0.0366 (3.66%) is quite low, reflecting the poor balance between precision and recall. Despite the high recall, the very low precision drags down the F1 score, indicating that the model's overall performance in terms of classifying fraud is weak.

    - Model has a high recall, meaning it is very effective at catching fraudulent transactions, but the very low precision suggests that the model is incorrectly flagging a large number of legitimate transactions as fraud, leading to a high number of false positives.

    - Low accuracy and F1 score indicates that while the model is good at detecting fraud, but it is poor at distinguishing between fraudulent and non-fraudulent transactions.

    - This performance might be acceptable in situations where catching most fraud is the top priority, even at the expense of a high number of false positives.

## Future Directions

- Explore the effect of Ensemble methods on performance (stacking, blending, bagging).

- Combining with more advanced models (LSTM or Transformer based) to extract features and feed into XGBoost. 

- Handling class imbalance through other means and gauging effects on performance (SMOTE, ADASYN, Deep Generative Model).

