<a id="readme-top"></a>


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br/>
<div align="center">
    <img src="images/project_cover.png" alt="Logo" width="1000" height="650">
  
  <h1 align="center"> Churn Prediction </h1>

  <p align="center">
    <h3>Customer churn classification of a credit card service with LGBM - Classifier as the main classification model.</h3>
    <br/>
    <a href="https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/tree/main/src"><strong>Explore the Docs and Functions »</strong></a>
    <br/><br/>
    <a href="https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/tree/main/notebooks">View Notebooks</a>
    ·
    <a href="https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <br>
  <summary>Table of Contents</summary>
  <br/>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#pre-requisites">Pre-requisites</a></li>
        <li><a href="#installation-of-libraries">Installation of libraries</a></li>
      </ul>
    </li>
    <li>
      <a href="#the-project">The Project</a></li>
      <ul>
        <li><a href="#1---business-understanding">1 - Business Understanding</a></li>
        <li><a href="#2---data-understanding">2 - Data Understanding</a></li>
        <li><a href="#3---data-preparation">3 - Data Preparation</a></li>
        <li><a href="#4---modeling">4 - Modeling</a></li>
        <li><a href="#5---evaluation">5 - Evaluation</a></li>
        <li><a href="#6---deployment">6 - Deployment</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<br/>

<!-- ABOUT THE PROJECT -->
## About The Project

<br/>

## Project Description 
In this project, I will be working with a dataset provided by **Kaggle**, where I will develop a churn-rate analysis. The goal is to identify the causes and reasons for customer churn from a banking institution in relation to credit card services. After understanding these causes and reasons, some machine learning models will be developed to predict potential customers who will be abandoning the credit card service of this institution. With these predictions, I will seek to develop solutions to prevent or reverse the churn of these customers.  

---

### CRISP-DM Methodology
This project follows the CRISP-DM (*Cross-Industry Standard Process for Data Mining*) framework applied to **Customer Retention & Churn Prediction**:
| **Stage** | **Objective** | **Methodological Execution** |
| :--- | :--- | :--- |
| **1. Business Understanding** | Mitigate revenue loss by identifying at-risk customers. | • **Target Definition**: Binary Classification (Churn: Yes/No).<br>• **KPIs**: Maximize **Lift** in retention campaigns & Revenue Saved vs. Cost. |
| **2. Data Understanding** | Detect patterns of friction and dissatisfaction. | • **EDA**: Distribution analysis (Detect Imbalance).<br>• **Hypothesis Testing**: Correlation Matrix & Independence Tests (Chi-Square). |
| **3. Data Preparation** | Construct a robust dataset for parametric modeling. | • **Scaling**: Standardization (Z-score) for coefficient comparability.<br>• **Encoding**: One-Hot Encoding for nominal variables.<br>• **Splitting**: Stratified Train/Test Split to preserve class ratio. |
| **4. Modeling** | Estimate Churn Probability | • **Algorithms**: Logistic Regression, SVM LinearSVC, KNN, Random Florest, XGBoost, LightGBM.<br>• **Inference**: Analyze **Odds Ratios** to determine feature elasticity. |
| **5. Evaluation** | Assess model reliability and financial impact. | • **Discrimination**: AUC-ROC & F1-Score & Recall.<br>• **Calibration**: Probability Calibration Curve (Reliability Diagram). |
| **6. Deployment** | Integrate insights into the CRM lifecycle. | • **Deliverable**: "High-Risk" Customer List for Marketing Squad.<br>• **Artifact**: Serialize model (`joblib`) for batch inference. |

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<br/>

## Built With
<br/>

- [![Databricks][Databricks Free]][Databricks Free-url]
- [![Language Python][Python]][Python-url]
- [![Apache][Apache Spark]][Apache Spark-url]
- [![PD][Pandas]][Pandas-url]
- [![NP][NumPy]][NumPy-url]
- [![Matplot][Matplotlib]][Matplotlib-url]
- [![Scipy][Scipy]][Scipy-url]
- [![Sklearn][scikit-learn]][scikit-learn-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

<br/>

**Clone the repository**
```sh
git clone https://github.com/OtnielGomes/Churn-Prediction-Credit-Card
```
<br/>

### Pre-requisites

> 📌 **This entire project was built using Databricks Free Edition**.

---

### 🧠 What is Databricks Free Edition?

**Databricks Free Edition** is the free version of the Databricks platform, designed for **students, educators, developers, and data enthusiasts**.  
It replaces the former *Community Edition* and offers a **serverless** environment with limited resources — ideal for **prototyping, learning, and collaboration**.

With it, you can:
- Create interactive notebooks (Python, SQL, Scala, R)
- Use **Databricks Assistant** for code suggestions and corrections
- Train machine learning models and build data pipelines
- Collaborate in real time with other users

---

### 📝 How to Sign Up

#### 1. Go to:  
[Databricks Free Edition – Microsoft Learn](https://learn.microsoft.com/en-us/azure/databricks/getting-started/free-edition)  
#### 2. Sign in with Google, GitHub, Microsoft, or another supported provider.  
#### 3. A **free workspace** will be automatically created for you.

---

### 🧭 First Steps in the Workspace

### 1. **Workspace**
- Organize your notebooks, scripts, and datasets
- Create folders and set sharing permissions

### 2. **Notebook**
- Interactive interface for writing and running code
- Supports **Python, SQL, R, Scala**

### 3. **Databricks Assistant**
- AI-powered helper that explains, suggests, and fixes code
- Works in notebooks and SQL editor


### Installation of Libraries

The installation of the required libraries is performed using the command:

```python
%pip install '..\requirements.txt'
```

This command is present in the first notebook of this project.

---

💡 **Note**:  
- In Jupyter/Databricks notebooks, the `%pip` magic command installs packages directly into the current environment.  
- If your `requirements.txt` file is located in a subdirectory or at a different path, make sure to update the path accordingly (e.g., `../requirements.txt`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<br/>

## The Project

<br/>

## 1 - Business Understanding  
---

### Project Challenge


The bank manager identified growth in the number of customers who are abandoning the credit card service.

Given this scenario, the main objective of the project will be to transform historical data into **actionable intelligence**, making it possible to understand the factors associated with churn and anticipate customer attrition risk.

Stakeholders expect the proposed solution to be capable of:

1. **Analyzing historical data** to identify patterns and variables related to churn.
2. **Developing a machine learning model** to estimate the probability of customer attrition.
3. **Supporting strategic retention actions**, prioritizing customers with the highest cancellation propensity.

> From a business perspective, the project aims to reduce customer losses, improve the efficiency of retention campaigns, and support data-driven decisions in the context of active customer relationship management.

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>

## 2 - Data Understanding

---

### Dataset Overview
This dataset contains information from 10,000 bank customers, including demographic, financial, and relationship-related attributes such as age, salary, marital status, credit card limit, and card category.

> These variables provide the analytical foundation for investigating behavioral patterns associated with customer attrition and for supporting the construction of predictive models.

---

### Data File
- **Data file**: `BankChurners.csv`

---

### Target Variable
The dependent target variable is **`Attrition_Flag`**, a categorical feature with binary classes:

1. **`Existing Customer`**
- Represents customers who remained active, that is, non-churners.

2. **`Attrited Customer`**
- Represents customers who discontinued their relationship with the credit card service, that is, churners.

> Since this is a **binary classification** problem, the target variable will be used to distinguish customers who remain in the base from those who are more likely to leave.

---

### Data Source
- **Dataset collected from Kaggle**:

[https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers?sort=votes&select=BankChurners.csv](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers?sort=votes&select=BankChurners.csv)

- **Original dataset reference**:

[https://leaps.analyttica.com/home](https://leaps.analyttica.com/home)

---
## Exploratory Data Analysis (EDA):
---

> The EDA will be conducted in three main stages: univariate, bivariate, and multivariate analysis.

### Univariate analysis

> **Evaluates one variable at a time, focusing on distribution, central tendency, dispersion, and outlier detection.**

---
### Bivariate analysis

> **Investigates the relationship between two variables, allowing the analysis of correlation, association, or differences between groups.**
---
### Multivariate analysis

> **Examines three or more variables simultaneously in order to identify more complex patterns, interactions, and joint behavior.**
---
<br/>

## Univariate Analysis:
---

<br/><br/>
<div align="center">
    <img src="images/hist_uni.png" alt="Histogram Univariate" width="900" height="750">
  </a>
</div>
<br/>

---

<br/><br/>
<div align="center">
    <img src="images/box_uni.png" alt="Box Plot Univariate" width="900" height="750">
  </a>
</div>
<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>

---

<br/><br/>
<div align="center">
    <img src="images/count_uni.png" alt="Count Plot Univariate" width="900" height="750">
  </a>
</div>
<br/>

---

<br/><br/>
<div align="center">
    <img src="images/target_count.png" alt="Target Count Univariate" width="900" height="500">
  </a>
</div>
<br/>

## Bi-Variate Analysis:
---
### Correlation of Variables x Churn

<br/><br/>
<div align="center">
    <img src="images/corr_bi.png" alt="Correlation Bivariate" width="900" height="900">
  </a>
</div>
<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>

---
<br/><br/>
<div align="center">
    <img src="images/hist_bi.png" alt="Histogram Bivariate width="500" height="900">
  </a>
</div>
<br/>

---
<br/><br/>
<div align="center">
    <img src="images/count_bi.png" alt="Count Plot Bivariate width="500" height="900">
  </a>
</div>
<br/>

---
### Statistical Tests

---
<br/><br/>
<div align="center">
    <img src="images/num_test.png" alt="Numerical Test Bivariate width="700" height="550">
  </a>
</div>
<br/>

---
<br/><br/>
<div align="center">
    <img src="images/cat_test.png" alt="Categorical Test Bivariate width="350" height="275">
  </a>
</div>
<br/>

### Multi-Variate Analysis:
---
<br/><br/>
<div align="center">
    <img src="images/multi.png" alt="Multi Analysis width="550" height="900">
  </a>
</div>
<br/>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>




## 3 - Data Preparation
---

- For the data preparation stage, **two distinct pipelines** were developed: one designed for **linear models** and the other for **tree-based models**.


> This separation was adopted because each family of algorithms has its own preprocessing requirements, especially with regard to **scaling numerical variables** and **encoding categorical variables**.


- As discussed in the EDA, the initial decision was to remove only the variables **`equip`** and **`wireless`**, since they showed high correlation with **`equipmon`** and **`wiremon`**, respectively.


- The other highly correlated variables were deliberately retained at this initial stage.


> This decision allows the modeling process to empirically evaluate how different algorithms handle **multicollinearity**, **informational redundancy**, and the possible **marginal predictive gain** associated with these variables.


- In both pipelines, the data preparation flow followed the same general structure:


- **Variable type optimization**, with the objective of ensuring structural consistency, reducing memory usage, and adapting the data to the computational requirements of the algorithms.

- **Feature engineering**, with the creation of new derived variables based on findings from the exploratory analysis and domain knowledge.

- **Model-family-specific preprocessing**, respecting the technical particularities of each modeling approach and ensuring compatibility between the transformed data and the algorithms used.

---


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>

## 4 - Modeling
---

### Model Evaluation and Selection Strategy
---

- The primary metric defined for this project will be **AUC-ROC**.


> Since the problem involves **imbalanced classes** and the central objective is to develop a model capable of estimating the **probability of churn**, AUC-ROC is appropriate because it provides a comprehensive view of the model's discriminative ability between the two classes.


- As secondary metrics, **F1-score** and **recall for the churn class** will also be monitored.


> The **F1-score** will be used to evaluate the balance between **precision** and **recall**, while **recall for the churn class** will receive special attention, since correctly identifying customers at higher risk of churn is essential for guiding retention actions.


> This choice is strategically relevant, since **retaining a customer** through engagement and loyalty campaigns is approximately **five times less costly** than **acquiring a new customer**.

- Initial training will be conducted using **cross-validation**, with the objective of increasing the robustness of model evaluation at this stage of the project.


> **5 folds** will be used, and the selection of the best model will consider not only the **highest mean AUC-ROC**, but also the **lowest variability** among the results observed across the different folds, seeking consistent performance and greater generalization capability.

- Simpler models, such as **Logistic Regression** and **Decision Tree Classifier**, will be tested.

- And more robust models, such as **Linear SVC** and **LightGBM**, will also be evaluated.

> The objective of this comparison is to verify whether the structure of the data benefits from algorithms with **greater modeling capacity**, making it possible to assess potential performance gains in relation to simpler approaches.


---

### Applying Cross Validation

<br/>
<div align="center">
    <img src="images/ml_crossvalidation.png" alt="ML Crossvalidation" width="1000" height="800">
  </a>
</div>
<br/>

---
### Training Neural NetWorks Models With PyTorch - Cross Validation

#### Data Preprocessing for Neural Network Models  

Neural networks benefit from feature scaling and encoding strategies that normalize value ranges and preserve meaningful relationships between categories, contributing to stable and efficient training. Based on this, I adopted the following preprocessing strategy:  

---

#### 1. Numerical Variables  

| **Technique**    | **Applied Variables**                                                                                                                                                                                        | **Justification**                                                                                                                |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **MinMaxScaler** | `months_on_book`, `customer_age`, `dependent_count`, `total_relationship_count`, `months_inactive_12_mon`, `contacts_count_12_mon`, `total_revolving_bal`, `avg_utilization_ratio`, `total_amt_chng_q4_q1`, `total_ct_chng_q4_q1`, `total_trans_ct` | Scales features to the [0, 1] range, preserving the original distribution shape and ensuring that all features contribute uniformly. |
| **RobustScaler** | `credit_limit`, `total_trans_amt`                                                                                                                                                                             | Reduces the influence of outliers and large value ranges, improving robustness in the presence of extreme values.               |

---

#### 2. Categorical Variables  

| **Technique**                  | **Applied Variables**                              | **Justification**                                                                                                                                                                |
|--------------------------------|----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OrdinalEncoder**             | `gender`, `marital_status`                         | Encodes nominal categories as integers for direct embedding or subsequent processing, without implying any intrinsic order.                                                     |
| **Custom OrdinalEncoder + Embedding Layer** | `education_level`, `income_category`, `card_category` | Preserves the natural order between categories (e.g., `High School` → `College` → `Graduate School`) for ordinal features and maps them to dense vectors through embeddings, allowing the network to learn richer relationships. |

---

**Note on Embeddings in PyTorch**  

Embedding layers are applied to **ordinal and nominal features** because they offer several key benefits:  


- **Richer semantic representation**   

  Transform categories into dense, continuous vectors that allow the network to capture both explicit and hidden relationships—hierarchies and inclusive similarities—that are impossible to express with fixed encodings like one-hot. 

- **Dimensionality reduction for high-cardinality features**

  Drastically reduce the size of the input space compared to one-hot encoding, which is particularly valuable for variables with many unique categories.  

- **Learnable, task-specific mappings**   

  Embedding vectors are optimized together with the rest of the network’s parameters, meaning that the representation adapts to the specific prediction task, improving accuracy and generalization.  

- **Performance and scalability**   

  Lower memory usage and computational load by avoiding sparse, high-dimensional matrices — enabling faster training and inference without sacrificing representational power.  

---

### Applying Cross Validation

<br/>
<div align="left">
    <img src="images/torch_crossvalidation_k1cm.png" alt="Torch Crossvalidation k1" width="200" height="200">
    <img src="images/torch_crossvalidation_k1metrics.png" alt="Torch Crossvalidation k1" width="550" height="200">
</div>

<br/>
<div align="left">
    <img src="images/torch_crossvalidation_k2cm.png" alt="Torch Crossvalidation k2" width="200" height="200">
    <img src="images/torch_crossvalidation_k2metrics.png" alt="Torch Crossvalidation k2" width="550" height="200">
</div>

<br/>
<div align="left">
    <img src="images/torch_crossvalidation_k3cm.png" alt="Torch Crossvalidation k3" width="200" height="200">
    <img src="images/torch_crossvalidation_k3metrics.png" alt="Torch Crossvalidation k3" width="550" height="200">
</div>

<br/>
<div align="left">
    <img src="images/torch_crossvalidation_k4cm.png" alt="Torch Crossvalidation k4" width="200" height="200">
    <img src="images/torch_crossvalidation_k4metrics.png" alt="Torch Crossvalidation k4" width="550" height="200">
</div>

<br/>
<div align="left">
    <img src="images/torch_crossvalidation_k5cm.png" alt="Torch Crossvalidation k5" width="200" height="200">
    <img src="images/torch_crossvalidation_k5metrics.png" alt="Torch Crossvalidation k5" width="550" height="200">
</div>

<br/>                                                                        

✅ Cross validation Metrics:

---
🔴 Loss: 0.027 - ☑️ Standard Deviation - Loss: 0.002934

---

🟠 Accuracy: 96.17% - ☑️ Standard Deviation - Accuracy: 0.005298

---

🔵 Precision: 84.58% - ☑️ Standard Deviation - Precision: 0.026960

---

🔵 NPV: 98.69% - ☑️ Standard Deviation - NPV: 0.002221

---

⚠️ Recall: 93.27% - ☑️ Standard Deviation - Recall: 0.012720

---

🎯 AUC-ROC: 99.10% - ☑️ Standard Deviation - AUC-ROC: 0.001672

---

<br/>

### Scores of Models

<br/> 
<div align="left">
    <img src="images/models_performance.png" alt="Models Performance" width="800" height="500">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<br/>

### 5 - Evaluation

At this stage, I analyze the comparative performance of **classical machine learning models** and the **PyTorch neural model**, focusing on both technical and business criteria.  
The comparison considers the top 3 models based on their performance during **cross-validation**.

---

### What is the main goal of a churn prediction project?

- **Customer retention**: prevent customer loss and extend their journey with the institution.
- **Cost reduction**: optimize retention campaigns.
- To achieve this, two indicators are essential: **CAC** and **LTV**.

---

### What are these indicators and how can they financially impact a banking institution?

**Customer Acquisition Cost (CAC)** — varies according to business model, channel, and customer profile.

**Average ranges observed in the market in recent years**:
- **Digital banks/fintechs**: From **US\$ 200** to **US\$ 700** per customer (optimized digital channels and segmented campaigns).
- **Traditional banks**: Above **US\$ 1,000** per customer (dependence on physical branches and mass media).
- **Aggressive campaigns**: From **US\$ 1,500** to **US\$ 2,500+**, especially for high-income or corporate customers.

**What makes up the CAC**:
- Advertising (TV, radio, print, digital media)
- Performance marketing (Google Ads, social media, affiliates)
- Sales salaries and commissions
- Events and sponsorships
- Financial incentives (cashback, bonuses, fee waivers)
- CRM and marketing automation tools/systems

**Strategy and return**:
- CAC is evaluated alongside **Lifetime Value (LTV)**.
- Customers using multiple products (account, credit card, investments, insurance) justify higher CACs as returns are spread over several years.

**Source of estimated values**:  
[Forbes – What Are Banks’ And Fintechs’ Real Customer Acquisition Costs?](https://www.forbes.com/sites/ronshevlin/2025/03/23/what-are-banks-and-fintechs-real-customer-acquisition-costs/)

---

### Model Performance

| **Model**               | **AUC-ROC** | **Std. Dev. (AUC-ROC)** | **Highlight** |
|-------------------------|-------------|--------------------------|---------------|
| Random Forest           | 98.26%      | ±0.001795                 | Good generalization and consistent AUC-ROC. |
| Gradient Boosting       | 99.18%      | ±0.001530                 | Highest AUC-ROC. |
| Neural Network (PyTorch)| 99.10%      | ±0.001672             | Excellent performance and good generalization. |

---

### Why prioritize the PyTorch model?

1. **Robust generalization**:
   - Lowest standard deviation, indicating superior consistency.
   - Dropout (`p=0.2`) and L2 regularization to mitigate overfitting.

2. **Adaptability to new data**:
   - More efficient architecture for unseen data and atypical patterns.
   - Ability to learn complex non-linear relationships.

3. **Business costs**:
   - **False negatives** cost 5 to 7 times more than false positives.  
     estimate: **US\$ 2,500** per lost customer vs. **US\$ 500** per unnecessary offer 
   - Architectural adjustments and loss functions (e.g., Focal Loss) prioritize **recall**.

---

### Hyperparameter Tuning

- The network architecture and its parameters/hyperparameters are already well-defined, and the current metrics are satisfactory for the project’s objectives.  
Therefore, the **hyperparameter tuning** process will be used to validate the parameters already tested and implemented, as well as to seek an **improvement in the AUC-ROC score**, focusing on **reducing variance** and **bias** across the models in the 5-fold cross-validation.

- Given the complexity of neural networks, it is challenging to precisely determine the ideal number of neurons per layer.  
Architectures with a progressive decrease in neurons — such as **256 → 128 → 64** — are often effective in many scenarios.  
However, it is still important to explore new possibilities and test different configurations in this regard.

- A more precise adjustment of the **learning rate** will also be performed, along with verification of the **L2 regularization (weight decay)**.  
The goal is to assess whether a lower value (`1e-5`) remains more effective than a higher value (`5e-4`) in the current context.

- Finally, the impact of **batch size** will be tested, with values of **128, 256, and 512**.  
The choice of larger batch sizes is justified by the use of **Batch Normalization** in the network layers.  
Using very small batches can negatively affect performance, since BatchNorm calculates the mean and variance of the input data in each mini-batch for normalization.  
With very small batches, these statistics may become unstable, which can harm the network’s performance.

---

### Final Training

- In this step, the parameters and hyperparameters obtained during the hypertuning process will be applied to a training and validation workflow, using **80%** of the data for training and **20%** reserved for validation.

- After training, the model will be saved and applied to the **test data** to verify its **actual performance** and confirm that its bias and variance are consistent and within expectations in relation to the training data.

<br/>
<div align="left">
    <img src="images/torch_crossvalidation_traincm.png" alt="Torch Crossvalidation train" width="200" height="200">
    <img src="images/torch_crossvalidation_train_metrics.png" alt="Torch Crossvalidation train" width="550" height="200">
</div>

---
## Final Testing

### Analyzing model performance on test data

<br/>
<div align="center">
    <img src="images/torch_crossvalidation_testcm.png" alt="Torch Test" width="400" height="400">
</div>

<br/><br/>
<div align="left">
    <img src="images/predictions_prob.png" alt="Predictions Prob" width="900" height="300">
</div>

<br/><br/>
<div align="center">
    <img src="images/precision_roc_curve.png" alt="Torch Crossvalidation k5" width="900" height="400">
</div>

---

### Financial Results

#### Final Metrics – Test Data

After applying the trained model to the test dataset, the following performance metrics were obtained. These results confirm the model’s ability to generalize well and maintain consistency on unseen data:

| **Metric**    | **Score** |
|---------------|-----------|
| **Loss**      | 0.027     |
| **Accuracy**  | 96.6%     |
| **Precision** | 85.7%     |
| **NPV**       | 99.0%     |
| **Recall**    | 94.8%     |
| **AUC-ROC**   | 99.2%     |

- The **low loss value** indicates strong convergence and minimal prediction error.  
- The **high recall** confirms the model’s effectiveness in identifying churn cases, aligning with the strategic goal of minimizing customer loss. The model captures **94.8%** of churners, enabling retention actions before potential service discontinuation.  
- The **NPV and precision** values demonstrate balanced performance across both classes.  
- The **AUC-ROC of 99.2%** reinforces the model’s high discriminative power and reliability in probabilistic churn prediction.

These metrics validate the robustness of the trained model and its suitability for deployment in retention strategies. The high AUC-ROC allows for more assertive campaigns targeting customers classified as churners. Considering the previously mentioned **Customer Acquisition Cost (CAC)** and **Lifetime Value (LTV)**, we can build a hypothetical scenario (as no official figures are available for this institution) to estimate the costs involved in retention and the potential loss of customers.

---

**Hypothetical Cost Scenario:**

- **Cost to acquire a customer:** US\$ 2,500  
- **Cost to retain a customer classified as a churner:** US$ 500  

Based on the confusion matrix:  
- **329 churners**  
- **1,672 non-churners**  
- The model classified **364 customers as churners**

#### 1. Cost of false positives (non-churners classified as churners):

- With a precision of **85.7%**, there were **52 false positives**.  
- Cost: 52 × US\$ 500 = **US\$ 26,000**  
- Total retention campaign cost: 364 × US\$ 500 = **US\$ 182,000**  
- Proportion of resources incorrectly allocated:  
  US\$ 26,000 / US\$ 182,000 = **14.28%**

This is considered satisfactory, as only a small portion of resources would be spent on customers who would not have left the service. Nevertheless, such actions may still positively impact customer loyalty.

#### 2. Preservation of customer acquisition investment (CAC):

- Total investment to acquire the 329 churners: 329 × US\$ 2,500 = **US\$ 822,500**  
- With a recall of **94.8%**, the model correctly identifies **312 churners**  
- Preserved value: 312 × US\$ 2,500 = **US\$ 780,000**

Therefore, the model has the potential to preserve up to **US\$ 780,000** in acquisition investment, provided that retention actions are effective.  
The high AUC-ROC reinforces confidence in making more aggressive decisions with low risk of wasting resources on customers not at risk of churn.

---

#### 3. Lifetime Value (LTV) Assessment

**Lifetime Value (LTV)** is a strategic metric that estimates the total value a customer generates for the institution over the course of their relationship.  
It helps assess whether investments in **acquisition (CAC)** and **retention** are financially justified.

The simplified formula for calculating LTV is:

**LTV = Average Ticket × Purchase Frequency × Average Relationship Duration**

For illustrative purposes, let’s consider the following estimated values based on credit card customers:

- **Average monthly ticket:** US\$ 150 *(estimated value, as no official data is available)*  
- **Purchase frequency:** monthly  
- **Average relationship duration:** 3 years (36 months), based on the bank’s customer history  

**Estimated LTV = 150 × 12 × 3 = US\$ 5,400**

This value represents the average return the bank can expect from each customer over three years.  
Comparing it to the **average CAC of US\$ 2,500**, we have an **LTV/CAC ratio of 2.16**, indicating a healthy and sustainable relationship — meaning the value generated per customer is more than double the acquisition cost.

This analysis reinforces the importance of effective retention strategies:  
- By preserving churners, the model not only avoids losing the acquisition investment but also **protects the customer’s future LTV**.

#### Potential Revenue Preserved:

- **Potential gross revenue** over a 3-year period for customers identified as churners and successfully retained:  
  312 × US\$ 5,400 = **US\$ 1,684,800**

- Additionally, with a recall of **94.8%**, the model has the potential to preserve up to **US\$ 780,000** in acquisition investment (CAC) and **US\$ 1.68 million** in future gross revenue (LTV), provided that retention actions are effective.

---

**Conclusion:**  
A predictive model with high AUC-ROC and recall not only reduces immediate losses but also **maximizes the long-term value** of customers, directly contributing to the institution’s profitability and sustainability.

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>

### 6 - Deployment  

In this step, I deploy the final churn prediction classifier, developed using the insights and statistical patterns identified during the Exploratory Data Analysis (EDA).  
The model processes individual customer data and returns:  

- **Churn probability** — the likelihood of the customer leaving.  
- **Key influencing factors** — the main behavioral and financial indicators driving the prediction.  
- **Actionable recommendations** — targeted suggestions to help reduce churn risk.  

This deployment enables data-driven decision-making, allowing the business to proactively implement retention strategies, improve customer engagement, and maximize lifetime value.

<br/>
<div align="center">
    <img src="images/churndemo.png" alt="Deploy Churn Demo" width="1000" height="800">
</div>

<br/>
<div align="center">
    <img src="images/nonchurndemo.png" alt="Deploy Non-Churn Demo" width="1000" height="800">
</div>

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
<br/>

## Roadmap

- [Notebook-1-EDA](https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/notebooks/0_EDA.ipynb)
- [Notebook-2-Modeling](https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/notebooks/1_Modeling.ipynb)


See the [open issues](https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
<br/>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<br/>
<a href="https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
<br/>

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/blob/main/LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

<br/>

## Contact

[![LinkedIn][linkedin-shield]][linkedin-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/OtnielGomes/Churn-Prediction-Credit-Card.svg?style=for-the-badge
[contributors-url]: https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/OtnielGomes/Churn-Prediction-Credit-Card.svg?style=for-the-badge
[forks-url]: https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/network/members

[stars-shield]: https://img.shields.io/github/stars/OtnielGomes/Churn-Prediction-Credit-Card.svg?style=for-the-badge
[stars-url]: https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/stargazers

[issues-shield]: https://img.shields.io/github/issues/OtnielGomes/Churn-Prediction-Credit-Card.svg?style=for-the-badge
[issues-url]: https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/issues

[license-shield]: https://img.shields.io/github/license/OtnielGomes/Churn-Prediction-Credit-Card.svg?style=for-the-badge
[license-url]: https://github.com/OtnielGomes/Churn-Prediction-Credit-Card/blob/main/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/otnielgomes

[Azure Databricks]: https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white
[Azure Databricks-url]:  https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=databricks


[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/

[Apache Spark]: https://img.shields.io/badge/Apache%20Spark-FDEE21?style=flat-square&logo=apachespark&logoColor=black
[Apache Spark-url]: https://spark.apache.org/

[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/

[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/

[Scipy]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white
[Scipy-url]: https://scipy.org/

[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

[Databricks Free]: https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white
[Databricks Free-url]: https://www.databricks.com/br/learn/free-edition
