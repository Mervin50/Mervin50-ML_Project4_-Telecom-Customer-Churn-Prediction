# ML_Project4: Telecom Customer Churn Prediction

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/9a90d92e-b19c-4e4a-80c0-62d9c48ef372)


# 1. Introduction: 
Customer churn prediction is to measure why customers are leaving a business. In this tutorial we will be looking at customer churn in telecom business. We will build a deep learning model to predict the churn and use precision,recall, f1-score to measure performance of our model.

# 2. Description of the dataset:

### 2.1 First 5 rows of our dataset

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/8c674e75-69f3-47d7-a83b-ad3564827afb)

### 2.2 Shape of our Dataset:

Our dataset has 7043 rows and 20 columns

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/0d60ddcb-28fb-4225-981d-7795f9d2f4f1)

### 2.3 Datatypes of our columns:

The datatypes of our columns :

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/39bdf133-9764-4cfa-88cc-dbe0b7fe366e)

# 3. Semi-preprocessing and Data Cleaning for EDA

### 3.1 Detecting Null values in our dataset:

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/0be140e8-636a-4e01-94ed-34966614350b)

### 3.2 Customer Gender and Churn Distributions

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/27f10228-d179-49e0-80d5-27193b502c2a)

Observation:
1. 26.6 % of customers switched to another firm.
2. Customers are 49.5 % female and 50.5 % male.

### 3.3 Churn Distribution by Gender: A Comparative Analysis

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/09fe8f3b-236a-4aa6-b9b2-85d46949a469)

Observation: 
1. There is negligible difference in customer percentage/ count who chnaged the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.

### 3.4 Distribution of Customer Contracts Based on Churn Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/07cf8980-9153-4b77-9e7c-a74e87ff919b)

Observation:
1. About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customrs with One Year Contract and 3% with Two Year Contract

### 3.5 Distribution of Customer Payment Methods

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/7cd4b19c-13a9-4bb5-8438-69c3fca839d4)

### 3.6 Customer Churn Analysis by Payment Method

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/d2c641e0-8b7a-4296-84f0-b3190132f1f5)

Observation:
1. Major customers who moved out were having Electronic Check as Payment Method.
2. Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.

### 3.7 Churn Analysis by Internet Service Type and Gender

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/d6a35e59-109f-4d82-8fee-7781559bf31c)

Observation:
1. A lot of customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate, this might suggest a dissatisfaction with this type of internet service.
2. Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service.

### 3.8 Distribution of Churn by Dependents Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/df5833a9-96c0-40d7-8a8a-ed0216118211)

1. Customers without dependents are more likely to churn

### 3.9 Churn Distribution Based on Partnership Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/0ea951b8-778b-436b-862d-25c9c3f0d5b7) 

1. Customers that doesn't have partners are more likely to churn

### 3.10 Churn Distribution Among Senior Citizens vs. Non-Senior Citizens

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/e3aee662-f1ab-4b6c-97db-ebf6bef44c6b)

Observation: 
1. It can be observed that the fraction of senior citizen is very less.
2. Most of the senior citizens churn.

### 3.11 Churn Distribution Based on Online Security Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/032e3193-3128-4412-8c22-65d78e98d6d2)

Observation: 
1. Most customers churn in the absence of online security,

### 3.12 Churn Distribution Based on Paperless Billing Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/f6a9109a-e008-4cca-a18d-b6e1ec7e1d30)

Observation:
1. Customers with Paperless Billing are most likely to churn.

### 3.13 Churn Distribution Based on Tech Support Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/357ea9b4-2944-4de7-9a2c-61d63d11533a)  

Observation: 
1. Customers with no TechSupport are most likely to migrate to another service provider.

### 3.14 Churn Distribution Based on Phone Service Subscription

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/0820e9ab-6aee-4c00-87e8-7c30ecf9b301)

Observation:
1. Very small fraction of customers don't have a phone service and out of that, 1/3rd Customers are more likely to churn.

### 3.15 Density Distribution of Monthly Charges by Churn Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/f1fa9680-c295-470a-b4bf-fdcd9d4cc439)

Observation:
1. Customers with higher Monthly Charges are also more likely to churn

### 3.16 otal Charges: A Comparative Analysis of Churned and Retained Customers

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/899a656c-0066-40c2-9a79-7ef8cb0e960d)

### 3.17 Box Plot of Tenure by Churn Status

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/f551f70b-4b30-41f2-8643-229530e3e50c)

Observation:
1. New customers are more likely to churn

### 3.18 Correlation Heatmap of Features

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/ed15fa89-0e86-4ef7-a398-814036c49d5c)

# 4. Preprocessing of data 

### 4.1 One-Hot Encoding: Transforming Categorical Data into Numerical Format

![image](https://github.com/user-attachments/assets/60e82922-ba25-48e6-b147-63761b01d2af)

We have used one-hot encoding is performed to convert categorical variables into a numerical format. This technique creates binary columns for each category, ensuring that the model treats them as distinct and independent features.

### 4.2 Normalization of output to enhancing Model Performance

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/8da86b52-6b64-4fbe-9be5-644f618c589b)  

Normalization is used in machine learning to scale input features to a similar range, which helps improve the convergence speed of optimization algorithms and enhances the overall performance of models.

# 5. Handling class imbalance using SMOTE

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/5419ef15-4822-47cf-b92b-6ca291d98288)  

SMOTE (Synthetic Minority Over-sampling Technique) is used in churn analysis to address class imbalance by generating synthetic samples for the minority class, which helps improve the model's ability to detect churn cases.

# 6. Model Building and Model Evaluation

### 6.1 Logisitc Regression  

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/4238483b-a50f-42d0-bbf3-e485413e9781)

Observation: 
The logistic regression model demonstrates a good overall accuracy of 77%, with a strong precision of 89% for the negative class (0) but a lower precision of 55% for the positive class (1). The recall for the positive class is relatively high at 74%, indicating that while the model is better at identifying positives, there is room for improvement in reducing false positives.

### 6.2 Random Forest Classifier

![image](https://github.com/user-attachments/assets/b145c5dc-8db7-4798-a563-d13646d18704)

Observation: The random forest model achieves an overall accuracy of 78%, with a precision of 87% for the negative class (0) and 58% for the positive class (1). While the model performs well in identifying the negative class, the relatively low precision and recall for the positive class indicate a need for further optimization to enhance its ability to detect positives effectively.

### 6.3 Gradient Boosting Classifier

![image](https://github.com/user-attachments/assets/dc94c01e-fd48-4b66-9c23-b853feab39f3)

Observation: The gradient boosting model exhibits an overall accuracy of 77%, with a high precision of 90% for the negative class (0) but a lower precision of 55% for the positive class (1). The model shows a commendable recall of 76% for the positive class, indicating its ability to capture many true positives, yet it still needs improvements in precision to minimize false positives effectively.

# 7. Customer Segmentation

![image](https://github.com/user-attachments/assets/5a26693a-1ad4-4510-a9a0-0c507fc0fbea)  

# 8. Model Evaluation (using ROC-AUC)

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/f7c73a2c-6ddb-4af3-94e4-18bcb105858a)  

ROC-AUC is used to evaluate the performance of a binary classification model by measuring its ability to discriminate between positive and negative classes across various threshold settings. It summarizes the trade-off between sensitivity (true positive rate) and specificity (false positive rate), providing a single metric that captures the model's overall effectiveness.

# 9. Important features for Customer churn prediction

![Screenshot 2024-11-05 164828](https://github.com/user-attachments/assets/3b71949a-5c7d-4dc0-bbfb-6a8e3fbffc3f)

# 10. Conclusion 

Customer churn is definitely bad to a firm ’s profitability. Various strategies can be implemented to eliminate customer churn. The best way to avoid customer churn is for a company to truly know its customers. This includes identifying customers who are at risk of churning and working to improve their satisfaction. Improving customer service is, of course, at the top of the priority for tackling this issue. Building customer loyalty through relevant experiences and specialized service is another strategy to reduce customer churn. Some firms survey customers who have already churned to understand their reasons for leaving in order to adopt a proactive approach to avoiding future customer churn.




















































