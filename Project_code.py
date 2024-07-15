#!/usr/bin/env python
# coding: utf-8

# In[478]:


# Accessing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve,auc,adjusted_rand_score, normalized_mutual_info_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from dmba import plotDecisionTree
from sklearn.naive_bayes import MultinomialNB
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import MinMaxScaler


# In[479]:


# Read the main table
CRSData = pd.read_excel('train.xlsx', sheet_name='train_5K')
CRSData.round(2).head(5)  # Display the first few rows of the dataframe


# In[480]:


CRSData.isnull().sum()


# In[481]:


# Dropping irrelevant columns
CRSData.drop(['Month' , 'Type_of_Loan', 'Credit_History_Age', 'SSN','Credit_Mix'], axis=1, inplace=True)


# In[482]:


#Treating missing values
CRSData['Num_of_Delayed_Payment'].fillna(CRSData['Num_of_Delayed_Payment'].median(), inplace=True)
CRSData['ChangedCreditLimit'].fillna(CRSData['ChangedCreditLimit'].median(), inplace=True)
CRSData['Num_Credit_Inquiries'].fillna(CRSData['Num_Credit_Inquiries'].median(), inplace=True)
CRSData['Amount_invested_monthly'].fillna(CRSData['Amount_invested_monthly'].mean(), inplace=True)
CRSData['Monthly_Balance'].fillna(CRSData['Monthly_Balance'].median(), inplace=True)


# In[483]:


#Removing missing values from payment behaviour
CRSclean_df = CRSData.dropna(subset=['Payment_Behaviour'])
CRSclean_df.isnull().sum()


# In[484]:


CRSclean_df.dtypes


# In[485]:


CRSclean_df.describe().round(2)


# In[486]:


#Removing 'NM' values in the column 'Payment of Min Amount'
CRSclean_df = CRSclean_df[CRSclean_df['Payment_of_Min_Amount'] != 'NM']


# In[487]:


numeric_cols = CRSclean_df.select_dtypes(exclude = "object").columns
cat_cols = CRSclean_df.select_dtypes(include = "object").columns
print(numeric_cols)
print(cat_cols)


# In[488]:


#Checking Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_df = CRSclean_df[numeric_cols]
vif_data = pd.DataFrame({
    "feature": vif_df.columns,
    "VIF": [variance_inflation_factor(vif_df.values, i) for i in range(len(vif_df.columns))]
})
print(vif_data.head(17).round(2))


# In[489]:


#Shows few variables as High multicollinearity, indicating that the predictor is highly correlated with other predictors.
#We will use feature selection for this at the later steps


# In[490]:


plt.figure(figsize= (11,11))
sns.heatmap(CRSclean_df[numeric_cols].corr().round(2),annot=True)


# In[491]:


#Visualising boxplot
# Set the number of rows and columns for the subplots grid
num_cols = 3  # Number of columns in the subplot grid
num_rows = int(np.ceil(len(numeric_cols) / num_cols))  # Calculate the number of rows needed
# Set the figure size for better visibility
plt.figure(figsize=(20, num_rows * 5))
# Create a boxplot for each numerical column
for i, col in enumerate(numeric_cols):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(y=CRSclean_df[col])
    plt.title(col)
    plt.xlabel('')
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()


# In[492]:


#Replacing outlier with median

CRS_o_df = CRSclean_df[numeric_cols].copy()

for col in numeric_cols:
    # Calculate the 0.05th and 99.95th percentiles
    Q1 = np.percentile(CRS_o_df[col], 0.05, interpolation='midpoint')
    Q3 = np.percentile(CRS_o_df[col], 99.95, interpolation='midpoint')
    median = CRS_o_df[col].median()
# Replace outliers with the median
CRS_o_df[col] = np.where((CRS_o_df[col] < Q1) | (CRS_o_df[col] > Q3), median,CRS_o_df[col])
CRS_o_df = CRS_o_df.round(4)

# Display the first few rows of the cleaned DataFrame
print(CRS_o_df)


# In[493]:


#Visualisation


# In[494]:


#Proportion of credit score
credit_score_counts =CRSclean_df['Credit_Score'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(credit_score_counts, labels=credit_score_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Credit Score Distribution')
plt.axis('equal') 
plt.show()


# In[495]:


#Proportion of occupation
Occupation_counts =CRSclean_df['Occupation'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(Occupation_counts, labels=Occupation_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('OccupationDistribution')
plt.axis('equal') 
plt.show()


# In[496]:


#Proportion of Payment Bheaviour
PB_counts =CRSclean_df['Payment_Behaviour'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(PB_counts, labels=PB_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Payment Behaviour Distribution')
plt.axis('equal') 
plt.show()


# In[497]:


#Number of credit card by credit score
credit_card_counts = CRSclean_df.groupby('Credit_Score')['Num_Credit_Card'].count()
plt.figure(figsize=(10, 6))
credit_card_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Num_Credit_Card by Credit_Score')
plt.xlabel('Credit Score')
plt.ylabel('Count of Num_Credit_Card')
plt.xticks(rotation=0)
plt.show()


# In[498]:


average_income_by_occupation = CRSclean_df.groupby('Occupation')['AnnualIncome'].mean()
plt.figure(figsize=(12, 6))
average_income_by_occupation.plot(kind='bar', color='skyblue')
plt.title('Average Annual Income by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Average Annual Income')
plt.xticks(rotation=45)
plt.show()


# In[499]:


#Occupaton by payment of min amount
fig = plt.figure(figsize= (17,9))
sns.countplot(data=CRSclean_df,x="Occupation",hue="Credit_Score")


# In[500]:


#Occupaton by payment of min amount
fig = plt.figure(figsize= (20,12))
sns.countplot(data=CRSclean_df,x="Payment_Behaviour",hue="Credit_Score")


# In[501]:


#Occupaton by payment of min amount
fig = plt.figure(figsize= (17,9))
sns.countplot(data=CRSclean_df,x="Occupation",hue="Payment_of_Min_Amount")


# In[502]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Scaled_data = scaler.fit_transform(CRS_o_df)


# In[503]:


column_names = ['Age', 'AnnualIncome', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
       'Num_Credit_Card', 'Interest_Rate', 'NumofLoan', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'ChangedCreditLimit', 'Num_Credit_Inquiries',
       'OutstandingDebt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Monthly_Balance']
scaled_df = pd.DataFrame(Scaled_data, columns=column_names)
scaled_df = scaled_df.round(4)
print(scaled_df)


# In[504]:


cat_df = CRSclean_df[cat_cols].copy()
print(cat_df)


# In[505]:


scaled_df.reset_index(drop=True, inplace=True)
cat_df.reset_index(drop=True, inplace=True)
combined_df = pd.concat([scaled_df, cat_df], axis=1)
combined_df = combined_df.round(4)
print(combined_df)


# In[506]:


combined_df['Credit_Score'].replace({"Poor":0, "Standard":1, "Good":2}, inplace=True)
combined_df['Payment_of_Min_Amount'].replace({"Yes":1, "No":0}, inplace=True)
combined_df = pd.get_dummies(combined_df, columns = ['Occupation', 'Payment_Behaviour'])
print(combined_df)


# In[507]:


#Feature selection


# In[508]:


X = combined_df.drop(['Credit_Score','ID','Customer_ID'] , axis=1)  # Features
y = combined_df['Credit_Score'] 


# In[509]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[510]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# In[511]:


rf_classifier.fit(X_train, y_train)


# In[512]:


#  want to see the importance of each feature
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
# Round the feature importances to 2 decimal places and sort them in descending order
rounded_feature_importances = feature_importances.round(4).sort_values(ascending=False)
# Print the rounded and sorted feature importances
print(rounded_feature_importances)


# In[513]:


#Going forward with these feature : OutstandingDebt, Interest_Rate ,Delay_from_due_date, 
#Num_of_Delayed_Payment, ChangedCreditLimit, Monthly_Balance, Credit_Utilization_Ratio, Amount_invested_monthly, Num_Credit_Inquiries, AnnualIncome


# In[514]:


#Graph for feature selection
plt.figure(figsize=(20, 6))
plt.plot(feature_importances.index, feature_importances.sort_values(ascending=False), marker='o', linestyle='-', color='b')
plt.title('feature_importances')
plt.xlabel('Index')
plt.ylabel('Column Name')
plt.grid(True)
plt.show()


# In[515]:


#Specifying x and Y
X1 = combined_df[['OutstandingDebt', 'Interest_Rate' ,'Delay_from_due_date', 'Num_of_Delayed_Payment', 'ChangedCreditLimit', 'Monthly_Balance', 'Credit_Utilization_Ratio', 'Amount_invested_monthly', 'Num_Credit_Inquiries', 'AnnualIncome']]
y1 = combined_df['Credit_Score']


# In[516]:


X1_train,X1_test,y1_train,y1_test= train_test_split(X1, y1, test_size=0.2, random_state=42)


# In[517]:


#MODEL ANALYSIS


# 1. Fitting KNN Model

# In[518]:


#Fitting the KNN model with three nearset neigbhours
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X1_train,np.ravel(y1_train))


# In[519]:


# Predicting on validation data
knn_pred=knn.predict(X1_test)


# In[520]:


#checking the accuracy score
accuracy=accuracy_score(y1_test,knn_pred)
print(f'accuracy:{accuracy}')


# In[521]:


# Train a classifier for different values of k
results = []
for k in range(1, 15):
     knn = KNeighborsClassifier(n_neighbors=k).fit(X1_train, y1_train)
     results.append({
    'k': k,
     'accuracy': accuracy_score(y1_test, knn.predict(X1_test))
})


# In[522]:


# Convert results to a pandas data frame
results = pd.DataFrame(results).round(4)
print(results)


# In[523]:


plt.plot(results["k"],results["accuracy"])


# K=9 is the optimal choice of nearest neighbours since its provides highest accuracy i.e. 63%

# In[524]:


# Tuning the Model by taking K=9
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X1_train,y1_train)
knn_pred=knn.predict(X1_test)
accuracy_1=accuracy_score(y1_test,knn_pred)
print("Accuracy of the KNN Model where k=9 is: ",round(results["accuracy"].max()*100,2),"%")


# 2. Decision Tree

# In[525]:


# fitthing Decision tree 
Tree=DecisionTreeClassifier(max_depth=2)
Tree.fit(X1_train,y1_train)


# In[526]:


# making prediction on test data
Tree_pred=Tree.predict(X1_test)
print(Tree_pred)


# In[527]:


accuracy_2=accuracy_score(y1_test,Tree_pred)
print(accuracy_2)


# In[528]:


# Graph
fig=plt.figure(figsize=(20,20))
_=tree.plot_tree(Tree,feature_names=X1.columns,filled=True)


# 3. Random Forest

# In[529]:


# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X1_train, y1_train)


# In[530]:


# Make predictions on the test set
y_pred = clf.predict(X1_test)


# In[531]:


# Evaluate the model
accuracy_3 = accuracy_score(y1_test, y_pred)
print(f'Accuracy: {accuracy_3:.2f}')


# In[532]:


# Print detailed classification report
print(classification_report(y1_test, y_pred))


# In[533]:


# Compute and plot confusion matrix
cm = confusion_matrix(y1_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1','Predicted 2'], yticklabels=['Actual 0', 'Actual 1','Actual 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# 4. Hierarchical Clustering

# In[534]:


# perform Hierarchical Clustering
Z = linkage(X1_train, method='ward')  # Other methods: 'single', 'complete', 'average', 'centroid', etc.


# accuracy = accuracy_score(y1_train, clusters)
# print(f'Clustering accuracy: {accuracy:.2f}')

# In[535]:


# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=X1_train.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()


# In[536]:


# Cut the dendrogram to form flat clusters
# The threshold can be set to determine the number of clusters
# Here, 't' is the threshold and 'criterion' specifies how the threshold is applied
clusters = fcluster(Z, t=3, criterion='maxclust')  # Form 3 clusters


# In[537]:


# Evaluate the Clustering
ari = adjusted_rand_score(y1_train, clusters)
nmi = normalized_mutual_info_score(y1_train, clusters)

print(f'Adjusted Rand Index: {ari:.2f}')
print(f'Normalized Mutual Information: {nmi:.2f}')


# #Interpretation: An ARI of 0.13 indicates a slight positive correlation between the clustering results and the true labels, but the clustering is not very effective. The clusters found by the hierarchical clustering algorithm do not align well with the actual classes.
# 
# An NMI of 0.15 indicates that there is some mutual information between the clustering results and the true labels, but it is quite low. This means that the clusters share some information with the actual classes, but overall, the clustering does not capture the true class structure well.

# In[538]:


# Add the cluster labels to the original training data
df_train = X_train.copy()
df_train['Cluster'] = clusters
print(df_train)


# 5. Naive Bayes

# In[539]:


# Assumptions: 1. Assume Independence of predictor Varaiable. 2. Requires Categorical Variable.


# In[540]:


# converting numerical data to categorical
X2_train=X1_train.astype('category')
y2_train=y1_train.astype('category')

X2_test=X1_test.astype('category')
y2_test=y1_test.astype('category')


# #Since the dataset contains negative values, we need to standardised the dataset by MinMaxScaler to perform Naive Bayes.

# In[541]:


scaler = MinMaxScaler(feature_range=(0, 1))
X2_train_scaled = scaler.fit_transform(X2_train)
X2_test_scaled = scaler.fit_transform(X2_test)


# In[542]:


nb=MultinomialNB(alpha=0.01)
nb.fit(X2_train_scaled,y2_train)


# In[543]:


# predict probabilities (Shows the belonging probabilities of each record to which class)
predProb_train = nb.predict_proba(X2_train_scaled)
print(predProb_train)
predProb_test = nb.predict_proba(X2_test_scaled)
print(predProb_test)


# In[544]:


# predict class membership (shows the class instead of probability by selecting the class with highest probability)
y_test_pred = nb.predict(X2_test_scaled)
print(y_test_pred)
y_train_pred = nb.predict(X2_train_scaled)
print(y_train_pred)


# In[545]:


accuracy_4 = accuracy_score(y2_test, y_test_pred)
print("Accuary is",accuracy_4)


# Confusion Matrix
conf_matrix = confusion_matrix(y2_test, y_test_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y2_test, y_test_pred)
print('Classification Report:')
print(class_report)


# In[546]:


#Comparison between different models
import matplotlib.pyplot as plt
model_names = ['KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes','Hierarchical Clustering']
accuracy_scores = [0.6281, 0.6046,  0.75, 0.52,0.12]
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'red', 'purple','orange'])
plt.title('Accuracy Scores of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')

# Display the accuracy scores on top of the bars
for i, score in enumerate(accuracy_scores):
    plt.text(i, score + 0.01, f'{score:.2f}', ha='center')
plt.show()


# In[ ]:




