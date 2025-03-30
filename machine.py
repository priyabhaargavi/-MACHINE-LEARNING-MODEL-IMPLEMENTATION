import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
#database
data=data = """v1,v2
ham,"Hey, how are you doing?"
spam,"Claim your free prize now! Visit: http://spam-prize.com"
ham,"Let's meet up for coffee tomorrow."
spam,"URGENT: Your account has been compromised. Reset your password immediately: http://spam-account.com"
ham,"It was great seeing you yesterday. Let's do it again soon!"
spam,"Limited time offer: Get a free iPhone. Click here: http://spam-iphone.com"
"""
# Save the dataset to a CSV file using UTF-8 encoding
with open("small_spam.csv", "w", encoding="utf-8") as f:
    f.write(data)
print("Small dataset saved successfully.")
# Load the dataset with UTF-8 encoding
df=pd.read_csv('small_spam.csv',encoding='utf-8', quotechar='"')
pd.set_option('display.max_colwidth', None)
# Preview the dataset
print("Dataset preview:")
print(df.head())
# Selecting relevant columns and renaming them
df=df[['v1','v2']]  # 'v1' is the label ('ham'/'spam'), 'v2' is the message
df.columns=['label', 'message']
# Convert 'ham' to 0 and 'spam' to 1 in the 'label' column
df['label']=df['label'].map({'ham': 0, 'spam': 1})
# Handle missing values (if any) by replacing NaNs with an empty string
df['message']=df['message'].fillna('')
# Remove any rows where the message column is empty
df=df[df['message'].str.strip() != '']
# Display the number of samples remaining after cleaning
print(f"Number of samples remaining after cleaning: {len(df)}")
# Check if there are enough samples to split
if len(df)>1:
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    # Convert text messages into numerical features using TF-IDF Vectorizer
    vectorizer=TfidfVectorizer(stop_words='english', min_df=1)  # Allow rare words to be included
    X_train_tfidf=vectorizer.fit_transform(X_train)
    X_test_tfidf=vectorizer.transform(X_test)
    # Train a Random Forest Classifier model
    model=RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    # Predict on the test set
    y_pred=model.predict(X_test_tfidf)
    # Evaluate the model performance
    accuracy=accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    # Display Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Plot the Confusion Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    # Save the trained model for future use (optional)
    joblib.dump(model, 'spam_classifier_model.pkl')
    print("\nModel has been saved as 'spam_classifier_model.pkl'.")
else:
    print("Not enough samples to split. Please check the dataset and preprocessing steps.")