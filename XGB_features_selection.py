import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("/content/drive/MyDrive/a new RNA m6A/dataset/main data for merged/train alll merged.csv")
X = df1.drop('Target', axis=1)
y = df1['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
feature_importances = xgb.feature_importances_ # Get feature importance scores

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}) # Create a DataFrame to store feature importance scores
k = 100  select
selected_features = feature_importance_df.head(k)['Feature']
selected_features_df = df1[selected_features] # selected features

selected_features_df['Target'] = df1['Target'] # adding lebels 1,0
selected_features_df.to_csv("/content/drive/MyDrive/a new RNA m6A/dataset/main data for merged/100-XGB-Features.csv", index=False)
#CSV file created
#-------------------------------------------
#at first take all the features
#then select XGB for selection
#then used evaluation
#get 100 important features
#creates a csv format numerical data