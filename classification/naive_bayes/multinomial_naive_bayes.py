# Data Preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, f1_score

spy_df = pd.read_csv("../../data/finance/csv_files/yearly_spy_data.csv")
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df['year'] = spy_df['Date'].dt.year
spy_df_extracted = spy_df[['year', 'Close']]

crime_df = pd.read_csv("../../data/crime/estimated_crimes_1979_2022.csv")
total_crime_df = crime_df[pd.isna(crime_df['state_abbr'])]
total_crime_extracted = total_crime_df[['year', 'violent_crime', 'homicide']]

merged_df_spy = pd.merge(total_crime_extracted, spy_df_extracted, on='year', how='inner')

scaler = MinMaxScaler()
features = merged_df_spy.iloc[:, :-1]
features_scaled = scaler.fit_transform(features)

normalizer = Normalizer()
target = merged_df_spy.iloc[:, -1:]
target = normalizer.fit_transform(target)

# Splitting into training and test
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2)

# Define and fit the Multinomial Naive Bayes model
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1-Score:", f1)
