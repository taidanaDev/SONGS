# Import Libraries
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("songs_dataset.csv")

print("Dataset shape:", data.shape)

# Encode categorical features
encoders = {}

for column in ['Genre','Name of Artist','Album','Country']:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(data[column])

    encoders[column] = encoder


# Features for prediction
X = data[['Genre','Duration of Song','Year of Release']]

# Target
y = np.log1p(data['Number of Streams'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

predictions = np.expm1(predictions)
y_test = np.expm1(y_test)

# Evaluate
print("\nModel Evaluation")
print("----------------")

print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))


# -------- TFIDF VECTOR SEARCH --------
data['search_text'] = data['search_text'].fillna("").astype(str)
vectorizer = TfidfVectorizer(max_features=50)

song_vectors = vectorizer.fit_transform(data["search_text"])


# Save everything
joblib.dump(model,"song_model.pkl")
joblib.dump(encoders,"label_encoders.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
joblib.dump(song_vectors,"song_vectors.pkl")

print("\nModels saved successfully")
