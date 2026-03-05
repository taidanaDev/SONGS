# # Import Libraries
# import pandas as pd
# import joblib
# import numpy as np

# from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load dataset
# data = pd.read_csv('songs_dataset.csv')
# print("Data shape:", data.shape)

# # Encoders
# encoders = {}

# for column in ['Genre', 'Name of Artist', 'Album', 'Country']:
    
#     encoder = LabelEncoder()
    
#     data[column] = encoder.fit_transform(data[column])
    
#     encoders[column] = encoder

# # Define features and target variable
# x = data[['Genre', 
#         #   'Name of Artist',
#         #   'Album',
#         #   'Country',
#           'Duration of Song',
#           'Year of Release']]

# y = np.log1p(data['Number of Streams'])

# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Train the Decision Tree Regressor
# # model = DecisionTreeRegressor(max_depth=5, random_state=42)
# model = LinearRegression()

# # model = RandomForestRegressor(
# #     n_estimators=300,
# #     max_depth=10,
# #     random_state=42
# # )
# model.fit(x_train, y_train)
# # Make predictions on the test set
# predictions = model.predict(x_test)
# predictions = np.expm1(predictions)
# y_test = np.expm1(y_test)

# # Evaluate Model
# r2 = r2_score(y_test, predictions)
# mae = mean_absolute_error(y_test, predictions)

# print("\nModel Evaluation")
# print("--------------------------")
# print("R2 Score:", r2)
# print("Mean Absolute Error:", mae)

# vibe_vectorizer = TfidfVectorizer(max_features=10)
# vibe_vectorizer.fit(data['search_text'])

# joblib.dump(model, "song_model.pkl")
# joblib.dump(encoders, "label_encoders.pkl")
# joblib.dump(vibe_vectorizer, "vibe_vectorizer.pkl")

# print("\nModel saved successfully!")

# print("\nModel and encoders saved successfully!")


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