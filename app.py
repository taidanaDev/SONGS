
# Import libraries
from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("songs_dataset.csv")

# Load trained prediction model
model = joblib.load("song_model.pkl")

# Load encoders
encoders = joblib.load("label_encoders.pkl")

# Load TF-IDF vector search models
vectorizer = joblib.load("vectorizer.pkl")
song_vectors = joblib.load("song_vectors.pkl")

# KNN RECOMMENDATION 
features = df[['Genre', 'Duration of Song', 'Year of Release']].copy()

features['Genre'] = encoders['Genre'].transform(df['Genre'])

knn = NearestNeighbors(n_neighbors=6, algorithm='auto')
knn.fit(features)


@app.route("/")
def home():
    return render_template("index.html")


# SEARCH (TF-IDF SIMILARITY)
@app.route("/search", methods=["POST"])
def search():

    query = request.form["query"].lower()

    query_vec = vectorizer.transform([query])

    similarity = cosine_similarity(query_vec, song_vectors)

    top_indices = similarity.argsort()[0][-5:][::-1]

    results = df.iloc[top_indices].to_dict(orient="records")

    recommendations = []

    if len(top_indices) > 0:

        idx = top_indices[0]

        distances, indices = knn.kneighbors(features.iloc[[idx]])

        recommendations = df.iloc[indices[0][1:]][
            ['Song Title','Name of Artist','Album','Genre']
        ].to_dict(orient="records")


    return render_template(
        "index.html",
        results=results,
        recommendations=recommendations
    )


# STREAM PREDICTION 
@app.route("/predict", methods=["POST"])
def predict():

    try:

        genre = request.form["genre"]
        duration = float(request.form["duration"])
        year = int(request.form["year"])

        genre_encoded = encoders["Genre"].transform([genre])[0]

        prediction_log = model.predict([[genre_encoded, duration, year]])

        prediction_streams = int(np.expm1(prediction_log[0]))

        prediction = f"{prediction_streams:,}"

    except Exception as e:

        prediction = f"Error in calculation: {str(e)}"

    return render_template("index.html", prediction=prediction)


# AUTOCOMPLETE 
@app.route("/autocomplete")
def autocomplete():

    query = request.args.get("q", "").lower()

    if not query:
        return {"suggestions": []}

    song_matches = df[
        df["Song Title"].str.lower().str.contains(query)
    ]["Song Title"].unique().tolist()

    artist_matches = df[
        df["Name of Artist"].str.lower().str.contains(query)
    ]["Name of Artist"].unique().tolist()

    genre_matches = df[
        df["Genre"].str.lower().str.contains(query)
    ]["Genre"].unique().tolist()

    all_suggestions = list(set(song_matches + artist_matches + genre_matches))

    return {"suggestions": all_suggestions[:10]}


if __name__ == "__main__":
    app.run(debug=True)