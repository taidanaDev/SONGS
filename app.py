
# GROUP 10 - SONGS SEARCH PREDICTION
# Import libraries
from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\Ken Bandiez\OneDrive\Desktop\SONGS\preprocessed_songs_dataset (1).csv") 

# Combine Searchable Text
df['search_text'] = (df['Song Title'].fillna("") + " " +
                     df['Name of Artist'].fillna("") + " " +
                     df['Album'].fillna("") + " " +
                     df['Genre'].fillna("")
                    ).astype(str)

# Load model 
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Song embeddings
song_embeddings = joblib.load("song_embeddings.pkl")


@app.route("/")
def home():
    return render_template("index.html")


# SEARCH AND RECOMMENDATIONS
# @app.route("/search", methods=["POST"])
# def search():

#     query = request.form["query"]

#     query_embedding = model.encode([query])

#     similarity = cosine_similarity(query_embedding, song_embeddings).flatten()

#     top_indices = similarity.argsort()[-10:][::-1]

#     # results = df.iloc[top_indices][
#     #     ["Song Title","Name of Artist","Album","Genre"]
#     #     ].to_dict(orient="records")
#     results = df.iloc[top_indices][
#         ["Song Title",
#         "Genre",
#         "Name of Artist",
#         "Album",
#         "Year of Release",
#         "Duration of Song",
#         "Country",
#         "Number of Streams" ]
#     ].to_dict(orient="records")

 
#     recommendations = []

#     if len(top_indices) > 0:

#         best_index = top_indices[0]

#         base_vector = song_embeddings[best_index]

#         rec_similarity = cosine_similarity(
#                 [base_vector],
#                 song_embeddings).flatten()

#         rec_indices = rec_similarity.argsort()[-6:][::-1]

#         # recommendations = df.iloc[rec_indices][
#         #         ["Song Title","Name of Artist","Album","Genre"]
#         #     ].to_dict(orient="records")
#         recommendations = df.iloc[rec_indices][
#         [
#         "Song Title",
#         "Genre",
#         "Name of Artist",
#         "Album",
#         "Year of Release",
#         "Duration of Song",
#         "Country",
#         "Number of Streams"
#         ]
#         ].to_dict(orient="records")

#     return render_template(
#         "index.html",
#         results=results,
#         recommendations=recommendations
#     )

# @app.route("/search", methods=["POST"])
# def search():
#     query = request.form.get("query", "")
#     active_filter = request.form.get("filter", "All") # Capture the filter type

#     query_embedding = model.encode([query])
#     similarity = cosine_similarity(query_embedding, song_embeddings).flatten()
#     top_indices = similarity.argsort()[::-1]
    
#     final_results = []
#     cols = ["Song Title", "Genre", "Name of Artist", "Album", "Year of Release", "Duration of Song", "Country", "Number of Streams"]
    
#     for idx in top_indices:
#         row = df.iloc[idx]
        
#         # Logic for functional filters
#         match = False
#         if active_filter == "All": 
#             match = True
#         elif active_filter == "Genre": 
#             match = query.lower() in str(row['Genre']).lower()
#         elif active_filter == "Artist": 
#             match = query.lower() in str(row['Name of Artist']).lower()
#         elif active_filter == "Year": 
#             match = query in str(row['Year of Release'])
#         elif active_filter == "Country": 
#             match = query.lower() in str(row['Country']).lower()
            
#         if match:
#             final_results.append(row[cols].to_dict())
            
#         if len(final_results) >= 10: break

#     # Recommendations logic (using the first result as base)
#     recommendations = []
#     if final_results:
#         # Find index of the first result in original dataframe to get its embedding
#         first_song_title = final_results[0]['Song Title']
#         base_idx = df[df['Song Title'] == first_song_title].index[0]
#         rec_sim = cosine_similarity([song_embeddings[base_idx]], song_embeddings).flatten()
#         rec_indices = rec_sim.argsort()[-7:][::-1]
#         recommendations = df.iloc[rec_indices][cols].to_dict(orient="records")

#     return render_template("index.html", results=final_results, recommendations=recommendations, last_query=query, last_filter=active_filter)

# @app.route("/search", methods=["POST"])
# def search():
#     query = request.form.get("query", "")
#     active_filter = request.form.get("filter", "All")

#     query_embedding = model.encode([query])
#     similarity = cosine_similarity(query_embedding, song_embeddings).flatten()
#     top_indices = similarity.argsort()[::-1]
    
#     final_results = []
#     cols = ["Song Title", "Genre", "Name of Artist", "Album", "Year of Release", "Duration of Song", "Country", "Number of Streams"]
    
#     for idx in top_indices:
#         row = df.iloc[idx]
#         match = False
#         if active_filter == "All": match = True
#         elif active_filter == "Genre": match = query.lower() in str(row['Genre']).lower()
#         elif active_filter == "Artist": match = query.lower() in str(row['Name of Artist']).lower()
#         elif active_filter == "Year": match = query in str(row['Year of Release'])
#         elif active_filter == "Country": match = query.lower() in str(row['Country']).lower()
            
#         if match:
#             final_results.append(row[cols].to_dict())
#         if len(final_results) >= 10: break

#     recommendations = []
#     if final_results:
#         first_song = final_results[0]['Song Title']
#         base_idx = df[df['Song Title'] == first_song].index[0]
#         rec_sim = cosine_similarity([song_embeddings[base_idx]], song_embeddings).flatten()
#         rec_indices = rec_sim.argsort()[-12:-2][::-1] # Skip the exact matches to keep it fresh
#         recommendations = df.iloc[rec_indices][cols].to_dict(orient="records")

#     return render_template("index.html", 
#                            results=final_results, 
#                            recommendations=recommendations, 
#                            last_query=query, 
#                            last_filter=active_filter)

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    active_filter = request.form.get("filter", "All")

    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, song_embeddings).flatten()
    top_indices = similarity.argsort()[::-1]
    
    final_results = []
    cols = ["Song Title", "Genre", "Name of Artist", "Album", "Year of Release", "Duration of Song", "Country", "Number of Streams"]
    
    for idx in top_indices:
        row = df.iloc[idx]
        
        match = False
        if active_filter == "All": 
            match = True
        elif active_filter == "Genre": 
            match = query.lower() in str(row['Genre']).lower()
        elif active_filter == "Artist": 
            match = query.lower() in str(row['Name of Artist']).lower()
        elif active_filter == "Year": 
            match = query in str(row['Year of Release'])
        elif active_filter == "Country": 
            match = query.lower() in str(row['Country']).lower()
            
        if match:
            song_data = row[cols].to_dict()
            # APPLY SENTENCE CASE
            song_data['Song Title'] = str(song_data['Song Title']).capitalize()
            song_data['Name of Artist'] = str(song_data['Name of Artist']).capitalize()
            final_results.append(song_data)
        
        if len(final_results) >= 10: break

    # Move exact match to top
    final_results.sort(key=lambda x: x['Song Title'].lower() == query.lower(), reverse=True)

    recommendations = []
    if final_results:
        # Get the original title for index lookup (before capitalization)
        # Or simply use the semantic results index
        base_title = final_results[0]['Song Title']
        # Lookup original index to get embedding
        # We use a case-insensitive match to find the original row
        orig_row = df[df['Song Title'].str.lower() == base_title.lower()]
        
        if not orig_row.empty:
            base_idx = orig_row.index[0]
            rec_sim = cosine_similarity([song_embeddings[base_idx]], song_embeddings).flatten()
            rec_indices = rec_sim.argsort()[-10:-2][::-1] 
            
            # Prepare recommendations with SENTENCE CASE
            for r_idx in rec_indices:
                r_row = df.iloc[r_idx][cols].to_dict()
                r_row['Song Title'] = str(r_row['Song Title']).capitalize()
                r_row['Name of Artist'] = str(r_row['Name of Artist']).capitalize()
                recommendations.append(r_row)

    return render_template("index.html", 
                           results=final_results, 
                           recommendations=recommendations, 
                           last_query=query, 
                           last_filter=active_filter)
# AUTOCOMPLETE 
@app.route("/autocomplete")
def autocomplete():

    query = request.args.get("q", "").lower()

    if not query:
        return jsonify({"suggestions": []})

    song_matches = df[
        df["Song Title"].str.lower().str.contains(query)
    ]["Song Title"].unique().tolist()

    artist_matches = df[
        df["Name of Artist"].str.lower().str.contains(query)
    ]["Name of Artist"].unique().tolist()

    genre_matches = df[
        df["Genre"].str.lower().str.contains(query)
    ]["Genre"].unique().tolist()

    album_matches = df[
        df["Album"].str.lower().str.contains(query)
    ]["Album"].unique().tolist()

    suggestions = list(set(
        song_matches +
        artist_matches +
        genre_matches +
        album_matches
    ))

    return jsonify({"suggestions": suggestions[:10]})


if __name__ == "__main__":
    app.run(debug=True)