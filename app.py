import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd


# Load the trained model
model = tf.keras.models.load_model("C:/Users/kl/Desktop/Test project/novel_recommender_model.h5")  # Ensure model.h5 is in the same directory

# Load book dataset (modify this path based on your data)
books_df = pd.read_csv("C:/Users/kl/Desktop/Test project/books_cleaned.csv")  # Make sure this contains book titles and IDs

# Shopify API details
SHOPIFY_STORE_URL = "https://r0ydah-w3.myshopify.com/api/2023-10/graphql.json"
SHOPIFY_ACCESS_TOKEN = "a221a23fbb36e3ec9a15e4e64d492b98"

app = Flask(__name__)

# Function to get book recommendations
def recommend_books(book_title):
    # Convert book title to model input (Modify this part based on your model's requirements)
    input_data = np.array([book_title])  # Example: You might need to convert it into embeddings
    predictions = model.predict(input_data)

    # Get top 5 recommended book IDs (Modify based on your output format)
    recommended_indices = predictions.argsort()[0][-5:][::-1]
    recommended_books = books_df.iloc[recommended_indices]['title'].tolist()

    return recommended_books


# Function to fetch book details from Shopify
def fetch_book_details(book_titles):
    query = """
    query GetBooks($bookTitles: [String!]) {
      products(first: 5, query: $bookTitles) {
        edges {
          node {
            id
            title
            description
            images(first: 1) { edges { node { url } } }
            priceRange { minVariantPrice { amount currencyCode } }
          }
        }
      }
    }
    """
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": SHOPIFY_ACCESS_TOKEN
    }
    variables = {"bookTitles": book_titles}
    
    response = requests.post(SHOPIFY_STORE_URL, json={"query": query, "variables": variables}, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("data", {}).get("products", {}).get("edges", [])
    else:
        return []

# API endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Get the JSON data from the request
        data = request.get_json()  # This is a safer way to parse JSON

        if not data or 'title' not in data:
            return jsonify({"error": "No book title provided"}), 400
        
        book_title = data['title']
        recommendations = recommend_books(book_title)
        return jsonify({"recommended_books": recommendations})

    except Exception as e:
        return jsonify({"error": f"Failed to decode JSON: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
