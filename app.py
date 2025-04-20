from flask import Flask, request, jsonify
from recommender import DeepFMRecommender

app = Flask(__name__)
recommender = DeepFMRecommender()
recommender.load_model()

@app.route('/')
def home():
    return "🎬 DeepFM Movie Recommender API is running!"

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    mood = request.args.get('mood', type=str, default=None)

    if user_id is None:
        return jsonify({'error': 'Missing user_id'}), 400

    recommendations = recommender.recommend(user_id_raw=user_id, mood=mood)
    return jsonify({'recommendations': recommendations})

if __name__ == "__main__":
    app.run(debug=True)
