# recommendation_system

# **Project Goal**
To build a hybrid movie recommender system that:

* Combines collaborative filtering and content-based filtering
* Stores and utilizes user history
* Uses DeepFM for learning feature interactions
* Adapts recommendations based on the user’s mood
* Exposes functionality through a Flask API

                    ┌────────────────────┐
                    │  CSV Movie Dataset │
                    └────────┬───────────┘
                             ▼
                ┌────────────────────────┐
                │   Feature Encoding     │
                │ - User/Movie IDs       │
                │ - Genre Binarization   │
                └────────┬───────────────┘
                         ▼
                ┌────────────────────────┐
                │    DeepFM Model (Keras)│
                │ - Embedding Layers     │
                │ - DNN + FM Components  │
                └────────┬───────────────┘
                         ▼
         ┌────────────────────────────────────┐
         │ Flask API                          │
         │ /recommend?user_id=X&mood=sad      │
         └────────────────┬───────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │  Recommendations Based on:         │
         │  - Ratings (Collaborative)         │
         │  - Genres (Content-Based)          │
         │  - Mood filtering (Personalized)   │
         └────────────────────────────────────┘
