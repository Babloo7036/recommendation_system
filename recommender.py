import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import pickle
import os

class DeepFMRecommender:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.movie_enc = LabelEncoder()
        self.genre_binarizer = MultiLabelBinarizer()
        self.model = None
        self.movies = None

    def load_data(self):
        ratings = pd.read_csv("data/ratings.csv")
        movies = pd.read_csv("data/movies.csv")
        ratings = ratings[['userId', 'movieId', 'rating']]
        movies = movies[['movieId', 'title', 'genres']]

        # Encode users and movies
        ratings['user'] = self.user_enc.fit_transform(ratings['userId'])
        ratings['movie'] = self.movie_enc.fit_transform(ratings['movieId'])

        # Process genres
        movies['genres'] = movies['genres'].str.split('|')
        self.genre_binarizer.fit(movies['genres'])
        genre_encoded = self.genre_binarizer.transform(movies['genres'])

        movies['movie'] = self.movie_enc.transform(movies['movieId'])
        genre_df = pd.DataFrame(genre_encoded, columns=self.genre_binarizer.classes_)
        movie_features = pd.concat([movies[['movie']], genre_df], axis=1)

        self.movies = movies
        self.movie_features = movie_features.set_index('movie')

        # Merge with ratings
        merged = ratings.merge(self.movie_features, left_on='movie', right_index=True)

        X_user = merged['user'].values
        X_movie = merged['movie'].values
        X_genres = merged[self.genre_binarizer.classes_].values
        y = merged['rating'].values

        return X_user, X_movie, X_genres, y

    def build_model(self, num_users, num_movies, num_genres):
        user_input = tf.keras.layers.Input(shape=(1,))
        movie_input = tf.keras.layers.Input(shape=(1,))
        genre_input = tf.keras.layers.Input(shape=(num_genres,))

        user_embed = tf.keras.layers.Embedding(num_users, 16)(user_input)
        movie_embed = tf.keras.layers.Embedding(num_movies, 16)(movie_input)

        user_vec = tf.keras.layers.Flatten()(user_embed)
        movie_vec = tf.keras.layers.Flatten()(movie_embed)

        concat = tf.keras.layers.Concatenate()([user_vec, movie_vec, genre_input])

        x = tf.keras.layers.Dense(128, activation='relu')(concat)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=[user_input, movie_input, genre_input], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self):
        X_user, X_movie, X_genres, y = self.load_data()

        num_users = len(self.user_enc.classes_)
        num_movies = len(self.movie_enc.classes_)
        num_genres = len(self.genre_binarizer.classes_)

        self.model = self.build_model(num_users, num_movies, num_genres)
        self.model.fit([X_user, X_movie, X_genres], y, epochs=5, batch_size=64)

        # Save model and encoders
        self.model.save("model/deepfm_model.h5")
        with open("model/encoders.pkl", "wb") as f:
            pickle.dump((self.user_enc, self.movie_enc, self.genre_binarizer), f)

    def load_model(self):
        self.model = tf.keras.models.load_model("model/deepfm_model.h5")
        with open("model/encoders.pkl", "rb") as f:
            self.user_enc, self.movie_enc, self.genre_binarizer = pickle.load(f)
        self.movies = pd.read_csv("data/movies.csv")
        self.movies['genres'] = self.movies['genres'].str.split('|')
        self.movies['movie'] = self.movie_enc.transform(self.movies['movieId'])
        genre_encoded = self.genre_binarizer.transform(self.movies['genres'])
        genre_df = pd.DataFrame(genre_encoded, columns=self.genre_binarizer.classes_)
        self.movie_features = pd.concat([self.movies[['movie']], genre_df], axis=1).set_index('movie')

    def recommend(self, user_id_raw, mood=None, top_n=10):
        if user_id_raw not in self.user_enc.classes_:
            return []

        user = self.user_enc.transform([user_id_raw])[0]
        movie_ids = self.movies['movie'].values
        genre_features = self.movie_features.loc[movie_ids][self.genre_binarizer.classes_].values

        users = np.full_like(movie_ids, user)

        preds = self.model.predict([users, movie_ids, genre_features], verbose=0).flatten()
        top_indices = preds.argsort()[-top_n:][::-1]

        rec_movies = self.movies.iloc[top_indices][['title', 'genres']]

        if mood:
            rec_movies = self.filter_by_mood(rec_movies, mood)

        return rec_movies.to_dict(orient='records')

    def filter_by_mood(self, df, mood):
        if mood == 'sad':
            return df[df['genres'].apply(lambda g: any(x in g for x in ['Comedy', 'Family']))]
        elif mood == 'happy':
            return df[df['genres'].apply(lambda g: any(x in g for x in ['Action', 'Adventure']))]
        return df
    