#Import packages
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Connect to Spotify's API and retrieve top N songs
load_dotenv()
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")
username = os.getenv("username")
redirect_uri = os.getenv("redirect_uri")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-top-read", username=username, 
                                               client_id=client_id, client_secret=client_secret, 
                                               redirect_uri=redirect_uri))

top_track_settings = sp.current_user_top_tracks(limit=100, time_range="medium_term") 
top_tracks_data = top_track_settings["items"]

#Connect to Genius API for Lyrics data 
genius_access_token = os.getenv("genius_access_token")
genius = lyricsgenius.Genius(genius_access_token)

top_tracks = [] #setting empty list and then will loop and append the top songs to this list

def get_lyrics(track_name, artist_name):
    try:
        # Search for lyrics
        song = genius.search_song(track_name, artist_name)
        return song.lyrics
    except Exception as e:
        print(f"Error fetching lyrics for {track_name} by {artist_name}: {e}")
        return None

for i, track in enumerate(top_tracks_data, start=1):
    artist_name = track['artists'][0]['name'] if track['artists'] and len(track['artists']) > 0 else 'Unknown Artist'
    print(f"{i}. {track['name']} by {artist_name}")
    
    # Get lyrics for the track
    lyrics = get_lyrics(track['name'], artist_name)
    
    if lyrics:
        top_tracks.append({
            "Name": track['name'],
            "Artist": artist_name,
            "Lyrics": lyrics,
            "Valence": sp.audio_features([track['id']])[0]['valence']
        })
    else:
        print(f"No lyrics found for {track['name']} by {artist_name}")

df = pd.DataFrame(data=top_tracks)

songs_to_drop = ["It's Not Living (If It's Not with You)", 
                 "I Don't Live Here Anymore (feat. Lucius)",
                 "Never Going Back Again - 2004 Remaster",
                 "Strobe - Radio Edit",
                 "Cute Without The 'E' (Cut From The Team) - Remastered"
                ]

#Create first dataset where label (y) is Spotify's valence score
df_songs_valence = df[~df['Name'].isin(songs_to_drop)]

df_songs_valence.head()
df_songs_valence.to_excel('top_tracks_updated.xlsx') #checks

df_songs_valence.loc[:, "Binary_Valence"] = df_songs_valence["Valence"].apply(lambda x: 1 if x >= 0.5 else 0)
df_songs_valence.drop("Valence", axis=1)

X = df_songs_valence.drop("Binary_Valence", axis=1)
y = df_songs_valence["Binary_Valence"]

X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(X, y, test_size=0.2, random_state=42)

#Create first dataset where label (y) is Spotify's valence score
df_songs_valence = df[~df['Name'].isin(songs_to_drop)]

df_songs_valence.head()
df_songs_valence.to_excel('top_tracks_updated.xlsx') #checks

df_songs_valence.loc[:, "Binary_Valence"] = df_songs_valence["Valence"].apply(lambda x: 1 if x >= 0.5 else 0)
df_songs_valence.drop("Valence", axis=1)

X = df_songs_valence.drop("Binary_Valence", axis=1)
y = df_songs_valence["Binary_Valence"]

X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(X, y, test_size=0.2, random_state=42)

#Create second dataset where label (y) was assigned by me in an Excel sheet
df_songs_self_assigned = pd.read_excel('top_tracks_self_assigned.xlsx')

X = df_songs_self_assigned.drop("Self Assigned Label", axis=1)
y = df_songs_self_assigned["Self Assigned Label"]

X_train_self, X_test_self, y_train_self, y_test_self = train_test_split(X, y, test_size=0.2, random_state=42)


#Modify both of the datasets to tokenize the lyrics into columns 

nltk.download('stopwords')
nltk.download('punkt')

def tokenize_and_clean(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

#Valence Dataset
df_songs_valence['Cleaned_Lyrics'] = df_songs_valence['Lyrics'].fillna('').apply(tokenize_and_clean)

tfidf_vectorizer_valence = TfidfVectorizer(max_features=1000)  
lyrics_tfidf_matrix_valence = tfidf_vectorizer_valence.fit_transform(df_songs_valence['Cleaned_Lyrics'])

X_combined_valence = pd.concat([df_songs_valence.drop(['Lyrics', 'Binary_Valence', 'Cleaned_Lyrics'], axis=1).reset_index(drop=True),
                                pd.DataFrame(lyrics_tfidf_matrix_valence.toarray())], axis=1)

X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(X_combined_valence, df_songs_valence["Binary_Valence"], test_size=0.2, random_state=42)

#Self Assigned Labels Dataset (Vectorize, update dfs and train/test split)
df_songs_self_assigned['Cleaned_Lyrics'] = df_songs_self_assigned['Lyrics'].fillna('').apply(tokenize_and_clean)

tfidf_vectorizer_self_assigned = TfidfVectorizer(max_features=1000)  # You can adjust max_features based on your dataset
lyrics_tfidf_matrix_self_assigned = tfidf_vectorizer_self_assigned.fit_transform(df_songs_self_assigned['Cleaned_Lyrics'])

X_combined_self_assigned = pd.concat([df_songs_self_assigned.drop(['Lyrics', 'Self Assigned Label', 'Cleaned_Lyrics'], axis=1).reset_index(drop=True),
                                      pd.DataFrame(lyrics_tfidf_matrix_self_assigned.toarray())], axis=1)

X_train_self_assigned, X_test_self_assigned, y_train_self_assigned, y_test_self_assigned = train_test_split(X_combined_self_assigned, df_songs_self_assigned["Self Assigned Label"], test_size=0.2, random_state=42)

#Implement Neural Network model for Valence dataset

model_valence = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=42)
model_valence.fit(X_train_valence, y_train_valence)
y_pred_valence = model_valence.predict(X_test_valence)

accuracy_valence = accuracy_score(y_test_valence, y_pred_valence)
print(f'Accuracy for df_songs_valence: {accuracy_valence}')

#Implement Neural Network model for Self Assigned dataset
model_self_assigned = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=42)
model_self_assigned.fit(X_train_self_assigned, y_train_self_assigned)
y_pred_self_assigned = model_self_assigned.predict(X_test_self_assigned)

accuracy_self_assigned = accuracy_score(y_test_self_assigned, y_pred_self_assigned)
print(f'Accuracy for df_songs_self_assigned: {accuracy_self_assigned}')


