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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

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
                ] #removing these songs because the API did not pull in the correct lyrics



#Create first dataset where label (y) is Spotify's valence score
df_songs_valence_1 = df[~df['Name'].isin(songs_to_drop)]
df_songs_valence_1.to_excel('top_tracks_updated.xlsx') #checks


df_songs_valence_1.loc[:, "Binary_Valence"] = df_songs_valence_1["Valence"].apply(lambda x: 1 if x >= 0.5 else 0)

#Tokenization

nltk.download('stopwords')
nltk.download('punkt')

additional_words_for_removal = ('chorus','bridge','verse','yes','yeah','liveget','tickets','intro','outro','lyrics')

def tokenize_and_clean(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words and word.lower() not in additional_words_for_removal]
    return ' '.join(words)

df_songs_valence_1['Cleaned_Lyrics'] = df_songs_valence_1['Lyrics'].fillna('').apply(tokenize_and_clean)
df_songs_valence['Cleaned_Lyrics']  = df_songs_valence_1['Lyrics'].fillna('').apply(tokenize_and_clean)

df_songs_valence = df_songs_valence_1.drop(["Name","Artist","Valence"], axis=1) #drop Name and Artist because they are not features (X)
#drop Valence because Binary_Valence has replaced it as the label (y)
#Leave Lyrics so tokenization can happen (remove later)

tfidf_vectorizer_valence = TfidfVectorizer(max_features=1000)  
lyrics_tfidf_matrix_valence = tfidf_vectorizer_valence.fit_transform(df_songs_valence['Cleaned_Lyrics'])

X_valence = pd.concat([df_songs_valence.drop(['Lyrics', 'Binary_Valence', 'Cleaned_Lyrics'], axis=1).reset_index(drop=True),
                                pd.DataFrame(lyrics_tfidf_matrix_valence.toarray())], axis=1)

X_valence.columns = X_valence.columns.astype(str)
X_valence = X_valence.dropna()

X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(X_valence, df_songs_valence["Binary_Valence"], test_size=0.2, random_state=42)

#Create second dataset where label (y) was assigned by me in an Excel sheet
df_songs_self_assigned = pd.read_excel('top_tracks_self_assigned.xlsx')

df_songs_self_assigned = df_songs_self_assigned.drop(["Name","Artist"],axis=1)
df_songs_self_assigned.head()

df_songs_self_assigned['Cleaned_Lyrics'] = df_songs_self_assigned['Lyrics'].fillna('').apply(tokenize_and_clean)

tfidf_vectorizer_self_assigned = TfidfVectorizer(max_features=1000)  
lyrics_tfidf_matrix_self_assigned = tfidf_vectorizer_self_assigned.fit_transform(df_songs_self_assigned['Cleaned_Lyrics'])

X_self_assigned = pd.concat([df_songs_self_assigned.drop(['Lyrics', "Self Assigned Label", 'Cleaned_Lyrics'], axis=1).reset_index(drop=True),
                                pd.DataFrame(lyrics_tfidf_matrix_self_assigned.toarray())], axis=1)

X_self_assigned.columns = X_self_assigned.columns.astype(str)
X_self_assigned = X_self_assigned.dropna()

X_train_self_assigned, X_test_self_assigned, y_train_self_assigned, y_test_self_assigned = train_test_split(X_self_assigned, df_songs_self_assigned["Self Assigned Label"], test_size=0.2, random_state=42)

#Naive Bayes Model
nb_model = MultinomialNB()

#Valence dataset
nb_model.fit(X_train_valence, y_train_valence)
y_pred_valence = nb_model.predict(X_test_valence)
y_pred_valence_train = nb_model.predict(X_train_valence)

accuracy_valence = accuracy_score(y_test_valence, y_pred_valence)
print(f"Accuracy for dataset with labels as valence score: {accuracy_valence}")

#Self Assigned Label dataset
nb_model.fit(X_train_self_assigned, y_train_self_assigned)
y_pred_self_assigned = nb_model.predict(X_test_self_assigned)
y_pred_self_assigned_train = nb_model.predict(X_train_self_assigned)

accuracy_self_assigned = accuracy_score(y_test_self_assigned, y_pred_self_assigned)
print(f"Accuracy for dataset with self assigned labels: {accuracy_self_assigned}")


#Evaluations and visualizations
df_songs_self_assigned_orig = pd.read_excel('top_tracks_self_assigned.xlsx')
df_final = pd.merge(df_songs_valence_1, df_songs_self_assigned_orig[['Name', 'Artist', 'Self Assigned Label']], on=['Name', 'Artist'], how='left')
y_pred_self_assigned_full = pd.Series(y_pred_self_assigned_train.tolist() + y_pred_self_assigned.tolist(), name='Predicted_Label_Self_Assigned')

# Add the self-assigned labels and predicted labels (same set of songs so only using self_assigned y_pred values)
df_final['Self Assigned Label'] = df_songs_self_assigned_orig['Self Assigned Label']
df_final['Predicted_Label'] = y_pred_self_assigned_full

df_final = df_final.drop(["Lyrics"], axis=1)
df_final

print("\nClassification Report for Valence Dataset:")
print(classification_report(y_test_valence, y_pred_valence))

print("\nClassification Report for Self Assigned Label Dataset:")
print(classification_report(y_test_self_assigned, y_pred_self_assigned))

def create_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.axes()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    
create_confusion_matrix(y_test_valence, y_pred_valence, "Confusion Matrix - Valence Dataset")
create_confusion_matrix(y_test_self_assigned, y_pred_self_assigned, "Confusion Matrix - Self Assigned Label Dataset")


df_final_w_lyrics = pd.merge(df_final, df_songs_valence_1[['Name', 'Artist', 'Cleaned_Lyrics']], on=['Name', 'Artist'], how='inner')

positive_text_predicted = ' '.join(df_final_w_lyrics[df_final_w_lyrics['Predicted_Label'] == 1]['Cleaned_Lyrics'])
negative_text_predicted = ' '.join(df_final_w_lyrics[df_final_w_lyrics['Predicted_Label'] == 0]['Cleaned_Lyrics'])

# Generate word clouds with distinct colors for predicted labels
create_wordcloud(positive_text_predicted, 'Positive Word Cloud - Predicted Labels')
create_wordcloud(negative_text_predicted, 'Negative Word Cloud - Predicted Labels')
