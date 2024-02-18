# Spotify-Sentiment-Analysis
This was my final project in my Machine Learning class. It is a sentiment analysis that takes my top songs on Spotify and assigns them a score of 0 or 1 (0 being more negative in sentiment and 1 being more positive). The results of the ML model are compared to the results of Spotify's valence score which is a 'measure of a song's positiveness' as well as the self-assigned labels I manually gave each songs. The 3 scores are compared side by side. 

I came up with the idea for this project after noticing that a lot of my favorite songs are upbeat songs with sad topics, which I found to be an interesting contrast. I was curious how Spotify's valence score would categorize these songs vs. a ML sentiment analysis vs. my own categorization. 

# Dataset
Dataset was pulled from 2 different APIs: 
Spotipy (Spotify's Python API) 
Lyricsgenuis 
Self Assigned Labels Excel file 

I used my top 50 songs of the year as the dataset (due to limitations of Spotify's API I was only able to pull 50 songs (which was then further reduced in the preprocessing step) however I still continued with the project despite the small size of the dataset). 

The API keys are located in a .env file in the same project folder.
