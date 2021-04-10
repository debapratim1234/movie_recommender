import pandas as pd
import numpy as np
import pyaudio
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################
r = sr.Recognizer()
my_mic = sr.Microphone(device_index=1)
def speak(message):
	engine=pyttsx3.init()
	rate=engine.getProperty('rate')
	engine.setProperty('rate',rate-10)
	engine.say(message)
	engine.runAndWait()
##Step 1: Read CSV File
df=pd.read_csv("movie_dataset.csv")
#print (df.columns)
##Step 2: Select Features
features = ['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df[feature]=df[feature].fillna(' ')
def combine_features(row):
	return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
df["combined_features"]=df.apply(combine_features,axis=1)
#print (df["combined_features"].head())
##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim=cosine_similarity(count_matrix)
print("System speaking...")
#movie_user_likes = "Interstellar"
with my_mic as source:
    speak("Hi Mr.Deb, what is the latest movie you've watched?")
    audio = r.listen(source)
movie_user_likes = r.recognize_google(audio)
print("Mr.Deb:"+movie_user_likes)
## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movies= list(enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies=sorted(similar_movies,key= lambda x:x[1],reverse=True)
## Step 8: Print titles of first 50 movies
i=0
print("System speaking...")
speak("Oh Nice ! Here is a list of ten movies that I would recommend you:")
for movie in sorted_similar_movies:
	speak(get_title_from_index(movie[0]))
	print(get_title_from_index(movie[0]))
	i=i+1
	if(i>10):
		break