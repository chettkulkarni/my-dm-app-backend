
import numpy as np
from flask import Flask,request
from flask_cors import CORS
from sklearn.externals import joblib
import tweepy
from tweepy import OAuthHandler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import nltk
nltk.download('wordnet')

from nltk import tokenize
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import sys
import os
import io
import re
from sys import path
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from string import punctuation, digits
from IPython.core.display import display, HTML
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report,recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import roc_auc_score,confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import tweepy
from tweepy import OAuthHandler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import nltk
from nltk import tokenize
from collections import defaultdict

















app = Flask(__name__)

cors = CORS(app)


@app.route('/pred',methods=['Post'])
def predict_1():
    
    # loaded_model = joblib.load('lr_Down.pkl')
    # vector=joblib.load('abc.pkl')
    # df=pd.read_csv('impeachtweets3.csv',encoding='latin')
    # df_orig=df
    # df=df[['Tweet']]
    # df=df.rename(columns={0: "Tweet"})
    # df['Lower']=df['Tweet'].str.lower()
    # df['Lower']=df['Lower'].str.replace('\d+', '')
    # df['Lower']=df['Lower'].str.replace(r"http\S+", '')
    # stop_words = set(stopwords.words('english')) 
    # df['tweet_without_stopwords'] = df['Lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # stemmer = SnowballStemmer("english")
    # lem=WordNetLemmatizer()

    # df['lemm'] = df['tweet_without_stopwords'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
    # df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    # df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
    # df=df.drop(columns=['Lower','tweet_without_stopwords','lemm','stemmed','Tweet'])
    # df=df.rename(columns={'cleaned_specials': "Tweet"})
    
    # vector2 = vector.transform(df['Tweet'])
    # df['Predicted_Class']=loaded_model.predict(vector2)
    # df_orig['Predicted_Class']=loaded_model.predict(vector2)

    # df_orig.to_csv(r'final.csv')
    df=pd.read_csv('final.csv',encoding='latin')
    predicted_class_counts=df.Predicted_Class.value_counts()
    return(predicted_class_counts.to_json(orient='records'))

@app.route('/dance',methods=['Post'])

def predict():
    consumer_key = "veNJxpsprtmX4qvysWCw3DHNG"
    consumer_secret = "rnbojQvAPe0guXUg15XMNb4TW7P9xeu9Yp2JvLEwQ452TUpx8q"
    access_key = "2422714549-zxSwCa4KHEM3n0Elve7fmkYyFDd5hMexMPRirud"
    access_secret = "W4RCAftVTF9kDLknXfC0Tsvh5M44mBrMrlWC1fLznHY5f"
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    
    loaded_model = joblib.load('lr_Down.pkl')
    vector=joblib.load('abc.pkl')
    twitter_api = tweepy.API(auth)


    keyword = ['impeachment','government']
    no_of_tweets = 1000
    tweets = []
    for i in keyword:
        tweets += [tweet._json for tweet in  twitter_api.search(i, count=no_of_tweets)]
        tweets_df = pd.DataFrame(tweets)
        tweets_df.head()
    tweets_df = pd.DataFrame(tweets)
    # print(tweets_df)
    df=pd.DataFrame(tweets_df['text'])
    df['Tweet']=df.text


    df['Lower']=df['Tweet'].str.lower()
    df['Lower']=df['Lower'].str.replace('\d+', '')
    df['Lower']=df['Lower'].str.replace(r"http\S+", '')
    stop_words = set(stopwords.words('english')) 
    df['tweet_without_stopwords'] = df['Lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    stemmer = SnowballStemmer("english")
    lem=WordNetLemmatizer()

    df['lemm'] = df['tweet_without_stopwords'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
    df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
    df=df.drop(columns=['Lower','tweet_without_stopwords','lemm','stemmed','Tweet'])
    df=df.rename(columns={'cleaned_specials': "Tweet"})
    
    vector2 = vector.transform(df['Tweet'])
    df['Predicted_Class']=loaded_model.predict(vector2)
    predicted_class_counts=df.Predicted_Class.value_counts()
    return(predicted_class_counts.to_json(orient='records'))











    #worcloud




if __name__=='__main__':
    app.run(host="127.0.0.1",port=5000)


