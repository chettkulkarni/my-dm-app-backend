
import tweepy
from tweepy import OAuthHandler
import nltk
from flask import Flask
from flask_cors import CORS
from requests_oauthlib import OAuth1

nltk.download('wordnet')
import pandas as pd
from sklearn.externals import joblib
from collections import Counter
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import json
words = set(nltk.corpus.words.words())
import re

app = Flask(__name__)

cors = CORS(app)


#water api to pull in impeached data and agrregate values hourly in pandas and then sending it to front end
@app.route('/water/', methods=['Post'])
def water():
	df=pd.read_csv('final.csv')
	df1=df.groupby(['Date','Predicted_Class']).count().reset_index()[['Date','Predicted_Class','Cleaned_Tweet']]
	df2=df1
	result = pd.merge(left=df1,right=df2, how='left', left_on='Date', right_on='Date')
	date=result[result.Cleaned_Tweet_x!=result.Cleaned_Tweet_y][['Date','Cleaned_Tweet_x','Cleaned_Tweet_y']]
	date['Cleaned_Tweet_y']=-date['Cleaned_Tweet_y']
	date=date.sort_values('Date',ascending=False).head(20)
	return date.drop_duplicates(subset ="Date").to_json(orient='records')

#pred api to get imeached data and return aggregated data to front end 
@app.route('/pred', methods=['Post'])
def predict():
	df = pd.read_csv('final.csv', encoding='latin')
	predicted_class_counts = df.Predicted_Class.value_counts()
	return (predicted_class_counts.to_json(orient='records'))

#pos api to get imeached data and return aggregated data to front end 
@app.route('/pos', methods=['Post'])
def pos():
    print('pos called')
    df = pd.read_csv('final.csv', encoding='latin')
    df = df[df['Predicted_Class'] == 0]
    # print(df.columns)
    # predicted_class_counts = df.Predicted_Class.value_counts()
    words=' '.join([word for word in df.Cleaned_Tweet if type(word)==str])
    x=words.split(" ")
    counterWords=Counter(x)
    # print(counterWords)
    x=[]
    value=[]
    terminator=0
    for i,j in counterWords.items():
        terminator+=1
        if terminator==100:
            break
        x.append(i)
        value.append(j)
    df=pd.DataFrame(x)
    df=df.rename(columns={0: "text"})
    df['value']=value
    df = df.sort_values(by=['value'],ascending=False)
    return (df.to_json(orient='records'))


#pred api to get imeached data and return aggregated data to front end 
@app.route('/neg', methods=['Post'])
def neg():
    df = pd.read_csv('final.csv', encoding='latin')
    df = df[df['Predicted_Class'] == 1]
    words=' '.join([word for word in df.Cleaned_Tweet if type(word)==str])
    x=words.split(" ")
    counterWords=Counter(x)
    # print(counterWords)
    x=[]
    value=[]
    terminator=0
    for i,j in counterWords.items():
        terminator+=1
        if terminator==100:
            break
        x.append(i)
        value.append(j)
    df=pd.DataFrame(x)
    df=df.rename(columns={0: "text"})
    df['value']=value
    df = df.sort_values(by=['value'],ascending=False)
    return (df.to_json(orient='records'))



# #Custom API's to get data from from front end,process the request 
# @app.route('/customwater', methods=['Post'])
# def customwater():
# 	customwater=pd.read_csv('customwater.csv')
# 	customwater=customwater.drop('Unnamed: 0',axis=1)
# 	return customwater.to_json(orient='records')

# @app.route('/custompos', methods=['Post'])
# def custompos():
# 	custompos=pd.read_csv('custompos.csv')
# 	custompos=custompos.drop('Unnamed: 0',axis=1)
# 	return custompos.to_json(orient='records')

# @app.route('/customneg', methods=['Post'])
# def customneg():
# 	customneg=pd.read_csv('customneg.csv')
# 	customneg=customneg.drop('Unnamed: 0',axis=1)
# 	return customneg.to_json(orient='records')

#Get Data from front end,process tweets and lean dataset and run the pickled model on it,and send it to front end

@app.route('/tweets/<string>', methods=['POST'])
def TweetPred(string):
	loaded_model = joblib.load('lr_Down.pkl')
	vector=joblib.load('abc.pkl')
	consumer_key = "veNJxpsprtmX4qvysWCw3DHNG"
	consumer_secret = "rnbojQvAPe0guXUg15XMNb4TW7P9xeu9Yp2JvLEwQ452TUpx8q"
	access_key = "2422714549-zxSwCa4KHEM3n0Elve7fmkYyFDd5hMexMPRirud"
	access_secret = "W4RCAftVTF9kDLknXfC0Tsvh5M44mBrMrlWC1fLznHY5f"
	def authorize_twitter_api():
	    auth = OAuthHandler(consumer_key, consumer_secret)
	    auth.set_access_token(access_key, access_secret)
	    return tweepy.API(auth)
	def fetch_tweets(keyword, no_of_tweets=10):
	    return twitter_api.search(keyword, count=no_of_tweets)
	twitter_api = authorize_twitter_api()
	keyword = string.split(" ")
	no_of_tweets = 100
	tweets = []
	for i in keyword:
	    tweets += [tweet._json for tweet in fetch_tweets(i, no_of_tweets)]
	data = pd.DataFrame(tweets)
	# data.to_csv('tweet_custom.csv')

	df=data

	df['Tweet']=df.text
	stemmer = SnowballStemmer("english")
	lem=WordNetLemmatizer()
	stop_words = set(stopwords.words('english')) 
	df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([re.sub('[^A-z]', '', y) for y in x.split()]))
	df['lemm'] = df['Tweet'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
	df['lemm'] = df['lemm'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
	df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
	df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
	df['cleaned_specials'] = df['stemmed'].str.replace('(\s+)(a|an|and|the)(\s+)', '')
	df=df.drop(columns=['lemm','stemmed','Tweet'])
	df=df.rename(columns={'cleaned_specials': "Tweet"})
	vector2 = vector.transform(df['Tweet'])
	df['Predicted_Class']=loaded_model.predict(vector2)
	# df.to_csv('tweet_custom.csv')
	data=df


	df = df[df['Predicted_Class'] == 1]
	words=' '.join([word for word in df.Tweet if type(word)==str])
	x=words.split(" ")
	counterWords=Counter(x)
	# print(counterWords)
	x=[]
	value=[]
	terminator=0
	for i,j in counterWords.items():
	    terminator+=1
	    if terminator==100:
	        break
	    x.append(i)
	    value.append(j)
	df=pd.DataFrame(x)
	df=df.rename(columns={0: "text"})
	df['value']=value
	df = df.sort_values(by=['value'],ascending=False)
	# df.to_csv('customneg.csv')
	customneg=df.to_json(orient='records')


	df=data

	# df['Tweet']=df.text
	# stemmer = SnowballStemmer("english")
	# lem=WordNetLemmatizer()
	# stop_words = set(stopwords.words('english')) 
	# df['lemm'] = df['Tweet'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
	# df['lemm'] = df['lemm'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
	# df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
	# df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
	# df['cleaned_specials'] = df['stemmed'].str.replace('(\s+)(a|an|and|the)(\s+)', '')
	# df=df.drop(columns=['lemm','stemmed','Tweet'])
	# df=df.rename(columns={'cleaned_specials': "Tweet"})
	# vector2 = vector.transform(df['Tweet'])
	# df['Predicted_Class']=loaded_model.predict(vector2)
	# # df.to_csv('tweet_custom.csv')
	# data=df


	df = df[df['Predicted_Class'] == 0]
	words=' '.join([word for word in df.Tweet if type(word)==str])
	x=words.split(" ")
	counterWords=Counter(x)
	# print(counterWords)
	x=[]
	value=[]
	terminator=0
	for i,j in counterWords.items():
	    terminator+=1
	    if terminator==100:
	        break
	    x.append(i)
	    value.append(j)
	df=pd.DataFrame(x)
	df=df.rename(columns={0: "text"})
	df['value']=value
	df = df.sort_values(by=['value'],ascending=False)
	# df.to_csv('custompos.csv')
	custompos=df.to_json(orient='records')

	df=data
	tweets_df=df
	tweets_df['Predicted_Class']=df['Predicted_Class']
	tweets_df['date']=pd.to_datetime(tweets_df['created_at']).dt.date
	tweets_df['hour']=pd.to_datetime(tweets_df['created_at']).dt.hour
	tweets_df['min']=pd.to_datetime(tweets_df['created_at']).dt.minute
	df=tweets_df
	df1=df.groupby(['created_at','Predicted_Class']).count().reset_index()[['created_at','Predicted_Class','Tweet','date','hour','min']]
	df2=df1
	df=df[['date','hour','min','Predicted_Class','Tweet',]]
	df1=df.groupby(['date','hour','min','Predicted_Class']).count().reset_index()[['date','hour','min','Predicted_Class','Tweet']]
	df2=df1
	result = pd.merge(left=df1,right=df2, how='left', left_on=['date','hour','min'], right_on=['date','hour','min'])
	date=result[result.Tweet_x!=result.Tweet_y][['date','hour','min','Tweet_x','Tweet_y']]
	date['Tweet_x']=-date['Tweet_x']
	date=date.sort_values(['date','hour'],ascending=False)
	date=date.drop_duplicates(subset =['date','hour','min'])
	customwater=date.to_json(orient='records')
	item=[]
	item.append(df.Predicted_Class.value_counts().to_json(orient='records'))
	item.append(customwater)
	item.append(custompos)
	item.append(customneg)
	item2=pd.DataFrame(item)

	# item=df.Predicted_Class.value_counts().to_json(orient='records')+','+customwater+','+custompos+','+customneg


	return item2.to_json(orient='index')





def createOAuth1Object(conf):
    return OAuth1(conf['consumerKey'],
                  client_secret=conf['consumerSecret'],
                  resource_owner_key=conf['accessToken'],
                  resource_owner_secret=conf['accessSecret'],
                  signature_method='HMAC-SHA1')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)