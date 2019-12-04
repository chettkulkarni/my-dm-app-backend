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

app = Flask(__name__)

cors = CORS(app)






@app.route('/customwater/<string>', methods=['Post'])
def customwater(string):
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
    tweets_df = pd.DataFrame(tweets)
    df = pd.DataFrame(tweets_df.text)
    df['Tweet']=df.text
    stemmer = SnowballStemmer("english")
    lem=WordNetLemmatizer()
    df['lemm'] = df['Tweet'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
    df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
    df['cleaned_specials']=df['cleaned_specials'].str.replace('\d+', '')
    df=df.drop(columns=['lemm','stemmed','Tweet'])
    df=df.rename(columns={'cleaned_specials': "Tweet"})
    tweets_df['Cleaned_Tweet']=df['Tweet']
    vector2 = vector.transform(df['Tweet'])
    df['Predicted_Class']=loaded_model.predict(vector2)
    tweets_df['Predicted_Class']=df['Predicted_Class']
    tweets_df['date']=pd.to_datetime(tweets_df['created_at']).dt.date
    tweets_df['hour']=pd.to_datetime(tweets_df['created_at']).dt.hour
    tweets_df['min']=pd.to_datetime(tweets_df['created_at']).dt.minute
    df=tweets_df
    df1=df.groupby(['created_at','Predicted_Class']).count().reset_index()[['created_at','Predicted_Class','Cleaned_Tweet','date','hour','min']]
    df2=df1
    df=df[['date','hour','min','Predicted_Class','Cleaned_Tweet',]]
    df1=df.groupby(['date','hour','min','Predicted_Class']).count().reset_index()[['date','hour','min','Predicted_Class','Cleaned_Tweet']]
    df2=df1
    result = pd.merge(left=df1,right=df2, how='left', left_on=['date','hour','min'], right_on=['date','hour','min'])
    date=result[result.Cleaned_Tweet_x!=result.Cleaned_Tweet_y][['date','hour','min','Cleaned_Tweet_x','Cleaned_Tweet_y']]
    date['Cleaned_Tweet_y']=-date['Cleaned_Tweet_y']
    date=date.sort_values(['date','hour'],ascending=False)
    date=date.drop_duplicates(subset =['date','hour','min'])
    return date.to_json(orient='records')

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






@app.route('/pred', methods=['Post'])
def predict():
    df = pd.read_csv('final.csv', encoding='latin')
    predicted_class_counts = df.Predicted_Class.value_counts()
    return (predicted_class_counts.to_json(orient='records'))

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


@app.route('/custompos/<string>', methods=['Post'])
def custompos(string):
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
    # string='chetan'
    keyword = string.split(" ")
    no_of_tweets = 100
    tweets = []
    for i in keyword:
        tweets += [tweet._json for tweet in fetch_tweets(i, no_of_tweets)]
        tweets_df = pd.DataFrame(tweets)
    print(tweets_df.text)
    df = pd.DataFrame(tweets_df.text)
    df['Tweet']=df.text
    stemmer = SnowballStemmer("english")
    lem=WordNetLemmatizer()

    df['lemm'] = df['Tweet'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
    df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
    df['cleaned_specials']=df['cleaned_specials'].str.replace('\d+', '')
    df=df.drop(columns=['lemm','stemmed','Tweet'])
    df=df.rename(columns={'cleaned_specials': "Tweet"})
    vector2 = vector.transform(df['Tweet'])
    df['Predicted_Class']=loaded_model.predict(vector2)
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


    return (df.to_json(orient='records'))

@app.route('/customneg/<string>', methods=['Post'])
def customneg(string):
    print(string)
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
    # string='chetan'
    keyword = string.split(" ")
    no_of_tweets = 100
    tweets = []
    for i in keyword:
        tweets += [tweet._json for tweet in fetch_tweets(i, no_of_tweets)]
        tweets_df = pd.DataFrame(tweets)
    print(tweets_df.text)
    df = pd.DataFrame(tweets_df.text)
    df['Tweet']=df.text
    stemmer = SnowballStemmer("english")
    lem=WordNetLemmatizer()

    df['lemm'] = df['Tweet'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
    df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
    df['cleaned_specials']=df['cleaned_specials'].str.replace('\d+', '')
    df=df.drop(columns=['lemm','stemmed','Tweet'])
    df=df.rename(columns={'cleaned_specials': "Tweet"})
    vector2 = vector.transform(df['Tweet'])
    df['Predicted_Class']=loaded_model.predict(vector2)
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


    return (df.to_json(orient='records'))


# @app.route('/predNew', methods=['Post'])
# def predict_new():
#     loaded_model = joblib.load('lr_Down.pkl')
#     vector=joblib.load('abc.pkl')
#     df=pd.read_csv('impeachtweets3.csv',encoding='latin')
#     df_orig=df
#     df=df[['Tweet']]
#     df=df.rename(columns={0: "Tweet"})
#     df['Lower']=df['Tweet'].str.lower()
#     df['Lower']=df['Lower'].str.replace('\d+', '')
#     df['Lower']=df['Lower'].str.replace(r"http\S+", '')
#     stop_words = set(stopwords.words('english'))
#     df['tweet_without_stopwords'] = df['Lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#     stemmer = SnowballStemmer("english")
#     lem=WordNetLemmatizer()

#     df['lemm'] = df['tweet_without_stopwords'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
#     df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
#     df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
#     df=df.drop(columns=['Lower','tweet_without_stopwords','lemm','stemmed','Tweet'])
#     df=df.rename(columns={'cleaned_specials': "Tweet"})

#     vector2 = vector.transform(df['Tweet'])
#     df['Predicted_Class']=loaded_model.predict(vector2)
#     df_orig['Predicted_Class']=loaded_model.predict(vector2)
#     df_orig['Cleaned_Tweet']=df.Tweet

#     df_orig.to_csv(r'final.csv')
#     # df = pd.read_csv('final.csv', encoding='latin')
#     # predicted_class_counts = df.Predicted_Class.value_counts()
#     return (predicted_class_counts.to_json(orient='records'))

conf = {
  "twitterEndpoint": "https://api.twitter.com",
  "consumerKey": "LxjMiK3zxdkGUADCxst6hO0lZ",
  "consumerSecret": "isY1RawP3HvYS3HsMwCQ3ZSiRqPoau67hnceIwS0klV3bK8PvE",
  "accessToken": "2422714549-4jdggTNWfG0HYDHbSxPWH73JBT6zuEhWocGDlil",
  "accessSecret": "dKUO6419UGmgzUg3wPunTq3U2vYr4Lcj4mAkzgQ14FyRS",
  "signature_method": "HMAC-SHA1"
}


@app.route('/tweets/<string>', methods=['POST'])
def TweetPred(string):
    # print('in here')
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
    print(keyword)
    no_of_tweets = 100
    tweets = []
    for i in keyword:
        tweets += [tweet._json for tweet in fetch_tweets(i, no_of_tweets)]
        tweets_df = pd.DataFrame(tweets)
    print(tweets_df.text)
    df = pd.DataFrame(tweets_df.text)
    df['Tweet']=df.text
    stemmer = SnowballStemmer("english")
    lem=WordNetLemmatizer()

    df['lemm'] = df['Tweet'].apply(lambda x: ' '.join([lem.lemmatize(y) for y in x.split()]))
    df['stemmed'] = df['lemm'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
    df['cleaned_specials'] = df['stemmed'].str.replace('[^A-z0-9 ]', '')
    df=df.drop(columns=['lemm','stemmed','Tweet'])
    df=df.rename(columns={'cleaned_specials': "Tweet"})
    vector2 = vector.transform(df['Tweet'])
    df['Predicted_Class']=loaded_model.predict(vector2)
    return df.Predicted_Class.value_counts().to_json(orient='records')





def createOAuth1Object(conf):
    return OAuth1(conf['consumerKey'],
                  client_secret=conf['consumerSecret'],
                  resource_owner_key=conf['accessToken'],
                  resource_owner_secret=conf['accessSecret'],
                  signature_method='HMAC-SHA1')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)



