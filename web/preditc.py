import pandas as pd
import numpy as np
import pickle
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
#https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef766
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm, tqdm_pandas
from bs4 import BeautifulSoup
#Plotly Tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
#https://stackoverflow.com/questions/55132071/series-object-has-no-attribute-iplot/55132247
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from plotly.offline import iplot
import plotly.graph_objs as go
import pickle5 as pickle
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import sklearn.metrics as metrics


_Data_Folder_Path="data"
_Final_Model="XGB_Features_10"
_Final_Features_List="FinalFeaturesList"
#_Raw_Records="Merged_Listing_NY_4"

# https://stackoverflow.com/a/47091490/4084039
import re
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#Method to clear Empty Spaces
def replaceSpaces(text):
    temp= str(text)
    temp=text.strip()
    temp=temp.replace('\\r', '')
    temp=temp.replace('\\"', '')
    temp=temp.replace('\\n', '')
    temp=temp.replace(' ', '_')
    return temp


# To get the results in 4 decemal points
SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)

    porter = PorterStemmer()
    pattern = re.compile('\W')

    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)

    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()

    return x
def getPrcocessedDataSet(allRecords,selectColumns):
    allRecords.fillna('na')
    allRecords['neighborhood_overview'] = allRecords["neighborhood_overview"].fillna("na").apply(preprocess)
    allRecords['neighborhood_overview_len']=allRecords.neighborhood_overview.apply(len)
    allRecords['host_response_rate']=allRecords["host_response_rate"].fillna('na').apply(preprocess)
    allRecords['host_acceptance_rate']= allRecords['host_acceptance_rate'].str.replace('%', '')
    allRecords['host_acceptance_rate']=allRecords['host_acceptance_rate'].astype('float')
    allRecords['host_acceptance_rate']=allRecords["host_acceptance_rate"].fillna(0)
    allRecords['host_response_rate']= allRecords['host_response_rate'].str.replace('na', '0')
    allRecords['host_response_rate']= allRecords['host_response_rate'].str.replace('percent', '')
    allRecords['host_response_rate']=allRecords['host_response_rate'].astype('float')
    allRecords["host_Since"] = pd.to_datetime(allRecords["host_since"])
    #https://stackoverflow.com/questions/57011334/how-to-find-number-of-days-between-today-and-future-date/57013179
    allRecords['host_age'] = ( pd.Timestamp('now')-allRecords['host_Since']).dt.days
    allRecords['description']=allRecords['description'].apply(preprocess)
    allRecords['description']=allRecords['description'].str.replace('br','')
    allRecords['description_count']= allRecords.description.apply(lambda x: len(x.split()))
    allRecords['amenities']=allRecords['amenities'].apply(preprocess)
    allRecords['amenties_count']= allRecords.amenities.apply(lambda x: len(x.split()))
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].fillna("na")
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].apply(replaceSpaces)
    # Label Ordinal Encoding for Bathroom text
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('na','0')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('baths','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('bath','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('shared','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('Shared','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('private','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('Private','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('Half','0.5')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('half','0.5')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('-','')
    allRecords['bathrooms_text']=allRecords['bathrooms_text'].str.replace('_','')
    allRecords=allRecords[selectColumns]
    cols = list(allRecords.columns)
    for i in cols:
        allRecords[i] = allRecords[i].apply(pd.to_numeric)
    allRecords['host_age'].fillna(allRecords['host_age'].mean(), inplace = True)
    return allRecords


def skewData(data):
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    power = power.fit(data)
    return power.transform(data)


#Make Final Predictions
def predict(recs):
    colNames= pd.read_csv(_Data_Folder_Path + "/" + _Final_Features_List)
    str_name=colNames['Unnamed: 0'].values
    processed_recs=getPrcocessedDataSet(recs,str_name)
    processed_recs= skewData(processed_recs)
    processed_recs=pd.DataFrame(processed_recs,columns=str_name)
    loaded_model = pickle.load(open(_Data_Folder_Path + "/" + _Final_Model, 'rb'))
    predictions=loaded_model.predict(processed_recs)
    return predictions