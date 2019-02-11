import matplotlib.pyplot as plt
import warnings
import requests
from io import BytesIO
import os
import numpy as np
from tqdm.autonotebook import tqdm
from PIL import  Image
from time import sleep
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from os import path, getcwd
from wordcloud import WordCloud, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, url_for, redirect, render_template
from flask import request
import json
import pandas as pd
app = Flask(__name__)


def load_data():
    foundations_details = pd.read_pickle('foundations_details2')
    foundations_reviews = pd.read_pickle('foundations_reviews')
    return foundations_details, foundations_reviews


def pre_pro(new_row):
    # import
    foundations_details, foundations_reviews = load_data()
    # the dataframe without reviews and 'reviewer_nickname' cloumn
    foundations_reviews_without_reviews = foundations_reviews[['Encoded_foundation_name',
           'r', 'g', 'b','helpful','not_helpful', 'review_rank', 'Encoded_Eye_color',
           'Encoded_Hair_color', 'reviewer_age', 'Encoded_Skin_type', 'Encoded_Skin_tone']]

    # make dummies 
    foundations_reviews_withdummies = pd.get_dummies(foundations_reviews_without_reviews)
    
    good_review =  foundations_reviews_withdummies[foundations_reviews_withdummies.review_rank > 50]
    y = good_review.pop('Encoded_foundation_name')
    X = good_review[['b', 'r', 'b', 'Encoded_Hair_color', 'Encoded_Skin_type', 'Encoded_Skin_tone']]
    
    fid = modeling(X, y, new_row)
    return fid



def foun_name(foundations_details, fid):
    getfname = foundations_details.loc[foundations_details.Encoded_foundation_name == fid, :].iloc[0,:]
    return getfname


def modeling(X, y, new_row):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # y pred
    y_pred = clf.predict(np.array(new_row).reshape(1, -1))
    
    return y_pred

    
def Encoded_Hair_color(Hair_color):
    EncodeHair_color= {'Auburn':0, 'Black':1, 'Blonde':2, 'Brunette':3, 'Gray':4, 'NA':5, 'Red':6}
    return EncodeHair_color[Hair_color]

                                    
def Encoded_Skin_type(Skin_type):
    EncodedSkin_type = {'Normal':0, 'Dry':1, 'Combination':2, 'Oily':3, 'Sensitive':4, 'NA': 5}
    return EncodedSkin_type[Skin_type]
    
def Encoded_Skin_tone(Skin_tone):
    EncodedSkin_tone = {'Dark':0, 'Deep':1, 'Ebony':2, 'Fair':3, 'Light':4, 'Medium':5, 'NA':6, 'Olive':7, 'Porcelain':8,
 'Tan':9}
    return EncodedSkin_tone[Skin_tone]

    

def get_rgb(foundations_reviews, url):
    return foundations_reviews.loc[(foundations_reviews.reviewer_skin_color_pic == url), ['r', 'g', 'b']].iloc[0,:]

def start(skinurl, Hc, Skt, Skto):
    foundations_details, foundations_reviews = load_data()
    rgb = get_rgb(foundations_reviews, skinurl)
    Hc = Encoded_Hair_color(Hc)
    Skt = Encoded_Skin_type(Skt)
    Skto = Encoded_Skin_tone(Skto)
    new_row = pd.DataFrame(columns=['r', 'b', 'g', 'Hair_color', 'Skin_type', 'Skin_tone'])
    new_row = [rgb[0], rgb[1], rgb[2], Hc, Skt, Skto]
    return pre_pro(new_row)

def wc(foundations_reviews, fid):  
    
    df_one_foun = foundations_reviews.loc[foundations_reviews.Encoded_foundation_name == fid, :]
    
    # number of characters 
    df_one_foun['word_count'] = df_one_foun.foundation_review.apply(lambda x: len(str(x).split(" ")))

    # lowercase
    df_one_foun['foundation_review'] = df_one_foun['foundation_review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # removing the speical characters
    df_one_foun['foundation_review'] = df_one_foun['foundation_review'].str.replace('[^\w\s]','')

    # removing the stop word 
    stop = stopwords.words('english')
    df_one_foun['foundation_review'] = df_one_foun['foundation_review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # find the frq word
    freq = pd.Series(' '.join(df_one_foun['foundation_review']).split()).value_counts()
    
    
    comment_words = ''
    for words in list(df_one_foun['foundation_review']): 
        comment_words = comment_words + words + ' '
        
    d = getcwd()
    ## join all documents in corpus
    text = comment_words
    ## image from PublicDomainPictures.net
    ## http://www.publicdomainpictures.net/view-image.php?image=232185&picture=family-gathering
    mask = np.array(Image.open(path.join(d, "makeup6.png")))
    wc = WordCloud(background_color="white", max_words=1000, mask=mask,
               max_font_size=90, random_state=42)
    wc.generate(text[:2200])
    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    wc.to_file("static/f1.png")


@app.route('/beautyinside', methods=['POST'])
def beautyinside():
  name = request.form['name']
  HairColor = request.form['HairColor']
  SkinType = request.form['SkinType']
  SkinTone = request.form['SkinTone']
  urlimage = request.form['n']
  foundations_details, foundations_reviews = load_data()
  fid = start(urlimage, HairColor, SkinType, SkinTone)
  fname = foun_name(foundations_details, fid[0])
  wc(foundations_reviews, fid[0])
  fname['name'] = name
  fname['rank'] = round(fname['rank'], 2)
  dict_fname = fname.to_dict()

  return render_template('Pre_foun.html', **dict_fname)
  #return json.dumps(str(dict_fname))

@app.route('/Pre_foun')
def Prepage():
  pass
  



if __name__ == '__main__':
  app.run(debug=True)

