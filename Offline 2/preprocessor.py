from bs4 import BeautifulSoup as bs
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
topics = []
def getTopicnames():
    with open('Data/topics.txt', 'r') as reader:
        line = reader.readline()
        while line != '':  # The EOF char is an empty string

            print(line, end='')
            topics.append(line[:-1])
            line = reader.readline()



getTopicnames()
print(topics)
def preprocessData():
    with open('Data\Test\Coffee.xml','r',encoding='utf-8') as file:
        content = file.readlines()
        content = "".join(content)
        soup = bs(content,'lxml')
        count = 0
        for item in soup.find("body"):
            if item != "\n":
            #  print("--------------------------------------------------")
            #   print(item["body"])
                count = count+1
        print(count)
    with open('Data\Training\Anime.xml','r',encoding='utf-8') as file:
        content = file.read()
        soup = bs(content,'lxml')
        count = 0
        for items in soup.findAll("posts"):
            for item in items:
                if item != "\n":
                #  print("------item:---------------------------------------------------")
                # print(item["body"])
                    text = item["body"]
                    count = count+1
        print(count)
preprocessData()