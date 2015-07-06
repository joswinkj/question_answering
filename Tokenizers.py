import pandas as pd
import pdb
from nltk import RegexpTokenizer
from nltk import word_tokenize

class SynonymTokenizer(object):
    def __init__(self):
        from nltk.stem import WordNetLemmatizer 
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc ,string_tokenize='[a-zA-Z0-9]+'):
    	from nltk.tokenize import RegexpTokenizer 
    	from nltk.corpus import stopwords
    	from nltk.corpus import wordnet as wn
    	#tokenizer = RegexpTokenizer(r'\w+')
    	tokenizer = RegexpTokenizer(string_tokenize)
        #words=[self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        words=[self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]
        mystops=(u'youtube',u'mine',u'this',u'that')
        stop_words=set(stopwords.words('english'))
        stop_words.update(mystops)
        stop_words=list(stop_words)
        words1= [i.lower() for i in words if i not in stop_words]
        words2= list(set(list({l.name() for word in words1 for s in wn.synsets(word) for l in s.lemmas()})+words1))
        
        return [i.lower() for i in words2 if i not in stop_words]

class LemmaTokenizer(object):
    def __init__(self):
        from nltk.stem import WordNetLemmatizer 
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc ):
    	from nltk.tokenize import RegexpTokenizer 
    	from nltk.corpus import stopwords
    	#tokenizer = RegexpTokenizer(r'\w+')
    	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        #words=[self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        words=[self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]
        mystops=(u'youtube',u'mine',u'this',u'that','facebook','com','google','www','http','https')
        stop_words=set(stopwords.words('english'))
        stop_words.update(mystops)
        
        stop_words=list(stop_words)
        return [i.lower() for i in words if i not in stop_words]

from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
class StemTokenizer(object):
    def __init__(self):
        self.mystops=(u'youtube',u'mine',u'this',u'that','facebook','com','google','www','http','https')
    def __call__(self, doc ):
        snowball_stemmer = SnowballStemmer('english')
    	#tokenizer = RegexpTokenizer(r'\w+')
        #words=[self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        words=[snowball_stemmer.stem(t) for t in word_tokenize(doc)]
        stop_words=set(stopwords.words('english'))
        stop_words.update(self.mystops)
        stop_words=list(stop_words)
        return [i.lower() for i in words if i not in stop_words]        

class SynonymStemTokenizer(object):
    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        from nltk.stem import SnowballStemmer
        self.wnl = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer('english')
    def __call__(self, doc ,string_tokenize='[a-zA-Z0-9]+'):
    	from nltk.tokenize import RegexpTokenizer
    	from nltk.corpus import stopwords
    	from nltk.corpus import wordnet as wn
    	#tokenizer = RegexpTokenizer(r'\w+')
    	tokenizer = RegexpTokenizer(string_tokenize)
        #words=[self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        words=[self.wnl.lemmatize(t) for t in tokenizer.tokenize(doc)]
        mystops=(u'youtube',u'mine',u'this',u'that')
        stop_words=set(stopwords.words('english'))
        stop_words.update(mystops)
        stop_words=list(stop_words)
        words1= [i.lower() for i in words if i not in stop_words]
        words2= list(set(list({l.name() for word in words1 for s in wn.synsets(word) for l in s.lemmas()})+words1))
        words3=list(set([self.snowball_stemmer.stem(t) for t in words2]))
        return [i.lower() for i in words3 if i not in stop_words]
