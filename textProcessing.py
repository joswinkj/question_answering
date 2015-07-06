import pandas as pd,numpy as np
import pdb
from Tokenizers import StemTokenizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import re,nltk.data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import cPickle as pickle

class changeToMatrix(object):
    '''initialize the class, give a query and location of data file, get_processed_data function will give \
    two document termpresence matrices on data and query '''
    def __init__(self,ngram_range=(1,1),tokenizer=StemTokenizer(),indexed_data=1):
        self.indexed_data = indexed_data
        if self.indexed_data !=1 :
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range,analyzer='word',lowercase=True,\
                                                  token_pattern='[a-zA-Z0-9]+',strip_accents='unicode',tokenizer=tokenizer)
            #self.vectorizer1 = TfidfVectorizer(ngram_range=ngram_range,analyzer='word',lowercase=True,\
            #    token_pattern='[a-zA-Z0-9]+',strip_accents='unicode',tokenizer=StemTokenizer())
            # self.vectorizer2 = CountVectorizer(ngram_range=ngram_range,analyzer='word',lowercase=True,\
            #     token_pattern='[a-zA-Z0-9]+',strip_accents='unicode',tokenizer=tokenizer)


    def load_ref_text(self,text_file):
        if self.indexed_data == 1:
            loc=open(text_file,'r')
            sentences1,chk2 ,self.vectorizer = pickle.load(loc)
            loc.close()
            return sentences1,chk2
        textfile = open(text_file,'r')
        lines=textfile.readlines()
        textfile.close()
        lines = ' '.join(lines)
        lines = lines.decode('utf8')
        lines = changeToMatrix.removeProbTexts([lines])
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = [ sent_tokenizer.tokenize(lines.strip()) ]
        sentences1 = [item.strip().strip('.') for sublist in sentences for item in sublist]
        #sentences2 = changeToMatrix.get_tagged_sentences(sentences1)
        sentences2 = sentences1
        sentences2 = changeToMatrix.removeStopWords(sentences2)
        sentences2 =[changeToMatrix.getSynonyms(sent,useNouns=1) for sent in sentences2]
        # pdb.set_trace()
        chk2=pd.DataFrame(self.vectorizer.fit_transform(sentences2).toarray(),columns=self.vectorizer.get_feature_names()).to_sparse(fill_value=0)
        loc = open(text_file+'.pickle','w')
        pickle.dump([sentences1,[chk2],self.vectorizer],loc)
        loc.close()
        return sentences1,[chk2]

    def load_query(self,text,tag=1):
        remove_list = ['what','where','when','who','why','how much','how many','how long','how']
        for i in remove_list:
            text=text.replace(i,'')
        text = changeToMatrix.removeStopWords([text])[0]
        if tag==1:
            text_tagged = pos_tag(word_tokenize(text))
            print text_tagged
            nouns=''
            notnouns=''
            for wrd,tkn in text_tagged:
                if tkn[0]=='N' or tkn[0]=='J' or tkn[0]=='V' or tkn[0]=='R':
                    nouns = nouns+' '+wrd
                else:
                    notnouns = notnouns+' '+wrd
        else:
            nouns = text
            notnouns = ''
        all_words=changeToMatrix.getSynonyms(nouns+' '+notnouns,useNouns=1) #will take time. synonyms of ref text is another option
        # print nouns+'\n'+notnouns
        chk_nouns=pd.DataFrame(self.vectorizer.transform([nouns]).toarray(),columns=self.vectorizer.get_feature_names()).to_sparse()
        chk_all=pd.DataFrame(self.vectorizer.transform([all_words]).toarray(),columns=self.vectorizer.get_feature_names()).to_sparse()
        #pdb.set_trace()
        #print np.sum(chk2.values)
        #chk1=pd.DataFrame(self.vectorizer1.transform([text]).toarray(),columns=self.vectorizer1.get_feature_names())
        return [chk_nouns,chk_all]

    def get_processed_data(self,query,data_loc):

        ref_sentences,ref_dataframes=self.load_ref_text(data_loc)#ref_dataframes is a list
        #print ref_dataframe.shape,len(ref_sentences)
        ref_querys=self.load_query(query,tag=0)
        return ref_sentences,ref_dataframes,ref_querys
    @staticmethod
    def get_tagged_sentences(sentences):
        from nltk import pos_tag
        tagged_sentences=[]
        for sent in sentences:
            tmp = pos_tag(sent.split())
            tmp_sent= ''
            for word,tag in tmp:
                if tag[0] in ['N','V','J','R']:
                    tmp_sent=tmp_sent+word+'/'+tag+' '
            tagged_sentences.append(tmp_sent)
        return tagged_sentences

    @staticmethod
    def removeStopWords(sentences,stopwords=None):
        '''
        :param sentences: list of sentences
        :param stopwords: list of stopwords
        :return:list of sentences without stopwords
        '''
        if stopwords==None:
            from nltk.corpus import stopwords
            stopwords=stopwords.words('english')
            stopwords.remove('most')
        sentences1=[]
        for sent in sentences:
            newsent=''
            for word in word_tokenize(sent):
                if word not in stopwords:
                    newsent = newsent+' '+word
            sentences1.append(newsent)
        return sentences1

    @staticmethod
    def getSynonyms(sentence,useNouns=0):
        '''
        :param sentence:single sentence
        :param useNouns: add synonyms for nouns. if 1, synonyms of nouns also will be used
        :return: sentence with synonyms
        '''
        sent1 = word_tokenize(sentence)
        sent_tagged = pos_tag(sent1)
        words2= list(set(list({l.name() for word,tag in sent_tagged if tag[0]!='N' or useNouns==1 for s in wn.synsets(word) for l in s.lemmas()})+sent1))
        sent2 = ' '.join(words2)
        return sent2
    @staticmethod
    def removeProbTexts(lines):
        lines = [re.sub('\\n',' ',sent) for sent in lines]
        lines = [re.sub('\n',' ',sent) for sent in lines]
        lines = [sent.replace('\\','') for sent in lines]
        lines = [sent.replace(".nn"," . ") for sent in lines]
        lines = [sent.replace(":n"," : ") for sent in lines]
        lines = [sent.replace(".n"," . ") for sent in lines]
        lines = [sent.replace('nnn==','').replace('==nn','') for sent in lines]
        lines = ' '.join(lines)
        return lines
