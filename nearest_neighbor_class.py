#nn_method = NearestNeighborMethod(n_results=5)
#nn_method.get_results("what is size of india")
class NearestNeighborMethod(object):
    def __init__(self,n_results=1,ngram_range=(1,1),tokenizer=SynonymTokenizer()):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.neighbors import NearestNeighbors
        self.countvec = CountVectorizer(ngram_range=ngram_range,analyzer='word',lowercase=True,\
            token_pattern='[a-zA-Z0-9]+',strip_accents='unicode',tokenizer=tokenizer)
        self.nbrs = NearestNeighbors(n_neighbors=n_results)

    def load_ref_text(self,text_file):
        import re,nltk.data
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer 
        textfile = open(text_file,'r')
        lines=textfile.readlines()
        textfile.close()
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = [ sent_tokenizer.tokenize(line.strip()) for line in lines]
        sentences1 = [item for sublist in sentences for item in sublist] 
        chk2=pd.DataFrame(self.countvec.fit_transform(sentences1).toarray(),columns=self.countvec.get_feature_names())
        chk2[chk2>1]=1
        return chk2,sentences1
        
        #text1 = []
        #for sent in text:
        #        new_words = []
        #        for word,word_type in sent:
        #                synonyms = list({l.name().lower() for s in wn.synsets(word) for l in s.lemmas()})
        #                new_words.append((synonyms,word_type))
        #        text1.append(new_words)
    

    def load_query(self,text):
        #print text
        chk2=pd.DataFrame(self.countvec.transform([text]).toarray(),columns=self.countvec.get_feature_names())
        #print chk2.shape
        chk2[chk2>1]=1
        return chk2

    def get_scores(self,ref_dataframe,ref_query,n_results=1):        
        self.nbrs.fit(ref_dataframe)
        return self.nbrs.kneighbors(ref_query)

        
    def get_results(self,query):
        ref_dataframe,ref_sentences=NearestNeighborMethod.load_ref_text(self,'india.txt')
        #print ref_dataframe.shape,len(ref_sentences)
        ref_query=NearestNeighborMethod.load_query(self,query)
        neighbors_index = NearestNeighborMethod.get_scores(self,ref_dataframe,ref_query)[1]
        #print type(neighbors_index)
        #print neighbors_index[0]
        neighbors = list( ref_sentences[i] for i in neighbors_index[0] )
        print neighbors
