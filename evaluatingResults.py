import numpy as np,pandas as pd
from rake import Rake
from nltk import word_tokenize

class EvaluateResults(object):
    def __init__(self,data,query):
        '''
        :param data: processed datas on which search is done
        :param query: search question
        :return:indices of the top answers
        '''
        self.data=data
        self.query=query

    def getDotProductSimilarity(self,n_results=1):
        '''
        both data and query are pandas dataframes
        :param n_results:no of results
        :return: list: indices of top answers
        '''
        res=np.sum(np.multiply(self.data.values,self.query.values),axis=1)
        res_ind=(-res).argsort()[:n_results]
        #pdb.set_trace()
        return list(res_ind)

    def getCosineSimilarity(self,n_results=1):
        '''
        both data and query are pandas dataframes
        :param n_results: no of results
        :return: two lists:indices of top answers,scores for top answers
        '''
        dot_product=np.sum(np.multiply(self.data.values,self.query.values),axis=1)
        data_mod = np.sum(np.multiply(self.data.values,self.data.values),axis=1)
        query_mod = np.sum(np.multiply(self.query.values,self.query.values))
        res = np.true_divide(dot_product,np.multiply(data_mod,query_mod))
        res_ind=(-res).argsort()[:n_results]
        res_out=[res[i] for i in res_ind]
        return list(res_ind),list(res_out)

    @staticmethod
    def get_phrases(sents,search_text,res_ind):
        '''
        :param sents: list of sentences for search
        :param search_text: search text
        :res_ind: indices of best matching sents
        :return: phrases from query and top results
        '''
        full_text=' . '.join([sents[i] for i in res_ind])
        full_text = full_text +' . '+search_text
        rake = Rake()
        keys = rake.run(full_text)
        print keys
        query_phrases=[]
        query_words=word_tokenize(search_text)
        for phr,score in keys:
            words=word_tokenize(phr)
            flag_present=1
            for word in words:
                if word not in query_words:
                    flag_present=0
            if flag_present==1:
                query_phrases.append((phr,score))
        print query_phrases
        ###change the phrase to all possible synonyms, find the phrase with maximum match
        ###look for the nearest answer type to that phrase
        return keys


    @staticmethod
    def get_top_from_two_scores(res1_ind,res2_ind,res1,res2,mult_factor_1=1,mult_factor_2=1):
        '''
         :param res1_ind : indices for res1
         :param res2_ind : indices for res2
         :param res1 : res1 score
         :param res2 : res2 score
         :param mult_factor_1 : multiply res1 by this,default 1
         :param mult_factor_2 : multiply res2 by this,default 1
         :return: list of indices of the data array with best score
        '''
        #import pdb
        #pdb.set_trace()
        res_ind_common = list(set(res1_ind)&set(res2_ind))
        res_score = [res1[res1_ind.index(i)]*mult_factor_1+res2[res2_ind.index(i)]*mult_factor_2 for i in res_ind_common]
        res_ind_sorted = (-np.array(res_score)).argsort()
        res_ind_final = [res_ind_common[ind] for ind in res_ind_sorted]
        res_score_final = [res_score[i] for i in res_ind_sorted]
        return res_ind_final,res_score_final
