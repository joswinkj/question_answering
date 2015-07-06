#######question answering system
####steps
#load text, create termdocument matrix using countvec - done in load_ref_text
#get query, create termdocument matrix using countvec _ done in  load_query
#get query, analyze query to identify return item - done in analyze query
#do matching, score the results, filter based on answer type, return the highest scored result

####todo
#remove stopwords before passing to countvec(probs with lemmatization)
######
import pandas as pd,numpy as np
import pdb
from nltk import RegexpTokenizer
#from Tokenizers import SynonymTokenizer
from Tokenizers import SynonymStemTokenizer
from Tokenizers import StemTokenizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from rake import Rake
from AnswerProcessing import AnswerProcessor
from textProcessing import changeToMatrix
from rake import Rake
from evaluatingResults import EvaluateResults

def runMainCode(query='how many people are there in india',loc='india.txt.pickle'):
    process_data = changeToMatrix(indexed_data=1)
    sentences,data_processed,query_processed=process_data.get_processed_data(query,loc)#data_processed and query_processed are lists
    print 'evaluating results'
    result_ind_nouns,res_nouns = EvaluateResults(data_processed[0],query_processed[0]).getCosineSimilarity(10)
    # result_ind_all,res_all = EvaluateResults(data_processed[0],query_processed[1]).getCosineSimilarity(10)
    #pdb.set_trace()
    # best_results_ind,best_results_score = EvaluateResults.get_top_from_two_scores(result_ind_nouns,result_ind_notnouns,res_nouns,res_notnouns)
    # res = [sentences[ind] for i,ind in enumerate(best_results_ind)]
    sent_nouns = [str(res_nouns[i])+' '+sentences[ind] for i,ind in enumerate(result_ind_nouns)]
    # sent_notnouns = [str(res_notnouns[i])+' '+sentences[ind] for i,ind in enumerate(result_ind_notnouns)]
    res = [sentences[ind] for i,ind in enumerate(result_ind_nouns)]
    # print '###### nouns #########'
    # print '\n'.join(sent_nouns)
    # print '###### all ##########'
    # print '\n'.join(sent_notnouns)
    # print res
    # print best_results_score
    print '############################'
    # pdb.set_trace()
    word, sent = AnswerProcessor(query,res,res_nouns).get_answer()


if __name__ == "__main__":
    import sys
    if len(sys.argv)>1:
        query=sys.argv[1]
        if len(sys.argv)>2:
            loc=sys.argv[2]
            runMainCode(query,loc)
        else:
            runMainCode(query)
    else:
        runMainCode()
