import pandas as pd,numpy as np
import pdb
from nltk import RegexpTokenizer
import re
from Tokenizers import SynonymTokenizer
from Tokenizers import SynonymStemTokenizer
from Tokenizers import StemTokenizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from rake import Rake
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import NERTagger

rr = Rake()
class AnswerProcessor(object):
    def __init__(self,query,answers,score=None):
        '''answers are a list of strings, query is a string,score is a list of scores for each answer '''
        self.query=query
        self.answers=answers
        self.score = score
        #self.question_types = {'who':'PERSON','whom':'PERSON','whose':'PERSON','where':'LOCATION'\
        #	,'when':('DURATION','DATE'),'how+adj/adv':'NUMBER','how long':'DURATION','how many':'NUMBER','how much':'NUMBER'}
        self.question_types = {'who':'PERSON','whom':'PERSON','whose':'PERSON','where':'LOCATION'\
        	,'when':'CD','how+adj/adv':'CD','how long':'CD','how many':'CD','how much':'CD'}
                ###for what, next noun will be the thing we are searching for
        self.question_type=None
        self.query_tag=None
        self.answers_tag=None
    def stringProcessing(self,only_query=1):
        ''' query is a string, answers is a list of strings. returns tuples with tags, with a list covering '''
        #pdb.set_trace()
        query_posTag = [pos_tag(word_tokenize(self.query))]
        query_nerTag = [AnswerProcessor.stanford_NER_tagger(self.query)]
        self.query_tag = AnswerProcessor.PosToNer(query_posTag,query_nerTag)
        if only_query==0:
            answers_posTag = [pos_tag(word_tokenize(answer)) for answer in self.answers]
            answers_nerTag = [AnswerProcessor.stanford_NER_tagger(answer) for answer in self.answers]
            self.answers_tag = AnswerProcessor.PosToNer(answers_posTag,answers_nerTag)

    
    def getAnswerType(self):
        ''' run only after running stringProcessing '''
        try:
        #pdb.set_trace()
            self.stringProcessing()
            two_words = self.query_tag[0][:2]
            #print self.query_tag
            for row,word_tag in enumerate(two_words):
                #print word_tag
                two_words[row]=(word_tag[0].lower(),word_tag[1])
            if two_words[0][0] != 'how':
                two_words = two_words[0]
                answer_type = self.question_types[two_words[0]]
            else:
                if two_words[1][0] not in ['long','many','much']:
                    if two_words[1][1].startswith('J') or two_words[1][1].startswith('R'):
                        answer_type = self.question_types['how+adj/adv']
                    else:
                        answer_type = 'top rated answer'
                else :
                    answer_type = self.question_types[two_words[0][0]+' '+two_words[1][0]]
        except:
            answer_type = 'top rated answer'
        return answer_type
    
    def getTopAnswer(self,answer_type,method='first'):
        if answer_type=='top rated answer':
            return self.answers[0] , 'No single answer'
        #pdb.set_trace()
        try:
            retWord,retInd = self.getBestAnswer(answer_type,method)
            return self.answers[retInd],retWord
        except:
            return self.answers[0] , 'No single answer'

    def getBestAnswer(self,answer_type,method='first'):
        if method=='first':
            retWord=None
            for ind,ans in enumerate(self.answers_tag):
                for word,tag in ans:
                    if not retWord:
                        if tag==answer_type:
                            retInd = ind
                            retWord = word
                    else:
                        if ind==retInd and tag==answer_type:
                            retWord=retWord+' '+word
                        else:
                            return retWord,retInd
            return None,None
        elif method=='useRakeInd':
            print '###################Getting best answer#########################'
            #prob 1.2 billion...only 1.2 coming
            # pos_tag useful only for numbers, need stanforNER tagger
            # pdb.set_trace()
            ind_answ,ret_word_top = self.get_rake_based_answ(answer_type)
            return ind_answ,ret_word_top

    def get_rake_based_answ(self,answer_type):
        top = 0
        # rr = Rake()
        for sent_id,sent in enumerate(self.answers):
            # pdb.set_trace()
            # sent_bck = re.sub(r',', '',sent) #this not needed as sometimes commas are good..
            # like Japanese surrendered on September 2, ending World War II...rake phrase would be september 2 ending world war
            # without the comma
            pos_to_insert = sent[self.answers_bestpos[sent_id]:].find(' ')
            if pos_to_insert != -1:
                tmp = sent[:self.answers_bestpos[sent_id]+pos_to_insert]+' qsxwdc '+\
                      sent[self.answers_bestpos[sent_id]+pos_to_insert:]
            else:
                tmp = sent + ' qsxwdc '
            tmp = re.sub(r',', ' ',tmp)
            tmp1 = pos_tag(word_tokenize(tmp))
            tmp2=AnswerProcessor.stanford_NER_tagger(tmp)[0]
            tmp3=AnswerProcessor.PosToNer([tmp1],[[tmp2]])[0]
            ans_candidates=[]
            ans_pos=[]
            ref_pos=[ind for ind,(word,tag) in enumerate(tmp3) if word=='qsxwdc']
            ref_pos = ref_pos[0]
            tmp3=[(wrdd,tgg) for wrdd,tgg in tmp3 if wrdd!='qsxwdc']
            #for each possible answer type, add its position to a list, then find min from ref position(pos of 'qsxwdc')
            for ind,(word,tag) in enumerate(tmp3):
                if tag==answer_type:
                    ans_candidates.append(word)
                    ans_pos.append(ind)
            if len(ans_pos)==0:
                continue
            # pdb.set_trace()
            ans_pos_relative = [abs(i-ref_pos) for i in ans_pos]
            # ans_pos_relative.sort()
            ans_ind_min = ans_pos_relative.index(min(ans_pos_relative))#index of ans_pos_relative with minimum distance from ref_pos\
                                                                        #final answer should include this
            ans_ind_max=ans_ind_min+1
            tmp=ans_pos[:]
            tmp.sort()
            for i in tmp:
                if i==tmp[ans_ind_min]+1:
                    ans_ind_max += 1
            tmp.sort(reverse=True)
            for i in tmp:
                if i==ans_pos[ans_ind_min]-1:
                    ans_ind_min -= 1

            ret_word=ans_candidates[ans_ind_min:ans_ind_max]
            if top==0:
                ret_word_top=ret_word
                ret_ind = sent_id
            # pdb.set_trace()
            tmp = rr.run(sent.lower())
            rake_exp = [ph for ph,scr in tmp if ' '.join(ret_word).lower() in ph]
            # pdb.set_trace()
            if len(rake_exp)==0:
                rake_exp.append('No Rake Expression')
            print ' '.join(ret_word) + ' -Rake phrase: '+rake_exp[0] +', Full Sentence: '+sent
        return ret_ind,ret_word_top

    @staticmethod
    def PosToNer(posTag,nerTag):
        '''method to add ner tag data to pos tag data. 
        posTag should be of form [[('It', 'PRP'), ('is', 'VBZ')..],[(u'Peninsular', u'JJ'), (u'India', u'NNP')..]]
        nerTag should be of form [[[(u'It', u'O'), (u'is', u'O')..]],[[(u'Peninsular', u'ORGANIZATION'), (u'India', u'ORGANIZATION')..]]]'''
        for row,sent in enumerate(nerTag):
            sent=sent[0]
            for col,word_tag in enumerate(sent):
                word,tag=word_tag[0],word_tag[1]
                if tag=='LOCATION' or tag=='PERSON' or tag=='ORGANIZATION':
                    posTag[row][col]=(word,tag)
        return posTag            
    
    @staticmethod    	
    def stanford_NER_tagger(sentence):
        st = NERTagger('/home/madan/Desktop/question_answering/stanford_named_entity_tagger/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz' \
        ,'/home/madan/Desktop/question_answering/stanford_named_entity_tagger/stanford-ner-2014-06-16/stanford-ner.jar' )
        return st.tag(word_tokenize(sentence))
    
    def getAnswer(self):
        self.stringProcessing()
        answer_type=self.getAnswerType()
        #pdb.set_trace()
        answer , single_answer = self.getTopAnswer(answer_type)
        return single_answer,answer

    def get_best_answer_rake(self,answertype):
        """ """
        wnl = WordNetLemmatizer()
        tmp = rr.run(self.query_cleaned.lower())
        query_phrases =[]
        for phr,score in tmp:
            words = word_tokenize(phr)
            words1 = list(set(list({l.name() for word in words for s in wn.synsets(wnl.lemmatize(word)) for l in s.lemmas()})+words))
            query_phrases.append(words1)
        query_phrases1=[i for l in query_phrases for i in l]
        query_words = list(set(query_phrases1))
        answers_pos = []
        #max_match = 0 # phrase with maximum match..suppose it is 3, plan is to multiply score of a sent with match of sent(suppose 2)/3
        ##run through all answers to get the phrases. look for the location of the best matching phrase
        self.rake_match = [] # max of rake phrase match with question words for each answer
        for score_ind,sent in enumerate(self.answers_cleaned):
            # pdb.set_trace()
            tmp = rr.run(sent.lower())
            phr_scores = []
            for phr,score in tmp:
                phr_words = []
                for word in word_tokenize(phr):
                    phr_words.append(wnl.lemmatize(word))
                n_match=0
                for word in phr_words:
                    if word in query_words:
                        n_match += 1
                phr_scores.append(n_match)
            best_phr_ind = [i for i, x in enumerate(phr_scores) if x == max(phr_scores)]
            ans_inds = []
            # pdb.set_trace()
            for ind in best_phr_ind:
                for m in re.finditer(tmp[ind][0],sent.lower()):
                    ans_inds.append((m.start()+m.end())/2)
            ans_ind = reduce(lambda x, y: x + y, ans_inds)/len(ans_inds)
            answers_pos.append(ans_ind)
            self.rake_match.append(max(phr_scores))
            print ans_ind,self.score[score_ind],self.rake_match[score_ind],self.answers[score_ind]
            # pdb.set_trace()
        self.answers_bestpos = answers_pos
        self.compute_rake_score()
        ans_final,word_final = self.getTopAnswer(answertype,'useRakeInd')
        return word_final,ans_final

    # def compute_final_score(self):


    def compute_rake_score(self):
        max_score = float(max(self.rake_match))
        rake_score = [i/max_score for i in self.rake_match]
        self.final_score = [i*self.score[ind] for ind,i in enumerate(rake_score) ]

    def string_cleaning(self,re_exp="[^a-zA-Z0-9.%]"):
        anss=[]
        for sent in self.answers:
            sent=re.sub(re_exp, ' ', sent)
            anss.append(sent)
        qq=re.sub(re_exp,' ',self.query)
        self.answers_cleaned = anss
        self.query_cleaned = qq



    def get_answer(self):
        '''
         uses answers and query to get the best answer
        '''
        #self.query_tag = pos_tag(self.query)
        self.string_cleaning()
        answer_type=self.getAnswerType()
        return self.get_best_answer_rake(answer_type)
