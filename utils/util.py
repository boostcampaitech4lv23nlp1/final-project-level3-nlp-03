import editdistance
from typing import *
import re
from heapq import heappush
from nltk.tag import pos_tag

def NLD(s1:str,s2:str) -> float:
    return editdistance.eval(s1.lower(),s2.lower()) / ((len(s1)+len(s2))/2) # normalized_levenshtein_distance

def check_answer(answer_list:List[str], words_list:List[str], boundary:int=0.2) -> float:
    similarity_score = 0
    for answer, word in zip(answer_list, words_list):
        # print('answer word', answer, word)
        ld_score = NLD(answer, word)

        # 각 단어의 레벤슈타인 거리가 0.2보다 크면 너무 차이가 많아서 정답이 아님
        if ld_score >= boundary:
            return 100 # 레벤슈타인의 최대값은 1이므로 100은 나올 수 없음

        else: similarity_score += ld_score
        
    return similarity_score / len(answer_list) # 가장 좋은 값은 0.0
    
def calculate_euclidean_mean(question_points:List[Tuple[int]], boxes:List[Tuple[int]]):
    euclidean = 0
    a_point = ((boxes[2] + boxes[0])/2, (boxes[3] + boxes[1])/2)
    for q_point in question_points:
        euclidean += ((a_point[0] - q_point[0])**2 + (a_point[1] - q_point[1])**2)**(1/2)
        
    return euclidean / len(question_points)
 
def clean_text(raw_string:str) -> str:
    #텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?^$.@*\※~ㆍ!…]','', raw_string)
 
    return text
 
def find_noun_ngram(questions:str, ngram:int) -> Set[Tuple[str]]:
    if ngram == 1: # unigram일 경우
        part_of_speech = {'NN', 'NNS','NNP', 'NNPS', 'POS','RP', 'CD', 'FW', 'VBG'}
    else:
        part_of_speech = {'NN', 'NNS','NNP', 'NNPS', 'IN', 'POS','RP', 'CD', 'FW', 'VBG', 'JJR', 'JJS', 'RBR', 'RBS'}
        
    result = set()
    questions:List[str] = questions.split()
    ngram_questions = [questions[i:i+ngram] for i in range(len(questions) - (ngram-1))]
    for question in ngram_questions:
        tmp_storage = []
        for tag in pos_tag(question):
            if tag[1] in part_of_speech:
                tmp_storage.append(clean_text(tag[0]))
                
        if len(tmp_storage) == ngram:
            result.add(tuple(tmp_storage))

    yield from result

def find_points(questions:str, words_list:List[str], boxes:List[List[int]], ngrams:int=3) -> List[Tuple[int]]:
    question_words = sum([[ngram_question for ngram_question in find_noun_ngram(questions, ngram)] for ngram in range(1, ngrams+1)], [])
    # boxes : (x1, y1, x2, y2)
    result = []
    for question in question_words:
        question_list = list(question)
        search_range = len(words_list) - (len(question_list)-1)
        for idx, i in enumerate(range(search_range)):
            nld = check_answer(question_list, words_list[i:i+len(question_list)], boundary=0.2)
            if nld != 100:
                if len(question) % 2 == 0:
                    bb1 =boxes[idx+(len(question)//2)-1]
                    bb2 = boxes[idx+(len(question)//2)]
                    # 
                    bp1 = ((bb1[2] + bb1[0])/2, (bb1[3] + bb1[1])/2)
                    bp2 = ((bb2[2] + bb2[0])/2, (bb2[3] + bb2[1])/2)
                    result.append(((bb2[0] + bb1[0])/2, (bb2[1] + bb1[1])/2))
                else:
                    bb =boxes[idx+ (len(question)//2)]
                    result.append(((bb[2] + bb[0])/2, (bb[3] + bb[1])/2))
                    
    return result

def find_candidates(answer_list:List[str], words_list:List[str], questions, boxes):
    nld_l = []
    search_range = len(words_list) - (len(answer_list)-1)
    for idx, i in enumerate(range(len(words_list))):
        nld = check_answer(answer_list, words_list[i:i+len(answer_list)])
        if nld != 100:
            # 각 원소 : normalized_levenshtein_distance, answer, start_idx, end_idx
            nld_l.append((nld, answer_list, idx, idx+len(answer_list)-1))
                
    if nld_l:
        nld_l.sort(key=lambda x: x[0]) # nld 최솟값 정렬
        if len(nld_l) == 1: # 하나만 뽑힘
            return nld_l[0][1], nld_l[0][2], nld_l[0][3]
        
        elif nld_l[0][0] == nld_l[1][0]: # 여러개 뽑힌 것들 중에 동일한 NLD 존재
            # 핵심 단어와의 유클리디안 거리를 통해 동일한 정답 중에서 최적을 선택
            question_points = find_points(questions, words_list, boxes, ngrams=3)
            
            if not question_points:
                return nld_l[0][1], nld_l[0][2], nld_l[0][3]
          
            candidates = []
            for q in nld_l:
                if q[0] == nld_l[0][0]:
                    euc_dist = calculate_euclidean_mean(question_points, boxes[q[2]])
                    heappush(candidates, [euc_dist, q[1], q[2], q[3]])
                else:
                    break

            return candidates[0][1], candidates[0][2], candidates[0][3]
        else: # 여러개 뽑힌 것들 중에 첫번째가 가장 적합함
            return nld_l[0][1], nld_l[0][2], nld_l[0][3]
    else:
        return None, 0, 0