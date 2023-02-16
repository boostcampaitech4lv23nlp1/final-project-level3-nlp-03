from transformers import PreTrainedTokenizerFast
from nltk.tag import pos_tag
import random
from typing import *
import re
from heapq import heappush
import datasets
import editdistance
import transformers
from datasets.utils.logging import set_verbosity_error
import torch

## nltk 에러가 난다면 nltk 데이터셋을 다운로드 해주시기 바랍니다.
## nltk.download('tagsets')
## nltk.download('averaged_perceptron_tagger')



set_verbosity_error()
class DecoderPreprocess():
    def __init__(self, tokenizer:PreTrainedTokenizerFast, max_length:int, stride:int, boundary):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.boundary = boundary
        
        self.question_start_token_id = tokenizer.convert_tokens_to_ids("<s_question>")
        self.question_end_token_id = tokenizer.convert_tokens_to_ids("</s_question>")
        self.answer_start_token_id = tokenizer.convert_tokens_to_ids("<s_answer>")
        self.answer_end_token_id = tokenizer.convert_tokens_to_ids("</s_answer>")

        self.ignore_id = tokenizer.pad_token_id

    def get_tok_label(self, answer, question):
        words = ["<s_question>"+question+"</s_question><s_answer>"+answer[-1]+"</s_answer>"+self.tokenizer.eos_token]
        boxes = [[0,0,0,0]]
        data = self.tokenizer(text = words,
                              boxes = boxes,
                              max_length=self.max_length,
                              padding = "max_length",
                              truncation = True,
                              return_tensors = "pt")
        return data['input_ids'].squeeze()
    
    def train(self, train_data:datasets.formatting.formatting.LazyBatch) -> transformers.tokenization_utils_base.BatchEncoding:
        """
        max_length = 512를 넘어가는 문장이 들어오게 되면 stride 길이 만큼 중첩해서 문장을 쪼개는 방식의 전처리 함수입니다.
        truncation = only_second로 고정시키며, 이는 첫번째로 들어오는 sentence는 고정시키고 반복적으로 넣어줍니다.
        그리고 second 문장인 context의 길이가 max_length를 넘어가게 되면 max_length만큼 짤라서 나눠서 tokenizer에 넣게 됩니다.
        question + context 문장이 512를 넘어가게 되면 384(max_length) + 128(stride) 길이만큼 토크나이징한 후에,
        question + context[나머지 길이 stride(128) + remainder(ex 124)]만큼 토크나이징을 진행합니다.
        return: 다음과 같은 키 밸류값을 가집니다.
            {
            'input_ids'(List[int]) : 토큰들을 id값으로 반환한 리스트
            'token_type_ids'(List[int]) : 문장을 구분해주는 리스트, BERT에서는 필수이지만, 나머지에서는 Optional합니다. ex) [0,0,0,1,1,1,2,2,2]
            'attention_mask'(List[int]) : attention을 적용할 문장일 경우 1, pad토큰일 경우 0으로 반환하는 리스트 ex) [1,1,1,1,1,0,0,0,0]
            'start_positions'(List[int]) : answer에 해당하는 토큰의 시작 인덱스를 반환하는 리스트
            'end_positions'(List[int]) : answer에 해당하는 토큰의 끝 인덱스를 반환하는 리스트
            }
        """

        '''
        tokenizer input
        question(str) : 찾고자 하는 answer에 대한 단일 question
        words(List[int], dim:(batch, sequence)) : ocr로 추출한 단어들의 리스트
        boxes(List[in], dim:(batch, point:4)) : ocr로 추출한 x, y 좌표를 width, height로 normalize한 좌표계
        '''

        tokenized_sentences = self.tokenizer(
            train_data['question'],
            train_data['words'],
            train_data['boxes'],
            truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
            max_length=self.max_length,
            stride=self.stride,
            return_token_type_ids=True, # BERT 모델일 경우에만 반환
            return_overflowing_tokens=True, # (List[int]) : 여러개로 쪼갠 문장들이 하나의 같은 context라는 것을 나타내주는 리스트, batch 단위일때 사용합니다.
            padding="max_length",
        )
        
        # batch 단위 안에서 각 데이터 및 max_length를 넘어가는 데이터를 포함함
        # ex) if batch = 4, max_length를 안넘을 경우 -> [0,0,0,0], 1,2번째 데이터가 max_len 넘을 경우 -> [0,1,0,1,0,0]
        overflow_to_sample_mapping = tokenized_sentences.pop("overflow_to_sample_mapping")

        # 정답지를 만들기 위한 리스트
        tokenized_sentences['image'] = []
        tokenized_sentences['labels'] = []
        tokenized_sentences['decoder_input_ids'] = []
        for i, example_index in enumerate(overflow_to_sample_mapping):
            # sequence가 속하는 example을 찾는다.
            answers = train_data['answers'][example_index]
            question = train_data['question'][example_index]
            tokenized_sentences['image'].append(train_data['image'][example_index])

            decoder_input_ids = self.get_tok_label(answers, question)
            tokenized_sentences['decoder_input_ids'].append(decoder_input_ids)
            labels = decoder_input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = self.ignore_id
            labels[labels == self.tokenizer.eos_token_id] = self.ignore_id
            labels[:torch.nonzero(labels == self.question_end_token_id).sum() + 1] = self.ignore_id
            tokenized_sentences['labels'].append(labels)

        return tokenized_sentences

    def val(self, val_data:datasets.formatting.formatting.LazyBatch) -> transformers.tokenization_utils_base.BatchEncoding:
        '''
        return: 다음과 같은 키 밸류값을 가집니다.
            {
            'input_ids'(List[int]) : 토큰들을 id값으로 반환한 리스트
            'token_type_ids'(List[int]) : 문장을 구분해주는 리스트, BERT에서는 필수이지만, 나머지에서는 Optional합니다. ex) [0,0,0,1,1,1,2,2,2]
            'attention_mask'(List[int]) : attention을 적용할 문장일 경우 1, pad토큰일 경우 0으로 반환하는 리스트 ex) [1,1,1,1,1,0,0,0,0]
            'questionId'(List[int]) : 해당 질문의 id를 의미합니다.
            }
            
        tokenizer input
        question(str) : 찾고자 하는 answer에 대한 단일 question
        words(List[int], dim:(batch, sequence)) : ocr로 추출한 단어들의 리스트
        boxes(List[in], dim:(batch, point:4)) : ocr로 추출한 x, y 좌표를 width, height로 normalize한 좌표계
        '''

        tokenized_sentences = self.tokenizer(
            val_data['question'],
            val_data['words'],
            val_data['boxes'],
            truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
            max_length=self.max_length,
            stride=self.stride,
            return_token_type_ids=True, # BERT 모델일 경우에만 반환
            return_overflowing_tokens=True, # (List[int]) : 여러개로 쪼갠 문장들이 하나의 같은 context라는 것을 나타내주는 리스트, batch 단위일때 사용합니다.
            padding="max_length",
        )
        
        # batch 단위 안에서 각 데이터 및 max_length를 넘어가는 데이터를 포함함
        # ex) if batch = 4, max_length를 안넘을 경우 -> [0,1,2,3], 1,2번째 데이터가 max_len 넘을 경우 -> [0,1,1,2,2,3]
        overflow_to_sample_mapping = tokenized_sentences.pop("overflow_to_sample_mapping")

        # 정답지를 만들기 위한 리스트
        tokenized_sentences['image'] = []
        tokenized_sentences["start_positions"] = []
        tokenized_sentences["end_positions"] = []
        tokenized_sentences['questionId'] = []
        tokenized_sentences['word_ids'] = []
        tokenized_sentences['labels'] = []
        tokenized_sentences['decoder_input_ids'] = []
        tokenized_sentences['prompt_end_index'] = []
        for i, example_index in enumerate(overflow_to_sample_mapping):
            input_ids = tokenized_sentences["input_ids"][i]
            words_idx_mapping = tokenized_sentences.word_ids(i)
            sequence_ids = tokenized_sentences.sequence_ids(i)
                
            # cls_token의 인덱스를 찾음
            if self.tokenizer._cls_token:
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
            elif 't5' in self.tokenizer.name_or_path:
                cls_index = input_ids.index(self.tokenizer('question')['input_ids'][0]) # t5의 question 토큰을 cls 토큰이라 가정
            else:
                cls_index = input_ids[0] # 첫번째 토큰
            
            # sequence가 속하는 example을 찾는다.
            answers = val_data['answers'][example_index]
            words = val_data['words'][example_index]
            question = val_data['question'][example_index]
            boxes = val_data['boxes'][example_index]
            tokenized_sentences['image'].append(val_data['image'][example_index])

            decoder_input_ids = self.get_tok_label(answers, question)
            tokenized_sentences['decoder_input_ids'].append(decoder_input_ids)
            labels = decoder_input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = self.ignore_id
            labels[:torch.nonzero(labels == self.question_end_token_id).sum() + 1] = self.ignore_id
            tokenized_sentences['labels'].append(labels)

            question_end_index = torch.nonzero(decoder_input_ids == self.question_end_token_id).sum()
            tokenized_sentences['prompt_end_index'].append(question_end_index)

            # 한 이미지/질문에 여러개의 정답이 있으므로 그 중에 random하게 선택
            start_positions = []
            end_positions = []
            for answer in answers:
                answer_list = answer.split()
                match, answer_start_index, answer_end_index = find_candidates(answer_list, words, question, boxes, self.boundary)
                     
                # 만약 찾지 못했다면 정답의 맨 뒤 문자부터 제거하여 재탐색 시작
                if not match and len(answer)>1:
                    for i in range(len(answer), 0, -1):
                        answer_i_list = (answer[:i-1] + answer[i:]).split()
                        match, answer_start_index, answer_end_index = find_candidates(answer_i_list, words, question, boxes, self.boundary)
                        
                        if match:
                            break
                
                if match:
                    # 정답이 있는 context(두번째 문장)의 시작 인덱스
                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1
                    
                    # 정답이 있는 token(두번째 문장)의 끝 인덱스
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    # answer가 현재 span을 벗어났는지 체크
                    if not (words_idx_mapping[token_start_index] <= answer_start_index and answer_end_index <= words_idx_mapping[token_end_index]):
                        start_positions.append(cls_index)
                        end_positions.append(cls_index)
                    else:
                        # token_start_index와 token_end_index를 answer의 시작점과 끝점으로 옮김
                        while token_start_index < len(words_idx_mapping) and words_idx_mapping[token_start_index] < answer_start_index:
                            token_start_index += 1
                        start_positions.append(token_start_index)
                        
                        while words_idx_mapping[token_end_index] > answer_end_index:
                            token_end_index -= 1
                        end_positions.append(token_end_index)
                else:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index) 

            if len(start_positions) > 1:
                if max(end_positions) != 0:
                    start_positions.sort(reverse=True)
                    end_positions.sort(reverse=True)
                    try: # [CLS] 토큰을 정답으로 한 경우를 제거
                        ans_i = random.randrange(len(end_positions[:end_positions.index(0)]))
                    except ValueError: # [CLS] 토큰을 정답으로 한 경우가 없다면
                        ans_i = random.randrange(len(end_positions))
                else: # [CLS] 토큰을 정답으로 한 경우만 존재함
                    ans_i = random.randrange(len(end_positions))
            else: # 단 하나의 정답만 존재
                ans_i = 0
                
            tokenized_sentences["start_positions"].append(start_positions[ans_i])
            tokenized_sentences["end_positions"].append(end_positions[ans_i])
            
            tokenized_sentences['questionId'].append(val_data['questionId'][example_index])
            tokenized_sentences["word_ids"].append([
                (o if sequence_ids[k] == 1 else None) for k, o in enumerate(words_idx_mapping)
            ]) # 원본에서 해당 인덱스를 참조하기 위해

        return tokenized_sentences

    def test(self, test_data:datasets.formatting.formatting.LazyBatch) -> transformers.tokenization_utils_base.BatchEncoding:
        '''
        tokenize된 question + context 문장 중에서
        question에 해당하는 offset_mapping을 None값으로 바꿔주고
        questionId라는 id값을 추가하여 반환합니다.
        return: 다음과 같은 키 밸류값을 가집니다.
            {
            'input_ids'(List[int]) : 토큰들을 id값으로 반환한 리스트
            'token_type_ids'(List[int]) : 문장을 구분해주는 리스트, BERT에서는 필수이지만, 나머지에서는 Optional합니다. ex) [0,0,0,1,1,1,2,2,2]
            'attention_mask'(List[int]) : attention을 적용할 문장일 경우 1, pad토큰일 경우 0으로 반환하는 리스트 ex) [1,1,1,1,1,0,0,0,0]
            'questionId'(List[int]) : 해당 질문의 id를 의미합니다.
            }
        '''
        
        tokenized_sentences = self.tokenizer(
            test_data['question'],
            test_data['words'],
            test_data['boxes'],
            truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
            max_length=self.max_length,
            stride=self.stride,
            return_token_type_ids=True, # BERT 모델일 경우에만 반환
            return_overflowing_tokens=True, # (List[int]) : 여러개로 쪼갠 문장들이 하나의 같은 context라는 것을 나타내주는 리스트, batch 단위일때 사용합니다.
            padding="max_length",
        )
        
        # batch 단위 안에서 각 데이터 및 max_length를 넘어가는 데이터를 포함함
        # ex) if batch = 4, max_length를 안넘을 경우 -> [0,0,0,0], 1,2번째 데이터가 max_len 넘을 경우 -> [0,1,0,1,0,0]
        overflow_to_sample_mapping = tokenized_sentences.pop("overflow_to_sample_mapping")

        # 정답지를 만들기 위한 리스트
        tokenized_sentences['image'] = []
        tokenized_sentences['questionId'] = []
        tokenized_sentences['word_ids'] = []
        for i, example_index in enumerate(overflow_to_sample_mapping):
            sequence_ids = tokenized_sentences.sequence_ids(i)

            tokenized_sentences['image'].append(test_data['image'][example_index])
            tokenized_sentences["questionId"].append(test_data["questionId"][example_index])
            tokenized_sentences["word_ids"].append([
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_sentences.word_ids(i))
            ])

        return tokenized_sentences

def NLD(s1:str,s2:str) -> float:
    return editdistance.eval(s1.lower(),s2.lower()) / ((len(s1)+len(s2))/2) # normalized_levenshtein_distance

def check_answer(answer_list:List[str], words_list:List[str], boundary:float=0.5) -> float:
    similarity_score = 0
    for answer, word in zip(answer_list, words_list):
        ld_score = NLD(answer, word)

        # 각 단어의 레벤슈타인 거리가 0.2보다 크면 너무 차이가 많아서 정답이 아님
        if ld_score >= boundary:
            return 100 # 레벤슈타인의 최대값은 1이므로 100은 나올 수 없음

        else: similarity_score += ld_score
        
    return similarity_score / len(answer_list) # 가장 좋은 값은 0.0
    
def calculate_euclidean_mean(question_points:List[Tuple[float]], boxes:List[Tuple[float]]):
    euclidean = 0
    a_point = ((boxes[2] + boxes[0])/2, (boxes[3] + boxes[1])/2)
    for q_point in question_points:
        euclidean += ((a_point[0] - q_point[0])**2 + (a_point[1] - q_point[1])**2)**(1/2)
        
    return euclidean / len(question_points)

 
def clean_text(raw_string:str) -> str:
    #텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?^$.@*\※~ㆍ!…]','', raw_string)
 
    return text
 
def find_noun_ngram(question:str, ngram:int) -> Set[Tuple[str]]:
    if ngram == 1: # unigram일 경우
        part_of_speech = {'NN', 'NNS','NNP', 'NNPS', 'POS','RP', 'CD', 'FW', 'VBG'}
    else:
        part_of_speech = {'NN', 'NNS','NNP', 'NNPS', 'IN', 'POS','RP', 'CD', 'FW', 'VBG', 'JJR', 'JJS', 'RBR', 'RBS'}
        
    result = set()
    question:List[str] = question.split()
    ngram_questions = [question[i:i+ngram] for i in range(len(question) - (ngram-1))]
    for question in ngram_questions:
        tmp_storage = []
        for tag in pos_tag(question):
            if tag[1] in part_of_speech:
                tmp_storage.append(clean_text(tag[0]))
                
        if len(tmp_storage) == ngram:
            result.add(tuple(tmp_storage))

    yield from result

def find_points(
    question:str,
    words_list:List[str],
    boxes:List[List[float]],
    ngrams:int=3
) -> List[Tuple[float]]:
    
    question_words = sum([[ngram_question for ngram_question in find_noun_ngram(question, ngram)] for ngram in range(1, ngrams+1)], [])
    # boxes : (x1, y1, x2, y2)
    result = []
    for question in question_words:
        if question:
            question_list = list(question)
            search_range = len(words_list) - (len(question_list)-1)
            for idx, i in enumerate(range(search_range)):
                nld = check_answer(question_list, words_list[i:i+len(question_list)], boundary=0.5)
                if nld != 100:
                    # 여러 단어들 중에서 중앙에 위치한 단어 뽑기 및 단어의 정중앙 좌표 위치 뽑아내기
                    if len(question) % 2 == 0: # ex) I love you, too -> love you
                        bb1 = boxes[idx+(len(question)//2)-1]
                        bb2 = boxes[idx+(len(question)//2)]

                        bp1 = ((bb1[2] + bb1[0])/2, (bb1[3] + bb1[1])/2)
                        bp2 = ((bb2[2] + bb2[0])/2, (bb2[3] + bb2[1])/2)
                        result.append(((bp2[0] + bp1[0])/2, (bp2[1] + bp1[1])/2))
                    else: # ex) I love you -> love
                        bb = boxes[idx+ (len(question)//2)]
                        result.append(((bb[2] + bb[0])/2, (bb[3] + bb[1])/2))
                    
    return result

def find_candidates(
    answer_list:List[str],
    words_list:List[str],
    question:str,
    boxes:List[List[float]],
    boundary=0.5
) -> Tuple[Optional[List[str]], float, float]:
    
    nld_l = []
    search_range = len(words_list) - (len(answer_list)-1)
    for idx, i in enumerate(range(search_range)):
        nld = check_answer(answer_list, words_list[i:i+len(answer_list)], boundary=boundary)
        if nld != 100:
            # 각 원소 : normalized_levenshtein_distance, answer, start_idx, end_idx
            nld_l.append((nld, answer_list, idx, idx+len(answer_list)-1))
                
    if nld_l:
        nld_l.sort(key=lambda x: x[0]) # nld 최솟값 정렬
        if len(nld_l) == 1: # 하나만 뽑힘
            return nld_l[0][1], nld_l[0][2], nld_l[0][3]
        
        elif nld_l[0][0] == nld_l[1][0]: # 여러개 뽑힌 것들 중에 동일한 NLD 존재
            # 핵심 단어와의 유클리디안 거리를 통해 동일한 정답 중에서 최적을 선택
            question_points = find_points(question, words_list, boxes, ngrams=3)
            
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