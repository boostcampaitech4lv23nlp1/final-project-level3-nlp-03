import collections
from tqdm.auto import tqdm
import json
import numpy as np
import editdistance
import datasets

class Metrics_nbest():
    def __init__(
        self,
        val_data:datasets.arrow_dataset.Dataset=None,
        val_dataset:datasets.arrow_dataset.Dataset=None,
        test_data:datasets.arrow_dataset.Dataset=None,
        test_dataset:datasets.arrow_dataset.Dataset=None,
        n_best_size:int=20,
        max_answer_length:int=20
    ):
        data_remove_columns = ['question', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'data_split', 'boxes']
        dataset_remove_columns = ['image', 'input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'start_positions', 'end_positions']
        dataset_remove_columns_t = ['image', 'input_ids', 'token_type_ids', 'attention_mask', 'bbox']
        self.do_test, self.do_test = False, False
        if val_data and val_dataset:
            self.val_examples = val_data.remove_columns(data_remove_columns) # 전처리 전의 원본 데이터인 example을 의미
            self.val_features = val_dataset.remove_columns(dataset_remove_columns) # 전처리가 완료된 dataset을 의미
            self.do_val = True
        if test_data and test_dataset:
            self.test_examples = test_data.remove_columns(data_remove_columns) # 전처리 전의 원본 데이터인 example을 의미
            self.test_features = test_dataset.remove_columns(dataset_remove_columns_t) # 전처리가 완료된 dataset을 의미
            self.do_test = True
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        
    def _make_features_per_example(self, examples, features) -> collections.defaultdict:
        # 각 문서의 id를 키값, index를 밸류값으로 하는 딕셔너리 생성(참조용)
        '''
        키값으로 인덱스, 밸류값으로 동일한 아이디를 가지는 문서들의 index를 가지는 리스트(example_id_to_index 참조)
        ex) features_per_example[defaultdict] : {0: [0], 1: [1], 2: [2], 3: [3, 4], 4: [5, 6]}
        3 : [3,4]인 경우 questionId가 동일하지만 문장의 길이가 max_length보다 길어서 truncation되서 나눠진 데이터
        '''
        
        example_id_to_index = {k: i for i, k in enumerate(examples["questionId"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(tqdm(features)):
            features_per_example[example_id_to_index[feature["questionId"]]].append(i)
            
        return features_per_example
    
    def make(self):
        if self.do_val:
            self.val_features_per_example = self._make_features_per_example(self.val_examples, self.val_features)
        if self.do_test:
            self.test_features_per_example = self._make_features_per_example(self.test_examples, self.test_features)
        
    def _set_save_dir(self, save_dir):
        self.save_dir = save_dir
        print('='*50, f'저장 경로는 {self.save_dir} 입니다.' , '='*50, sep='\n\n')

    def compute_val(
        self,
        all_start_logits:np.ndarray,
        all_end_logits:np.ndarray,
        epoch:int=1
    ):

        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        # Prediction 시작(전체 example들에 대한 main Loop)
        for example_index, example in enumerate(tqdm(self.val_examples)):
            # 해당하는 현재 example_index(key) ex) 3 : [3,4]에서 feature_indices는 [3,4]에 해당됩니다.
            feature_indices = self.val_features_per_example[example_index]
            prelim_predictions = []
            
            # 현재 example에 대한 모든 feature 생성합니다.
            for feature_index in feature_indices: # ex) [3,4]
                # 각 featureure에 대한 모든 prediction을 가져옵니다.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                # logit과 original context의 logit을 mapping합니다.
                word_idx_mapping = self.val_features[feature_index]["word_ids"]

                # n_best size만큼 큰 값 순으로 인덱스 정렬 및 reverse slicing([int:int:-1])
                start_indexes = np.argsort(start_logits)[-1 : (-self.n_best_size - 1) : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : (-self.n_best_size - 1) : -1].tolist()
                
                # n_best_size^2 만큼 완전탐색
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # out-of-scope answers는 고려하지 않습니다.
                        if (
                            start_index >= len(word_idx_mapping) # max_len을 벗어난 경우
                            or end_index >= len(word_idx_mapping)
                            or word_idx_mapping[start_index] is None # context가 아닌 question이나 special token일 경우
                            or word_idx_mapping[end_index] is None
                        ): continue

                        # 길이(end - start)가 < 0 또는 길이가 > max_answer_length(하이퍼 파라미터)인 answer도 고려하지 않습니다.
                        if (
                            end_index < start_index # 길이가 0 미만인 경우
                            or end_index - start_index + 1 > self.max_answer_length # max_answer_length보다 긴 경우
                        ): continue

                        # n_best_size내에서 고려할 수 있는 모든 경우를 추가합니다.
                        prelim_predictions.append(
                            {
                                "offsets": (
                                    word_idx_mapping[start_index],
                                    word_idx_mapping[end_index],
                                ),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                        
            # feature_indices(ex) [3,4])에 대한 탐색을 끝내고
            # 모든 truncation 문장을 포함해서 가장 좋은 `n_best_size` predictions만 유지합니다.
            predictions = sorted(prelim_predictions,
                                 key=lambda x: x["score"], reverse=True # 내림차순
                                 )[:self.n_best_size] # n_best_size만큼 남기기

            # offset을 사용하여 original context에서 predict answer text를 수집합니다.
            words = example["words"]
            for pred in predictions:
                offsets = pred.pop("offsets") # offsets key의 value값을 pop  (start, end)
                if np.isnan(offsets[0]) or np.isnan(offsets[1]): #[CLS] 토큰 등 에측에 실패한 경우
                    pred['pred'] = ''
                else:
                    pred_word = ' '.join(words[int(offsets[0]) : int(offsets[1])+1])
                    pred["pred"] = pred_word #predictions에 {'pred' : predict answer text}를 추가

            # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["pred"] == ""):
                predictions.insert( # 예측에 실패했으므로 가장 높은 점수로써 기본 0값을 넣습니다.
                    0, {"pred": "",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0}
                )

            # 모든 점수의 소프트맥스를 계산합니다
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # 예측값에 확률을 포함합니다.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # best prediction을 선택합니다. all_predictions에 id에 해당하는 가장 높은 확률[0]의 예상 text를 추가합니다.
            all_predictions[example["questionId"]] = {'pred' : predictions[0]["pred"], 'answers' : example['answers']}

            # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
            all_nbest_json[example["questionId"]] = [{'answers': example['answers']}]
            for pred in predictions:
                all_nbest_json[example["questionId"]].append(
                    { k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items() }
                )

        # all_predictions와 n_best를 json파일로 내보내기
        with open(f'save/{self.save_dir}/predictions_{epoch}.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + '\n')

        with open(f'save/{self.save_dir}/nbest_predictions_{epoch}.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + '\n')

        # 실제 계산단계
        return self._compute_ANLS(all_predictions)

    def compute_test(
        self,
        all_start_logits:np.ndarray,
        all_end_logits:np.ndarray,
    ):

        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = []

        # Prediction 시작(전체 example들에 대한 main Loop)
        for example_index, example in enumerate(tqdm(self.test_examples)):
            # 해당하는 현재 example_index(key) ex) 3 : [3,4]에서 feature_indices는 [3,4]에 해당됩니다.
            feature_indices = self.test_features_per_example[example_index]
            prelim_predictions = []
            
            # 현재 example에 대한 모든 feature 생성합니다.
            for feature_index in feature_indices: # ex) [3,4]
                # 각 featureure에 대한 모든 prediction을 가져옵니다.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                # logit과 original context의 logit을 mapping합니다.
                word_idx_mapping = self.test_features[feature_index]["word_ids"]

                # n_best size만큼 큰 값 순으로 인덱스 정렬 및 reverse slicing([int:int:-1])
                start_indexes = np.argsort(start_logits)[-1 : (-self.n_best_size - 1) : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : (-self.n_best_size - 1) : -1].tolist()
                
                # n_best_size^2 만큼 완전탐색
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # out-of-scope answers는 고려하지 않습니다.
                        if (
                            start_index >= len(word_idx_mapping) # max_len을 벗어난 경우
                            or end_index >= len(word_idx_mapping)
                            or word_idx_mapping[start_index] is None # context가 아닌 question이나 special token일 경우
                            or word_idx_mapping[end_index] is None
                        ): continue

                        # 길이(end - start)가 < 0 또는 길이가 > max_answer_length(하이퍼 파라미터)인 answer도 고려하지 않습니다.
                        if (
                            end_index < start_index # 길이가 0 미만인 경우
                            or end_index - start_index + 1 > self.max_answer_length # max_answer_length보다 긴 경우
                        ): continue

                        # n_best_size내에서 고려할 수 있는 모든 경우를 추가합니다.
                        prelim_predictions.append(
                            {
                                "offsets": (
                                    word_idx_mapping[start_index].item(),
                                    word_idx_mapping[end_index].item(),
                                ),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                        
            # feature_indices(ex) [3,4])에 대한 탐색을 끝내고
            # 모든 truncation 문장을 포함해서 가장 좋은 `n_best_size` predictions만 유지합니다.
            predictions = sorted(prelim_predictions,
                                 key=lambda x: x["score"], reverse=True # 내림차순
                                 )[:self.n_best_size] # n_best_size만큼 남기기

            # offset을 사용하여 original context에서 predict answer text를 수집합니다.
            words = example["words"]
            for pred in predictions:
                offsets = pred.pop("offsets") # offsets key의 value값을 pop  (start, end)
                if np.isnan(offsets[0]) or np.isnan(offsets[1]): #[CLS] 토큰 등 에측에 실패한 경우
                    pred['pred'] = ''
                else:
                    pred_word = ' '.join(words[int(offsets[0]) : int(offsets[1])+1])
                    pred["pred"] = pred_word #predictions에 {'pred' : predict answer text}를 추가

            # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["pred"] == ""):
                predictions.insert( # 예측에 실패했으므로 가장 높은 점수로써 기본 0값을 넣습니다.
                    0, {"pred": "",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0}
                )

            # 모든 점수의 소프트맥스를 계산합니다
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # 예측값에 확률을 포함합니다.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # best prediction을 선택합니다. all_predictions에 id에 해당하는 가장 높은 확률[0]의 예상 text를 추가합니다.
            all_predictions.append({
                'answer':predictions[0]["pred"],
                'questionId':[example["questionId"]]
            })

        # all_predictions와 n_best를 json파일로 내보내기
        with open(f'save/{self.save_dir}/submission.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + '\n')
            
    def _compute_ANLS(self, all_predictions:dict):
        score = 0
        # all predictions key : question_id / value : {pred, answer}
        for prediction in all_predictions.values():
            pred = prediction['pred']
            scores = []
            for answer in prediction['answers']:
                ed = editdistance.eval(answer.lower(), pred.lower())
                NL = ed/max(len(answer),len(pred))
                scores.append(1-NL if NL<0.5 else 0)
            score += max(scores)
        return score / len(all_predictions)