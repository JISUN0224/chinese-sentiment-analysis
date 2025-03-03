#기존 규칙기반 감성코드

import sys
import gzip
from collections import defaultdict
from itertools import product
import jieba
import csv
import pandas as pd

class Struct(object):
    def __init__(self, word, sentiment, pos, value, class_value):
        self.word = word
        self.sentiment = sentiment
        self.pos = pos
        self.value = value
        self.class_value = class_value

class Result(object):
    def __init__(self, score, score_words, not_word, degree_word):
        self.score = score
        self.score_words = score_words
        self.not_word = not_word
        self.degree_word = degree_word

class Score(object):
    # 감성 분류(대분류:소분류)
    score_class = {'乐': ['PA', 'PE'],
                   '好': ['PD', 'PH', 'PG', 'PB', 'PK'],
                   '怒': ['NA'],
                   '哀': ['NB', 'NJ', 'NH', 'PF'],
                   '惧': ['NI', 'NC', 'NG'],
                   '恶': ['NE', 'ND', 'NN', 'NK', 'NL'],
                   '惊': ['PC']
                   }
    # ICTPOS 3.0
    POS_MAP = {
        'noun': 'n',
        'verb': 'v',
        'adj': 'a',
        'adv': 'd',
        'nw': 'al',  # 인터넷 용어
        'idiom': 'al',
        'prep': 'p',
    }

    # 부정어 추가
    NOT_DICT = set(['不', '不是', '不大', '没', '无', '非', '莫', '弗', '毋',
                    '勿', '未', '否', '别', '無', '休', '不一定是', '不一定', '不太'])

    def __init__(self, sentiment_dict_path, degree_dict_path, stop_dict_path):
        self.sentiment_struct, self.sentiment_dict = self.load_sentiment_dict(sentiment_dict_path)
        self.degree_dict = self.load_degree_dict(degree_dict_path)
        self.stop_words = self.load_stop_words(stop_dict_path)

    def load_stop_words(self, stop_dict_path):
        stop_words = [w for w in open(stop_dict_path).readlines()]
        return stop_words

    def remove_stopword(self, words):
        words = [w for w in words if w not in self.stop_words]
        return words

    def load_degree_dict(self, dict_path):
        """정도 부사 사전 불러오기
        Args:
            dict_path: 정도 부사 사전 파일 경로, 파일 형식 word\tdegree
                       정도 부사 분류: 极其, 很, 较, 稍, 欠, 超
       Returns:
            dict = {word: degree}
        """
        degree_dict = {}
        with open(dict_path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip()
                word, degree = line.split('\t')
                degree = float(degree)
                degree_dict[word] = degree
        return degree_dict

    def load_sentiment_dict(self, dict_path):
        """감성 사전 불러오기
        Args:
            dict_path: 감성 사전 파일 경로. GitHub README.md 참조
        Returns:
            dict = {(word, postag): 극성}
        """
        sentiment_dict = {}
        sentiment_struct = []

        with open(dict_path, 'r', encoding='UTF-8') as f:
            for index, line in enumerate(f):
                if index == 0:  # title
                    continue
                items = line.split('\t')
                word = items[0]
                pos = items[1]
                sentiment = items[4]
                intensity = items[5]  # 1, 3, 5, 7, 9 분류, 9 최대치, 1 최소치.
                polar = items[6]  # 극성

                # 단어의 품사를 ICTPOS 품사 체계로 변환
                pos = self.__class__.POS_MAP[pos]
                intensity = int(intensity)
                polar = int(polar)

                # 감성 경향의 표현 형태 변환, 음수는 부정, 0은 중립, 양수는 긍정
                # 숫자의 절대 값은 극성의 강도를 나타냄 // 3 범주로 나누어 극성: 긍정(+1), 중립(0), 부정(-1)을 나타내며,
                # 숫자의 크기는 극성의 강도를 나타냄
                value = None
                if polar == 0:  # 중립
                    value = 0
                elif polar == 1:  # 긍정
                    value = intensity
                elif polar == 2:  # 부정
                    value = -1 * intensity
                else:  # 무효
                    continue

                key = word
                sentiment_dict[key] = value

                # 대응하는 대분류 찾기
                for item in self.score_class.items():
                    key = item[0]
                    values = item[1]
                    for x in values:
                        if (sentiment == x):
                            class_value = key
                sentiment_struct.append(Struct(word, sentiment, pos,value, class_value))
            return  sentiment_struct, sentiment_dict


    def findword(self, text):  # 텍스트에 포함된 감정 단어 찾기
        word_list = []
        for item in self.sentiment_struct:
            if item.word in text:
                word_list.append(item)
        return word_list

    def classify_words(self, words):
        # 이 3개의 키는 단어의 인덱스(색인)임

        sen_word = {}
        not_word = {}
        degree_word = {}
        # sent, not, degree에 해당하는 항목을 찾기; words는 분리된 단어들의 목록임
        for index, word in enumerate(words):
            if word in self.sentiment_dict and word not in self.__class__.NOT_DICT and word not in self.degree_dict:
                sen_word[index] = self.sentiment_dict[word]
            elif word in self.__class__.NOT_DICT and word not in self.degree_dict:
                not_word[index] = -1
            elif word in self.degree_dict:
                degree_word[index] = self.degree_dict[word]
        return sen_word, not_word, degree_word

    def get2score_position(self, words):
        sen_word, not_word, degree_word = self.classify_words(words)  # 딕셔너리

        score = 0
        start = 0
        # 모든 감성 단어, 부정 단어, 정도 부사의 위치(색인, 순서)를 저장하는 리스트
        sen_locs = sen_word.keys()
        not_locs = not_word.keys()
        degree_locs = degree_word.keys()
        senloc = -1
        # 문장의 모든 단어(단어의 절대 위치)에 대해 반복
        for i in range(0, len(words)):
            if i in sen_locs:
                W = 1  # 감성 단어 간의 가중치 재설정
                not_locs_index = 0
                degree_locs_index = 0

                # senloc는 감성 단어 위치 목록의 인덱스, 이전의 sen_locs는 감성 단어의 위치가 다시 자르고 나서의 목록의 위치 인덱스
                senloc += 1
                if (senloc == 0):  # 첫 번째 감성 단어, 앞에 부정 단어, 정도 부사가 있는지 확인
                    start = 0
                elif senloc < len(sen_locs):  # 이전 감성 단어와의 사이에 부정 단어, 정도 부사가 있는지 확인
                    start = previous_sen_locs

                for j in range(start, i):  # 단어 사이의 상대적 위치
                    # 만약 부정 단어가 있다면
                    if j in not_locs:
                        W *= -1
                        not_locs_index = j
                    # 만약 정도 부사가 있다면
                    elif j in degree_locs:
                        W *= degree_word[j]
                        degree_locs_index = j

                # 부정 단어와 정도 부사의 위치 판단: 1) 부정사(앞)+정도부사(뒤) = 정도 부사 강도 감소 不是很；
                # 2) 정도사(앞)+부정사(뒤)= 정도 부사 강도 그대로 很不是
                if ((not_locs_index > 0) and (degree_locs_index > 0)):
                    if (not_locs_index < degree_locs_index):
                        degree_reduce = (float(degree_word[degree_locs_index] / 2))
                        W += degree_reduce
                score += W * float(sen_word[i])  # 감성 단어 점수를 직접 추가
                previous_sen_locs = i
        return score

    # get2score
    def get2score(self, text):
        word_list = self.findword(text)  ##텍스트에 포함된 정도 어휘 찾은 다음 각각의 값을 누적
        pos_score = 0
        pos_word = []
        neg_score = 0
        neg_word = []
        for word in word_list:
            if (word.value > 0):
                pos_score = pos_score + word.value
                pos_word.append(word.word)
            else:
                neg_score = neg_score + word.value
                neg_word.append(word.word)
        print("pos_score=%d; neg_score=%d" % (pos_score, neg_score))

    def getscore(self, text):
        word_list = self.findword(text)  ##텍스트에 포함된 어떤 정도 단어가 있는지 찾기
        # 정도 부사 + 부정어 추가
        not_w = 1
        not_word = []
        for notword in self.__class__.NOT_DICT:  # 부정어
            if notword in text:
                not_w = not_w * -1
                not_word.append(notword)
        degree_word = []
        for degreeword in self.degree_dict.keys():
            if degreeword in text:
                degree = self.degree_dict[degreeword]
                degree_word.append(degreeword)
        # 7개의 대분류에 해당하는 단어를 찾아 각각의 점수= 단어 극성 * 단어 가중치를 계산
        result = []
        for key in self.score_class.keys():  # 7개의 대분류 구분
            score = 0
            score_words = []
            for word in word_list:

                if (key == word.class_value):
                    score = score + word.value
                    score_words.append(word.word)
            if score > 0:
                score = score + degree
            elif score < 0:
                score = score - degree
            score = score * not_w

            x = '{}_score={}; word={}; nor_word={}; degree_word={};'.format(key, score, score_words, not_word,
                                                                             degree_word)
            print(x)
            result.append(x)
        return result

#깃허브 첨부파일 참고
if __name__ == '__main__':
    sentiment_dict_path = "sentiment_words_chinese.tsv"
    degree_dict_path = "degree_dict.txt"
    stop_dict_path = "stop_words.txt"

    # 파일 읽기
    f = open('분석할 파일.csv', encoding='utf-8')
    data = pd.read_csv(f)

    # 파일 쓰기
    c = open("Result.csv", "w", newline='', encoding='utf-8')
    writer = csv.writer(c)
    writer.writerow(["no", "review", "score"])

    # 문장 분리 기능, 부정어 정도어 위치 판단
    score = Score(sentiment_dict_path, degree_dict_path, stop_dict_path)

    n = 1
    for temp in data['评论']:
        tlist = []
        words = [x for x in jieba.cut(temp)]  # 단어 분리
        words_ = score.remove_stopword(words)
        print(words_)

        # 단어 분리 -> 부정어 정도어 간에 위치 -> 점수 누적
        result = score.get2score_position(words_)
        print(result)

        tlist.append(str(n))
        tlist.append(words)
        tlist.append(str(result))
        writer.writerow(tlist)
        n = n + 1

        # 문장 -> 전체 문장 부정어 정도어 판별 -> 긍정 부정 단어 분리
        #score.get2score(temp)
        #score.getscore(text)
    c.close()

2.사전 수정 및 보완

2-1 빈도+엔트로피 문자열 문자열 주변에 나타나는 좌측 문맥(Left Context)과 우측 문맥(Right Context)의 분포를 계산하여 엔트로피를 측정합니다.

  from collections import Counter
import jieba  # 중국어 형태소 분석기
import math
import re

# 텍스트 파일에서 데이터 읽기
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data

# 기존 감성 사전 파일에서 첫 번째 열 단어만 읽기
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lexicon = set(line.split('\t')[0] for line in file if line.strip())  # 첫 번째 열만 추출
    return lexicon

# 외부 인접 정보 엔트로피 계산 함수
def calculate_entropy(tokens):
    token_freq = Counter(tokens)
    total_tokens = len(tokens)
    entropy = 0
    for freq in token_freq.values():
        prob = freq / total_tokens
        entropy -= prob * math.log2(prob)
    return entropy

# 내부 결속도 계산 함수
def calculate_cohesion(tokens, substr, freq_counter):
    substr_len = len(substr)
    if substr_len <= 1:
        return 0
    total_freq = freq_counter[substr]
    left_freq = freq_counter.get(substr[:-1], 1)  # 왼쪽 부분 빈도
    right_freq = freq_counter.get(substr[1:], 1)  # 오른쪽 부분 빈도
    return total_freq / (left_freq * right_freq)

# 새로운 단어 후보 식별 함수
def identify_new_words(freq_counter, cohesion_counter, entropy_counter, freq_threshold, cohesion_threshold, entropy_threshold):
    new_words = []
    for substr, freq in freq_counter.items():
        if freq >= freq_threshold:
            cohesion = cohesion_counter.get(substr, 0)
            entropy = entropy_counter.get(substr, 0)
            if cohesion >= cohesion_threshold and entropy >= entropy_threshold:
                new_words.append(substr)
    return new_words

# 텍스트 전처리 (중국어만 남기기)
def preprocess_text(text):
    return re.sub(r'[^\u4e00-\u9fff]', '', text)

# 메인 실행 함수
def main(file_path, lexicon_path):
    # 텍스트 데이터 로드 및 전처리
    text_data = read_text_file(file_path)
    text_data = preprocess_text(text_data)

    # 기존 감성 사전 로드 (첫 번째 열만 가져오기)
    existing_lexicon = load_lexicon(lexicon_path)

    # 중국어 형태소 분석 (jieba 사용)
    tokens = list(jieba.cut(text_data))

    # 빈도 계산
    freq_counter = Counter(tokens)

    # 내부 결속도 계산
    cohesion_counter = {}
    for substr in freq_counter:
        cohesion_counter[substr] = calculate_cohesion(tokens, substr, freq_counter)

    # 외부 엔트로피 계산
    entropy_counter = {}
    for substr in freq_counter:
        entropy_counter[substr] = calculate_entropy(tokens)

    # 임계값 설정
    freq_threshold = 8
    cohesion_threshold = 1.0
    entropy_threshold = 0.5

    # 새로운 단어 후보 식별
    new_words = identify_new_words(freq_counter, cohesion_counter, entropy_counter, freq_threshold, cohesion_threshold, entropy_threshold)

    # 기존 감성 사전 단어 필터링
    filtered_words = [word for word in new_words if word not in existing_lexicon]

    # 결과 출력
    print("Identified New Words:", filtered_words)

# 파일 경로 입력
if __name__ == "__main__":
    file_path = "your_text_file.txt"  # 텍스트 파일 경로
    lexicon_path = "your_lexicon_file.tsv"  # 감성 사전 파일 경로
    main(file_path, lexicon_path)


2-2 N-gram

from collections import Counter, defaultdict
import jieba.posseg as pseg  # 품사 태깅을 위한 모듈
import re

# 텍스트 파일에서 데이터 읽기
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data

# 불용어 파일에서 불용어 리스트 읽기
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    return stopwords

# 텍스트 전처리 (중국어만 남기기)
def preprocess_text(text):
    return re.sub(r'[^\u4e00-\u9fff]', '', text)

# n-gram 생성 함수 (품사 기반 필터링)
def generate_pos_based_ngrams(tokens, n, pos_filter):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        gram = tokens[i:i + n]
        # 품사 태그 필터링
        if all(pos in pos_filter for _, pos in gram):
            ngrams.append(' '.join([word for word, _ in gram]))
    return ngrams

# 메인 실행 함수
def main(file_path, stopwords_path, min_frequency=5, ngram_range=(2, 3), pos_filter={'n', 'v', 'a'}):
    # 텍스트 데이터 로드 및 전처리
    text_data = read_text_file(file_path)
    text_data = preprocess_text(text_data)

    # 불용어 리스트 로드
    stopwords = load_stopwords(stopwords_path)

    # 품사 태깅 및 불용어 제거
    tokens = [(word, flag) for word, flag in pseg.cut(text_data) if word not in stopwords]

    # n-gram 빈도 계산
    ngram_results = defaultdict(list)
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams = generate_pos_based_ngrams(tokens, n, pos_filter)
        ngram_counter = Counter(ngrams)
        filtered_ngrams = [(ngram, freq) for ngram, freq in ngram_counter.items() if freq >= min_frequency]
        ngram_results[n] = sorted(filtered_ngrams, key=lambda x: x[1], reverse=True)

    # 결과 출력
    for n, ngrams in ngram_results.items():
        print(f"\n=== {n}-gram Results ===")
        print(f"{'N-gram':<30} {'Frequency':<10}")
        print("=" * 40)
        for ngram, freq in ngrams:
            print(f"{ngram:<30} {freq:<10}")

# 파일 경로 입력
if __name__ == "__main__":
    file_path = "your_text_file.txt"  # 텍스트 파일 경로
    stopwords_path = "your_text_file.txt"  # 불용어 파일 경로
    min_frequency = 3  # 최소 빈도수
    ngram_range = (2, 3, 4)  # 2-gram과 3-gram
    pos_filter = {'n', 'v', 'a'}  # 명사(n), 동사(v), 형용사(a)

    main(file_path, stopwords_path, min_frequency, ngram_range, pos_filter)


2-3 PMI

from collections import Counter, defaultdict
import jieba
import math
import re

# 텍스트 파일 읽기
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 감성 사전 로드 (첫 번째 열만 가져오기)
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.split('\t')[0].strip() for line in file if line.strip())

# 텍스트 전처리 (중국어만 남기기)
def preprocess_text(text):
    return re.sub(r'[^\u4e00-\u9fff]', '', text)

# N-gram 빈도 계산
def calculate_ngram_frequency(tokens, n=2):
    ngrams = [''.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)

# PMI 계산
def calculate_pmi(word, context_word, word_freq, total_tokens, co_occurrence_freq):
    p_word = word_freq[word] / total_tokens
    p_context_word = word_freq[context_word] / total_tokens
    p_joint = co_occurrence_freq[(word, context_word)] / total_tokens
    if p_joint == 0 or p_word == 0 or p_context_word == 0:
        return 0
    return math.log2(p_joint / (p_word * p_context_word))

# jieba에 등록된 단어 필터링
def filter_out_registered_words(neologisms):
    registered_words = set(jieba.dt.FREQ.keys())  # jieba에 등록된 어휘 가져오기
    return [(word, freq, pmi) for word, freq, pmi in neologisms if word not in registered_words]

# 신조어 추출
def extract_neologisms_with_pmi(tokens, lexicon, freq_threshold, pmi_threshold):
    total_tokens = len(tokens)

    # 단어 빈도 계산
    word_freq = Counter(tokens)

    # 단어 쌍 빈도 (공동 출현 빈도) 계산
    co_occurrence_freq = defaultdict(int)
    for i in range(len(tokens) - 1):
        for j in range(i + 1, min(i + 5, len(tokens))):  # 윈도우 크기 5
            co_occurrence_freq[(tokens[i], tokens[j])] += 1

    # PMI 기반 신조어 후보 계산
    neologisms = []
    for word, freq in word_freq.items():
        if word in lexicon or freq < freq_threshold:
            continue

        # 감성 단어와의 PMI 계산
        for lex_word in lexicon:
            pmi = calculate_pmi(word, lex_word, word_freq, total_tokens, co_occurrence_freq)
            if pmi >= pmi_threshold:
                neologisms.append((word, freq, pmi))
                break

    return neologisms

# 메인 실행 함수
def main(text_file_path, lexicon_file_path):
    # 텍스트 데이터 및 감성 사전 로드
    text_data = read_text_file(text_file_path)
    lexicon = load_lexicon(lexicon_file_path)

    # 텍스트 전처리 및 형태소 분석
    text_data = preprocess_text(text_data)
    tokens = list(jieba.cut(text_data))

    # 임계값 설정
    freq_threshold = 10
    pmi_threshold = 8.0

    # 신조어 추출
    neologisms = extract_neologisms_with_pmi(tokens, lexicon, freq_threshold, pmi_threshold)

    # jieba에 등록된 단어 필터링
    filtered_neologisms = filter_out_registered_words(neologisms)

    # 결과 출력
    print(f"{'Neologism':<10} {'Frequency':<10} {'PMI':<10}")
    print("=" * 30)
    for word, freq, pmi in sorted(filtered_neologisms, key=lambda x: x[2], reverse=True):
        print(f"{word:<10} {freq:<10} {pmi:<10.1f}")

# 파일 경로 설정 및 실행
if __name__ == "__main__":
    text_file_path = "your_text_file.txt"  # 텍스트 파일 경로
    lexicon_file_path = "your_lexicon_file.tsv"  # 감성 사전 파일 경로
    main(text_file_path, lexicon_file_path)


3.문법 규칙 추가

역접 관계 但是, 然而, 却, 可是, 反而, 不过/ (0, 1)
점층 관계 不但……而且, 并且, 而且……还, / (1, 1.5)
가정 관계 如果, 要是, 假如, 假使, 倘若/(1, 0.5)
인과 관계 因为……所以, 是因为, 由于……因此, 之所以……是因为/ (0.5, 1)
극성 전환 难道，以为~原本, 好久没有, 差点
물음표(?), 느낌표(!) 가중치 추가


##문장규칙 추가 감성분석
import re
import pandas as pd

class Struct(object):
    def __init__(self, word, sentiment, pos, value, class_value):
        self.word = word
        self.sentiment = sentiment
        self.pos = pos
        self.value = value
        self.class_value = class_value

class Result(object):
    def __init__(self, score, score_words, not_word, degree_word):
        self.score = score
        self.score_words = score_words
        self.not_word = not_word
        self.degree_word = degree_word

class Score(object):
    score_class = {
        '乐': ['PA', 'PE'],
        '好': ['PD', 'PH', 'PG', 'PB', 'PK'],
        '怒': ['NA'],
        '哀': ['NB', 'NJ', 'NH', 'PF'],
        '惧': ['NI', 'NC', 'NG'],
        '恶': ['NE', 'ND', 'NN', 'NK', 'NL'],
        '惊': ['PC']
    }

    POS_MAP = {
        'noun': 'n',
        'verb': 'v',
        'adj': 'a',
        'adv': 'd',
        'nw': 'al',
        'idiom': 'al',
        'prep': 'p',
    }

    NOT_DICT = set([
        '不', '不是', '不大', '没', '无', '非', '莫', '弗', '毋', '没有',
        '勿', '未', '否', '無', '休', '不一定是', '不一定', '不太', '甭', '从未', '未曾', '决不', '绝不', '没法', '不可能', '无法', '不见得', '毫无', '绝非', '并非'
    ])

    TRANSITION_WORDS = {"但是", "可是", "但", "可", "然而", "不过", "却", "最后"}
    PROGRESSIVE_WORDS = {"不仅", "而且", "更有甚者"}
    HYPOTHETICAL_WORDS = {"如果", "假如", "例如"}
    CAUSAL_WORDS = {"因为", "所以"}

    # 예외 단어로 '未来' 추가
    EXCEPTION_WORDS = {"未来","不要"}

    def __init__(self, sentiment_dict_path, degree_dict_path, stop_dict_path, user_dict_path=None):
        self.sentiment_struct, self.sentiment_dict = self.load_sentiment_dict(sentiment_dict_path)
        self.degree_dict = self.load_degree_dict(degree_dict_path)
        self.stop_words = self.load_stop_words(stop_dict_path)

    def load_stop_words(self, stop_dict_path):
        stop_words = [w.strip() for w in open(stop_dict_path, encoding='utf-8').readlines()]
        return stop_words

    def remove_stopword(self, words):
        words = [w for w in words if w not in self.stop_words]
        return words

    def load_degree_dict(self, dict_path):
        degree_dict = {}
        with open(dict_path, 'r', encoding='UTF-8') as f:
            for line in f:
                word, degree = line.strip().split('\t')
                degree_dict[word] = float(degree)
        return degree_dict

    def load_sentiment_dict(self, dict_path):
        sentiment_dict = {}
        sentiment_struct = []

        with open(dict_path, 'r', encoding='UTF-8') as f:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                items = line.split('\t')
                word = items[0]
                pos = items[1]
                sentiment = items[4]
                intensity = int(items[5])
                polar = int(items[6])

                pos = self.__class__.POS_MAP.get(pos, pos)
                value = intensity if polar == 1 else -intensity if polar == 2 else 0

                sentiment_dict[word] = value

                for key, values in self.score_class.items():
                    if sentiment in values:
                        class_value = key
                        break

                sentiment_struct.append(Struct(word, sentiment, pos, value, class_value))

        return sentiment_struct, sentiment_dict

    def find_words(self, text):
        """텍스트에서 감정 단어, 부정어, 정도 부사의 위치를 추적하여 저장"""
        sen_word = {}
        not_word = {}
        degree_word = {}
        covered_ranges = []  # 이미 처리된 범위를 기록할 리스트

        # 예외 단어는 먼저 처리하여, 나누어지지 않도록 합니다.
        for word in self.EXCEPTION_WORDS:
            start = 0
            while start < len(text):
                index = text.find(word, start)
                if index != -1:
                    # 이미 처리된 구간인지 확인
                    if not any(start <= index < end for start, end in covered_ranges):
                        sen_word[index] = self.sentiment_dict.get(word, 0)
                        covered_ranges.append((index, index + len(word)))  # 해당 단어 범위 기록
                        print(f"예외 단어 찾음: {word} (위치: {index}, 값: {self.sentiment_dict.get(word, 0)})")
                    start = index + len(word)
                else:
                    break

        # 감정 단어를 길이순으로 정렬하여 긴 단어부터 검색
        sorted_sentiment_words = sorted(self.sentiment_dict.keys(), key=len, reverse=True)

        # 긴 감정 단어를 텍스트에서 먼저 검색하고 위치 저장
        for word in sorted_sentiment_words:
            start = 0
            while start < len(text):
                index = text.find(word, start)
                if index != -1:
                    # 이미 처리된 구간인지 확인
                    if not any(start <= index < end for start, end in covered_ranges):
                        sen_word[index] = self.sentiment_dict[word]
                        covered_ranges.append((index, index + len(word)))  # 해당 단어 범위 기록
                        print(f"감정 단어 찾음: {word} (위치: {index}, 값: {self.sentiment_dict[word]})")
                    start = index + len(word)
                else:
                    break

        # ***부정어와 정도 부사를 하나로 합쳐서 처리, 긴 단어부터 우선 처리***
        combined_words = {word: 'negation' for word in self.NOT_DICT}
        combined_words.update({word: 'degree' for word in self.degree_dict})

        sorted_combined_words = sorted(combined_words.keys(), key=len, reverse=True)

        # 부정어와 정도 부사를 길이순으로 처리
        for word in sorted_combined_words:
            start = 0
            while start < len(text):
                index = text.find(word, start)
                if index != -1:
                    # 이미 처리된 구간인지 확인
                    if not any(start <= index < end for start, end in covered_ranges):
                        if combined_words[word] == 'negation':
                            not_word[index] = -1
                            print(f"부정어 찾음: {word} (위치: {index})")
                        elif combined_words[word] == 'degree':
                            degree_word[index] = self.degree_dict[word]
                            print(f"정도 부사 찾음: {word} (위치: {index}, 값: {self.degree_dict[word]})")
                        covered_ranges.append((index, index + len(word)))  # 해당 단어 범위 기록
                    start = index + len(word)
                else:
                    break

        return sen_word, not_word, degree_word

    def get2score_position(self, text):
        sen_word, not_word, degree_word = self.find_words(text)  # 감정 단어, 부정어, 정도부사 위치 추적
        score = 0
        sorted_sen_keys = sorted(sen_word.keys())
        max_distance = 7  # 부정어와 감정 단어 사이 허용 범위

        for i in sorted_sen_keys:
            W = 1  # 기본 가중치

            # 부정어와 감정 단어 사이 거리를 고려하여 극성 반전
            for not_index in not_word:
                if 0 < i - not_index <= max_distance:
                    W *= -1  # 부정어가 일정 거리 내에 있을 때 반전
                    print(f"부정어 적용: 감정 단어 위치 {i}, 부정어 위치 {not_index}, 가중치 {W}")

            # 감정 단어 앞에 정도 부사가 있으면 가중치에 해당 정도 곱하기
            for degree_index in degree_word:
                if degree_index < i:
                    W *= degree_word[degree_index]
                    print(f"정도 부사 적용: 감정 단어 위치 {i}, 정도 부사 위치 {degree_index}, 가중치 {W}")

            # 가중치(W)와 감정 단어의 값을 곱해 점수에 반영
            current_score = W * sen_word[i]
            print(f"점수 계산: 감정 단어 위치 {i}, 값 {sen_word[i]}, 가중치 {W}, 현재 점수 {current_score}")  # 계산 과정 출력
            score += current_score

        return score

    def apply_sentence_type_rules(self, sentence, sentiment_score):
        """문장 유형 규칙을 적용하여 감정 점수를 조정합니다."""
        if any(phrase in sentence for phrase in ['难道', '以为', '本以为','哪是','本来']):
            return -sentiment_score

        if sentence.endswith('！'):
            return sentiment_score * 1.5

        return sentiment_score

    def apply_inter_sentence_rules(self, sentence, previous_score):
        """문장 간 규칙을 적용하여 감정 점수를 조정합니다."""
        if any(word in sentence for word in self.TRANSITION_WORDS):
            return 0.5, 2.0  # 전환 관계, 앞의 감정은 약화, 뒤의 감정은 강화
        if any(word in sentence for word in self.PROGRESSIVE_WORDS):
            return 1.0, 1.5  # 점진 관계, 뒤의 감정 강화
        if any(word in sentence for word in self.HYPOTHETICAL_WORDS):
            return 1.0, 0.5  # 가정 관계, 뒤의 감정 약화
        if any(word in sentence for word in self.CAUSAL_WORDS):
            return 0.5, 1.0  # 인과 관계
        return 1, 1  # 기본값

    def split_sentences(self, text):
        """쉼표와 문장 구분 기호로 텍스트를 문장 단위로 나눕니다."""
        sentences = re.split(r'[。，,]', text)
        return [sentence for sentence in sentences if sentence.strip()]  # 빈 문자열 제거

    def calculate_sentiment(self, text):
        sentences = self.split_sentences(text)
        overall_score = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # 물음표를 기준으로 문장을 나누고, 각 절을 처리
            clauses = re.split(r'[?？]', sentence)
            clause_scores = []

            for j, clause in enumerate(clauses):
                sentiment_score = self.get2score_position(clause)

                # 문장 유형 규칙 적용
                sentiment_score = self.apply_sentence_type_rules(clause, sentiment_score)

                # 물음표가 있는 절 처리 (마지막 절이 아니면 물음표로 끝난 것으로 간주)
                if j < len(clauses) - 1:
                    if sentiment_score > 0:
                        sentiment_score = -sentiment_score  # 양성 감정이면 음성으로 변환
                    if sentiment_score == 0:
                        sentiment_score = -10  # 0점이면 -10 적용

                clause_scores.append(sentiment_score)

            # 문장 간 규칙 적용 (앞 문장과의 관계)
            if i > 0:
                prev_weight, curr_weight = self.apply_inter_sentence_rules(sentence, overall_score)
                overall_score *= prev_weight
                clause_scores[0] *= curr_weight  # 첫 번째 절에만 가중치 적용

            # 각 절의 점수를 합산하여 전체 점수에 반영
            overall_score += sum(clause_scores)

        print(f"문장: {text}, 전체 감정 점수: {overall_score}")
        return overall_score


# 아래 부분은 텍스트를 입력받아 바로 감정 분석을 실행하고 결과를 출력합니다.

if __name__ == '__main__':
    sentiment_dict_path = "your file path.tsv"
    degree_dict_path = "your file path.txt"
    stop_dict_path = "your file path.txt"

    score_calculator = Score(sentiment_dict_path, degree_dict_path, stop_dict_path)


    # 리뷰 데이터를 읽어옵니다.
    data = pd.read_excel('your file path.xlsx', engine='openpyxl')
    # 결과를 저장할 빈 데이터프레임 생성
    results_list = []

    # 리뷰 데이터를 순회하며 감정 점수를 계산합니다.
    for idx, review_text in enumerate(data['Review'], start=1):
        if pd.isna(review_text):
            review_text = ""
        else:
            review_text = str(review_text)

        sentiment_score = score_calculator.calculate_sentiment(review_text)

        # 결과를 리스트에 추가합니다.
        results_list.append({'Review': review_text, 'Sentiment_Score': sentiment_score})

    # 리스트를 데이터프레임으로 변환합니다.
    results_df = pd.DataFrame(results_list)

    # 엑셀 파일로 저장
    results_df.to_excel("/content/drive/MyDrive/감성분석/Jisun_Test_Results.xlsx", index=False)

