import os
import string

import gensim
import numpy as np
from gensim import corpora
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from utils import read_json_data

datasets = ['ICLR.json', 'NIPS.json']
datast_path = '../datasets'

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
Lda = gensim.models.ldamodel.LdaModel


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def gen_author(dic):
    man_sum = len(dic)
    if man_sum == 0:
        return [0, 0]
    score_sum = 0
    for key, val in dic.items():
        score_sum += val['score']

    return [man_sum, score_sum / man_sum]


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    tagged_sent = pos_tag(punc_free.split())
    normalized = " ".join(lemma.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_sent)
    return normalized


def get_clean_docs(data):
    docs = []
    for key, content in data.items():
        docs.append(clean(content.get('title', '')).split())
    return docs


def cal_entropy(lis):
    array = np.array(lis)
    return float(-np.sum(array * np.log2(array)))


def main():
    data = read_json_data(os.path.join(datast_path, 'datasetv0.json'))
    docs = get_clean_docs(data)
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    ldamodel = Lda(doc_term_matrix, id2word=dictionary, minimum_probability=0.0)
    topics = ldamodel.print_topics(100)
    for dataset in datasets:
        data = read_json_data(os.path.join(datast_path, dataset))
        docs = get_clean_docs(data)
        dictionary = corpora.Dictionary(docs)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
        res = []
        for doc, id in zip(doc_term_matrix, data.keys()):
            res.append([int(id)])
            prop = [b for a, b in ldamodel[doc]]
            res[-1].extend(prop)
            res[-1].append(cal_entropy(prop))
            res[-1].append(data[id].get('year', 2017))
            res[-1].extend(gen_author(data[id].get('authors', {})))
        with open(dataset+'.txt','w') as f:
            for r in res:
                for num in r:
                    f.write(str(num))
                    f.write('\t')
                f.write('\n')


if __name__ == '__main__':
    main()

# doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
# doc2 = "My father spends a lot of time driving my sister around to dance practice."
# doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
# doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
# doc5 = "Health experts say that Sugar is not good for your lifestyle."
#
# # 整合文档数据
# doc_complete = [doc1, doc2, doc3, doc4, doc5]
#
# doc_clean = [clean(doc).split() for doc in doc_complete]
# print(doc_clean)
#
# # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
# dictionary = corpora.Dictionary(doc_clean)
#
# # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#
# # 使用 gensim 来创建 LDA 模型对象
# Lda = gensim.models.ldamodel.LdaModel
#
# # 在 DT 矩阵上运行和训练 LDA 模型
# ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
# print(ldamodel[doc_term_matrix[0]])
#
# # 输出结果
# print(ldamodel.print_topics(num_topics=3, num_words=3))
