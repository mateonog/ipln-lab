import pandas
corpus = pandas.read_csv("corpus_humor_training.csv",encoding='utf-8')

print(corpus.columns)

for text in corpus['text'][:1000]:
    print(text + '\n')
