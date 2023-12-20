import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt


filenames = ['oliver twist charles dickens.txt', 'greatexpectations charles dickens.txt',
             'adventures of huckleberry mark twain.txt', 'adventures of tom sawyer mark twain.txt']

def tokenization(filenames):
    alldataframes = []

    for filename in filenames:
        with open(filename, 'r', encoding = "utf8") as file:
            text = file.read()

            beginning = r"\bCHAPTER I\b"
            matches = list(re.finditer(beginning, text, flags = re.IGNORECASE))

            ending = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"
            matches2 = re.search(ending, text)

            if matches:
                lastoccur = matches[-1].start()
            else:
                lastoccur = 0

            if matches2:
                endoccur = matches2.start()
                text = text[lastoccur:endoccur]


            text = re.sub(r'CHAPTER I', '', text, flags = re.IGNORECASE)
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            words = text.split()

            data = pd.DataFrame(words, columns = ['Words'])
            alldataframes.append(data)

    return alldataframes

# print(tokenization(filenames))

def termfrequency(filenames):
    dataframes = tokenization(filenames)
    tflist = []

    for i, data in enumerate(dataframes):
        count = data.groupby(['Words']).size().sort_values(ascending = True).reset_index(name = 'Count')
        totalwords = len(data)
        count['Term Frequency'] = count['Count'] / totalwords
        newframe = count.drop('Count', axis = 1)
        vectorize = np.array(newframe)

        thefilename = filenames[0 + i]
        newframe.to_csv(f'{thefilename}_TFVALUES.csv', index = False)
        tflist.append((thefilename, vectorize))

    return tflist

# print(termfrequency(filenames))

def inversetermfrequency(filenames):
    dataframes = tokenization(filenames)
    N = len(dataframes)
    termlists = []
    tdf = {}

    for data in dataframes:
        termlist = data['Words'].tolist()
        termlists.append(termlist)

    for termlist in termlists:
        unique = set(termlist)
        for term in unique:
            if term in tdf:
                tdf[term] += 1
            else:
                tdf[term] = 0

    idf = {}
    for term, terms in tdf.items():
        idf[term] = math.log(N / (1 + terms))

    newdata = pd.DataFrame(list(idf.items()), columns = ['Term', 'IDF'])
    newdata.sort_values(by = 'IDF', ascending = True).to_csv('IDF Values', index = False)

    return idf

# print(inversetermfrequency(filenames))

def plottopwords(tfidfvector, n=25, title=None, bar_color = 'green'):
    tfidfvector['TFIDF'] = pd.to_numeric(tfidfvector['TFIDF'])
    topwords = tfidfvector.nlargest(n, 'TFIDF')
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.bar(topwords['Term'], topwords['TFIDF'], color = bar_color)

    ax.set_xlabel('TF-IDF Score')
    ax.set_ylabel('Term')

    if title is not None:
        ax.set_title('Document: {}'.format(title))

    plt.xticks(rotation = 90)


def tfidf(filenames):
    tflist = termfrequency(filenames)
    idf = inversetermfrequency(filenames)

    tfidflist = []

    for filename, tfvector in tflist:
        tfidfvector = tfvector.copy()
        for i, term in enumerate(tfidfvector[:, 0]):
            tfidfvector[i, 1] *= idf.get(term, 0)
        tfidflist.append((filename, tfidfvector))
        newdata = pd.DataFrame(tfidfvector, columns = ['Term', 'TFIDF'])
        newdata.sort_values(by = 'TFIDF', ascending = False).to_csv(f"{filename}_TFIDFVALUES.csv", index = False)

        plottopwords(newdata, n = 25, title = filename)


    plt.show()

    return tfidflist

tfidf(filenames)

