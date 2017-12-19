import os
import csv
from string import punctuation
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

num_topics = 5

def get_topics():
	directory = os.getcwd()
	directory = os.path.join(directory, "bbc-fulltext", "bbc")
	toReturn = []
	for filename in os.listdir(directory):
		if not filename == "README.TXT":
			toReturn.append(filename)
	return toReturn

def clean_string(s):
    toReturn = ''.join(c for c in s if c not in punctuation)
    toReturn = ' '.join(toReturn.split())
    return toReturn

def read_in_and_save(topic):
    directory = os.getcwd()
    directory = os.path.join(directory, "bbc-fulltext", "bbc", topic)
    toReturn = []
    counter = 0
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as file:
            if counter == 50:
                break
            else:
                data = file.read()
                data = clean_string(data)
                toReturn.append(data)
                #toReturn.append(data.split())
            counter += 1
    return toReturn

topics = get_topics()
documents = []
topic_lengths = {}
for topic in topics:
    docs_list = read_in_and_save(topic)
    documents = documents + docs_list
    topic_lengths[topic] = len(docs_list)

#vectorizer = CountVectorizer(min_df=1, stop_words = "english")
vectorizer = TfidfVectorizer(stop_words = "english", sublinear_tf=True)
X = vectorizer.fit_transform(documents)
idf = vectorizer.idf_
idf = dict(zip(vectorizer.get_feature_names(), idf))
tf_feature_names = vectorizer.get_feature_names()
for key in idf.keys():
    if key not in tf_feature_names:
        print(key)
        
# store term frequencies in a list where each element represents a document 
documents_tf = []
for i in range(len(documents)):
    documents_tf.append(Counter(documents[i].split()))
    
# store UNIVERSAL term frequencies in a list where each element represents a term 
documents_split = []
for i in range(len(documents)):
    documents_split = documents_split + documents[i].split()
universal_term_freq_dict = Counter(documents_split)
universal_term_freq = []
for term in tf_feature_names:
    universal_term_freq.append(universal_term_freq_dict[term])
    
    
# create a new list of dictionaries where each element represents a document 
# {'word index' : tf*idf}

documents_word_indices = []
#documents_word_indices = pd.DataFrame()
documents_tfidf_values = []
#documents_tfidf_values = pd.DataFrame()
for i in range(len(documents_tf)):
    current_word_indices = ""
    current_tfidf_values = ""
    for key in documents_tf[i].keys():
        if key in idf.keys():
            #current[key] = documents_tf[i][key] * idf[key]
            #current[tf_feature_names.index(key)] = documents_tf[i][key] * idf[key]
            current_word_indices = " ".join((current_word_indices, str(tf_feature_names.index(key))))
            #current_word_indices.append(tf_feature_names.index(key))
            current_tfidf_values = " ".join((current_tfidf_values, str(round(documents_tf[i][key] * idf[key], 0))))
            #current_tfidf_values.append(documents_tf[i][key] * idf[key])
    documents_word_indices.append([current_word_indices])
    documents_tfidf_values.append([current_tfidf_values])
    #documents_word_indices[i] = current_word_indices
    #documents_tfidf_values[i] = current_tfidf_values

#data = vectorizer.fit_transform(tf_feature_names).toarray()

"""
# output the idf values to a csv
with open('idfvalues.csv', 'w') as f:
    w = csv.DictWriter(f, idf.keys())
    w.writeheader()
    w.writerow(idf)
"""
with open("word_indices.csv", "w", newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(documents_word_indices)

with open("tfidf_values.csv", "w", newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(documents_tfidf_values)

# output the feature names to a csv as "vocab"
with open("vocab.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in tf_feature_names:
        writer.writerow([val]) 
        
# output the universal term frequencies as "total_term_freq"
with open("total_term_freq.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in universal_term_freq:
        writer.writerow([val]) 

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
display_topics(lda, tf_feature_names, 10)


	