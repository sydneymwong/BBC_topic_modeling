import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

num_topics = 5

def get_topics():
	directory = os.getcwd()
	directory = os.path.join(directory, "bbc-fulltext", "bbc")
	toReturn = []
	for filename in os.listdir(directory):
		if not filename == "README.TXT":
			toReturn.append(filename)
	return toReturn

def read_in_and_save(topic):
	directory = os.getcwd()
	directory = os.path.join(directory, "bbc-fulltext", "bbc", topic)
	toReturn = []
	for filename in os.listdir(directory):
		with open(os.path.join(directory, filename)) as file:
			data = file.read()
			toReturn.append(data)
			#toReturn.append(data.split())
	return toReturn

topics = get_topics()
documents = []
topic_lengths = {}
for topic in topics:
    docs_list = read_in_and_save(topic)
    documents = documents + docs_list
    topic_lengths[topic] = len(docs_list)

vectorizer = CountVectorizer(min_df=1, stop_words = "english")
X = vectorizer.fit_transform(documents)
tf_feature_names = vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
display_topics(lda, tf_feature_names, 10)


	