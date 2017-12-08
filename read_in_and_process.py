import os

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
			toReturn.append(data.split())
	return toReturn

topics = get_topics()
documents = {}
for topic in topics:
	documents[topic] = read_in_and_save(topic)

	