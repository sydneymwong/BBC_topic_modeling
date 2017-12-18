#install.packages("LDAvis")
#install.packages("tm")
#install.packages("lda")

library(LDAvis)
library(tm)
library(lda)
stop_words <- stopwords("SMART")

# Get a list of topics which are the directory names inside the "bbc" folder
get_topics <- function() {
  directory <- getwd()
  directory <- file.path(directory, "Github", "BBC_topic_modeling", "bbc-fulltext", "bbc")
  toReturn = list.dirs(path = directory, full.names = FALSE, recursive = FALSE)
}

# Get a list of the documents
read_in_and_save <- function(topic) {
  directory <- getwd()
  directory <- file.path(directory, "Github", "BBC_topic_modeling", "bbc-fulltext", "bbc", topic)
  list.files(path = directory)
}


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


# Fit LDA

K <- 20
G <- 5000
alpha <- 0.02
eta <- 0.02


fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

NewsArticles <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

json <- createJSON(phi = NewsArticles$phi, 
                   theta = NewsArticles$theta, 
                   doc.length = NewsArticles$doc.length, 
                   vocab = NewsArticles$vocab, 
                   term.frequency = NewsArticles$term.frequency)

serVis(json, out.dir = 'vis', open.browser = FALSE)