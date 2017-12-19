#install.packages("LDAvis")
#install.packages("tm")
#install.packages("lda")
#install.packages("servr")

library(LDAvis)
library(tm)
library(lda)
library(servr)

# stop_words <- stopwords("SMART")
# 
# # Get a list of topics which are the directory names inside the "bbc" folder
# get_topics <- function() {
#   directory <- getwd()
#   directory <- file.path(directory, "Github", "BBC_topic_modeling", "bbc-fulltext", "bbc")
#   toReturn = list.dirs(path = directory, full.names = FALSE, recursive = FALSE)
# }
# 
# topics <- get_topics()
# 
# # Get a list of the documents
# read_in_and_save <- function(topic) {
#   directory <- getwd()
#   directory <- file.path(directory, "Github", "BBC_topic_modeling", "bbc-fulltext", "bbc", topic)
#   for filename in list.files(path = directory) {
#     read.delim(file.path(directory, filename), sep = "\n", header = FALSE)
#   }
# }

directory <- getwd()
#directory <- file.path(directory, "Github", "BBC_topic_modeling")
index <- read.csv(file.path(directory, "word_indices.csv"), header = FALSE)
value <- read.csv(file.path(directory, "tfidf_values.csv"), header = FALSE)
vocab <- read.csv(file.path(directory, "vocab.csv"), header = FALSE)
vocab <- as.character(vocab[,1])
term.frequency <- read.csv(file.path(directory, "total_term_freq.csv"), header = FALSE)
term.frequency <- as.integer(term.frequency[,1])

# Put data in the form LDA wants
documents <- list()
for (i in seq(1, length(index))) {
  #print(i)
  current_index <- strsplit(as.character(index[i, 1][1]), " ")
  current_index <- as.integer(current_index[[1]][-1])
  current_value <- strsplit(as.character(value[i, 1][1]), " ")
  current_value <- as.integer(current_value[[1]][-1])
  mymatrix <- rbind(as.integer(current_index), as.integer(current_value))
  documents[[i]] <- mymatrix
}

# Fit LDA

K <- 5
G <- 5000
alpha <- 0.02
eta <- 0.02


fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))
doc.length <- sapply(documents, function(x) sum(x[2, ]))

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

serVis(json, out.dir = file.path(directory, 'vis'), open.browser = FALSE)