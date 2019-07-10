

install.packages(c("ggplot2", "e1071", "caret", "quanteda", 
                   "irlba", "randomForest"))


spam.raw <- read.csv("spam.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-16")
View(spam.raw)


spam.raw <- spam.raw[, 1:2]
names(spam.raw) <- c("Label", "Text")
View(spam.raw)


length(which(!complete.cases(spam.raw)))


spam.raw$Label <- as.factor(spam.raw$Label)



prop.table(table(spam.raw$Label))




spam.raw$TextLength <- nchar(spam.raw$Text)



library(ggplot2)

ggplot(spam.raw, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")


library(caret)
help(package = "caret")



set.seed(32984)
indexes <- createDataPartition(spam.raw$Label, times = 1,
                               p = 0.7, list = FALSE)

train <- spam.raw[indexes,]
test <- spam.raw[-indexes,]


prop.table(table(train$Label))
prop.table(table(test$Label))




library(quanteda)


# Tokenize SMS text messages.
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)


# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)


# Use quanteda's built-in stopword list for English.

train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[357]]


# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[357]]


# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)


# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:50]



# Setup a the feature data frame with labels.
train.tokens.df <- cbind(Label = train$Label, data.frame(train.tokens.dfm))


# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))


# creating stratified folds for 10-fold cross validation repeated 
# 3 times (i.e., create 30 random stratified samples)
set.seed(48743)
cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)



#install.packages("doSNOW")
library(doSNOW)


# Time the code execution
start.time <- Sys.time()


# Create a cluster to work on 10 logical cores.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)



rpart.cv.1 <- train(Label ~ ., data = train.tokens.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)


# stop cluster.
stopCluster(cl)

 
total.time <- Sys.time() - start.time
total.time


#results.
rpart.cv.1



# function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))

  log10(corpus.size / doc.count)
}

# calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}


# normalizing via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)


# calculating the IDF vector that we will use - both
# for training data and for test data
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)


# TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])


# Transpose
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])


# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]


# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))


# Make a clean data frame using the same process as before.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))


# Time the code execution
start.time <- Sys.time()


cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)


rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)


stopCluster(cl)


total.time <- Sys.time() - start.time
total.time

# results.
rpart.cv.2

library(irlba)


# Time the code execution
start.time <- Sys.time()


train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)

total.time <- Sys.time() - start.time
total.time


View(train.irlba$v)



sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document


document.hat[1:10]
train.irlba$v[1, 1:10]




# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).

train.svd <- data.frame(Label = train$Label, train.irlba$v)



cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)


start.time <- Sys.time()


rpart.cv.4 <- train(Label ~ ., data = train.svd, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)


stopCluster(cl)


total.time <- Sys.time() - start.time
total.time


rpart.cv.4


cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

 start.time <- Sys.time()


rf.cv.1 <- train(Label ~ ., data = train.svd, method = "rpart", 
                 trControl = cv.cntrl, tuneLength = 7)


 stopCluster(cl)

 total.time <- Sys.time() - start.time
 total.time




# results.
rf.cv.1

# confusion matrix
confusionMatrix(train.svd$Label, rf.cv.1$finalModel$predicted)






# add text length feature
train.svd$TextLength <- train$TextLength


cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

start.time <- Sys.time()

rf.cv.2 <- train(Label ~ ., data = train.svd, method = "rpart",
                 trControl = cv.cntrl, tuneLength = 7, 
                 importance = TRUE)

stopCluster(cl)

total.time <- Sys.time() - start.time
total.time


# results.
rf.cv.2

#confusion matrix
confusionMatrix(train.svd$Label, rf.cv.2$finalModel$predicted)

# we check which feature was most effective.

varImpPlot(rf.cv.1$finalModel)
varImpPlot(rf.cv.2$finalModel)




#textlength feature drastically increased the ccuracy of the model
#for cosine similarities
install.packages("lsa")
library(lsa)

train.similarities <- cosine(t(as.matrix(train.svd[, -c(1, ncol(train.svd))])))



spam.indexes <- which(train$Label == "spam")

train.svd$SpamSimilarity <- rep(0.0, nrow(train.svd))
for(i in 1:nrow(train.svd)) {
  train.svd$SpamSimilarity[i] <- mean(train.similarities[i, spam.indexes])  
}


# visualizing results using the mighty ggplot2
ggplot(train.svd, aes(x = SpamSimilarity, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 0.05) +
  labs(y = "Message Count",
       x = "Mean Spam Message Cosine Similarity",
       title = "Distribution of Ham vs. Spam Using Spam Cosine Similarity")

# we again perform cv
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

start.time <- Sys.time()
 
set.seed(932847)
rf.cv.3 <- train(Label ~ ., data = train.svd, method = "rpart",
                 trControl = cv.cntrl, tuneLength = 7,
                 importance = TRUE)

 stopCluster(cl)

total.time <- Sys.time() - start.time
total.time


# results.
rf.cv.3

# confusion matrix
confusionMatrix(train.svd$Label, rf.cv.3$finalModel$predicted)

# this increased the specificity to 1
#now we check how effective this featurewas.

varImpPlot(rf.cv.3$finalModel)




# As we have trained our trining data, now we process our test data which we have divided in the begining.
# we have to go through a number of steps of pipelining

# Tokenization.
test.tokens <- tokens(test$Text, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)

# Lower case the tokens.
test.tokens <- tokens_tolower(test.tokens)

# Stopword removal.
test.tokens <- tokens_select(test.tokens, stopwords(), 
                             selection = "remove")

# Stemming.
test.tokens <- tokens_wordstem(test.tokens, language = "english")


# Convert to quanteda document-term frequency matrix.
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)


train.tokens.dfm
test.tokens.dfm


test.tokens.dfm <- dfm_select(test.tokens.dfm, pattern = train.tokens.dfm,
                              selection = "keep")
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.dfm




# projecting the term counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)
str(test.tokens.df)

# Lastly, calculate TF-IDF for our training corpus.
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)
View(test.tokens.tfidf[1:25, 1:25])

# Transpose
test.tokens.tfidf <- t(test.tokens.tfidf)

# incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])




# SVD matrix factorization
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))


# add Label and TextLength.
test.svd <- data.frame(Label = test$Label, test.svd.raw, 
                       TextLength = test$TextLength)


# calculate SpamSimilarity for all the test documents
# create a spam similarity matrix.

test.similarities <- rbind(test.svd.raw, train.irlba$v[spam.indexes,])
test.similarities <- cosine(t(test.similarities))



test.svd$SpamSimilarity <- rep(0.0, nrow(test.svd))
spam.cols <- (nrow(test.svd) + 1):ncol(test.similarities)
for(i in 1:nrow(test.svd)) {
 
  test.svd$SpamSimilarity[i] <- mean(test.similarities[i, spam.cols])  
}



test.svd$SpamSimilarity[!is.finite(test.svd$SpamSimilarity)] <- 0


# predictions on the test data set using our trained.
preds <- predict(rf.cv.3, test.svd)


# confusion matrix
confusionMatrix(preds, test.svd$Label)




# overfitting is doing far better on the training data as
# evidenced by CV than doing on a test dataset.
# here it is the use of the spam similarity feature.
# The hypothesis here is that spam features varies
# highly, espeically over time. so average spam cosine similarity 
# is likely to overfit to the training data.
# let's rebuild a mighty random forest without the spam similarity feature.

train.svd$SpamSimilarity <- NULL
test.svd$SpamSimilarity <- NULL


cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

start.time <- Sys.time()

set.seed(254812)
rf.cv.4 <- train(Label ~ ., data = train.svd, method = "rpart",
                  trControl = cv.cntrl, tuneLength = 7,
                  importance = TRUE)

stopCluster(cl)

Total time of execution on workstation was
total.time <- Sys.time() - start.time
total.time


rf.cv.4.RData


# Make predictions and create confusion matrix
preds <- predict(rf.cv.4, test.svd)
confusionMatrix(preds, test.svd$Label)
