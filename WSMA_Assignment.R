#setting the working directory

setwd("D:/BABI/BABI-17th Residency/WSMA/Assignment/Assignment_work")
getwd()

#loading the csv file

data <- read.csv("Dataset.csv")
View(data)

#loading text mining packages

library(tm) 
library(SnowballC)
library(wordcloud)

#loading the ML packages

library(mlr)
library(dplyr)
library(explore)

#loading the data split packages

library(caTools)
library(caret)

#text mining

corpusstc <- Corpus(VectorSource(data$description)) #converting the column description into unigrams

corpusstc <- tm_map(corpusstc, tolower) #converting the unigrams in description column into lower case
corpusstc <- tm_map(corpusstc, removePunctuation) #remove the punctuation marks
corpusstc <- tm_map(corpusstc, removeWords, c(stopwords("english"))) #removing the stop words
corpusstc <- tm_map(corpusstc, stemDocument) #stemming the document

#forming a word cloud based on the number of occurences of the words

wordcloud(corpusstc, colors = rainbow(7), max.words = 100)

#Sentiment Analysis

library(sentimentr)
library(syuzhet)

STsentiment = get_nrc_sentiment(as.character(corpusstc))
Score = data.frame(colSums(STsentiment[,]))
names(Score) = "Score"
Score = cbind("sentiment" = rownames(Score), Score)
rownames(Score) = NULL
ggplot(data = Score, aes(x = sentiment, y = Score))+
  geom_bar(aes(fill=sentiment), stat = "identity") +
  theme(legend.position = "none") +
  xlab("Sentiment") + ylab("Score") + ggtitle("Sentiment Score derived from Description")

#Forming the DTM

frequenciesstc <- DocumentTermMatrix(corpusstc) 

sparsestc <- removeSparseTerms(frequenciesstc, 0.995) #removing the least occurring words
stcsparse <- as.data.frame(as.matrix(sparsestc)) #forming the DTM

View(stcsparse)

colnames(stcsparse) <- make.names(colnames(stcsparse))

#copying the existing columns in the dataset to the newly formed DTM

stcsparse$deal <- data$deal
stcsparse$askedFor <- data$askedFor
stcsparse$exchangeForStake <- data$exchangeForStake
stcsparse$valuation <- data$valuation

str(stcsparse$deal)
stcsparse$deal <- as.factor(stcsparse$deal) #converting the dependent variable into categorical variable

summary(stcsparse$deal)

data %>% select(askedFor, exchangeForStake, valuation, deal) %>% explain_tree(target = deal)

#creating a copy of the dataset

dtmdata <- stcsparse

#splitting of data (80-20 ratio)

set.seed(1234)

train.index <- createDataPartition(stcsparse$deal, p = .8, list = FALSE)
train <- stcsparse[train.index,]
test  <- stcsparse[-train.index,]

#CART Model

library(rpart)
library(rpart.plot)

library(DataExplorer)
library(greybox)
library(rattle)

SharktankCart = rpart(deal ~ ., data=train, method="class")
rpart.plot(SharktankCart)

#printing the cart tree in a pdf document

pdf("Cart.pdf")
fancyRpartPlot(SharktankCart, palettes=c("Greys", "Oranges"), 
               main = "CART model before adding variable ratio")
dev.off()

#CART Diagram
prp(SharktankCart, extra=2, main = "CART Tree before adding variable ratio")

summary(SharktankCart)
attributes(SharktankCart)

#confusion matrix

train$predict.class <- predict(SharktankCart, train, type="class")
train$predict.score <- predict(SharktankCart, train)

confmattrain = table(train[,c("deal","predict.class")])
confusionMatrix(confmattrain)

#predicting for the test dataset

test$predict.class <- predict(SharktankCart, test, type="class")
test$predict.score <- predict(SharktankCart, test)

confmattest = table(test[,c("deal","predict.class")])
confusionMatrix(confmattest)

#Logistic Regression (80:20 split)

library(ROCR)
library(lmtest)
library(pscl)

set.seed(1234)
split <- createDataPartition(stcsparse$deal, p = .8, list = FALSE)
train_lt <- stcsparse[split,]
test_lt  <- stcsparse[-split,]

lg_model <- glm(deal ~ .,data = train_lt,family = binomial())

summary(lg_model)

pR2(lg_model)

#Confusion matrix of train data

pred<-predict(lg_model,train_lt,type='response')
mat_tab<-table(train_lt$deal, pred > 0.5)
mat_tab

lt_acc <- sum(diag(mat_tab))/sum(mat_tab)
lt_acc*100

#Prediction and Confusion matrix of test data

tdata <- predict(lg_model,test_lt, type="response")
t_confmat <- table(test_lt$deal, tdata > 0.5)
t_confmat
test_acc <- sum(diag(t_confmat))/sum(t_confmat)
test_acc*100

#Random Forest

#loading the required package for random forest

library(randomForest)

set.seed(1234)
split_rf <- createDataPartition(stcsparse$deal, p = .8, list = FALSE)
train_rf <- stcsparse[split_rf,]
test_rf  <- stcsparse[-split_rf,]

model_rf<-randomForest(deal~.,data=train_rf,importance=T)
model_rf

plot(model_rf, main = "Plot of random forest model before adding variable ratio")

varImpPlot(model_rf, main = "Variable importance plot before adding variable ratio")

#confusion matrix

train_rf$predict.class <- predict(model_rf, train_rf, type="class")
train_rf$predict.score <- predict(model_rf, train_rf)

confmattrain_rf = table(train_rf[,c("deal","predict.class")])
confusionMatrix(confmattrain_rf)

#predicting for the test dataset

test_rf$predict.class <- predict(model_rf, test_rf, type="class")
test_rf$predict.score <- predict(model_rf, test_rf)

confmattest_rf = table(test_rf[,c("deal","predict.class")])
confusionMatrix(confmattest_rf)

#addition of new variable -- Ratio

dtmdata$ratio = dtmdata$askedFor/dtmdata$valuation

View(dtmdata)

#modelling

set.seed(1234)

train.index_aft <- createDataPartition(dtmdata$deal, p = .8, list = FALSE)
train_aft <- stcsparse[train.index_aft,]
test_aft  <- stcsparse[-train.index_aft,]

#CART model

SharktankCart_aft = rpart(deal ~ ., data=train_aft, method="class")
rpart.plot(SharktankCart_aft)

#printing the cart tree in a pdf document

pdf("Cart_after.pdf")
fancyRpartPlot(SharktankCart_aft, palettes=c("Greys", "Oranges"),
               main = "CART tree after adding variable ratio")
dev.off()

#CART Diagram
prp(SharktankCart_aft, extra=2, main = "CART tree after adding variable ratio")

summary(SharktankCart_aft)
attributes(SharktankCart_aft)

#confusion matrix

train_aft$predict.class <- predict(SharktankCart_aft, train_aft, type="class")
train_aft$predict.score <- predict(SharktankCart_aft, train_aft)

confmattrain_aft = table(train_aft[,c("deal","predict.class")])
confusionMatrix(confmattrain_aft)

#predicting for the test dataset

test_aft$predict.class <- predict(SharktankCart_aft, test_aft, type="class")
test_aft$predict.score <- predict(SharktankCart_aft, test_aft)

confmattest_aft = table(test_aft[,c("deal","predict.class")])
confusionMatrix(confmattest_aft)

#Logistic Regression (80:20 split)

set.seed(1234)
split_af <- createDataPartition(dtmdata$deal, p = .8, list = FALSE)
train_lt.af <- stcsparse[split_af,]
test_lt.af  <- stcsparse[-split_af,]

lg_model.aft <- glm(deal ~ .,data = train_lt.af,family = binomial())

summary(lg_model.aft)

pR2(lg_model.aft)

#Confusion matrix of train data

pred_aft<-predict(lg_model.aft,train_lt.af,type='response')
mat_tab.af<-table(train_lt.af$deal, pred_aft > 0.5)
mat_tab.af

lt_acc_aft <- sum(diag(mat_tab.af))/sum(mat_tab.af)
lt_acc_aft*100

#Prediction and Confusion matrix of test data

tdata_aft <- predict(lg_model.aft,test_lt.af, type="response")
t_confmat.aft <- table(test_lt.af$deal, tdata_aft > 0.5)
t_confmat.aft

test_acc.aft <- sum(diag(t_confmat.aft))/sum(t_confmat.aft)
test_acc.aft*100

#Random Forest

set.seed(1234)
split_rf.aft <- createDataPartition(dtmdata$deal, p = .8, list = FALSE)
train_rf.aft <- stcsparse[split_rf.aft,]
test_rf.aft  <- stcsparse[-split_rf.aft,]

model_rf.aft<-randomForest(deal~.,data=train_rf.aft,importance=T)
model_rf.aft

plot(model_rf.aft, main = "Random forest model after adding variable ratio")

varImpPlot(model_rf.aft, main = "Variable importance plot after adding variable ratio")

#confusion matrix

train_rf.aft$predict.class <- predict(model_rf.aft, train_rf.aft, type="class")
train_rf.aft$predict.score <- predict(model_rf.aft, train_rf.aft)

confmattrain_rf.aft = table(train_rf.aft[,c("deal","predict.class")])
confusionMatrix(confmattrain_rf.aft)

#predicting for the test dataset

test_rf.aft$predict.class <- predict(model_rf.aft, test_rf.aft, type="class")
test_rf.aft$predict.score <- predict(model_rf.aft, test_rf.aft)

confmattest_rf.aft = table(test_rf.aft[,c("deal","predict.class")])
confusionMatrix(confmattest_rf.aft)
