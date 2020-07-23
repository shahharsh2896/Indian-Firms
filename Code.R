library(readr)
library(ROCR)
library(DMwR)
library('smotefamily')

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
IMB579_XLS_ENG <-IMB579_XLS_ENG[,c(2:10)]
IMB579_XLS_ENG<- as.data.frame(IMB579_XLS_ENG)

SampleData <- SMOTE(IMB579_XLS_ENG[,-9],IMB579_XLS_ENG[,9])
nrow(SampleData$data) #2370 rows
SampleData <- SampleData$data
colnames(SampleData)[9] <- "Manipulater"
SampleData$Manipulater <- as.factor(SampleData$Manipulater)
str(SampleData)


#CART Model
set.seed(1234)
index <- sample(2, nrow(SampleData), replace = T, prob = c(0.75,0.25))
TrainData <- SampleData[index == 1, ]
TestData <- SampleData[index == 2, ]

library(rpart)
man_rpart <- rpart(Manipulater ~ ., data = TrainData, parms = list(split = "gini"))

printcp(man_rpart) 
opt <- which.min(man_rpart$cptable[,"xerror"])
opt
cp1 <- man_rpart$cptable[opt, "CP"]
cp1
#0.01

# We can use the rpart.plot to plot the decision tree
library(rpart.plot)
rpart.plot(man_rpart)

# Print the decision tree and take a look at the summary of rpart
print(man_rpart)
summary(man_rpart)
pred_Test_class<- predict(man_rpart, newdata = TestData, type = "class")
(mean(pred_Test_class == TestData$Manipulater))*100
#87.39054%

confusionMatrix(pred_Test_class, TestData$Manipulater, positive = "Yes")
#pred_Test_class  No   Yes
#  No             238  31
#  Yes            41   261

###Accuracy    : 87.39054%
###Sensitivity : 0.8938       
###Specificity : 0.8530   


#Stepwise LOGIsTIC Regression

set.seed(1234)
null <- glm(Manipulater ~ 1, data= TrainData,family="binomial") # only includes one variable
full <- glm(Manipulater ~ ., data= TrainData,family="binomial") # includes all the variables
logitModel <- step(null, scope = list(lower = null, upper = full), direction = "both")
summary(logitModel)

mylogit = glm( Manipulater ~ ACCR + DSRI + SGI + AQI + GMI + LEVI + 
                 DEPI, family = "binomial", data = TrainData)
summary(mylogit)

pred_prob <- predict(mylogit,newdata = TestData, type = "response")
table(pred_prob > 0.5 ,TestData$Manipulater )
#Getting Probability cut off point using ROC curve
pred <- prediction( predictions = pred_prob, TestData$Manipulater)
perf <- performance(pred,"tpr","fpr")
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x-0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]],
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)}

print(opt.cut(perf, pred))
table(pred_prob >  0.4324689,TestData$Manipulater)
pred_Class <- as.factor(ifelse(pred_prob > 0.4324689,"Yes","No"))
with(mylogit, null.deviance - deviance) #1091.685
(1-mean(pred_Class != TestData$Manipulater))*100
#        No   Yes
#FALSE   239  35
#TRUE    40   257

###Accuracy    : 86.86515%
###Sensitivity : 0.8709677     
###Specificity : 0.8664384 


#RANDOM FOREST

set.seed(1234)
library(randomForest)
library(caret)
fit = randomForest(Manipulater ~ ., data=TrainData,
                   importance=TRUE, proximity=TRUE)
fit
predict.rf <- predict(fit,newdata = TestData)
confusionMatrix(predict.rf, TestData$Manipulater, positive = "Yes")
importance(fit)
varImpPlot(fit)

#           Actual
#Prediction  No   Yes
# No         267  0
# Yes        12   292

###Accuracy    : 97.9%
###Sensitivity : 1.0000       
###Specificity : 0.9570 


#ADABOOST
set.seed(1234)
man.adaboost <- boosting(Manipulater ~ ., data = TrainData, mfinal = 10, control = rpart.control(maxdepth = 1))
man.adaboost

# trees show the weaklearners used at each iteration
man.adaboost$trees
man.adaboost$trees[[1]]

# weights returns the voting power
man.adaboost$weights

# prob returns the confidence of predictions
man.adaboost$prob

# class returns the predicted class
man.adaboost$class

# votes indicates the weighted predicted class
man.adaboost$votes

#importance returns important variables
man.adaboost$importance

table(man.adaboost$class, TrainData$Manipulater, dnn = c("Predicted Class", "Observed Class"))
#                 Observed Class
#Predicted Class  No   Yes
#            No   791  265
#           Yes   130  613

errorrate <- 1 - sum(man.adaboost$class == TrainData$Manipulater) /length(TrainData$Manipulater)
errorrate

# To get predicted class on test data we can use predict function
pred <- predict(man.adaboost,newdata = TestData)

#                 Observed Class
#Predicted Class  No   Yes
#           No    242  75
#           Yes   37   217

###   Accuracy : 80.3853%
###Sensitivity : 0.7432
###Specificity : 0.8674

# However if you use predict.boosting, you can change mfinal
man.predboosting <- predict.boosting(man.adaboost, newdata = TestData)

# errorevol calculates errors at each iteration of adaboost
err.train <- errorevol(man.adaboost,TrainData)
err.test <- errorevol(man.adaboost,TestData)

plot(err.test$error, type = "l", ylim = c(0,1), col = "red", lwd = 2)
lines(err.train$error, cex = 0.5, col = "blue", lty = 2, lwd = 2)

