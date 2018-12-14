rm(list=ls(all=TRUE))

#setwd("~/Downloads/20180901_Batch_45_CSE_7305c_CUTe_PartB/Data/")
bankdata <- read.csv("bankdata.csv")
names(bankdata)
str(bankdata)
summary(bankdata)

#Keeping original data
OriginalData <- bankdata

bankdata$target <- as.factor(as.character(bankdata$target))
str(bankdata)

sum(is.na(bankdata))
bankdata <- subset(bankdata, select = -c(Attr37))
summary(bankdata)
sum(is.na(bankdata))
head(bankdata)

colMeans(is.na(bankdata)) > 0.8

A <- rowSums(is.na(bankdata))[rowSums(is.na(bankdata))>5]
length(A)


bankdata <- as.data.frame(bankdata)
class(bankdata)


library(DMwR)
imputedData <- centralImputation(bankdata)
sum(is.na(imputedData))

attach(imputedData)
library(caret)
set.seed(123)
rec <- createDataPartition(target, times = 1, list = F, p = 0.8)

x_train <- imputedData[rec,]
x_test <- imputedData[-rec,]

str(x_train)
str(x_test)


dim(x_train)
dim(x_test)

######################################### KNN Algorithm ##################################### 
# Setting levels for both training and validation data
levels(x_train$target) <- make.names(levels(factor(x_train$target)))
levels(x_test$target) <- make.names(levels(factor(x_test$target)))

# Setting up train controls
repeats = 3
numbers = 10
tunel = 10

set.seed(1234)
x = trainControl(method = "repeatedcv",
                 number = numbers,
                 repeats = repeats,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

model1 <- train(target~. , data = x_train, method = "knn",
                preProcess = c("center","scale"),
                trControl = x,
                metric = "ROC",
                tuneLength = tunel)

# Summary of model
model1
plot(model1)

# Validation
valid_pred <- predict(model1,x_test, type = "prob")
valid_pred
#Storing Model Performance Scores
library(ROCR)
pred_val <-prediction(valid_pred[,2],x_test$target)

# Calculating Area under Curve (AUC)
perf_val <- performance(pred_val,"auc")
perf_val

# Plot AUC
perf_val <- performance(pred_val, "tpr", "fpr")
plot(perf_val, col = "green", lwd = 1.5)

#Calculating KS statistics
ks <- max(attr(perf_val, "y.values")[[1]]-(attr(perf_val, "x.values")[[1]]))
ks

test_pred <- predict(model1,x_test)
confusionMatrix(test_pred, x_test$target)

###############Continuing with PreProcessing post KNN


##Splitting the Target variable to a new dataset in Train and Test
#Train
x_trainWithoutTarget <- subset(x_train, select = -c(target))
names(x_trainWithoutTarget)
y_trainOnlyTarget <- subset(x_train, select = c(target))
names(y_trainOnlyTarget)

#Test
x_testWithoutTarget <- subset(x_test, select = -c(target))
names(x_testWithoutTarget)
y_testOnlyTarget <- subset(x_test, select = c(target))
names(y_testOnlyTarget)
correlation_withoutTarget <- cor(x_trainWithoutTarget, method = "s")
correlation_withoutTarget


##Observing correlations of the Independent variables
library(corrplot)
corrplot(correlation_withoutTarget)

corrplot(correlation_withoutTarget,title = "Correlation Plot", method = "square", outline = T, 
         addgrid.col = "darkgray", order="hclust", mar = c(4,0,4,0), 
         addrect = 4, rect.col = "black", rect.lwd = 5,cl.pos = "b", tl.col = "indianred4", 
         tl.cex = 0.25, cl.cex = 0.25)

library(RColorBrewer)
corrplot(correlation_withoutTarget, method = "color", outline = T, addgrid.col = "darkgray", 
         order="hclust", addrect = 4, rect.col = "black", rect.lwd = 5, cl.pos = "b", 
         tl.col = "indianred4", tl.cex = 0.25, cl.cex = 0.25, addCoef.col = "white", 
         number.digits = 2, number.cex = 0.25, 
         col = colorRampPalette(c("darkred","white","midnightblue"))(100))

##Standardization of Train and Test
scale_train <- preProcess(x_train, !(names(x_train) %in% (target)), method = c("center", "scale"))
scale_test <- preProcess(x_test,!(names(x_test) %in% (target)), method = c("center", "scale"))

train <- predict(scale_train, x_train)
test <- predict(scale_test, x_test)


#################### PCA
library(caret)
train1 <- preProcess(x_trainWithoutTarget, method=c("center","scale"))
test1 <- preProcess(x_testWithoutTarget, method = c("center", "scale"))

train1 <- predict(train1, x_trainWithoutTarget)
test1<- predict(test1, x_testWithoutTarget)

attach(train1)
standardized_pca <- princomp(train1)
pca$loadings
summary(standardized_pca)
standardized_pca$scores

pca_train <- predict(standardized_pca, train1)
pca_test <- predict(standardized_pca, test1)

#Checking how many components are really required
plot(standardized_pca)
p.variance.explained <- standardized_pca$sdev^2 / sum(standardized_pca$sdev^2)
barplot(100*p.variance.explained, las = 2, xlab = '', ylab = "% Variance")

#Lets say 19 components
pca_x <- data.frame(pca_train[,1:19],"NEW" = y_trainOnlyTarget)
pca_y <- data.frame(pca_test[,1:19], "NEW" = y_testOnlyTarget)


############################## LOGISTIC REGRESSION MODEL - USING PCA ################################
log_reg_pca <- glm(target~., 
               data = pca_x, family = "binomial")
summary(log_reg_pca)

##Predicting on train data
predvalues_logRegPCA_train <- predict(log_reg_pca, newdata = pca_x, type = "response")

###Choosing a cutoff point
##Using ROC curve
library(ROCR)
predvalues_logRegPCA_trainROCR <- prediction(predvalues_logRegPCA_train, pca_x$target)
##Extract performance measures (True Positive Rate and False Positive Rate) 
##using the “performance()” function from the ROCR package
##The performance() function from the ROCR package helps us extract metrics such as 
##True positive rate, False positive rate etc. from the prediction object, we created above.
##Two measures (y-axis = tpr, x-axis = fpr) are extracted
perf_logReg_pca <- performance(predvalues_logRegPCA_trainROCR, measure = "tpr", 
                               x.measure = "fpr")
plot(perf_logReg_pca, col=rainbow(10), colorize = T, print.cutoffs.at=seq(0,1,0.1))

#Extract the AUC score of the ROC curve and store it in a variable named “auc”
perf_logReg_pca_auc <- performance(predvalues_logRegPCA_trainROCR, measure = "auc")

#Access the auc score from the performance object
logReg_pca_auc <- perf_logReg_pca_auc@y.values[[1]]
print(logReg_pca_auc)

##Deciding on Cutoff Value
###Based on the trade-off between TPR and FPR, depending on the business domain, a call on the
###cutoff has to be made. Here, a cutoff of 0.1 can be chosen

###Predictions on Test data
predvalues_logReg_pca_test <- predict(log_reg_pca, newdata = pca_y, type = "response")
pred_class_logReg_pca_test <- ifelse(predvalues_logReg_pca_test > 0.1, 1, 0)

pred_class_logReg_pca_test <- factor(pred_class_logReg_pca_test)
table(pred_class_logReg_pca_test)

library(ROCR)
predvalues_logReg_pca_testROCR <- prediction(predvalues_logReg_pca_test, pca_y$target)
#TEST ----- #Extract the AUC score of the ROC curve and store it in a variable named “auc”
test_logReg_pca_auc <- performance(predvalues_logReg_pca_testROCR, measure = "auc")

logReg_test_pca_auc <- test_logReg_pca_auc@y.values[[1]]
print(logReg_test_pca_auc)

##Evaluation Metrics for classification
###Manual Computation using Confusion Matrix
conf_matrix_pca <- table(pca_y$target, pred_class_logReg_pca_test)
print(conf_matrix_pca)
conf_matrix_sensitivity_pca <- conf_matrix_pca[1,1]/sum(conf_matrix_pca[1,])
conf_matrix_sensitivity_pca
conf_matrix_specificity_pca <- conf_matrix_pca[2,2]/sum(conf_matrix_pca[2,])
conf_matrix_specificity_pca
conf_matrix_accuracy_pca <- sum(diag(conf_matrix_pca))/sum(conf_matrix_pca)
conf_matrix_accuracy_pca


str(pred_class_logReg_pca_test);str(pca_y$target)
summary(log_reg_pca)


## Loading DMwr to balance the unbalanced class
library(DMwR)
## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
balanced.data <- SMOTE(target~., train, perc.over = 600, k = 5, perc.under = 200)

class(balanced.data)

as.data.frame(table(balanced.data$target))

#Unique values in Train and Test
apply(train, 2, function(x){length(unique(x))})
apply(test, 2, function(x){length(unique(x))})

################### LOGISTIC REGRESSION MODEL - Using SMOTE and Correlation ###################
log_reg <- glm(balanced.data$target~(Attr61+Attr60+Attr36+Attr64+Attr9+Attr34+Attr55+Attr53+Attr4
                                     +Attr25+Attr38+Attr8+Attr17+Attr33+Attr40+Attr5+Attr46+Attr21
                                     +Attr56+Attr24+Attr13+Attr16+Attr45+Attr11+Attr57+Attr27+Attr39
                                     +Attr48+Attr59+Attr15+Attr41+Attr58+Attr2+Attr51+Attr30+Attr62
                                     +Attr6+Attr29+Attr20+Attr43+Attr44), 
               data = balanced.data, family = "binomial")
summary(log_reg)

##Predicting on train data
predvalues_logReg_train <- predict(log_reg, newdata = balanced.data, type = "response")

###Choosing a cutoff point
##Using ROC curve
library(ROCR)
predvalues_logReg_trainROCR <- prediction(predvalues_logReg_train, balanced.data$target)
##Extract performance measures (True Positive Rate and False Positive Rate) 
##using the “performance()” function from the ROCR package
##The performance() function from the ROCR package helps us extract metrics such as 
##True positive rate, False positive rate etc. from the prediction object, we created above.
##Two measures (y-axis = tpr, x-axis = fpr) are extracted
perf <- performance(predvalues_logReg_trainROCR, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(10), colorize = T, print.cutoffs.at=seq(0,1,0.1))

#Extract the AUC score of the ROC curve and store it in a variable named “auc”
perf_logReg_auc <- performance(predvalues_logReg_trainROCR, measure = "auc")

#Access the auc score from the performance object
logReg_auc <- perf_logReg_auc@y.values[[1]]
print(logReg_auc)

##Deciding on Cutoff Value
###Based on the trade-off between TPR and FPR, depending on the business domain, a call on the
###cutoff has to be made. Here, a cutoff of 0.1 can be chosen

###Predictions on Test data
predvalues_logReg_test <- predict(log_reg, newdata = test, type = "response")
pred_class_logReg_test <- ifelse(predvalues_logReg_test > 0.1, 1, 0)

pred_class_logReg_test <- factor(pred_class_logReg_test)
table(pred_class_logReg_test)

library(ROCR)
predvalues_logReg_testROCR <- prediction(predvalues_logReg_test, test$target)
#TEST ----- #Extract the AUC score of the ROC curve and store it in a variable named “auc”
test_logReg_auc <- performance(predvalues_logReg_testROCR, measure = "auc")

logReg_test_auc <- test_logReg_auc@y.values[[1]]
print(logReg_test_auc)

##Evaluation Metrics for classification
###Manual Computation using Confusion Matrix
conf_matrix <- table(test$target, pred_class_logReg_test)
print(conf_matrix)
conf_matrix_sensitivity <- conf_matrix[1,1]/sum(conf_matrix[1,])
conf_matrix_sensitivity
conf_matrix_specificity <- conf_matrix[2,2]/sum(conf_matrix[2,])
conf_matrix_specificity
conf_matrix_accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
conf_matrix_accuracy


str(pred_class_logReg_test);str(test$target)
summary(log_reg)

################################### LOGISTIC REGRESSION MODEL-2 ###############################
###Improve the model using stepAIC
library(MASS)
log_reg_modelAIC <- stepAIC(log_reg, direction = "both")
summary(log_reg_modelAIC)

###Use vif to find any multi-collinearity
library(car)
log_reg_vif <- vif(log_reg_modelAIC)
log_reg_vif

#New Logistic Model based on StepAIC and VIF (using only StepAIC)
log_reg2 <- glm(train$target ~ Attr36 + Attr64 + Attr9 + Attr55 + 
                  Attr53 + Attr4 + Attr25 + Attr38 + Attr33 + Attr40 + Attr46 + 
                  Attr56 + Attr13 + Attr16 + Attr11 + Attr57 + Attr27 + Attr48 + 
                  Attr59 + Attr58 + Attr51 + Attr62 + Attr29 + Attr20 + Attr43 + 
                  Attr44, 
                data = train, family = "binomial")
summary(log_reg2)

##Predicting on train data
predvalues_logReg2_train <- predict(log_reg2, newdata = train, type = "response")

###Choosing a cutoff point
##Using ROC curve
library(ROCR)
predvalues_logReg2_trainROCR <- prediction(predvalues_logReg2_train, train$target)
##Extract performance measures (True Positive Rate and False Positive Rate) 
##using the “performance()” function from the ROCR package
##The performance() function from the ROCR package helps us extract metrics such as 
##True positive rate, False positive rate etc. from the prediction object, we created above.
##Two measures (y-axis = tpr, x-axis = fpr) are extracted
perf_logReg2 <- performance(predvalues_logReg2_trainROCR, measure = "tpr", x.measure = "fpr")
plot(perf_logReg2, col=rainbow(10), colorize = T, print.cutoffs.at=seq(0,1,0.1))

#Extract the AUC score of the ROC curve and store it in a variable named “auc”
perf_logReg2_auc <- performance(predvalues_logReg2_trainROCR, measure = "auc")

#Access the auc score from the performance object
logReg2_auc <- perf_logReg2_auc@y.values[[1]]
print(logReg2_auc)

##Deciding on Cutoff Value
###Based on the trade-off between TPR and FPR, depending on the business domain, a call on the
###cutoff has to be made. Here, a cutoff of 0.1 can be chosen

###Predictions on Test data
predvalues_logReg2_test <- predict(log_reg2, newdata = test, type = "response")
pred_class_logReg2_test <- ifelse(predvalues_logReg2_test > 0.1, 1, 0)

pred_class_logReg2_test <- factor(pred_class_logReg2_test)
table(pred_class_logReg2_test)

library(ROCR)
predvalues_logReg2_testROCR <- prediction(predvalues_logReg2_test, test$target)
#TEST ----- #Extract the AUC score of the ROC curve and store it in a variable named “auc”
test_logReg2_auc <- performance(predvalues_logReg2_testROCR, measure = "auc")

logReg2_test_auc <- test_logReg2_auc@y.values[[1]]
print(logReg2_test_auc)

##Evaluation Metrics for classification
###Manual Computation using Confusion Matrix
conf_matrix_logReg2 <- table(test$target, pred_class_logReg2_test)
print(conf_matrix_logReg2)
conf_matrix_sensitivity_logReg2 <- conf_matrix_logReg2[1,1]/sum(conf_matrix_logReg2[1,])
conf_matrix_sensitivity_logReg2
conf_matrix_specificity_logReg2 <- conf_matrix_logReg2[2,2]/sum(conf_matrix_logReg2[2,])
conf_matrix_specificity_logReg2
conf_matrix_accuracy_logReg2 <- sum(diag(conf_matrix_logReg2))/sum(conf_matrix_logReg2)
conf_matrix_accuracy_logReg2


str(pred_class_logReg2_test);str(test$target)
summary(log_reg2)
################################### NAIVE-BAYES MODEL ####################################
library(e1071)
nb_model = naiveBayes(balanced.data$target~., data = balanced.data)
nb_model

predvalues_nb_train = predict(nb_model, train)
table(predvalues_nb_train, train$target)

predvalues_nb_test = predict(nb_model, test)
table(predvalues_nb_test, test$target)

###Manual Computation using Confusion Matrix
conf_matrix_nb <- table(test$target, predvalues_nb_test)
print(conf_matrix_nb)
conf_matrix_sensitivity_nb <- conf_matrix_nb[1,1]/sum(conf_matrix_nb[1,])
conf_matrix_sensitivity_nb
conf_matrix_specificity_nb <- conf_matrix_nb[2,2]/sum(conf_matrix_nb[2,])
conf_matrix_specificity_nb
conf_matrix_accuracy_nb <- sum(diag(conf_matrix_nb))/sum(conf_matrix_nb)
conf_matrix_accuracy_nb
conf_matrix_precision_nb <- conf_matrix_nb[2,1]/sum(conf_matrix_nb[,1])
conf_matrix_precision_nb

confusionMatrix(data=predvalues_nb_test,  
                reference=test$target)

table(train$target)
table(test$target)

############################### RANDOM FOREST ###############################
library(randomForest)
library(e1071)  

balanced.data_train_rf <- subset(balanced.data, select = c(Attr36,Attr64,Attr9,Attr55,Attr53,Attr4,Attr25,Attr38,
                                     Attr33,Attr40,Attr46,Attr56,Attr13,Attr16,Attr11,
                                     Attr57,Attr27,Attr48,Attr59,Attr58,Attr51,Attr62,
                                     Attr29,Attr20,Attr43,Attr44,target))

test_rf <- subset(test, select = c(Attr36,Attr64,Attr9,Attr55,Attr53,Attr4,Attr25,Attr38,
                                   Attr33,Attr40,Attr46,Attr56,Attr13,Attr16,Attr11,
                                   Attr57,Attr27,Attr48,Attr59,Attr58,Attr51,Attr62,
                                   Attr29,Attr20,Attr43,Attr44,target))

rf = randomForest(target~.,  
                  ntree = 100,
                  data = balanced.data_train_rf)
plot(rf) 
varImp(rf)

## Important variables according to the model
varImpPlot(rf,  
           sort = T,
           n.var=27,
           main="Variable Importance")

predicted.response <- predict(rf, test_rf)


confusionMatrix(data=predicted.response,  
                reference=test_rf$target)

train_rf <- subset(train, select = c(Attr36,Attr64,Attr9,Attr55,Attr53,Attr4,Attr25,Attr38,
                                                           Attr33,Attr40,Attr46,Attr56,Attr13,Attr16,Attr11,
                                                           Attr57,Attr27,Attr48,Attr59,Attr58,Attr51,Attr62,
                                                           Attr29,Attr20,Attr43,Attr44,target))

test_rf <- subset(test, select = c(Attr36,Attr64,Attr9,Attr55,Attr53,Attr4,Attr25,Attr38,
                                   Attr33,Attr40,Attr46,Attr56,Attr13,Attr16,Attr11,
                                   Attr57,Attr27,Attr48,Attr59,Attr58,Attr51,Attr62,
                                   Attr29,Attr20,Attr43,Attr44,target))

rf = randomForest(target~.,  
                  ntree = 500,
                  data = train_rf)
plot(rf) 
varImp(rf)

## Important variables according to the model
varImpPlot(rf,  
           sort = T,
           n.var=27,
           main="Variable Importance")

predicted.response <- predict(rf, test_rf)


confusionMatrix(data=predicted.response,  
                reference=test_rf$target)
