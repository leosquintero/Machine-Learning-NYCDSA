library(ggplot2)
library(tidyverse)
library(dummies)
library(fastDummies)
library(class)
library(corrplot)
library(caret)

#####Data Cleaning#####


#uploading data sets
train = read.csv("train.csv", stringsAsFactors = FALSE)

#### wrangling and cleaning ####

# Taking a look at the head
head(train)


# taking a look at the dimentions
dim(train)

#taking a look at the structure
str(train)

# taking a look at the summary
summary(train)

colnames(train)

# Droping column ID
train <- train[, -1]


#### Missing Data #### 

# Percentage of data missing in the data frame (0.05889565)
sum(is.na(train)) / (nrow(train) *ncol(train))

# counting missing values per column
data.frame(sapply(train, function(y) sum(length(which(is.na(y))))))

# Imputing missing data with the mean of each numerical column
for(i in 1:ncol(train)){
    train[is.na(train[,i]), i] <- mean(train[,i], na.rm = TRUE)
}

# Erasing columns with too many missing values
train <- train[ , -which(names(train) %in% c("Alley","PoolQC", "Fence", 
                                             "MiscFeature", "FireplaceQu"))]

# Deleting rows with remaining missing values
train <- train[complete.cases(train), ]

# Counting missing values again
sum(is.na(train)) / (nrow(train) *ncol(train))


#### Dummies ####

# dummifying Categorical(Factor) columns 
t <- dummyVars("~.", data = train, drop2nd = TRUE)
train <- data.frame(predict(t, newdata = train))


#### EDA ####

# evaluate relevant variables and plot with and without outliers
qplot(train$GrLivArea, train$SalePrice, main = "With Outliers")

# Deleting outliers
train <- train[-which(train$GrLivArea > 4000),]
 
#plot without outliers
qplot(train$GrLivArea, train$SalePrice, main = "Without Outliers")



# Selecting relevant variables to subset

train.model<- lm(SalePrice ~ .,  data = train)
summary(train.model)



















