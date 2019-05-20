library(ggplot2)
library(tidyverse)
library(dummies)
library(class)
library(corrplot)
library(caret)
library(broom)

#####Data Cleaning#####


#uploading data sets
train = read.csv("./Data/train.csv", stringsAsFactors = FALSE)

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

# transforming numerical columns that should be considered categorical
train$MSSubClass = as.character(train$MSSubClass)
train$YearRemodAdd = as.character(train$YearRemodAdd)
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


# Running a preliminary multiple linear model to evaluate the relevance of all variables
model<- lm(SalePrice ~ .,  data = train)
summary(model)


#### Feature Selection ####

#Selecting all variables with P value < 0.05
tm <- tidy(model)

# visualise dataframe of the model (using non scientific notation of numbers)
options(scipen = 999)
train_rel <- tm$term[tm$p.value < 0.05]

#obtaining a data frame with the column names that match the previous condition
train_rel <- train %>% 
    select(train_rel)

#adding Saleprice to new data set
train_rel["SalePrice"] <- train$SalePrice

# running a nwe lm including only features with p-value smaller than 0.05
summary(lm(SalePrice ~ .,  data = train_rel))


#### Data Visualization ####

hist(train_rel$SalePrice, probability = F)

# evaluate relevant variables and plot with and without outliers
qplot(train_rel$LotArea, train_rel$SalePrice, main = "With Outliers")

# Deleting outliers
train_rel <- train_rel[-which(train_rel$LotArea > 30000),]
 
#plot without outliers
qplot(train_rel$LotArea, train_rel$SalePrice, main = "Without Outliers")


ggplot(train_rel, aes(SalePrice))+
    geom_area(stat = "bin")



# writing file with cleaned dummified data
write.csv(train, "train_wrangled", row.names=FALSE)
write.csv(train_rel, "train_relevant", row.names=FALSE )
