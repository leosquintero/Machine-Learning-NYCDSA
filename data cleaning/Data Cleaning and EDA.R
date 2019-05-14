library(ggplot2)
library(tidyverse)
library(dummies)

#####Data Cleaning#####


#uploading data sets
train = read.csv("./train.csv", stringsAsFactors = FALSE)
test = read.csv("./test.csv", stringsAsFactors = FALSE)

#### wrangling and cleaning ####

# Taking a look at the head
head(train)
head(test)

# taking a look at the dimentions
dim(train)

#taking a look at the structure
str(train)

# taking a look at the summary
summary(train)

colnames(train)

# Droping column ID
train <- train[, -1]


#### Missing values #### 

# Percentage of data missing in the data frame (0.05889565)
sum(is.na(train)) / (nrow(train) *ncol(train))

# counting missing values per column
data.frame(sapply(train, function(y) sum(length(which(is.na(y))))))

# replacing NA by the mean of each numerical column
for(i in 1:ncol(train)){
    train[is.na(train[,i]), i] <- mean(train[,i], na.rm = TRUE)
}

# Erasing columns with too many missing values
train <- train[ , -which(names(train) %in% c("Alley","PoolQC", "Fence", 
                                             "MiscFeature"))]



# Select only columns that add value to a property and subset
# dummify Categorical variables if necessary
# drop rows with NAN or impute (case sensitive)
#dummify
dummy(train$Street, 'Pave')
train$Street <- ifelse(train$Street == "Pave", 1, 0)

#### EDA ####

qplot(train$GrLivArea, train$SalePrice, main = "With Outliers")




