library(ggplot2)
library(tidyverse)


#####Data Cleaning#####


#uploading data sets
train = read.csv("./train.csv", stringsAsFactors = FALSE)
test = read.csv("./test.csv", stringsAsFactors = FALSE)

#### EDA ####

# Taking a look at the head
head(train)
head(test)

# taking a look at the dimentions
dim(train)
dim(test)

#taking a look at the structure
str(train)
str(test)

# taking a look at the summary
summary(train)
summary(test)


colnames(train)


# counting missing values per column and assigning to missing_val
missing_val <- sapply(train, function(y) sum(length(which(is.na(y)))))

