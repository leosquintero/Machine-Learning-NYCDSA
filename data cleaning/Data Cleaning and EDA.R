library(ggplot2)
library(tidyverse)
library(dummies)
library(fastDummies)
library(class)
library(corrplot)
library(caret)

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
train$YrSold = as.character(train$YrSold)
train$MoSold = as.character(train$MoSold)
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
train.model<- lm(SalePrice ~ .,  data = train)
summary(train.model)


# Spliting the Data set into relevant and irrelevant sets.
train_rel <- train %>% 
    select(c("MSZoningC..all.", "MSZoningFV", "LotArea", "StreetGrvl", "LandContourLow", "LotConfigCulDSac", 
             "LotConfigFR2", "LandSlopeGtl", "LandSlopeMod", "NeighborhoodEdwards", "NeighborhoodMitchel", 
             "NeighborhoodNAmes", "NeighborhoodNoRidge", "NeighborhoodNridgHt", "NeighborhoodNWAmes", 
             "NeighborhoodStoneBr", "Condition1RRAe", "Condition2PosN", "Condition2RRAe", "HouseStyle2Story",
             "OverallQual", "OverallCond", "YearBuilt","YearRemodAdd1950","YearRemodAdd1951","YearRemodAdd1952",
             "YearRemodAdd1953","YearRemodAdd1954", "YearRemodAdd1955", "YearRemodAdd1956","YearRemodAdd1957",
             "YearRemodAdd1958", 'YearRemodAdd1959', "YearRemodAdd1960", 'YearRemodAdd1961', 
             "YearRemodAdd1962", "YearRemodAdd1963", 'YearRemodAdd1964', "YearRemodAdd1965",
             "YearRemodAdd1966", "YearRemodAdd1967", "YearRemodAdd1968", "YearRemodAdd1969",
             "YearRemodAdd1970", "YearRemodAdd1971", "YearRemodAdd1972", "YearRemodAdd1973",
             "YearRemodAdd1974", "YearRemodAdd1975", "YearRemodAdd1976", "YearRemodAdd1977", 
             "YearRemodAdd1978", "YearRemodAdd1979", "YearRemodAdd1980", "YearRemodAdd1981",
             "YearRemodAdd1982", "YearRemodAdd1983", "YearRemodAdd1984", "YearRemodAdd1985",
             "YearRemodAdd1986", "YearRemodAdd1987", "YearRemodAdd1988", "YearRemodAdd1989",
             "YearRemodAdd1990", "YearRemodAdd1991", "YearRemodAdd1992", "YearRemodAdd1993",
             "YearRemodAdd1994", "YearRemodAdd1995", "YearRemodAdd1996", "YearRemodAdd1997",
             "YearRemodAdd1998", "YearRemodAdd1999", "YearRemodAdd2000", "YearRemodAdd2001",
             "YearRemodAdd2002", "YearRemodAdd2003", "YearRemodAdd2004", "YearRemodAdd2005",
             "YearRemodAdd2006", "YearRemodAdd2007", "YearRemodAdd2008", "YearRemodAdd2009",
             "YearRemodAdd2010", "RoofStyleFlat", "RoofStyleGable", "RoofStyleGambrel",
             "RoofStyleHip", "RoofStyleMansard", "RoofMatlClyTile", "RoofMatlCompShg", "RoofMatlRoll", "RoofMatlTar.Grv",
             "RoofMatlWdShake", "Exterior1stBrkFace", "Exterior2ndImStucc", "MasVnrTypeBrkCmn", "MasVnrTypeBrkFace",
             "MasVnrArea", "ExterQualEx", "FoundationBrkTil", "FoundationCBlock", "FoundationPConc", "FoundationStone",
             "FoundationWood","X1stFlrSF","X2ndFlrSF","KitchenAbvGr","KitchenQualEx", "BsmtQualEx", "BsmtExposureAv", 
             "BsmtExposureGd", "BsmtFinType1LwQ", "BsmtFinSF1","BsmtFinSF2","FunctionalMin2","FunctionalMod",
             "FunctionalSev","GarageType2Types","GarageArea","GarageQualEx", 
             "GarageCondEx","BsmtUnfSF","WoodDeckSF", "PoolArea", "SalePrice"))

summary(lm(SalePrice ~ .,  data = train_rel))


# reading file with cleaned data
write.csv(train, "train_wrangled")
write.csv(train_rel, "train_relevant")


#### EDA ####

hist(train$SalePrice, probability = F)

lines(density(train_rel$SalePrice), col = "red")


# evaluate relevant variables and plot with and without outliers
qplot(train_rel$GrLivArea, train_rel$SalePrice, main = "With Outliers")

# Deleting outliers
train_rel <- train_rel[-which(train_rel$GrLivArea > 4000),]
 
#plot without outliers
qplot(train_rel$GrLivArea, train_rel$SalePrice, main = "Without Outliers")


pairs(train_rel)
# Selecting relevant variables to subset











getOption("max.print", default = 1500)








