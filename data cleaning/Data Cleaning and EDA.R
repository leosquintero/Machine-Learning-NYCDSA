library(ggplot2)
library(tidyverse)
library(dummies)
library(class)
library(corrplot)
library(caret)
library(broom)
library(purrr)
library(ggplot2)
library(gridExtra)
library(corrplot)

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




#### Feature Selection ####

#creating new feature adding the total area of the house
train <- train %>% 
    mutate(TotalSF = TotalBsmtSF + X1stFlrSF + X2ndFlrSF, 
                TotalBsmtSF = NULL, X1stFlrSF = NULL, X2ndFlrSF = NULL)

train <- train[ , -which(names(train) %in% c("BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF"))]


# Running a preliminar multiple linear model to evaluate the relevance of all variables
model<- lm(SalePrice ~ .,  data = train)
summary(model)

#Selecting all variables with P value < 0.04
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

# visualizing lineal model
model1 <- lm(SalePrice ~ .,  data = train_rel)
plot(model1)


# drawing a historgram of SalePrice
hist(train_rel$SalePrice, probability = F, breaks = 50, main = "Sale price hist",
     xlab = "sale Price", col = "skyblue")

# evaluate relevant variables and plot with and without outliers
qplot(train_rel$LotArea, train_rel$SalePrice, main = "With Outliers",xlab = "Lot Area", ylab = "Sale Price")

# Deleting outliers
train_rel <- train_rel[-which(train_rel$LotArea > 30000),]

#plot without outliers
qplot(train_rel$LotArea, train_rel$SalePrice, main = "Without Outliers",xlab = "Lot Area", ylab = "Sale Price")


# Evaluating the distribution 
ggplot(train_rel, aes(x = SalePrice, fill = ..count..)) +
    geom_histogram(binwidth = 5000) +
    ggtitle("Figure 1 Histogram of SalePrice") +
    ylab("Count of houses") +
    xlab("Housing Price") + 
    theme(plot.title = element_text(hjust = 0.5))

#log term of SalePrice sinse the data is skewed to the left
train_rel$SalePrice <- log(train_rel$SalePrice)

# re-evaluating the distribution
ggplot(train_rel, aes(x = SalePrice, fill = ..count..)) +
    geom_histogram(binwidth = 0.05) +
    ggtitle("Figure 2 Histogram of log SalePrice") +
    ylab("Count of houses") +
    xlab("Housing Price normalized") + 
    theme(plot.title = element_text(hjust = 0.5))

# Plotting Sale price compared to lotArea
plot_1 <- ggplot(train_rel, aes(x= SalePrice, y = LotArea)) +
    geom_point()+
    geom_smooth(method=lm , color="blue", se=FALSE) +
    ggtitle("Scatterplot of Sale Price vs Lot Area") +
    theme(plot.title = element_text(hjust = 0.4))


# Plotting Sale price compared to Year Built
plot_2 <- ggplot(train_rel, aes(x= SalePrice, y = YearBuilt)) +
    geom_point()+
    geom_smooth(method=lm , color="blue", se=FALSE) +
    ggtitle("Scatterplot of Sale Price vs Year Built") +
    theme(plot.title = element_text(hjust = 0.4))


# Plotting Sale price compared to Wood Deck SF
plot_3 <- ggplot(train_rel, aes(x= SalePrice, y = WoodDeckSF)) +
    geom_point()+
    geom_smooth(method=lm , color="blue", se=FALSE) +
    ggtitle("Scatterplot of Sale Price vs Wood Deck SF") +
    theme(plot.title = element_text(hjust = 0.4))

#  Plotting Sale price compared to Total Square feet
plot_4 <- ggplot(train_rel, aes(x= SalePrice, y = TotalSF)) +
    geom_point()+
    geom_smooth(method=lm , color="blue", se=FALSE) +
    ggtitle("Scatterplot of Sale Price vs Total Square feet") +
    theme(plot.title = element_text(hjust = 0.4))

#  Plotting Sale price compared to Above grade (ground) living area square fee
plot_5 <- ggplot(train_rel, aes(x= SalePrice, y = GrLivArea)) +
    geom_point()+
    geom_smooth(method=lm , color="blue", se=FALSE) +
    ggtitle("Scatterplot of Sale Price vs Above grade (ground) living area square feet") +
    theme(plot.title = element_text(hjust = 0.4))

#  Plotting Sale price compared to Masonry veneer area in square feet
plot_6 <- ggplot(train_rel, aes(x= SalePrice, y = MasVnrArea)) +
    geom_point()+
    geom_smooth(method=lm , color="blue", se=FALSE) +
    ggtitle("Scatterplot of Sale Price vs Masonry veneer area in square feet") +
    theme(plot.title = element_text(hjust = 0.4))

grid.arrange(plot_1, plot_2,plot_3,plot_4)
grid.arrange(plot_3,plot_4, plot_5,plot_6)

# Showing the corelation between some numeric variables
c1 <- train %>% select("GarageYrBlt","LotArea", "YearBuilt", "WoodDeckSF", "TotalSF",
                       "GrLivArea", "MasVnrArea",'OverallQual','OverallCond','YearBuilt')
corrplot(cor(c1), method = 'number', tl.col = "blue" )

# writing file with cleaned dummified data
write.csv(train_rel, "train_relevant", row.names=FALSE )
