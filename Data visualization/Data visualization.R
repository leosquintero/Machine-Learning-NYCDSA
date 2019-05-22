library(ggplot2)
library(tidyverse)
library(corrplot)

#### Data visualization #####

train <- read.csv('./Data/train_relevant')


model <- lm(SalePrice ~ .,  data = train)

plot(model)





qplot(train$LotArea, train$SalePrice, main = "Without Outliers")
