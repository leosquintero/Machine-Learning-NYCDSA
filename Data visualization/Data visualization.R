library(ggplot2)
library(tidyverse)
library(corrplot)


train_rel <- read.csv('./Data/train_relevant')





x <- train[1:10]
corrplot(train[1:8], pch = 21)


qplot(train$LotArea, train$SalePrice, main = "Without Outliers")

scatterplotMatrix()