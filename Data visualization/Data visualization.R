library(ggplot2)
library(tidyverse)
library(corrplot)


train_rel <- read.csv('./Data/train_relevant')



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
