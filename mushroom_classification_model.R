library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)
library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue)
library(h2o) 
library(scorecard)

path <- dirname(getSourceEditorContext()$path)
setwd(path)

#Importing data
raw <- read.csv("mushrooms.csv")

raw %>% skim()
raw %>% glimpse()

#Checking the balance of the label. We can see that it is balanced.
raw$class %>% table() %>% prop.table()

#Converting the label to factor. "1" will correspond to poisonous mushrooms.
raw$class <- raw$class %>% 
  factor(levels = c("'p'","'e'"),
         labels = c(1,0))

#Q2. Apply Cross-validation;
h2o.init()
raw.num <- raw %>%
  select_if(is.numeric)

raw.chr <- raw %>%
  select_if(is.character)

drop_single_valued_cols <- function(df) {
  num_unique <- sapply(df, function(x) length(unique(x)))
  keep_cols <- which(num_unique > 1)
  return(df[, keep_cols])
}

raw.chr <- drop_single_valued_cols(raw.chr)

raw.chr <- dummyVars(" ~ .", data = raw.chr) %>% 
  predict(newdata = raw.chr) %>% 
  as.data.frame()

final <- cbind(setNames(data.frame(raw$class), "class"),raw.chr,raw.num) %>%
  select(class,everything())

final <- final %>% as.h2o()

folds <- 5
cross_val_model <- h2o.gbm(x = raw.chr %>%  names(), y = 'class', training_frame = final,
                           nfolds = folds, seed = 1234)

#We can see that we are getting a perfect AUC score of 1 for cross validation.
print(h2o.auc(cross_val_model, xval = TRUE))

# --------------------------------- Modeling ----------------------------------
#Q1. Build classification model with h2o.automl();

# Splitting the data ----
h2o_data <- final %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- raw.chr %>% names()


# Fitting h2o model ----
model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  nfolds = 10, seed = 123,
  max_runtime_secs = 200)

#We can see that many models have led to an AUC score of 1.
model@leaderboard %>% as.data.frame() 

#Choosing the best model
model <- model@leader 

#Making predictions for test data
pred <- model %>% h2o.predict(newdata = test) %>% 
  as.data.frame() %>% select(p1,predict)

#Getting the confusion matrix. The accuracy of our model is 100%
conf_matrix <- confusionMatrix(pred$predict, test$class %>% as.vector() %>% 
                                 factor(levels = c(0,1),
                                        labels = c(0,1)))
conf_matrix

#We will now calculate precision and recall manually
results <- cbind(pred$predict %>% as.character() %>% as.numeric(), test$class %>% as.vector() %>% as.numeric()) %>% as.data.frame()
names(results) <- c('pred','actual')

FP <- results %>% filter(pred == 1 & actual == 0) %>% nrow()
TP <- results %>% filter(pred == 1 & actual == 1) %>% nrow()
TN <- results %>% filter(pred == 0 & actual == 0) %>% nrow()
FN <- results %>% filter(pred == 0 & actual == 1) %>% nrow()

accuracy <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2*precision*recall)/(precision+recall)

#We got 100% everywhere.
tibble(accuracy=accuracy,
       precision=precision,
       recall=recall,
       f1_score=f1)

#Q3. Find threshold by max F1 score;
model %>% h2o.performance(newdata = test) %>%
  h2o.find_threshold_by_max_metric('f1')

#Q4. Calculate Accuracy, AUC, GİNİ.
#Checking for overfitting. Calculating AUC and GINI. Accuracy has been calculated above.
#We are getting maximum results everywhere. There is definetily no overfitting.
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)
