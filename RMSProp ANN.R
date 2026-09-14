library(tidyverse)
library(gridExtra)

## 2.1 Installing keras in R
install.packages("keras")
install.packages("tensorflow")

library(keras)
library(tensorflow)
reticulate::install_python('3.10:latest')
install_tensorflow()
install_keras()

## 2.2 A binary classification job with keras
devtools::install_github("https://github.com/jmsallan/BAdatasets")
library(BAdatasets)
library(tidyverse)
library(caret)
library(keras)

red <- WineQuality$red

red <- red %>% mutate(good = ifelse(quality >= 6, 1, 0))  #factor must be encoded as numeric
red <- red %>% select(-quality)

set.seed(2020)
inTrain <- createDataPartition(red$good, p=0.8, list = FALSE)
red_train <- red %>% slice(inTrain)
red_test <- red %>% slice(-inTrain)

x_train <- red_train %>% 
  select(-good) %>% 
  scale() #we get a matrix doing this

y_train <- to_categorical(red_train$good, 2)

x_test <- red_test %>% 
  select(-good) %>% 
  scale()

y_test <- to_categorical(red_test$good, 2)

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

model %>% fit(
  x_train, y_train, 
  epochs = 100, 
  batch_size = 256,
  validation_split = 0.2,
  verbose=0
)

model %>% evaluate(x_train, y_train)

model %>% evaluate(x_test, y_test)

keraspred <- model %>% predict_classes(x_test)
confusionMatrix(as.factor(keraspred), as.factor(red_test$good), positive="1")

## 2.3 A numerical prediction job with keras
library(mlbench)
data("BostonHousing")
bh2 <- BostonHousing %>% select(-tax)
bh2 <- bh2 %>% mutate(chas=as.numeric(chas)) #transforming factor chas

set.seed(2020)
inTrain <- createDataPartition(bh2$medv, p=0.8, list=FALSE)
bh2_train <- bh2 %>% slice(inTrain)
bh2_test <- bh2 %>% slice(-inTrain)

bh_train <- bh2_train %>% 
  select(-medv) %>% 
  scale()

label_train <- bh2_train$medv

bh_test <- bh2_test %>% 
  select(-medv) %>% 
  scale()

label_test <- bh2_test$medv

bh_model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = ncol(bh_train)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

bh_model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error"))

fit_bh <- bh_model %>% fit(
  bh_train,
  label_train,
  epochs = 500,
  validation_split = 0.1,
  verbose = 0
)

bh_model %>% evaluate(bh_train, label_train)

bh_model %>% evaluate(bh_test, label_test)

plot(fit_bh)

rm(bh_model)

bh_model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = ncol(bh_train)) %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 1)

bh_model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error"))

fit_bh <- bh_model %>% fit(
  bh_train,
  label_train,
  epochs = 100,
  validation_split = 0.2,
  verbose = 1
)

plot(fit_bh)

bh_model %>% evaluate(bh_train, label_train)

bh_model %>% evaluate(bh_test, label_test)

keras_train <- bh_model %>% predict(bh_train)
keras_test <- bh_model %>% predict(bh_test)

postResample(keras_train[,1], bh2_train$medv)

postResample(keras_test[,1], bh2_test$medv)
