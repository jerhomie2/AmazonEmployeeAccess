library(tidyverse)
library(tidymodels)
library(vroom)

train <- vroom("./AmazonEmployeeAccess/train.csv")
test <- vroom("./AmazonEmployeeAccess/test.csv")

train$ACTION <- factor(train$ACTION)

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())
prepped_recipe <- prep(amazon_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

logModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## Put into a workflow here
log_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(logModel) %>%
  fit(data=train)

## Make predictions
amazon_predictions <- predict(log_wf,
                              new_data=test,
                              type="prob") # "class"(yes or no) or "prob"(probability)


kaggle_submission <- amazon_predictions %>% 
  select(.pred_1) %>%
  rename(ACTION=.pred_1) %>%
  mutate(id = row_number())

## Write out the file
vroom_write(x=kaggle_submission, file="./AmazonEmployeeAccess/LogPreds.csv", delim=",")


