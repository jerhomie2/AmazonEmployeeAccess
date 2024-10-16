library(tidyverse)
library(tidymodels)
library(vroom)

train <- vroom("./AmazonEmployeeAccess/train.csv")
test <- vroom("./AmazonEmployeeAccess/test.csv")

#-----EDA------



#-----Recipe-----
amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())
prepped_recipe <- prep(amazon_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

ncol(baked_train)
