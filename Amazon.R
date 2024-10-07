library(tidyverse)
library(tidymodels)
library(vroom)

train <- vroom("./AmazonEmployeeAccess/train.csv")
test <- vroom("./AmazonEmployeeAccess/test.csv")

#-----EDA------




amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = .001)
prepped_recipe <- prep(amazon_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

ncol(baked_train)
