library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

train <- vroom("./train.csv")
test <- vroom("./test.csv")

train$ACTION <- factor(train$ACTION)

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(),outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_nominal_predictors())
prepped_recipe <- prep(amazon_recipe)
baked_train <- bake(prepped_recipe, new_data=train) 
baked_test <- bake(prepped_recipe, new_data=test)

knn_mod <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(knn_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #, f_meas, sens, recall, spec, precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
amazon_predictions <- final_wf %>%
  predict(new_data = test, type="prob") # "class"(yes or no) or "prob"(probability)

kaggle_submission <- amazon_predictions %>% 
  select(.pred_1) %>%
  rename(ACTION=.pred_1) %>%
  mutate(id = row_number())

## Write out the file
vroom_write(x=kaggle_submission, file="./KKNpreds.csv", delim=",")

