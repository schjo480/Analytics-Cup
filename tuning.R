### Library imports ###
library(data.table)
library(ggplot2)
library(tidyr)
library(dplyr)
library(lubridate)
library(openxlsx)
library(tidymodels)
library(tidyverse)
library(gbm)
library(xgboost)
library(finetune)

##  Setting seed ###
set.seed(2022)

### Data import ###
setwd("~/analytics_cup")
transactions <- fread("Data/Training_Data_AC2022/transactions.csv")
customers <- fread("Data/Training_Data_AC2022/customers.csv")
geo <- fread("Data/Training_Data_AC2022/geo.csv")


### Data Cleaning ###
## transactions ##
transactions <- transactions %>%
  mutate("CUSTOMER" = as.numeric(gsub('""', "", transactions$CUSTOMER)))

transactions <- transactions %>%
  mutate("END_CUSTOMER" = ifelse(transactions$END_CUSTOMER %in% c("Yes", "No"), NA, transactions$END_CUSTOMER))

transactions <- transactions %>%
  mutate("MO_CREATED_DATE" = as.POSIXct(transactions$MO_CREATED_DATE, format = "%d.%m.%Y %H:%M"))

transactions <- transactions %>%
  mutate("SO_CREATED_DATE" = as.POSIXct(transactions$SO_CREATED_DATE, format = "%d.%m.%Y %H:%M"))

transactions <- transactions %>%
  mutate("OFFER_STATUS" = as.factor(ifelse(toupper(transactions$OFFER_STATUS) %in% c("WIN", "WON", "LOST", "LOSE"), ifelse(toupper(transactions$OFFER_STATUS) %in% c("WIN", "WON"), 1, 0), NA)))

# impute ISIC according to TECH category
isic_tech_match <- data.table(tech_match = c("S", "C", "F", "BP", "FP", "EPS", "E"), 
                              isic_match = c(6419, 4322, 4321, 4322, 4659, 5510, 8610))
for(c in transactions[is.na(ISIC)]$MO_ID) {
  transactions <- transactions[MO_ID == c, ISIC := ifelse(is.na(ISIC) & TECH %in% isic_tech_match$tech_match, 
                                                          isic_tech_match[tech_match %in% TECH, isic_match], ISIC)]
}
# impute most used sales location
transactions[, SALES_LOCATION := ifelse(is.na(SALES_LOCATION), "Geneva West", SALES_LOCATION)]


## customers ##
customers <- customers %>%
  mutate("REV_CURRENT_YEAR" = as.numeric(gsub('""', "", customers$REV_CURRENT_YEAR)))

customers <- customers %>%
  mutate("CREATION_YEAR" = gsub("/", ".", customers$CREATION_YEAR))

customers <- customers %>%
  mutate("CREATION_YEAR" = gsub("01.01.", "", customers$CREATION_YEAR))

customers <- customers %>%
  mutate("COUNTRY" = as.factor(ifelse(customers$COUNTRY %in% c("France"), "FR", "CH")))

customers <- customers[, REV_CURRENT_YEAR := NULL] #Leave out REV_CURRENT_YEAR (identical to REV_CURRENT_YEAR.1)


## geo ##
geo <- geo[, SALES_LOCATION := ifelse(is.na(SALES_LOCATION), "Velizy", SALES_LOCATION)]

### Merging all data tables ###
df <- merge(transactions, geo, by = "SALES_LOCATION")

df <- df %>% mutate("COUNTRY" = as.factor(COUNTRY))

df <- df[, END_CUSTOMER := ifelse(is.na(END_CUSTOMER), CUSTOMER, END_CUSTOMER)]

df <- left_join(df, customers, by = c("COUNTRY", "CUSTOMER"))
df <- df[, REV_CURRENT_YEAR.1 := ifelse(is.na(REV_CURRENT_YEAR.1), median(REV_CURRENT_YEAR.1, na.rm = T), REV_CURRENT_YEAR.1)]
df <- df[, REV_CURRENT_YEAR.2 := ifelse(is.na(REV_CURRENT_YEAR.2), median(REV_CURRENT_YEAR.2, na.rm = T), REV_CURRENT_YEAR.2)]
df <- df[, OWNERSHIP := ifelse(is.na(OWNERSHIP), "Privately Owned/Publicly Traded", OWNERSHIP)]
df <- df[, CURRENCY := ifelse(is.na(CURRENCY), sample(CURRENCY, 1), CURRENCY)]
df <- df[, SALES_OFFICE := ifelse(is.na(SALES_OFFICE), "Bezons", SALES_OFFICE)]
df <- df[, CREATION_YEAR := ifelse(is.na(CREATION_YEAR), 2004, CREATION_YEAR)]


### Ready df for Random Forest ###
df <- df %>% 
  mutate(across(.cols = where(~is.character(.)),
                .fns = ~as.factor(.)),
         across(.cols = c(ISIC, CREATION_YEAR, OFFER_STATUS),
                .fns = ~as.factor(.))
  ) %>%  select(-c(CUSTOMER, MO_ID, SO_ID, END_CUSTOMER, MO_CREATED_DATE, SO_CREATED_DATE))


### Splitting Training & Test Data ###
df_train <- df[!is.na(OFFER_STATUS), -c("TEST_SET_ID")]
df_test <- df %>% anti_join(df_train)


## Select only from df_train, since don't have OFFER_STATUS for df_test ##
# Train-test split
train_test_split <- initial_split(df_train, prop = 0.80, strata=OFFER_STATUS) # selects stratified for churners to keep proportion of churners equal
train_df <- training(train_test_split)
test_df <- testing(train_test_split)


# 10-fold cross-validation
folds <- train_df %>% vfold_cv(v = 10)

mset <- metric_set(accuracy)
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning

## Define recipe ##
gb_recipe <- recipe(OFFER_STATUS ~ ., data = train_df) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_dummy(all_nominal_predictors())

## Define gb tree with hyperparameters to be tuned ##
gb_tree <- boost_tree(mode = "classification", # binary response
                      trees = tune(),
                      mtry = tune(),
                      tree_depth = tune(),
                      learn_rate = tune(),
                      loss_reduction = tune(),
                      min_n = tune())


## Define workflow for training ##
gb_wf <- 
  workflow() %>% 
  add_model(gb_tree) %>% 
  add_recipe(gb_recipe)

## Tuned hyperparameters of the tree found out in tuning.R ##
gb_tune <- gb_wf %>%
  tune_grid(folds,
            metrics = mset,
            control = control,
            grid = crossing(trees = c(200, 500, 1000, 10000),
                            mtry = c(5, 10, 15, 20, 23),
                            tree_depth = 5:14,
                            learn_rate = c(0.01, 0.015, 0.02, 0.025),
                            loss_reduction = c(0.01, 0.05, 0.1),
                            min_n = c(5, 7, 9, 11, 13, 15)))

gb_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))

# ROC Curve
test_predictions_full <- data.table(test_df %>% select(OFFER_STATUS))
test_predictions_full <- bind_cols(test_predictions_full, formula1 = test_predictions1)
test_predictions_melted <- melt(test_predictions_full, id.vars = "OFFER_STATUS", variable.name = "Model", value.name = "Prediction")
ggroc <- ggplot(test_predictions_melted, aes(d=as.numeric(OFFER_STATUS)-1, 
                                             m=Prediction,  
                                             color=Model)) +
  geom_roc() +
  geom_abline()
ggroc
