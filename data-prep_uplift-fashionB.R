library(data.table)
library(xgboost)
library(xtable)

set.seed(123456789)

data <- fread("/Users/hauptjoh/projects/research/dynamic_cost_targeting/data/FashionB.csv")
#data <- data[campaignValue==500 & campaignMov==5000,]

#### Data Cleaning ####
# The data is part of a randomized trial that showed 0 average treatment effect and no treatment heterogeneity
# We retain both groups to maximize the available data and drop the control group assignment 
data[, controlGroup := NULL]

# The data logs the information of a user at the time of the treatment decision
# In the available data, the page view at which the treatment decision takes place is randomized to allow
# randomization of the optimal timing of the treatment 
# We abstract from the timing decision and assume that the treatment is played after a fixed number of page views
# In application and for similar campaigns, treatment is typically enabled after the 5th page view when some customer
# data is available for the current session
# Fix the time of playing the coupon to the fifth view
data <- data[ViewCount ==5, ]

# Total number of observations at 5th page view
nrow(data)
mean(data$converted)
# Table B.7: Quantiles over conversion values
median(data[data$converted==1]$checkoutAmount)
xtable(transpose(data.frame(round(quantile(data$checkoutAmount[data$checkoutAmount>0], c(0,0.01,0.05,0.25,0.5,0.75,0.9,0.95,0.99,1)),2))))

summary(data$checkoutAmount[data$checkoutAmount>0])
plot(density(data$checkoutAmount[data$checkoutAmount>0]))

dropVars <- c(
  # Internal Information on the Campaign Run during Data Collection
  "checkoutDiscount", "trackerKey", "epochSecond","campaignId", "campaignMov", "campaignValue","campaignUnit", "targetViewCount","ViewCount","label","dropOff", "confirmed", 
              "aborted", "campaignTags",
  # Invariant Variables
  "ViewedBefore(cart)", "SecondsSinceTabSwitch", "TabSwitchPer(product)", "TimeToFirst(cart)", "SecondsSinceFirst(cart)", "TabSwitchOnLastScreenCount", "TotalTabSwitchCount",
  # Aggregates with Majority of Values Missing
  "RecencyOfPreviousSessionInHrs", "FrequencyOfPreviousSessions"
  )

data[, c(dropVars):=NULL]

# Fix missing values for count variables and other variables where NA is equivalent to 0
data[is.na(InitCartNonEmpty), InitCartNonEmpty:=0]
data[is.na(MonetaryDiscountValueOfPreviousSessions), MonetaryDiscountValueOfPreviousSessions:=0]
data[is.na(MonetaryValueOfPreviousSessions), MonetaryValueOfPreviousSessions:=0]

data[is.na(TriggerEventsSinceLastOnThisScreenType), TriggerEventsSinceLastOnThisScreenType:=0]       
data[is.na(TriggerEventsSinceLastOnThisPage), TriggerEventsSinceLastOnThisPage:=0]

# Impute missing values for variables w.r.t to last visit for unseen users
data[is.na(DurationLastVisitInSeconds), DurationLastVisitInSeconds:= mean(data$DurationLastVisitInSeconds, na.rm = TRUE)]
data[is.na(ViewCountLastVisit), ViewCountLastVisit:=mean(data$ViewCountLastVisit, na.rm = TRUE)]

## Drop 'SecondsSince' and 'TimeTo' variables b/c exessive missing values
data[, grep(colnames(data), pattern = "HoursSince", value = TRUE) := NULL]
data[, grep(colnames(data), pattern = "SecondsSince", value = TRUE) := NULL]
data[, grep(colnames(data), pattern = "TimeSince", value = TRUE) := NULL]
data[, grep(colnames(data), pattern = "TimeTo", value = TRUE) := NULL]

# Feature preparation 
# Calculate average over historical value (M of RFM)
data[, MonetaryValueOfPreviousSessions := ifelse(is.na(MonetaryValueOfPreviousSessions/PreviousVisitCount), 0, MonetaryValueOfPreviousSessions/PreviousVisitCount) ]
data[, MonetaryDiscountValueOfPreviousSessions := ifelse(is.na(MonetaryDiscountValueOfPreviousSessions/PreviousVisitCount),0,MonetaryDiscountValueOfPreviousSessions/PreviousVisitCount)]
data[, MonetaryValueOfPreviousSessions := log(MonetaryValueOfPreviousSessions+1)]
data[, MonetaryDiscountValueOfPreviousSessions := log(MonetaryDiscountValueOfPreviousSessions+1)]

colnames(data) <- gsub("\\(|\\)", "", colnames(data))

# One-hot encode time variables
data_time <- data[,c("HourOfDay", "DayOfWeek")]
data_time[, HourOfDay := cut(HourOfDay, breaks = 8, labels=paste0("3hBlock", 1:8))]
data_time[, DayOfWeek := factor(DayOfWeek)]

data <- cbind(data, model.matrix(~.-1 , data_time[,c("HourOfDay", "DayOfWeek")]))
data[, c("HourOfDay", "DayOfWeek") := NULL]


#### Treatment Effect Simulation ####
tau_model_linear <- function(X, ATE, tau_range=0.2, tau_min=-Inf, tau_max=+Inf){
  X <- as.matrix(X)
  no_var <- ncol(X)
  W3 <- matrix(rnorm(no_var, 0, 1), nrow=no_var, ncol=1)
  
  o <- c(X %*% W3)
  o <- (o - mean(o))* (tau_range) / (quantile(o, 0.95) - quantile(o, 0.05))
  o <- o + ATE
  #o <- o / sd(o)
  o[o<tau_min] <- tau_min
  o[o>tau_max] <- tau_max
  return(o)
}

tau_model <- function(X, hidden_layer, ATE, tau_range=0.2, tau_min=-Inf, tau_max=+Inf){
  X <- as.matrix(X)
  no_var <- ncol(X)
  W1 <- matrix(rnorm(no_var*hidden_layer, 0, 1), nrow=no_var, ncol=hidden_layer)
  #W2 <- matrix(rnorm(hidden_layer*hidden_layer, 0, 1), nrow=hidden_layer, ncol=hidden_layer)
  W3 <- matrix(rnorm(hidden_layer,        0, 1), nrow=hidden_layer, ncol=1)
  
  h <- dlogis(X %*% W1)
  
  o <- c(h %*% W3)
  o <- (o - mean(o))* (tau_range) / (quantile(o, 0.95) - quantile(o, 0.05))
  o <- o + ATE
  #o <- o / sd(o)
  o[o<tau_min] <- tau_min
  o[o>tau_max] <- tau_max
  return(o)
}


# Selection of variables deemed relevated for treatment effect based on business intuition
treatment_covariates = c("VisitorKnown", "WasConvertedBefore", "HasAbortedBefore", 
                         "TotalClickCount", "MonetaryValueOfPreviousSessions",
                         "ChannelIsEMAIL", "ChannelIsPAID", "ChannelIsSEARCH", 
                         "InitCartNonEmpty",  
                         "CartQuantity", "NormalizedCartSum")
X_tau <- data[, ..treatment_covariates]
for(var in treatment_covariates){
  set(x=X_tau, j=var, value=(X_tau[[var]]-mean(X_tau[[var]]))/sd(X_tau[[var]]) )
}


# Train gradient boosted trees to predict the checkout amount for simulated purchasers
library(xgboost)

data_checkout <- data[data$converted==1,]
tr_idx <- round(nrow(data_checkout)*0.8)

data_checkout_tr <- as.matrix(data_checkout[1:tr_idx,])
data_checkout_val <- as.matrix(data_checkout[(tr_idx+1):nrow(data_checkout),])

data_checkout_tr <- xgb.DMatrix(data_checkout_tr[, 3:(ncol(data))], label=data_checkout_tr[, "checkoutAmount"])
data_checkout_val <- xgb.DMatrix(data_checkout_val[, 3:(ncol(data))], label=data_checkout_val[, "checkoutAmount"])
data_xgb <- xgb.DMatrix(as.matrix(data)[, 3:(ncol(data))])

xgb_checkout <- xgb.train(data=data_checkout_tr, watchlist = list(tr = data_checkout_tr, val=data_checkout_val),
                 nrounds=10000, early_stopping_rounds=100, verbose=1,
                 params=list(eta=0.05, max_depth=4, min_child_weight=2,
                             subsample=0.8, colsample_by_tree=0.8,
                             objective="reg:linear", base_score=60))


# Train gradient boosted trees to predict conversion probability for all shoppers
data_converted <- data
tr_idx <- round(nrow(data_converted)*0.8)

data_converted_tr <- as.matrix(data_converted[1:tr_idx,])
data_converted_val <- as.matrix(data_converted[(tr_idx+1):nrow(data_converted),])

data_converted_tr <- xgb.DMatrix(data_converted_tr[, 3:(ncol(data))], label=data_converted_tr[, "converted"])
data_converted_val <- xgb.DMatrix(data_converted_val[, 3:(ncol(data))], label=data_converted_val[, "converted"])

xgb_conversion <- xgb.train(data=data_converted_tr, watchlist = list(tr = data_converted_tr, val=data_converted_val),
                 nrounds=10000, early_stopping_rounds=100, verbose=1,
                 params=list(eta=0.05, max_depth=4, min_child_weight=2,
                             subsample=0.8, colsample_by_tree=0.8,
                             objective="binary:logistic", eval_metric="auc"))



# Simulation
#TAU_CONVERSION <- tau_model_linear(X_tau, ATE=0.05, tau_range = 0.1, tau_min=-0.1, tau_max=0.15)
TAU_CONVERSION <- tau_model(X_tau, hidden_layer=3*ncol(X_tau), ATE=0.05, tau_range = 0.1, tau_min=-0.1, tau_max=0.15)
plot(density(TAU_CONVERSION))
quantile(TAU_CONVERSION, probs=seq(0,1,0.1))

TAU_BASKET <- tau_model_linear(X_tau, ATE=1, tau_range = 10, tau_min=-10, tau_max=10)
#TAU_BASKET <- tau_model(X_tau, hidden_layer=3*ncol(X_tau),  ATE=1, tau_range = 10, tau_min=-10, tau_max=10)
plot(density(TAU_BASKET))
quantile(TAU_BASKET, probs=seq(0,1,0.1))

# Assign treatment
data$TREATMENT <- rbinom(nrow(data), 1, prob=0.5)


# Conversion assignment
conv_pred <- predict(xgb_conversion, data_xgb)
C <- matrix(0, ncol=2, nrow=nrow(data))
for(i in 1:nrow(C)){
    C[i,2] <- rbinom(n = 1, size = 1, prob= pmax( 0, pmin( conv_pred[i] + TAU_CONVERSION[i], 1)))
    C[i,1] <- rbinom(n = 1, size = 1, prob= conv_pred[i])
  }
data$converted <- ifelse(data$TREATMENT==1, C[,2], C[,1])

# Checkout assignment
# Impute checkout amount for switched purchasers
basket_pred <- predict(xgb_checkout, data_xgb)
potential_basket <- ifelse(data$checkoutAmount==0, basket_pred, data$checkoutAmount)
Y <- cbind(Y0=potential_basket, Y1=potential_basket + TAU_BASKET) #* 100

# Set checkout amount for switched non-purchasers to 0
data[converted==0, "checkoutAmount"] <- 0

# Set checkout amount for switched purchasers to correct amount
data[converted==1 & TREATMENT==1, "checkoutAmount"] <- Y[,2][data$converted==1 & data$TREATMENT==1]
data[converted==1 & TREATMENT==0, "checkoutAmount"] <- Y[,1][data$converted==1 & data$TREATMENT==0]


# Calculate the overall treatment effect on the profit
data$TREATMENT_EFFECT_CONVERSION <- TAU_CONVERSION
data$TREATMENT_EFFECT_BASKET     <- Y[,2] - Y[,1]
potential_response <- cbind("0" = conv_pred, "1" = conv_pred + TAU_CONVERSION) * Y
data$TREATMENT_EFFECT_RESPONSE <- potential_response[,2] - potential_response[,1]

plot(density(data$TREATMENT_EFFECT_RESPONSE))
quantile(data$TREATMENT_EFFECT_RESPONSE, probs=seq(0,1,0.1))
# Approximate optimal targeted ratio
MARGIN <- 0.3
mean( (data$TREATMENT_EFFECT_RESPONSE*MARGIN) > ((conv_pred+TAU_CONVERSION)*10) )

fwrite(data, "~/Downloads/fashionB_clean_nonlinear.csv")
