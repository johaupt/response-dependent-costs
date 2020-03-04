library(data.table)
library(xgboost)

data <- fread("FashionB.csv")

#### Data Cleaning ####

# Drop purchases below 1 Euro and above 300 Euro
data <- data[converted == 0 | checkoutAmount>=1]
data <- data[checkoutAmount<=300] #quantile(data$checkoutAmount[data$checkoutAmount>0], prob=0.99)]

# Drop control group assignment
# Prior evaluation shows no ATE or ITE
# so we choose to keep both groups
data[, controlGroup := NULL]

# Fix the time of playing the coupon
data <- data[ViewCount ==5, ]

summary(data$checkoutAmount[data$checkoutAmount>0])
plot(density(data$checkoutAmount[data$checkoutAmount>0]))

dropVars <- c(
  # Campaign information
  "checkoutDiscount", "trackerKey", "epochSecond","campaignId", "campaignMov", "campaignValue","campaignUnit", "targetViewCount","ViewCount","label","dropOff", "confirmed", 
              "aborted", "campaignTags",
  # Invariant
  "ViewedBefore(cart)", "SecondsSinceTabSwitch", "TabSwitchPer(product)", "TimeToFirst(cart)", "SecondsSinceFirst(cart)", "TabSwitchOnLastScreenCount", "TotalTabSwitchCount",
  # Aggregates with missing
  "RecencyOfPreviousSessionInHrs", "FrequencyOfPreviousSessions"
  )

data[, c(dropVars):=NULL]

# Fix missing values
data[is.na(InitCartNonEmpty), InitCartNonEmpty:=0]
data[is.na(MonetaryDiscountValueOfPreviousSessions), MonetaryDiscountValueOfPreviousSessions:=0]
data[is.na(MonetaryValueOfPreviousSessions), MonetaryValueOfPreviousSessions:=0]

data[is.na(TriggerEventsSinceLastOnThisScreenType), TriggerEventsSinceLastOnThisScreenType:=0]       
data[is.na(TriggerEventsSinceLastOnThisPage), TriggerEventsSinceLastOnThisPage:=0]

data[is.na(DurationLastVisitInSeconds), DurationLastVisitInSeconds:=0]
data[is.na(ViewCountLastVisit), ViewCountLastVisit:=0]

## Drop all 'SecondsSince' and 'TimeTo' variables
data[, grep(colnames(data), pattern = "HoursSince", value = TRUE) := NULL]
data[, grep(colnames(data), pattern = "SecondsSince", value = TRUE) := NULL]
data[, grep(colnames(data), pattern = "TimeSince", value = TRUE) := NULL]
data[, grep(colnames(data), pattern = "TimeTo", value = TRUE) := NULL]

# Feature preparation
data[, MonetaryValueOfPreviousSessions := ifelse(is.na(MonetaryValueOfPreviousSessions/PreviousVisitCount), 0, MonetaryValueOfPreviousSessions/PreviousVisitCount) ]
data[, MonetaryDiscountValueOfPreviousSessions := ifelse(is.na(MonetaryDiscountValueOfPreviousSessions/PreviousVisitCount),0,MonetaryDiscountValueOfPreviousSessions/PreviousVisitCount)]

# Clean outliers
data <- data[NormalizedCartSum < quantile(NormalizedCartSum, p=0.9999)]
data <- data[MonetaryDiscountValueOfPreviousSessions < quantile(MonetaryDiscountValueOfPreviousSessions, p=0.9999)]
data <- data[MonetaryValueOfPreviousSessions < quantile(MonetaryValueOfPreviousSessions, p=0.999)]

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
  o[o<tau_min] <- tau_min
  o[o>tau_max] <- tau_max
  return(o)
}

tau_model <- function(X, hidden_layer, ATE){
  X <- as.matrix(X)
  no_var <- ncol(X)
  W1 <- matrix(rnorm(no_var*hidden_layer, 0, 1), nrow=no_var, ncol=hidden_layer)
  W3 <- matrix(rnorm(hidden_layer,        0, 1), nrow=hidden_layer, ncol=1)
  
  h <- dlogis(X %*% W1)
  
  o <- c(h %*% W3)
  o <- o - mean(o) + ATE
  o <- o / sd(o)
  return(o)
}

treatment_covariates = c("VisitorKnown", "HasConfirmedBefore", "WasConvertedBefore", 
                         "HasAbortedBefore",  "TotalClickCount",
                         "ChannelIsEMAIL", "ChannelIsPAID", "ChannelIsSEARCH", 
                         "InitCartNonEmpty", "CartQuantity", "NormalizedCartSum")
X_tau <- data[, ..treatment_covariates]
for(var in treatment_covariates){
  set(x=X_tau, j=var, value=(X_tau[[var]]-mean(X_tau[[var]]))/sd(X_tau[[var]]) )
}


# Train gradient boosted trees to predict the checkout amount for purchasers
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
                             objective="reg:linear", base_score=60,
                             seed=123456789))

# Train gradient boosted trees to predict conversion for shoppers
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
                             objective="binary:logistic", eval_metric="auc",
                             seed=123456789))


# Simulation
set.seed(123456789)
TAU_CONVERSION <- tau_model_linear(X_tau, ATE=0.05, tau_range = 0.1, tau_min=-0.1, tau_max=0.15)
#TAU_CONVERSION <- tau_model(X_tau, hidden_layer=3*ncol(X_tau), ATE=0.05)
plot(density(TAU_CONVERSION))
quantile(TAU_CONVERSION, probs=seq(0,1,0.1))

TAU_BASKET <- tau_model_linear(X_tau, ATE=1, tau_range = 10, tau_min=-10, tau_max=10)
#TAU_BASKET <- tau_model(X_tau, hidden_layer=3*ncol(X_tau), ATE=1) * mean(data$checkoutAmount[data$converted==1])
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
# Check approximate optimal targeted ratio for comparison to models
MARGIN <- 0.3
mean( (data$TREATMENT_EFFECT_RESPONSE*MARGIN) > ((conv_pred+TAU_CONVERSION)*10) )

fwrite(data, "~/Downloads/fashionB_clean.csv")
