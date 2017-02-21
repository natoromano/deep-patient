# libraries
library(dplyr)
library(glmnet)
library(PRROC)
library(data.table)
library(randomForest)
library(methods)

targets <- c(9, 21, 22, 24, 27, 32, 35, 37, 40, 63, 69, 70, 73, 77)
# targets <- c(21, 24, 40, 77, 20)

n_groups <- 20
group_size <- 500
print(paste("Group size", group_size))

# Data --------------------------------------------------------------------

# Training groups
# Load training data
z.train <- fread(
  '/scratch/users/naromano/deep-patient/shah/x.train.encoded.small.txt',
  header=F
) %>% as.data.frame()
z.test <- fread(
  '/scratch/users/naromano/deep-patient/shah/x.test.encoded.small.txt',
  header=F
) %>% as.data.frame()

groups = list()
for (i in 1:n_groups) {
  groups[[i]] = sample(1:nrow(z.train), size=group_size, replace=F)
}

ytrain <- read.table(
  '/scratch/users/kjung/ehr-repr-learn/data/y.val.txt',
  header = T
)
ytest <- read.table(
  '/scratch/users/kjung/ehr-repr-learn/data/y.test.txt',
  header = T
)

ytrain <- ytrain[, -c(1, 2, 3)]
ytest <- ytest[, -c(1, 2, 3)]

ytrain[ytrain > 1] = 1
ytest[ytest > 1] = 1

# Model -------------------------------------------------------------------

alpha_ <- 1
print(paste("glmnet", alpha_))

test_on_group <- function(t, i) {
  
  Ytrain <- as.factor(ytrain[, t])[groups[[i]]]
  Yval <- ytest[, t]
  
  model <- cv.glmnet(
    as.matrix(z.train[groups[[i]], ]), Ytrain,
    nfolds = 5, family = "binomial", type = "auc", alpha = alpha_
  )

  pred <- predict(model, Xtest, type = "response")
  pred1 <- pred[Yval == 1]
  pred0 <- pred[Yval == 0]
  
  pr <- pr.curve(scores.class0 = pred1, scores.class1 = pred0)
  auprc <- pr$auc.integral
  
  roc <- roc.curve(scores.class0 = pred1, scores.class1 = pred0)
  auroc <- roc$auc
  
  # predictions <- prediction(pred[,1], Yval)
  # perf <- performance(predictions, measure = "auc")
  # auc <- perf@y.values[[1]]
  
  return(data.frame(
    "auprc" = auprc,
    "auroc" = auroc
  ))
}


test_target <- function(t) {
  
  print(paste("-----", t, "-----"))
  res <- plyr::ldply(1:n_groups, function(i) test_on_group(t, i))
  print(paste(t, "AUROC:", mean(res$auroc), "  AUPRC:", mean(res$auprc)))
  print("  ---  ")
  
  return(data.frame(
    "target" = t,
    "meanAUROC" = mean(res$auroc),
    "semAUROC" = sd(res$auroc)/sqrt(n_groups),
    "meanAUPRC" = mean(res$auprc),
    "semAUPRC" = sd(res$auprc)/sqrt(n_groups)
  ))
  
}


results <- plyr::ldply(targets, test_target)
write.csv(
  results,
  file = paste("/home/naromano/deep-patient/glmnet_", group_size, 
               ".csv", sep = "")
)
