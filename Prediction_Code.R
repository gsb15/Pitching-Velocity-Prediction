####Pitch Velocity Prediction 
library(rms)
library(reshape2)
library(gbm) #GBM
library(dplyr)
library(ggplot2)
library(kernlab)      # SVM methodology
library(e1071)        # SVM methodology        
library(RColorBrewer) # customized coloring of plots
library(randomForest) #General Random Forest
library(ranger) #Optimal Random Forest 
library(pmsampsize) #a priori sample size calculation
library(caret)
library(sjPlot)
library(readxl)
library(glmnet)
library(factoextra) #Viz PCA 
#princomp : base R function for PCA

#Data Frame 
fb_biomech <- read.csv("Insert your file path here.csv")

#Sample Size 
pmsampsize(type = "c",  rsquared = 0.89, parameters = 30, intercept = 36.1, sd = 3.41)
#264

#Complete Case (missing data are very small)
fb_biomech_avg_2$complete <- complete.cases(fb_biomech_avg_2)

fb_ml2 <- fb_biomech_avg_2 %>% 
  filter(complete == "TRUE")

fb_ml2 <- fb_ml2 %>% 
  select(-complete)


#Taking out Level for Primary Analyses
fb_ml2 <- fb_ml2 %>% 
          select(-Level)

#################################
#Regression Model
#################################

statcv <- train(BALL_RELEASE_SPEED ~  Back_Leg_GRF_mag_max + Hip_Shoulders_Sep_Footstrike + Lead_Leg_GRF_mag_max + rcs(Pitching_Elbow_Ang_Vel_max, 3) +
                  Pitching_Elbow_Angle_Max_Shoulder_Rot + rcs(Pitching_Elbow_Angle_Footstrike,4) +  Pitching_Shoulder_Angle_Footstrikeabd + 
                  Pitching_Shoulder_MER + rcs(Pitching_Shoulder_Angle_ReleaseAbduction,3) +  Trunk_Angle_Release_ForwardFlex + 
                  Trunk_Angle_Release_LateralTilt +  rcs(Pelvis_Ang_Vel_max, 3) + Pitching_Humerus_Ang_Vel_max + Thorax_Ang_Vel_max + 
                  Time_Diff_pelvis_trunk + rcs(STRIDE_LENGTH_MEAN_PERCENT, 3),
                data = fb_ml2, method = "glm", family = "gaussian", 
                trControl = trainControl("repeatedcv", number = 10)
)
coef(statcv$finalModel)
confint(statcv$finalModel)
statcv$results



statcv_predict <- predict.train(statcv, newdata = fb_ml2, type = "raw")

mean(statcv_predict) 
sd(statcv_predict) 
min(statcv_predict) 
max(statcv_predict) 


actual_v_predict_combined <- cbind(fb_ml2, statcv_predict)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ statcv_predict,family="gaussian",x=TRUE,y=TRUE, data = actual_v_predict_combined)
mod_combined_2
confint(mod_combined_2) 


#Calibration Plot 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = actual_v_predict_combined, aes(x = statcv_predict, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  xlim(25, 45) + 
  ylim(25, 45) + 
  theme_minimal()
dev.off()


#Ten Fold CV for Performance Metrics 

nfolds=10

results <- matrix(nrow = nfolds,ncol = 3)

folds <- caret::createFolds(fb_ml2$BALL_RELEASE_SPEED, k=10)



for(i in 1:nfolds){
  
  cv_samp <- fb_ml2[-folds[[i]],]
  
  cv_test <- fb_ml2[folds[[i]],]
  
  
  model <- glm(BALL_RELEASE_SPEED ~  Back_Leg_GRF_mag_max + Hip_Shoulders_Sep_Footstrike + Lead_Leg_GRF_mag_max + rcs(Pitching_Elbow_Ang_Vel_max, 3) +
                 Pitching_Elbow_Angle_Max_Shoulder_Rot + rcs(Pitching_Elbow_Angle_Footstrike,4) +  Pitching_Shoulder_Angle_Footstrikeabd + 
                 Pitching_Shoulder_MER + rcs(Pitching_Shoulder_Angle_ReleaseAbduction,3) +  Trunk_Angle_Release_ForwardFlex + 
                 Trunk_Angle_Release_LateralTilt +  rcs(Pelvis_Ang_Vel_max, 3) + Pitching_Humerus_Ang_Vel_max + Thorax_Ang_Vel_max + 
                 Time_Diff_pelvis_trunk + rcs(STRIDE_LENGTH_MEAN_PERCENT, 3),
               data = cv_samp, family = "gaussian")
  
  pr_cv <- predict(model,type="response", newdata = cv_samp)
  
  pr_cv2 <- cbind(cv_samp, pr_cv)
  
  test_predict <- predict(model,type="response", newdata = cv_test)
  
  test_predict2 <- cbind(cv_test, test_predict)
  
  app_rsquare_model <-   caret::R2(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,1] <- app_rsquare_model
  
  app_cslope_model <- glm(BALL_RELEASE_SPEED ~ pr_cv, family= "gaussian", data= pr_cv2)
  
  results[i,2] <- summary(app_cslope_model)$coefficients[2,1]
  
  
  
  RMSE <- caret::RMSE(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,3] <- RMSE
  
}

results2 <- as.data.frame(results)

colnames(results2) <- c("r-square","c_slope", "RMSE")

mean(results2$`r-square`) 
mean(results2$c_slope) 
mean(results2$RMSE) 




###############################
#Gradient Boosting Machine
###############################

# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)
#81

#Finding Optimal Tuning Parameters for GBM for Pitch Velow 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(17)
  
  # train model
  gbm.tune <- gbm(
    formula = BALL_RELEASE_SPEED ~ .,
    distribution = "gaussian",
    data = fb_ml2,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = 0.75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  arrange(min_RMSE) #0.3 shrinkage the best top 3/4, interaction 3 3/4, min 5 for 3/4
#bag 0.8/0.65 alternating, optimal trees ~all over the place, Min RMSE at 2.57

#More Precise Grid 
hyper_grid <- expand.grid(
  shrinkage = c(.1, .15, .2, .25, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15, 20), #Range the same but adding one more min number
  bag.fraction = c(0.65, 0.80), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)


for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(17)
  
  # train model
  gbm.tune <- gbm(
    formula = BALL_RELEASE_SPEED ~ .,
    distribution = "gaussian",
    data = fb_ml2,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = 0.75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}


hyper_grid %>% 
  arrange(min_RMSE)  
#Best Model 
#shrinkage interaction.depth n.minobsinnode bag.fraction optimal_trees    min_RMSE
# 0.30                 3             5         0.80            72      2.57

#Training Actual GBM Model 
gbm.final <- gbm(
  formula = BALL_RELEASE_SPEED ~ .,
  distribution = "gaussian",
  data = fb_ml2,
  n.trees = 2500,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 5,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm.final) 



#Visualizing Influence of Difference VARS 
summary(
  gbm.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

pred_gbm <- predict(gbm.final, n.trees = gbm.final$n.trees, fb_ml2)

summary(pred_gbm)

sd(pred_gbm) 

caret::RMSE(pred_gbm, fb_ml2$BALL_RELEASE_SPEED)

caret::R2(pred_gbm, fb_ml2$BALL_RELEASE_SPEED)

gbmactual_v_predict_combined <- cbind(fb_ml2, pred_gbm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_gbm,family="gaussian",x=TRUE,y=TRUE, data = gbmactual_v_predict_combined)
mod_combined_2 
confint(mod_combined_2) 


#Calibration Plot 
png("INsert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = gbmactual_v_predict_combined, aes(x = pred_gbm, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  theme_minimal() 
dev.off()


#########################################
#Support Vector Machine 
#########################################

tune.out <- tune(svm, BALL_RELEASE_SPEED ~., data = fb_ml2, kernel = "radial",
                 ranges = list(cost = c(0.1,1,10,100,1000),
                               gamma = c(0.5,1,2,3,4)), 
                 tunecontrol=tune.control(cross=10))
# show best model
tune.out$best.model
#Kernel radial
#Cost: 10, 
#Gamma: 0.5, 
#epsilon: 0.1

svmfit <- svm(BALL_RELEASE_SPEED ~ ., data = fb_ml2, kernel = "radial", cost = 10, gamma = 0.5, epsilon = 0.1)
# plot results
plot(svmfit)
summary(svmfit)


#Getting Final RMSE 

pred_svm <- predict(svmfit, fb_ml2, type = "raw")
# Summarizing the distribution of linear predictor 
mean(pred_svm) 
sd(pred_svm) 
min(pred_svm) 
max(pred_svm) 

caret::RMSE(pred_svm, fb_ml2$BALL_RELEASE_SPEED)


nrow(youth_ht_ml)
nrow(pred_svm)

svmactual_v_predict_combined <- cbind(fb_ml2, pred_svm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_svm,family="gaussian",x=TRUE,y=TRUE, data = svmactual_v_predict_combined)
mod_combined_2 #1.09
confint(mod_combined_2) 

png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = svmactual_v_predict_combined, aes(x = pred_svm, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  theme_minimal() 
dev.off()


############################################################################
#Random Forest 
############################################################################

#Finding a good first mtry to know where to put the hyper grid parameters
# names of features
features <- setdiff(names(fb_ml2), "BALL_RELEASE_SPEED")

set.seed(17)

m2 <- tuneRF(
  x          = fb_ml2[features],
  y          = fb_ml2$BALL_RELEASE_SPEED,
  ntreeTry   = 500,
  mtryStart  = 3,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)

#Lowest error is at 9 mtry 

#Creating Hyper Grid for tuning parameters 

hyper_grid2 <- expand.grid(
  mtry       = seq(1, 12, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

hyper_grid2 <- expand.grid(
  mtry       = seq(5, 11, by = 1),
  node_size  = seq(3, 7, by = 1),
  sampe_size = c(.55, .70),
  OOB_RMSE   = 0
)

#Did both tuning iterations here

# total number of combinations
nrow(hyper_grid2)
## [1] 224
nrow(fb_ml_elbow)

#
for(i in 1:nrow(hyper_grid2)) {
  
  # train model
  model <- ranger(
    formula         = BALL_RELEASE_SPEED ~ ., 
    data            = fb_ml2, 
    num.trees       = 500,
    mtry            = hyper_grid2$mtry[i],
    min.node.size   = hyper_grid2$node_size[i],
    sample.fraction = hyper_grid2$sampe_size[i],
    seed            = 17
  )
  
  # add OOB error to grid
  hyper_grid2$OOB_RMSE[i] <- sqrt(model$prediction.error)
}



hyper_grid2 %>% 
  arrange(OOB_RMSE) #Mtry 5, node size 5, samp size 0.550, RMSE = 2.89

hyper_grid2 %>% 
  arrange(OOB_RMSE)#Mtry: 5, Node size 5, Sam size 0.55, RMSE: 2.89

#Getting Deeper into tuning 
OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = BALL_RELEASE_SPEED~ ., 
    data            = fb_ml2, 
    num.trees       = 500,
    mtry            = 5,
    min.node.size   = 5,
    sample.fraction = .55,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20) #Peak of Hist RMSE at ~2.91


#Looking at Importance of Different VARs in Random Forest 

optimal_ranger <- ranger(
  formula         =  BALL_RELEASE_SPEED~ ., 
  data            = fb_ml2, 
  num.trees       = 500,
  mtry            = 5,
  min.node.size   = 5,
  sample.fraction = .55)


summary(optimal_ranger)


#Getting Final RMSE 

pred_forest <- predict(optimal_ranger, fb_ml2)
pred_forest2 <- pred_forest$predictions #Have to add the extra step for ranger....

# Summarizing the distribution of linear predictor 
mean(pred_forest2) 
sd(pred_forest2) 
min(pred_forest2) 
max(pred_forest2) 

caret::RMSE(pred_forest$predictions, fb_ml2$BALL_RELEASE_SPEED)


nrow(youth_ht_ml)
nrow(pred_svm)

forestactual_v_predict_combined <- cbind(fb_ml2, pred_forest2)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_forest2,family="gaussian",x=TRUE,y=TRUE, data = forestactual_v_predict_combined)
mod_combined_2 #1.49
confint(mod_combined_2) 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = forestactual_v_predict_combined, aes(x = pred_forest2, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  xlim(25, 40) + 
  ylim(25, 40)+
  theme_minimal() 
dev.off()


###############################################################
#Sensitivity Analyses 
###############################################################
fb_sens <- fb_ml2 %>% 
  select(-Pitching_Humerus_Ang_Vel_max, -Pitching_Elbow_Ang_Vel_max)

#Training Actual GBM Model 
gbm.final <- gbm(
  formula = BALL_RELEASE_SPEED ~ .,
  distribution = "gaussian",
  data = fb_sens,
  n.trees = 2500,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 5,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm.final) 



#Visualizing Influence of Difference VARS 
summary(
  gbm.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

pred_gbm <- predict(gbm.final, n.trees = gbm.final$n.trees, fb_sens)

summary(pred_gbm)

sd(pred_gbm) 

caret::RMSE(pred_gbm, fb_sens$BALL_RELEASE_SPEED)



gbmactual_v_predict_combined <- cbind(fb_sens, pred_gbm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_gbm,family="gaussian",x=TRUE,y=TRUE, data = gbmactual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2) 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = gbmactual_v_predict_combined, aes(x = pred_gbm, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  theme_minimal() 
dev.off()



statcv_s <- train(BALL_RELEASE_SPEED ~  Back_Leg_GRF_mag_max + Hip_Shoulders_Sep_Footstrike + Lead_Leg_GRF_mag_max + rcs(Pitching_Elbow_Ang_Vel_max, 3) +
                    Pitching_Elbow_Angle_Max_Shoulder_Rot + rcs(Pitching_Elbow_Angle_Footstrike,4) +  Pitching_Shoulder_Angle_Footstrikeabd + 
                    Pitching_Shoulder_MER + rcs(Pitching_Shoulder_Angle_ReleaseAbduction,3) +  Trunk_Angle_Release_ForwardFlex + 
                    Trunk_Angle_Release_LateralTilt +  rcs(Pelvis_Ang_Vel_max, 3) + Pitching_Humerus_Ang_Vel_max + Thorax_Ang_Vel_max + 
                    Time_Diff_pelvis_trunk + rcs(STRIDE_LENGTH_MEAN_PERCENT, 3),
                  data = fb_ml2, method = "glm", family = "gaussian", 
                  trControl = trainControl("repeatedcv", number = 10)
)
summary(statcv_s$finalModel)
confint(statcv_s$finalModel)
statcv_s$results


statcv_predict <- predict.train(statcv_s, newdata = fb_ml2, type = "raw")

mean(statcv_predict) 
sd(statcv_predict) 
min(statcv_predict) 
max(statcv_predict) 


actual_v_predict_combined <- cbind(fb_ml2, statcv_predict)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ statcv_predict,family="gaussian",x=TRUE,y=TRUE, data = actual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2) 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = actual_v_predict_combined, aes(x = statcv_predict, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  theme_minimal()
dev.off()


#####################################
#HS
#####################################

fb_models2 <- fb_biomech_avg_2 %>% 
  select(BALL_RELEASE_SPEED, Back_Leg_GRF_mag_max, Hip_Shoulders_Sep_Footstrike, Lead_Leg_GRF_mag_max, Pitching_Elbow_Ang_Vel_max, 
         Pitching_Elbow_Angle_Max_Shoulder_Rot, Pitching_Elbow_Angle_Footstrike, Pitching_Shoulder_Angle_Footstrikeabd, 
         Pitching_Shoulder_MER, Pitching_Shoulder_Angle_ReleaseAbduction, Trunk_Angle_Release_ForwardFlex,
         Trunk_Angle_Release_LateralTilt, Pelvis_Ang_Vel_max, Pitching_Humerus_Ang_Vel_max, Thorax_Ang_Vel_max, 
         Time_Diff_pelvis_trunk, STRIDE_LENGTH_MEAN_PERCENT, Level)




fb_hs <- fb_models2 
fb_hs$complete <- complete.cases(fb_hs)

fb_hs2 <- fb_hs %>% 
  filter(complete == "TRUE")
fb_hs2 <- fb_hs2 %>% 
  select(-complete)

fb_HS <- fb_hs2 %>% 
  filter(Level == "HS")

fb_HS2 <- fb_HS %>% 
  select(-Level)

#GBM 

gbm.final <- gbm(
  formula = BALL_RELEASE_SPEED ~ .,
  distribution = "gaussian",
  data = fb_HS2,
  n.trees = 2500,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 5,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm.final) 



#Visualizing Influence of Difference VARS 
summary(
  gbm.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

pred_gbm <- predict(gbm.final, n.trees = gbm.final$n.trees, fb_HS2)

summary(pred_gbm)

sd(pred_gbm) 

caret::RMSE(pred_gbm, fb_HS2$BALL_RELEASE_SPEED)



gbmactual_v_predict_combined <- cbind(fb_HS2, pred_gbm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_gbm,family="gaussian",x=TRUE,y=TRUE, data = gbmactual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2) 



statcv <- train(BALL_RELEASE_SPEED ~  Back_Leg_GRF_mag_max + Hip_Shoulders_Sep_Footstrike + Lead_Leg_GRF_mag_max + rcs(Pitching_Elbow_Ang_Vel_max, 3) +
                  Pitching_Elbow_Angle_Max_Shoulder_Rot + rcs(Pitching_Elbow_Angle_Footstrike,4) +  Pitching_Shoulder_Angle_Footstrikeabd + 
                  Pitching_Shoulder_MER + rcs(Pitching_Shoulder_Angle_ReleaseAbduction,3) +  Trunk_Angle_Release_ForwardFlex + 
                  Trunk_Angle_Release_LateralTilt +  rcs(Pelvis_Ang_Vel_max, 3) + Pitching_Humerus_Ang_Vel_max + Thorax_Ang_Vel_max + 
                  Time_Diff_pelvis_trunk + rcs(STRIDE_LENGTH_MEAN_PERCENT, 3),
                data = fb_HS2, method = "glm", family = "gaussian", 
                trControl = trainControl("repeatedcv", number = 10)
)
summary(statcv$finalModel)
confint(statcv$finalModel)
statcv$results


statcv_predict <- predict.train(statcv, newdata = fb_HS2, type = "raw")

mean(statcv_predict) 
sd(statcv_predict) 
min(statcv_predict) 
max(statcv_predict) 


actual_v_predict_combined <- cbind(fb_HS2, statcv_predict)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ statcv_predict,family="gaussian",x=TRUE,y=TRUE, data = actual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2) #C Slope: 1.00 (95%CI: 0.84, 1.16)




#####################################
#Center and Scale Sensitivity Analyses 
######################################

preProcValues <- caret::preProcess(fb_ml2, method = c("center", "scale")) #Pre-processing 

fb_ml_center_scale <- predict(preProcValues, fb_ml2) #Integrating pre-processing to data 
View(fb_ml_center_scale) #Checking 


statcv <- train(BALL_RELEASE_SPEED ~  Back_Leg_GRF_mag_max + Hip_Shoulders_Sep_Footstrike + Lead_Leg_GRF_mag_max + rcs(Pitching_Elbow_Ang_Vel_max, 3) +
                  Pitching_Elbow_Angle_Max_Shoulder_Rot + rcs(Pitching_Elbow_Angle_Footstrike,4) +  Pitching_Shoulder_Angle_Footstrikeabd + 
                  Pitching_Shoulder_MER + rcs(Pitching_Shoulder_Angle_ReleaseAbduction,3) +  Trunk_Angle_Release_ForwardFlex + 
                  Trunk_Angle_Release_LateralTilt +  rcs(Pelvis_Ang_Vel_max, 3) + Pitching_Humerus_Ang_Vel_max + Thorax_Ang_Vel_max + 
                  Time_Diff_pelvis_trunk + rcs(STRIDE_LENGTH_MEAN_PERCENT, 3),
                data = fb_ml_center_scale, method = "glm", family = "gaussian", 
                trControl = trainControl("repeatedcv", number = 10)
)
coef(statcv$finalModel)
confint(statcv$finalModel)
statcv$results



statcv_predict <- predict.train(statcv, newdata = fb_ml_center_scale, type = "raw")

mean(statcv_predict) 
sd(statcv_predict) 
min(statcv_predict) 
max(statcv_predict) 


actual_v_predict_combined <- cbind(fb_ml_center_scale, statcv_predict)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ statcv_predict,family="gaussian",x=TRUE,y=TRUE, data = actual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2)


#Calibration Plot 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = actual_v_predict_combined, aes(x = statcv_predict, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  xlim(25, 45) + 
  ylim(25, 45) + 
  theme_minimal()
dev.off()


#Ten Fold CV for Performance Metrics 

nfolds=10

results <- matrix(nrow = nfolds,ncol = 3)

folds <- caret::createFolds(fb_ml_center_scale$BALL_RELEASE_SPEED, k=10)



for(i in 1:nfolds){
  
  cv_samp <- fb_ml_center_scale[-folds[[i]],]
  
  cv_test <- fb_ml_center_scale[folds[[i]],]
  
  
  model <- glm(BALL_RELEASE_SPEED ~  Back_Leg_GRF_mag_max + Hip_Shoulders_Sep_Footstrike + Lead_Leg_GRF_mag_max + rcs(Pitching_Elbow_Ang_Vel_max, 3) +
                 Pitching_Elbow_Angle_Max_Shoulder_Rot + rcs(Pitching_Elbow_Angle_Footstrike,4) +  Pitching_Shoulder_Angle_Footstrikeabd + 
                 Pitching_Shoulder_MER + rcs(Pitching_Shoulder_Angle_ReleaseAbduction,3) +  Trunk_Angle_Release_ForwardFlex + 
                 Trunk_Angle_Release_LateralTilt +  rcs(Pelvis_Ang_Vel_max, 3) + Pitching_Humerus_Ang_Vel_max + Thorax_Ang_Vel_max + 
                 Time_Diff_pelvis_trunk + rcs(STRIDE_LENGTH_MEAN_PERCENT, 3),
               data = cv_samp, family = "gaussian")
  
  pr_cv <- predict(model,type="response", newdata = cv_samp)
  
  pr_cv2 <- cbind(cv_samp, pr_cv)
  
  test_predict <- predict(model,type="response", newdata = cv_test)
  
  test_predict2 <- cbind(cv_test, test_predict)
  
  app_rsquare_model <-   caret::R2(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,1] <- app_rsquare_model
  
  app_cslope_model <- glm(BALL_RELEASE_SPEED ~ pr_cv, family= "gaussian", data= pr_cv2)
  
  results[i,2] <- summary(app_cslope_model)$coefficients[2,1]
  
  
  
  RMSE <- caret::RMSE(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,3] <- RMSE
  
}

results2 <- as.data.frame(results)

colnames(results2) <- c("r-square","c_slope", "RMSE")

mean(results2$`r-square`) #0.46
mean(results2$c_slope) #1.01
mean(results2$RMSE) #0.74


#######
#GBM 
#######

#Training Actual GBM Model 
gbm.final <- gbm(
  formula = BALL_RELEASE_SPEED ~ .,
  distribution = "gaussian",
  data = fb_ml_center_scale,
  n.trees = 2500,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 5,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm.final) 

pred_gbm <- predict(gbm.final, n.trees = gbm.final$n.trees, fb_ml_center_scale)

summary(pred_gbm)

sd(pred_gbm) 

caret::RMSE(pred_gbm, fb_ml_center_scale$BALL_RELEASE_SPEED)

caret::R2(pred_gbm, fb_ml2$BALL_RELEASE_SPEED)

gbmactual_v_predict_combined <- cbind(fb_ml_center_scale, pred_gbm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_gbm,family="gaussian",x=TRUE,y=TRUE, data = gbmactual_v_predict_combined)
mod_combined_2 
confint(mod_combined_2) 


#######################
#Random Forest 
#######################


optimal_ranger <- ranger(
  formula         =  BALL_RELEASE_SPEED~ ., 
  data            = fb_ml_center_scale, 
  num.trees       = 500,
  mtry            = 5,
  min.node.size   = 5,
  sample.fraction = .55)


summary(optimal_ranger)


#Getting Final RMSE 

pred_forest <- predict(optimal_ranger, fb_ml_center_scale)
pred_forest2 <- pred_forest$predictions #Have to add the extra step for ranger....

# Summarizing the distribution of linear predictor 
mean(pred_forest2) 
sd(pred_forest2) 
min(pred_forest2) 
max(pred_forest2) 

caret::RMSE(pred_forest$predictions, fb_ml_center_scale$BALL_RELEASE_SPEED)


nrow(youth_ht_ml)
nrow(pred_svm)

forestactual_v_predict_combined <- cbind(fb_ml_center_scale, pred_forest2)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_forest2,family="gaussian",x=TRUE,y=TRUE, data = forestactual_v_predict_combined)
mod_combined_2 #1.49
confint(mod_combined_2) 



#######################
#Support Vector Machine 
#######################

svmfit <- svm(BALL_RELEASE_SPEED ~ ., data = fb_ml_center_scale, kernel = "radial", cost = 10, gamma = 0.5, epsilon = 0.1)
# plot results
plot(svmfit)
summary(svmfit)


#Getting Final RMSE 

pred_svm <- predict(svmfit, fb_ml_center_scale, type = "raw")
# Summarizing the distribution of linear predictor 
mean(pred_svm) 
sd(pred_svm) 
min(pred_svm) 
max(pred_svm) 

caret::RMSE(pred_svm, fb_ml_center_scale$BALL_RELEASE_SPEED) #0.10


nrow(youth_ht_ml)
nrow(pred_svm)

svmactual_v_predict_combined <- cbind(fb_ml_center_scale, pred_svm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_svm,family="gaussian",x=TRUE,y=TRUE, data = svmactual_v_predict_combined)
mod_combined_2 
confint(mod_combined_2) 



##########################################################################
#Principal Component Data Driven Predictor Selection Sensitivity Analysis 
##########################################################################

#Original Data Frame
pitch_biomech_notlong

#Splitting into Outcome and Predictors 
pitch_velo_only <- pitch_biomech_notlong %>% 
  select(BALL_RELEASE_SPEED)


pitch_predictors <- pitch_biomech_notlong %>% 
  select(-BALL_RELEASE_SPEED)


#Splitting Predictors into Kinetics and Kinematics 

pitch_predictors_kinetics <- pitch_biomech_notlong %>% 
  select(Back_Leg_GRF_mag_max, Lead_Leg_GRF_mag_max)

str(pitch_predictors_kinetics)

pitch_predictors_kinematics <- pitch_biomech_notlong %>%
  select(-Back_Leg_GRF_mag_max, -Lead_Leg_GRF_mag_max)

str(pitch_predictors_kinematics)

#1st PCA, no kinetics
res.pca_kinematics <- princomp(pitch_predictors_kinematics)

fviz_eig(res.pca_kinematics) #Vix Eigen Values 

fviz_pca_var(res.pca_kinematics,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
) #Viz Variables within different pCA components 


# Eigenvalues
eig.val_kinematics <- get_eigenvalue(res.pca_kinematics)
eig.val_kinematics

# Results for Variables
res.var <- get_pca_var(res.pca_kinematics)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 


########Kinematics 3 PCA's###########################

kinematics_pca_three_components <- res.pca_kinematics$scores[,1:3]


#Combining PCA kinematics, kinetics, and Pitch Velo into one data frame 

pca_full_pitch <- cbind(pitch_velo_only, pitch_predictors_kinetics, kinematics_pca_three_components)


####2nd PCA, Both Kinematics and Kinetics 
res.pca_both <- princomp(pitch_predictors)

fviz_eig(res.pca_both) #Vix Eigen Values 

fviz_pca_var(res.pca_both,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
) #Viz Variables within different pCA components 


# Eigenvalues
eig.val_both <- get_eigenvalue(res.pca_both)
eig.val_both

# Results for Variables
res.var.both <- get_pca_var(res.pca_both)
res.var.both$coord          # Coordinates
res.var.both$contrib        # Contributions to the PCs
res.var.both$cos2           # Quality of representation 

##############Kinematics + Kineitcs (Both) 3 Components ###############################

both_pca_three_components <- res.pca_both$scores[,1:3]


#Combining PCA kinematics, kinetics, and Pitch Velo into one data frame 

pca_both_pitch <- cbind(pitch_velo_only, both_pca_three_components)



#Adding Kinematics PCA to Model 



#Statistical Analyses 
statcv <- train(BALL_RELEASE_SPEED ~ . ,
                data = pca_full_pitch, method = "glm", family = "gaussian", 
                trControl = trainControl("repeatedcv", number = 10)
)
coef(statcv$finalModel)
confint(statcv$finalModel)
statcv$results



statcv_predict <- predict.train(statcv, newdata = pca_full_pitch, type = "raw")

mean(statcv_predict) 
sd(statcv_predict) 
min(statcv_predict)
max(statcv_predict) 


actual_v_predict_combined <- cbind(pca_full_pitch, statcv_predict)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ statcv_predict,family="gaussian",x=TRUE,y=TRUE, data = actual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2) 


#Calibration Plot 



#Plot Itself 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = actual_v_predict_combined, aes(x = statcv_predict, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  xlim(25, 45) + 
  ylim(25, 45) + 
  theme_minimal()
dev.off()


#Ten Fold CV for Performance Metrics 

nfolds=10

results <- matrix(nrow = nfolds,ncol = 3)

folds <- caret::createFolds(pca_full_pitch$BALL_RELEASE_SPEED, k=10)



for(i in 1:nfolds){
  
  cv_samp <- pca_full_pitch[-folds[[i]],]
  
  cv_test <- pca_full_pitch[folds[[i]],]
  
  
  model <- glm(BALL_RELEASE_SPEED ~  .,
               data = cv_samp, family = "gaussian")
  
  pr_cv <- predict(model,type="response", newdata = cv_samp)
  
  pr_cv2 <- cbind(cv_samp, pr_cv)
  
  test_predict <- predict(model,type="response", newdata = cv_test)
  
  test_predict2 <- cbind(cv_test, test_predict)
  
  app_rsquare_model <-   caret::R2(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,1] <- app_rsquare_model
  
  app_cslope_model <- glm(BALL_RELEASE_SPEED ~ pr_cv, family= "gaussian", data= pr_cv2)
  
  results[i,2] <- summary(app_cslope_model)$coefficients[2,1]
  
  
  
  RMSE <- caret::RMSE(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,3] <- RMSE
  
}

results2 <- as.data.frame(results)

colnames(results2) <- c("r-square","c_slope", "RMSE")

mean(results2$`r-square`)
mean(results2$c_slope) 
mean(results2$RMSE) 



##############################################
#Gradient Boosting Machine 
#############################################

gbm.final <- gbm(
  formula = BALL_RELEASE_SPEED ~ .,
  distribution = "gaussian",
  data = pca_full_pitch,
  n.trees = 2500,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 5,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm.final) 



#Visualizing Influence of Difference VARS 
summary(
  gbm.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

pred_gbm <- predict(gbm.final, n.trees = gbm.final$n.trees, pca_full_pitch)

summary(pred_gbm)

sd(pred_gbm) 

caret::RMSE(pred_gbm, pca_full_pitch$BALL_RELEASE_SPEED)
#3.338892e-08
caret::R2(pred_gbm, pca_full_pitch$BALL_RELEASE_SPEED)

gbmactual_v_predict_combined <- cbind(pca_full_pitch, pred_gbm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_gbm,family="gaussian",x=TRUE,y=TRUE, data = gbmactual_v_predict_combined)
mod_combined_2 
confint(mod_combined_2) 


#Plot Itself 

png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = gbmactual_v_predict_combined, aes(x = pred_gbm, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  theme_minimal() 
dev.off()


#########Kineamtic + Kinetics PCA Models 


#Adding Kinematics PCA to Model 



#Statistical Analyses 
statcv <- train(BALL_RELEASE_SPEED ~ . ,
                data = pca_both_pitch, method = "glm", family = "gaussian", 
                trControl = trainControl("repeatedcv", number = 10)
)
coef(statcv$finalModel)
confint(statcv$finalModel)
statcv$results



statcv_predict <- predict.train(statcv, newdata = pca_both_pitch, type = "raw")

mean(statcv_predict) 
sd(statcv_predict) 
min(statcv_predict)
max(statcv_predict) 


actual_v_predict_combined <- cbind(pca_both_pitch, statcv_predict)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ statcv_predict,family="gaussian",x=TRUE,y=TRUE, data = actual_v_predict_combined)
mod_combined_2 #1.00
confint(mod_combined_2) 


#Calibration Plot 



#Plot Itself 


png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = actual_v_predict_combined, aes(x = statcv_predict, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  xlim(25, 45) + 
  ylim(25, 45) + 
  theme_minimal()
dev.off()


#Ten Fold CV for Performance Metrics 

nfolds=10

results <- matrix(nrow = nfolds,ncol = 3)

folds <- caret::createFolds(pca_both_pitch$BALL_RELEASE_SPEED, k=10)



for(i in 1:nfolds){
  
  cv_samp <- pca_full_pitch[-folds[[i]],]
  
  cv_test <- pca_full_pitch[folds[[i]],]
  
  
  model <- glm(BALL_RELEASE_SPEED ~  .,
               data = cv_samp, family = "gaussian")
  
  pr_cv <- predict(model,type="response", newdata = cv_samp)
  
  pr_cv2 <- cbind(cv_samp, pr_cv)
  
  test_predict <- predict(model,type="response", newdata = cv_test)
  
  test_predict2 <- cbind(cv_test, test_predict)
  
  app_rsquare_model <-   caret::R2(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,1] <- app_rsquare_model
  
  app_cslope_model <- glm(BALL_RELEASE_SPEED ~ pr_cv, family= "gaussian", data= pr_cv2)
  
  results[i,2] <- summary(app_cslope_model)$coefficients[2,1]
  
  
  
  RMSE <- caret::RMSE(pr_cv2$pr_cv, pr_cv2$BALL_RELEASE_SPEED)
  
  results[i,3] <- RMSE
  
}

results2 <- as.data.frame(results)

colnames(results2) <- c("r-square","c_slope", "RMSE")

mean(results2$`r-square`)
mean(results2$c_slope) 
mean(results2$RMSE) 



##############################################
#Gradient Boosting Machine 
#############################################

gbm.final <- gbm(
  formula = BALL_RELEASE_SPEED ~ .,
  distribution = "gaussian",
  data = pca_full_pitch,
  n.trees = 2500,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 5,
  bag.fraction = 0.8,
  train.fraction = 1,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm.final) 



#Visualizing Influence of Difference VARS 
summary(
  gbm.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

pred_gbm <- predict(gbm.final, n.trees = gbm.final$n.trees, pca_full_pitch)

summary(pred_gbm)

sd(pred_gbm) 

caret::RMSE(pred_gbm, pca_full_pitch$BALL_RELEASE_SPEED)
#3.338892e-08
caret::R2(pred_gbm, pca_full_pitch$BALL_RELEASE_SPEED)

gbmactual_v_predict_combined <- cbind(pca_full_pitch, pred_gbm)

# C-slope
mod_combined_2 <- glm(BALL_RELEASE_SPEED ~ pred_gbm,family="gaussian",x=TRUE,y=TRUE, data = gbmactual_v_predict_combined)
mod_combined_2 
confint(mod_combined_2) 


#Plot Itself 

png("Insert Your File Path Here.png", width = 3000, height = 3000, units = 'px', res = 500)
ggplot(data = gbmactual_v_predict_combined, aes(x = pred_gbm, y = BALL_RELEASE_SPEED)) + 
  geom_point() +
  geom_abline(slope = 1, color = "blue", size = 2, alpha = 0.6) + # 45 degree line indicating perfect calibration
  geom_smooth(method = lm, se = T, color = "red", size = 1, alpha = 0.8) +
  xlab("Predicted Pitch Velocity (m/s)") +
  ylab("Observed Pitch Velocity (m/s)") +
  theme_minimal() 
dev.off()



