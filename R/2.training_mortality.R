# feature selection
source('Rscript/train.R')
palette <- c("#313695", "#D73027")

res <- split_dataset(SF_X, SF_y, 0.8, 123)
train_SF_X <- res$train_x
train_SF_y <- res$train_y   
test_SF_X <- res$test_x
test_SF_y <- res$test_y 

train_SF_patients <- rownames(train_SF_X)
test_SF_patients <- rownames(test_SF_X)
 
if(F){
  control <- rfeControl(functions=rfFuncs, method="cv", number=10)
  # run the RFE algorithm
  results <- rfe(train_SF_X, train_SF_y$status, sizes=c(1:dim(train_SF_X)[2]), rfeControl=control,
                 metric = "ROC",
                 preProc = c("center", "scale")
  )
  # summarize the results
  print(results)
  # list the chosen features
  rfe_features <- predictors(results)
  # plot the results
  plot(results, type=c("g", "o"))
}

if(T){
  library(glmnet)
  library(plotmo)
  train_SF_y$status
  tmp_y <- train_SF_y
  tmp_y$status <- factor(tmp_y$status, levels=c('0', '1'))
  cox_model <- cv.glmnet(as.matrix(train_SF_X), tmp_y$status, family = 'binomial')
  if(T){
    pdf('result/lasso.pdf')
    plot_glmnet(cox_model$glmnet.fit, s=cox_model$lambda.min) #, col= brewer.pal(dim(train_SF_X)[2], 'Dark2'))
    library(ggpubr)
    tmp <- data.frame(coef(cox_model, s='lambda.min')[-1,])
    colnames(tmp) <- c('Coefficient')
    tmp$Feature <- rownames(tmp)
    tmp <- tmp[order(tmp$Coefficient), ]
    tmp$color <- 'redundant'
    tmp$color[tmp$Coefficient > 0] <- 'positive'
    tmp$color[tmp$Coefficient < 0] <- 'negative'
    tmp$Coefficient <- round(tmp$Coefficient, 4)
    f <- ggbarplot(tmp, 'Feature', 'Coefficient', color='color', fill='color', palette = c('steelblue', 'red', 'gray'),
                   label=T, #repel=T,
              ylab=paste0('Coefficient (s=',round(cox_model$lambda.min,3),')')) + coord_flip()
    print(f)
    dev.off()
  }
  write.csv(coef(cox_model, s='lambda.min')[-1,], 'result/lasso_coef.csv')
  full_features <- colnames(train_SF_X)
  features <- coef(cox_model, s='lambda.min')[-1,]
  features <- names(features[abs(features) > 0])
  features
} 


# training model
model_list <- train_models(train_SF_X[, features], 
                           factor(train_SF_y$status, levels = c('0', '1')), 
                           c('LR', 'SVM', 'KNN', 'RF', 'GBDT', 'NN'))

# save(model_list, features, file='best_model.RData')
# predict model
if(T){
  models <- c('LR', 'SVM', 'KNN', 'RF', 'GBDT', 'NN', 'Ensemble')
  train_SF_pred_df <- pred_models(model_list, train_SF_X[, features], train_SF_y)
  #ens_model <- train_ensemble(train_SF_pred)
  #train_SF_pred_df <- pred_ensemble(ens_model, train_SF_pred)
  f1 <- plot_roc(train_SF_pred_df, models, 'train_SF')
  
  test_SF_pred_df <- pred_models(model_list, test_SF_X[, features], test_SF_y)
  #test_SF_pred_df <- pred_ensemble(ens_model, test_SF_pred)
  f2 <- plot_roc(test_SF_pred_df, models, 'test_SF')
  
  OV_pred_df <- pred_models(model_list, OV_X[, features], OV_y)
  #OV_pred_df <- pred_ensemble(ens_model, OV_pred)
  f3 <- plot_roc(OV_pred_df, models, 'OV')
  
  CHWH_pred_df <- pred_models(model_list, CHWH_X[, features], CHWH_y)
  f4 <- plot_roc(CHWH_pred_df, models, 'CHWH')

  
  if(T){
    pdf('result/roc_legend.pdf', width = 15, height = 10)
    f <- ggarrange(f1, f2, f3, f4, ncol=2, nrow=2)
    print(f)
    dev.off()
    pdf('result/roc.pdf', width = 10, height = 10)
    f <- ggarrange(f1, f2, f3, f4, ncol=2, nrow=2, legend = F)
    print(f)
    dev.off()
  }
  f1 <- my_survival_analysis(train_SF_pred_df, 'train_SF', palette = palette)$fig
  f2 <- my_survival_analysis(test_SF_pred_df, 'test_SF', palette = palette)$fig
  f3 <- my_survival_analysis(OV_pred_df, 'OV', palette = palette)$fig
  f4 <- my_survival_analysis(CHWH_pred_df, 'CHWH', palette = palette)$fig
  if(T){
    pdf('result/surv.pdf', width = 7, height = 10)
    f <- ggarrange(f1, f2, f3, f4, ncol=2, nrow=2)
    print(f)
    dev.off()
  }
  if(T){
    pdf('result/cal.pdf')
    val.prob(train_SF_df$Ensemble, train_SF_df$status, m=10, smooth=F, xlab="Predicted Probability (Ensemble on Train SF)")
    val.prob(test_SF_df$Ensemble, test_SF_df$status, m=10, smooth=F, xlab="Predicted Probability (Ensemble on Test SF)")
    val.prob(OV_df$Ensemble, OV_df$status, m=10, smooth=F, xlab="Predicted Probability (Ensemble on OV )")
    val.prob(CHWH_df$Ensemble, CHWH_df$status, smooth=F, m=10, xlab="Predicted Probability (Ensemble on CHWH)")
    dev.off()
  }
  OV_df <- cbind(OV_X, OV_pred_df)
  OV_df$`Patient ID` <- rownames(OV_X)
  write.csv(OV_df, 'result/OV.csv')
  train_SF_df <- cbind(train_SF_X, train_SF_pred_df) 
  train_SF_df$`Patient ID` <- rownames(train_SF_X)
  write.csv(train_SF_df, 'result/train_SF.csv')
  test_SF_df <- cbind(test_SF_X, test_SF_pred_df)
  test_SF_df$`Patient ID` <- rownames(test_SF_X)
  write.csv(test_SF_df, 'result/test_SF.csv')
  CHWH_df <- cbind(CHWH_X, CHWH_pred_df)
  CHWH_df$`Patient ID` <- rownames(CHWH_X)
  write.csv(CHWH_df, 'result/CHWH.csv')  

metrics_df <- calculate_metrics(train_SF_pred_df, models)
metrics_df$Dataset <- 'train_SF'
tmp <- calculate_metrics(test_SF_pred_df, models)
tmp$Dataset <- 'test_SF'
metrics_df <- rbind(metrics_df, tmp)
tmp <- calculate_metrics(OV_pred_df, models)  
tmp$Dataset <- 'OV'
metrics_df <- rbind(metrics_df, tmp)
tmp <- calculate_metrics(CHWH_pred_df, models)
tmp$Dataset <- 'CHWH'
metrics_df <- rbind(metrics_df, tmp)
write.csv(metrics_df, 'result/performance.csv')

library(caret)
library(gbm)
importance_df <- get_feature_importance(model_list, features)
write.csv(importance_df, 'result/importance.csv')

tmp <- importance_df[importance_df$Model == 'Ensemble', ]
sorted_features <- tmp[order(tmp$Importance), ]$Feature
importance_df$Feature <- factor(importance_df$Feature, levels=sorted_features)
importance_df$Log10_Importance <- log10(importance_df$Importance)
if(T){
  pdf('result/importance.pdf', width = 4, height = 5)
  library(ggpubr)
  tmp <- importance_df[importance_df$Model %in% c('LR', 'SVM', 'GBDT', 'NN', 'Ensemble'), ]
  tmp$Model <- factor(tmp$Model, levels=c('Ensemble', 'LR', 'SVM', 'GBDT', 'NN'))
  f <- ggplot() + 
    geom_point(data=tmp, mapping=aes(Model, Feature, size=Importance, color=Model)) +
        ggsci::scale_fill_nejm() + ggsci::scale_color_nejm() + theme_bw() +
    theme(axis.text.x = element_text(angle=30, hjust=1, vjust=1),
          axis.text.y = element_text(angle=30),
          axis.title.x = element_blank(),
          axis.title.y = element_blank()) 
  print(f)
  dev.off()
}
  }

get_feature_importance <- function(model_list, features){
  df <- NULL
  models <- names(model_list$model_list)
  for (m in models){
    print(m)
    tmp <-  tryCatch({
      varImp(model_list$model_list[[m]]$finalModel, scale = F)
    }, error = function(e) {
      varImp(model_list$model_list[[m]], scale = F)$importance
    })
    if (m %in% c('SVM', 'KNN', 'NN')){
      tmp$Overall <- tmp$Yes
      tmp$No <- NULL
      tmp$Yes <- NULL
    }
    tmp$Model <- m
    tmp <- tmp[rownames(tmp) != '.weights', ]
    tmp$Importance <- tmp$Overall
    tmp_scale <- function(x){(x-min(x))/(max(x)-min(x))*100+10}
    tmp$Importance <- tmp_scale(tmp$Importance)
    tmp$Feature <- features
    tmp$Overall <- NULL
    df <- rbind(df, tmp)
  }
  ens_tmp <- data.frame(row.names=features, Model='Ensemble', 
                        Importance=df[df$Model == 'LR', ]$Importance*0.25 + 
                          df[df$Model == 'SVM', ]$Importance*0.3 + 
                          df[df$Model == 'GBDT', ]$Importance*0.1 +  
                          df[df$Model == 'NN', ]$Importance*0.35,
                        Feature = features)
  df <- rbind(df, ens_tmp)
  df$Model <- factor(df$Model, levels = c(models, 'Ensemble'))
  return(df)
}



