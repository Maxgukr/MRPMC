
# training models
pred_models <- function(model_list, X, y){
  calibrate_model_list <- model_list$calibrate_model_list
  model_list <- model_list$model_list
  df <- y
  for (m in names(model_list)){
    df[[m]] <- pred_func(model_list[[m]], calibrate_model_list[[m]], X)
  }
  rownames(df) <- rownames(X)
  df$Ensemble <- df$LR*0.25 + df$SVM*0.3 + + df$GBDT*0.1 + df$NN*0.35
  df$Cluster <- as.factor(as.numeric(df$Ensemble >= 0.5))
  return(df)
}

train_model <- function(X, y, method=NULL){
  set.seed(1)
  library(caret)
  y <- factor(y, levels=c(0, 1, "No", "Yes"))
  y[y == '0'] <- 'No'
  y[y == '1'] <- 'Yes'
  y <- factor(y, levels=c("No", "Yes"))
  objControl <- trainControl(method='cv', number=10, returnResamp='none',
                             summaryFunction = twoClassSummary, classProbs = TRUE)
  model_weights <- ifelse(y == "No",
                          (1/table(y)[1]) * 0.2,
                          (1/table(y)[2]) * 0.8)
  model <- train(X, y,
                 method=method, 
                 trControl=objControl, 
                 weights = model_weights,
                 metric = "ROC",
                 preProc = c("center", "scale"))
  pred_y <- predict(model, newdata = X, type='prob')[[2]]
  
  tmp <- data.frame(X=pred_y, y=y)
  calibrate_model = train(y ~ X, data=tmp,
                          method='glm', 
                          family='binomial',
                          trControl=objControl, 
                          metric = "ROC",
                          preProc = c("center", "scale"))
  return (list(model=model, calibrate_model=calibrate_model))
}

train_models <- function(X, y, models){
  
  model_list <- list()
  calibrate_model_list <- list()
  
  for (m in models){
    print(m)
    if (m == 'LR'){
      res <- train_model(X, y, method='bayesglm')
    }
    if (m == 'SVM'){
      res <- train_model(X, y, method='svmLinear')
    }
    if (m == 'KNN'){
      res <- train_model(X, y, method='knn')
    }
    if (m == 'RF'){
      res <- train_model(X, y, method='rf')
    }
    if (m == 'GBDT'){
      res <- train_model(X, y, method='gbm')
    }
    if (m == 'NN'){
      res <- train_model(X, y, method='avNNet')
    }
    model_list[[m]] <- res$model
    calibrate_model_list[[m]] <- res$calibrate_model 
  }
  
  return (list(model_list=model_list, calibrate_model_list=calibrate_model_list))
}

pred_func <- function(model, calibrate_model, X){
  pred_y <- predict(model, newdata = X, type='prob')[[2]]
  tmp <- data.frame(X=pred_y)
  calibrated_y <- predict(calibrate_model, newdata = tmp, type = "prob")[[2]]
  return (calibrated_y)
}





calculate_metrics <- function(df, models){
  library(rms)
  tmp <- NULL
  for(m in models){
    cm <- confusionMatrix(factor(as.numeric(df[[m]]>= 0.5), levels=c('1','0')), factor(df$status, levels=c('1','0')), mode='everything')
    metrics <- epiR::epi.tests(dat=cm$table)
    values <- c()
    # AUC
    print(m)
    value <- round(ci.auc(df$status, df[[m]]), 4)
    value <- paste0(value[[2]], ' (', value[[1]], ' - ', value[[3]], ')')
    values <- c(values, value)
    for (metric in c('diag.acc', 'se', 'sp', 'ppv', 'npv')){
      value <- round(metrics$rval[[metric]], 4)
      value <- paste0(value$est, ' (', value$lower, ' - ', value$upper, ')')
      values <- c(values, value)
    }
    # kappa
    value <- round(cm$overall[[2]], 4)
    values <- c(values, value)
    # F1
    value <- round(cm$byClass[[7]], 4)
    values <- c(values, value)
    # Brier
    value <- val.prob(df[[m]], df$status, m=10, pl=F)[['Brier']]
    values <- c(values, value)
    
    names(values) <- c('AUC(95% CI)', 'ACC(95% CI)', 'SE(95% CI)', 'SP(95% CI)', 'PPV(95% CI)', 'NPV(95% CI)', 'Kappa', 'F1', 'Brier')
    tmp <- rbind(tmp, values)
  }
  tmp <- data.frame(tmp)
  colnames(tmp) <- c('AUC(95% CI)', 'ACC(95% CI)', 'SE(95% CI)', 'SP(95% CI)', 'PPV(95% CI)', 'NPV(95% CI)', 'Kappa', 'F1', 'Brier')
  rownames(tmp) <- models
  return(tmp)
}



plot_roc <- function(pred_df, models, title){
  library(pROC)
  library(ggpubr)
  library(RColorBrewer)
  roc_list = list()
  for (c in models){
    rocobj <- roc(pred_df$status, as.numeric(pred_df[[c]]))
    roc_list[[paste0('AUC for ', c, ' = ', round(rocobj$auc, 4))]] <- rocobj
  }
  ggroc(roc_list, legacy.axes = TRUE) +
    theme_classic() +
    ggtitle(title) + ggsci::scale_color_nejm() +
    labs(x = "1 - Specificity",
         y = "Sensitivity")
}

my_survival_analysis <- function(df, 
                                 title='',
                                 palette=c('steelblue', 'red'),
                                 xmax=NULL, timeby=NULL
){
  library(survival)
  library(survminer)
  if (is.null(xmax)){
    xlim <- NULL
  }else{
    xlim <- c(0, xmax)
  }
  
  s_fit <- survfit(Surv(time, status) ~ Cluster, data=df)
  surv_res <- summary(s_fit)
  surv_cols <- lapply(c(2:6, 8:11) , function(x) surv_res[x])
  surv_df <- do.call(data.frame, surv_cols)
  surv_df$Desc <- title
  
  
  cox_fit = coxph(Surv(time, status) ~ Cluster, data=df)
  print(cox_fit)
  conf <- as.data.frame(round(summary(cox_fit)$conf.int, digits=2))
  
  text=paste('HR ', conf$`exp(coef)`, 
             ' (95% CI ', conf$`lower .95`, "-", conf$`upper .95`, ")\n", 
             surv_pvalue(s_fit, data=df)$method, " ",
             surv_pvalue(s_fit, data=df)$pval.txt, 
             sep="")
  fig <- ggsurvplot(s_fit, data=df,
                    conf.int = T,
                    pval=text, pval.size=3, pval.coord=c(0, 0.1),
                    risk.table = TRUE, tables.theme = clean_theme(), tables.y.text=FALSE,
                    cumcensor=TRUE, tables.height = 0.2,
                    palette=palette,
                    xlim=xlim, break.time.by=timeby,
                    xlab = "Time from onset",
                    title=title
  )
  
  fig <- ggarrange(fig$plot, fig$table, ncol=1, nrow=2, heights=c(4,1))
  return(list(fig=fig,
              surv_df=surv_df,
              pvalue=surv_pvalue(s_fit, data=df)$pval
  ))
}


split_dataset <- function(x, y, ratio, seed){
  
  smp_size <- floor(ratio * nrow(x))
  ## set the seed to make your partition reproducible
  set.seed(seed)
  train_ind <- sample(seq_len(nrow(x)), size = smp_size)
  
  train_x <- x[train_ind, ]
  test_x <- x[-train_ind, ]
  train_y <- y[train_ind, ]
  test_y <- y[-train_ind, ]
  
  return(list(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y))
}






