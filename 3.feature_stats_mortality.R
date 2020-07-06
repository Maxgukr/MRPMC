library(readxl)


if(T){
  OV_X <- cbind(OV_c_X, OV_n_X)
  SF_X <- cbind(SF_c_X, SF_n_X)
  CHWH_X <- cbind(CHWH_c_X, CHWH_n_X)
  
  OV_y <- lab_test[lab_test$ID %in% OV_patients, c('Death', 'Hospital length')]
  rownames(OV_y) <- lab_test$ID[lab_test$ID %in% OV_patients]
  colnames(OV_y) <- c('status', 'time')
  all.equal(rownames(OV_y), rownames(OV_X))
  
  SF_y <- lab_test[lab_test$ID %in% SF_patients, c('Death', 'Hospital length')]
  rownames(SF_y) <- lab_test$ID[lab_test$ID %in% SF_patients]
  colnames(SF_y) <- c('status', 'time')
  all.equal(rownames(SF_y), rownames(SF_X))
  
  CHWH_y <- lab_test[lab_test$ID %in% CHWH_patients, c('Death', 'Hospital length')]
  rownames(CHWH_y) <- lab_test$ID[lab_test$ID %in% CHWH_patients]
  colnames(CHWH_y) <- c('status', 'time')
  all.equal(rownames(CHWH_y), rownames(CHWH_X))
}

features <- colnames(OV_X)
features
rownames(lab_test) <- lab_test$ID
raw_features_df <- lab_test[, c(features, 'Death', 'Hospital length')]
colnames(raw_features_df) <- c(features, 'status', 'time')


library(ggpubr)



source('../Rscript/utils.R')





if (T){
  cox_fit <- coxph(Surv(time, status) ~ ., raw_features_df)
  res_df <- process_cox_fit(cox_fit)
  write.csv(res_df, 'result/all_feature_cox.csv')
  cox_f <- volcano_plot(res_df, 'Death')
  pdf('result/all_feature_cox.pdf')
  print(cox_f)
  dev.off()
}

if (T){
  cox_fit <- coxph(Surv(time, status) ~ ., raw_features_df[rownames(raw_features_df) %in% OV_patients, ])
  res_df <- process_cox_fit(cox_fit)
  write.csv(res_df, 'result/OV_feature_cox.csv')
  cox_f <- volcano_plot(res_df, 'Death')
  pdf('result/OV_feature_cox.pdf')
  print(cox_f)
  dev.off()
}

if (T){
  cox_fit <- coxph(Surv(time, status) ~ ., raw_features_df[rownames(raw_features_df) %in% SF_patients, ])
  res_df <- process_cox_fit(cox_fit)
  write.csv(res_df, 'result/SF_feature_cox.csv')
  cox_f <- volcano_plot(res_df, 'Death')
  pdf('result/SF_feature_cox.pdf')
  print(cox_f)
  dev.off()
}

if (T){
  cox_fit <- coxph(Surv(time, status) ~ ., raw_features_df[rownames(raw_features_df) %in% CHWH_patients, ])
  res_df <- process_cox_fit(cox_fit)
  res_df
  write.csv(res_df, 'result/CHWH_feature_cox.csv')
  cox_f <- volcano_plot(res_df, 'Death')
  pdf('result/CHWH_feature_cox.pdf')
  print(cox_f)
  dev.off()
}

