library(readxl)

source('../Rscript/predict.R')

lab_test <- read_excel('mortality_lab_test.xlsx')



tmp <- data.frame(lab_test)
colnames(tmp) <- colnames(lab_test)
lab_test <- tmp
# remove endswith Grade1 and Grade2
lab_test <- lab_test[, colnames(lab_test)[!(endsWith(colnames(lab_test), 'Grade1') | endsWith(colnames(lab_test), 'Grade2') | endsWith(colnames(lab_test), 'Grade'))]]

selected_features <- read_excel('selected_features.xls')$Feature

# check duplicate patient
tmp <- data.frame(table(lab_test$ID))
tmp[tmp$Freq != 1, ]

lab_test$Gender[lab_test$Gender == '男'] <- 1
lab_test$Gender[lab_test$Gender == '女'] <- 0
# lab_test$Temp_max_39 <- as.numeric(lab_test$Temperature >= 39)
lab_test <- lab_test[order(lab_test$Hospital, lab_test$Death), ]

lab_test$Death <- lab_test$Death - 1


feature_type_df <- read_excel('feature_type.xlsx')
cate_cols <- feature_type_df[feature_type_df$Type == 'Categorical', ]$Feature
cate_cols <- cate_cols[cate_cols %in% selected_features]
length(cate_cols)

cont_cols <- feature_type_df[feature_type_df$Type == 'Continuous', ]$Feature
cont_cols <- cont_cols[cont_cols %in% selected_features]
length(cont_cols)
for (c in cont_cols){
  if(c %in% colnames(lab_test)){
    lab_test[[c]] <- as.numeric(lab_test[[c]])
  }
}

annotation_colors <- list(
  'Hospital' = c('GG' = '#E7298A', 'ZF' = '#7570B3', 'XY' = 'orange'),
  'Death' = c('0'='steelblue', '1'='red')
)
annotation_df <- data.frame(row.names = lab_test$`ID`, 
                            Hospital=lab_test$Hospital,
                            Death=factor(lab_test$Death, levels=c(0, 1)))


X <- lab_test[, c(cont_cols, cate_cols)]
rownames(X) <- lab_test$ID

OV_patients <- lab_test[lab_test$Hospital == 'GG', ]$`ID`
SF_patients <- lab_test[lab_test$Hospital == 'ZF', ]$`ID`
CHWH_patients <- lab_test[lab_test$Hospital == 'XY', ]$`ID`











# filter
n_X <- X[, colnames(X) %in% c(cont_cols)]
rownames(n_X) <- rownames(X)

n_f1 <- plot_heatmap(n_X, annotation_df, annotation_colors)
write.csv(n_X, 'result/numeric_X.csv')


ratio <- 0.05
filter_out_features <- c()
for (h in unique(lab_test$Hospital)){
  tmp <- n_X[rownames(n_X) %in% lab_test[lab_test$Hospital == h, ]$ID, ]
  filter_out_features <- union(filter_out_features, colnames(tmp)[colSums(is.na(tmp))/dim(tmp)[1] >= ratio])
}

n_f2 <- plot_heatmap(n_X[, filter_out_features], annotation_df, annotation_colors)
write.csv(n_X[, filter_out_features], 'result/numeric_filter_out_X.csv')


filtered_n_X <- n_X[, !(colnames(n_X) %in% filter_out_features)]

n_f3 <- plot_heatmap(filtered_n_X, annotation_df, annotation_colors)
write.csv(filtered_n_X, 'result/numeric_filtered_X.csv')

if(T){
  library(ggpubr)
  pdf('result/heatmap_numeric_filter.pdf', width=10, height = 12)
  f <- ggarrange(n_f1, n_f2, n_f3,
                 ncol=1, nrow=3)
  print(f)
  dev.off()
}

c_X <- X[, colnames(X) %in% cate_cols]
rownames(c_X) <- rownames(X)
for (c in cate_cols){
  if (c %in% colnames(c_X)){
    c_X[[c]] <- as.factor(c_X[[c]])
  }
}

c_f1 <- plot_cate_heatmap(c_X, annotation_df, annotation_colors)
write.csv(c_X, 'result/cate_X.csv')

filter_out_c_features <- c()
for (h in unique(lab_test$Hospital)){
  tmp <- c_X[rownames(c_X) %in% lab_test[lab_test$Hospital == h, ]$ID, ]
  filter_out_c_features <- union(filter_out_c_features, colnames(tmp)[colSums(is.na(tmp))/dim(tmp)[1] >= ratio])
}

c_f2 <- plot_cate_heatmap(c_X[, filter_out_c_features], annotation_df, annotation_colors)
write.csv(c_X[, filter_out_c_features], 'result/cate_filter_out_X.csv')


filtered_c_X <- c_X[, !(colnames(c_X) %in% filter_out_c_features)]

c_f3 <- plot_cate_heatmap(filtered_c_X, annotation_df, annotation_colors)
write.csv(filtered_c_X, 'result/cate_filtered_X.csv')
if(T){
  library(ggpubr)
  pdf('result/heatmap_categorical_filter.pdf', width=20, height = 10)
  f <- ggarrange(c_f1, c_f2, c_f3, align='hv', widths = c(4,2,3), labels = 'auto',
                 ncol=3, nrow=1)
  print(f)
  dev.off()
}





library(missForest)
# impute
OV_filtered_n_X <- filtered_n_X[rownames(filtered_n_X) %in% OV_patients, ]
rownames(OV_filtered_n_X) <- rownames(filtered_n_X)[rownames(filtered_n_X) %in% OV_patients]
SF_filtered_n_X <- filtered_n_X[rownames(filtered_n_X) %in% SF_patients, ]
rownames(SF_filtered_n_X) <- rownames(filtered_n_X)[rownames(filtered_n_X) %in% SF_patients]
CHWH_filtered_n_X <- filtered_n_X[rownames(filtered_n_X) %in% CHWH_patients, ]
rownames(CHWH_filtered_n_X) <- rownames(filtered_n_X)[rownames(filtered_n_X) %in% CHWH_patients]
write.csv(OV_filtered_n_X, 'result/OV_numeric_filtered_X.csv')
write.csv(SF_filtered_n_X, 'result/SF_numeric_filtered_X.csv')
write.csv(CHWH_filtered_n_X, 'result/CHWH_numeric_filtered_X.csv')


# impute
OV_filtered_c_X <- filtered_c_X[rownames(filtered_c_X) %in% OV_patients, ]
rownames(OV_filtered_c_X) <- rownames(filtered_c_X)[rownames(filtered_c_X) %in% OV_patients]
SF_filtered_c_X <- filtered_c_X[rownames(filtered_c_X) %in% SF_patients, ]
rownames(SF_filtered_c_X) <- rownames(filtered_c_X)[rownames(filtered_c_X) %in% SF_patients]
CHWH_filtered_c_X <- filtered_c_X[rownames(filtered_c_X) %in% CHWH_patients, ]
rownames(CHWH_filtered_c_X) <- rownames(filtered_c_X)[rownames(filtered_c_X) %in% CHWH_patients]
write.csv(OV_filtered_c_X, 'result/OV_cate_filtered_X.csv')
write.csv(SF_filtered_c_X, 'result/SF_cate_filtered_X.csv')
write.csv(CHWH_filtered_c_X, 'result/CHWH_cate_filtered_X.csv')

OV_filtered_X <- cbind(OV_filtered_n_X, OV_filtered_c_X)
SF_filtered_X <- cbind(SF_filtered_n_X, SF_filtered_c_X)
CHWH_filtered_X <- cbind(CHWH_filtered_n_X, CHWH_filtered_c_X)

set.seed(123)
res <- missForest(OV_filtered_X)
OV_X <- res$ximp
for (c in cate_cols){
  if (c %in% colnames(OV_X)){
    OV_X[[c]] <- as.numeric(OV_X[[c]])
  }
}
res <- missForest(SF_filtered_X)
SF_X <- res$ximp
for (c in cate_cols){
  if (c %in% colnames(SF_X)){
    SF_X[[c]] <- as.numeric(SF_X[[c]])
  }
}
res <- missForest(CHWH_filtered_X)
CHWH_X <- res$ximp
for (c in cate_cols){
  if (c %in% colnames(CHWH_X)){
    CHWH_X[[c]] <- as.numeric(CHWH_X[[c]])
  }
}

if(T){
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

OV_n_X <- OV_X[colnames(OV_X) %in% cont_cols]
OV_c_X <- OV_X[colnames(OV_X) %in% cate_cols]
SF_n_X <- SF_X[colnames(SF_X) %in% cont_cols]
SF_c_X <- SF_X[colnames(SF_X) %in% cate_cols]
CHWH_n_X <- CHWH_X[colnames(CHWH_X) %in% cont_cols]
CHWH_c_X <- CHWH_X[colnames(CHWH_X) %in% cate_cols]



missing_count_df <- NULL
OV_res <- process_impute(OV_filtered_n_X, annotation_df, annotation_colors)
missing_count_df <- rbind(missing_count_df, colSums(is.na(OV_filtered_n_X)))
OV_n_X <- OV_res$imputed_X
missing_count_df <- rbind(missing_count_df, colSums(is.na(OV_n_X)))
write.csv(OV_n_X, 'result/OV_cont_X.csv')

SF_res <- process_impute(SF_filtered_n_X, annotation_df, annotation_colors)
missing_count_df <- rbind(missing_count_df, colSums(is.na(SF_filtered_n_X)))
SF_n_X <- SF_res$imputed_X
missing_count_df <- rbind(missing_count_df, colSums(is.na(SF_n_X)))
write.csv(SF_n_X, 'result/SF_cont_X.csv')

CHWH_res <- process_impute(CHWH_filtered_n_X, annotation_df, annotation_colors)
missing_count_df <- rbind(missing_count_df, colSums(is.na(CHWH_filtered_n_X)))
CHWH_n_X <- CHWH_res$imputed_X
missing_count_df <- rbind(missing_count_df, colSums(is.na(CHWH_n_X)))
write.csv(CHWH_n_X, 'result/CHWH_cont_X.csv')

rownames(missing_count_df) <- c(
  'OV (before imputation)',
  'OV (after imputation)',
  'SF (before imputation)',
  'SF (after imputation)',
  'CHWH (before imputation)',
  'CHWH (after imputation)'
)
write.csv(missing_count_df, 'result/cont_missing_count.csv')


if(T){
  library(ggpubr)
  pdf('result/heatmap_numeric_filter_impute.pdf', width=15, height = 15)
  f <- ggarrange(SF_res$before, SF_res$after,
                 OV_res$before, OV_res$after,
                 CHWH_res$before, CHWH_res$after,
                 ncol=2, nrow=3)
  print(f)
  dev.off()
}

missing_count_df <- NULL

OV_res <- process_cate_impute(OV_filtered_c_X, annotation_df, annotation_colors)
missing_count_df <- rbind(missing_count_df, colSums(is.na(OV_filtered_c_X)))
OV_c_X <- OV_res$imputed_X
missing_count_df <- rbind(missing_count_df, colSums(is.na(OV_c_X)))
write.csv(OV_c_X, 'result/OV_cate_X.csv')

SF_res <- process_cate_impute(SF_filtered_c_X, annotation_df, annotation_colors)
missing_count_df <- rbind(missing_count_df, colSums(is.na(SF_filtered_c_X)))
SF_c_X <- SF_res$imputed_X
missing_count_df <- rbind(missing_count_df, colSums(is.na(SF_c_X)))
write.csv(SF_c_X, 'result/SF_cate_X.csv')

CHWH_res <- process_cate_impute(CHWH_filtered_c_X, annotation_df, annotation_colors)
missing_count_df <- rbind(missing_count_df, colSums(is.na(CHWH_filtered_c_X)))
CHWH_c_X <- CHWH_res$imputed_X
missing_count_df <- rbind(missing_count_df, colSums(is.na(CHWH_c_X)))
write.csv(CHWH_c_X, 'result/CHWH_cate_X.csv')

rownames(missing_count_df) <- c(
  'OV (before imputation)',
  'OV (after imputation)',
  'SF (before imputation)',
  'SF (after imputation)',
  'CHWH (before imputation)',
  'CHWH (after imputation)'
)
write.csv(missing_count_df, 'result/cate_missing_count.csv')

if(T){
  library(ggpubr)
  pdf('result/heatmap_cate_filter_impute.pdf', width=15, height = 30)
  f <- ggarrange(SF_res$before, SF_res$after,
                 OV_res$before, OV_res$after,
                 CHWH_res$before, CHWH_res$after,
                 ncol=2, nrow=3)
  print(f)
  dev.off()
}

tmp <- cbind(rownames(OV_X), OV_X, OV_y$status)
colnames(tmp) <- c('ID', colnames(OV_X), 'Death')
library(openxlsx)
write.xlsx(tmp, 'result/OV.xlsx')

tmp <- cbind(rownames(SF_X), SF_X, SF_y$status)
colnames(tmp) <- c('ID', colnames(SF_X), 'Death')
write.xlsx(tmp, 'result/SF.xlsx')

tmp <- cbind(rownames(CHWH_X), CHWH_X, CHWH_y$status)
colnames(tmp) <- c('ID', colnames(CHWH_X), 'Death')
write.xlsx(tmp, 'result/CHWH.xlsx')

if(T){
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

filtered_features <- colnames(OV_X)

rownames(lab_test) <- lab_test$ID
raw_features_df <- lab_test[, c(filtered_features, 'Death', 'Hospital length')]
colnames(raw_features_df) <- c(filtered_features, 'status', 'time')


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


