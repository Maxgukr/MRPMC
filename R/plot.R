


volcano_plot <- function(res_df, title){
  res_df$neglog10p <- -log10(res_df$`Pr(>|z|)`)
  res_df$sig[res_df$`Pr(>|z|)` < 0.05] <- 'significant'
  res_df$sig[res_df$`Pr(>|z|)` >= 0.05] <- 'not significant'
  res_df$Prognosis[res_df$HR > 1] <- 'high risk'
  res_df$Prognosis[res_df$HR <= 1] <- 'low risk'
  res_df$Prognosis <- paste(res_df$sig, res_df$Prognosis)
  res_df$Prognosis[res_df$sig == 'not significant'] <- 'not significant'
  selects <- rownames(res_df[res_df$`Pr(>|z|)` < 0.05, ])
  vol_palette <- c('gray', 'gray','red', 'steelblue')
  names(vol_palette) <- c('NA low risk', 'not significant', 'significant high risk', 'significant low risk')
  ggscatter(res_df, 'HR', 'neglog10p', color='Prognosis', 
            palette = vol_palette,
            label='Row.names', label.select = selects, repel=T,
            ylab='-log10(pvalue)', 
            title=title) + geom_vline(xintercept=1, linetype='dotted')
}


plot_heatmap <- function(x, annotation_df, annotation_colors){
  library(pheatmap)
  library(RColorBrewer)
  pheatmap(x, cluster_rows = F, cluster_cols = F, show_rownames = F, na_co = 'black',
           annotation_row = annotation_df, annotation_colors = annotation_colors,
           color = colorRampPalette(rev(brewer.pal(n = 7, name ="Spectral")))(100)
  )[[4]]
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

plot_cate_heatmap <- function(x, annotation_df, annotation_colors){
  library(ggplot2)
  library(reshape2)
  library(RColorBrewer)
  color_palette = rainbow(10)
  names(color_palette) <- c(0:8)
  # converting data to long form for ggplot2 use
  x$ID <- rownames(x)
  df <- melt(x, id.vars = 'ID')
  df$value <- factor(df$value)
  f1 <- ggplot(df, aes(variable, ID)) + geom_tile(aes(fill = value)) + 
    scale_fill_manual(values=color_palette, na.value='black') +
    theme(
      axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    ) + xlab('') + ylab('')
  tmp <- annotation_df
  tmp <- tmp[rownames(x), ]
  tmp$ID <- factor(rownames(tmp), levels=row.names(x))
  df <- melt(tmp, id.vars = 'ID')
  f2 <- ggplot(df, aes(variable, ID)) + geom_tile(aes(fill = value)) + 
    scale_fill_manual(values=c(annotation_colors$Hospital, annotation_colors$Death), na.value='black') +
    theme(
      axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    ) + xlab('') + ylab('')
  
  
  ggarrange(f2, f1, align = 'hv', widths = c(dim(annotation_df)[[2]]*3, dim(x)[2]))
}









