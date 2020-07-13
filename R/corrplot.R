library(dplyr)
library(corrgram)
library(corrplot)
library(reshape2)
library(plyr)
library(tidyr)
library(doBy)
library(RColorBrewer)
cols<-brewer.pal(8, "RdBu")
pal<-colorRampPalette(cols)

str(Circlize_ml_new)

mydata <- Circlize_ml_new[,c("BUN", "DDimer", "SpO2", "RR", 
                             "LBC", "ALB", "Age","PLT")]

str(mydata)

mycirc <- cor(mydata, use= "na.or.complete", method = "spearman")

##coorplot
corrplot(mycirc, method = "circle", tl.col="black", tl.pos="top", type = "upper", order = "hclust", addrect = 3, col = pal(100))
corrplot(mycirc,add=TRUE, type="lower", method="number",order="AOE", diag=FALSE,tl.pos="n", cl.pos="n", col = pal(100))
