
library(ggplot2)
library(ggbeeswarm)
library(RColorBrewer)

cols<-brewer.pal(10, "RdYlBu")
pal<-colorRampPalette(cols)
mycolors<-pal(100)

Circlize_ml_new$Death <- as.factor(Circlize_ml_new$Death)
mydata <- data.frame(Circlize_ml_new)
mydatanew <- mydata

#BUN
#mydatanew$BUN <- log10(mydatanew$BUN) 
BUNplot <- 
  ggplot(mydatanew, mapping = aes(Death, BUN, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank())+
  labs(title="BUN", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))+
  ylim(0,20)

#DDimer
#mydatanew$DDimer <- log10(mydatanew$DDimer) 
DDimerplot <- 
  ggplot(mydatanew, mapping = aes(Death, DDimer, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank()) +
  labs(title="D-Dimer", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))+
  scale_y_log10()

#SpO2
#mydatanew$SpO2 <- log10(mydatanew$SpO2) 
SpO2plot <- 
  ggplot(mydatanew, mapping = aes(Death, SpO2, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank())+
  labs(title="SpO2", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))

#RR
#mydatanew$RR <- log10(mydatanew$RR) 
RRplot <- ggplot(mydatanew, mapping = aes(Death, RR, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank()) +
  labs(title="RR", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))+
  scale_y_continuous(limits = c(10,35), breaks = c(seq(10, 35, 5)))

#LBC
#mydatanew$LBC <- log10(mydatanew$LBC) 
LBCplot <- 
  ggplot(mydatanew, mapping = aes(Death, LBC, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank()) +
  labs(title="LBC", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))

#ALB
#mydatanew$ALB <- log10(mydatanew$ALB) 
ALBplot <- ggplot(mydatanew, mapping = aes(Death, ALB, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank()) +
  labs(title="ALB", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))+
  scale_y_continuous(limits = c(10,60), breaks = c(seq(10, 60, 10)))

#Age
#mydatanew$Age <- log10(mydatanew$Age) 
Ageplot <- 
  ggplot(mydatanew, mapping = aes(Death, Age, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank()) +
  labs(title="Age", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))


#PLT
PLTplot <- 
  ggplot(mydatanew, mapping = aes(Death, PLT, color = Death)) + 
  geom_jitter(aes(colour=factor(Death)), width = 0.20, size = 0.25) + 
  geom_boxplot(width=0.5, alpha = 0.25, color = "black", size = 0.25)+
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        legend.key=element_blank()) +
  labs(title="PLT", x = "", y = "")+
  theme(legend.position="none", plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=30, hjust = 1, vjust=1))+
  scale_color_manual(values = c("Discharge"= "#313695", "Death"="#D73027"))

library(grid)

grid.newpage()  ###新建图表版面
pushViewport(viewport(layout = grid.layout(2,4)))
vplayout <- function(x,y){viewport(layout.pos.row = x, layout.pos.col = y)} 

print(BUNplot, vp = vplayout(1,1))  
print(DDimerplot, vp = vplayout(1,2))  
print(SpO2plot, vp = vplayout(1,3))              
print(RRplot , vp = vplayout(1,4)) 
print(LBCplot, vp = vplayout(2,1))
print(ALBplot, vp = vplayout(2,2))  
print(Ageplot , vp = vplayout(2,3)) 
print(PLTplot , vp = vplayout(2,4)) 
