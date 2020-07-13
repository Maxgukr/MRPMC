dput(names(TotalPatients_ML))

TotalPatients_ML$Label <- as.factor(TotalPatients_ML$Label)
TotalPatients_ML$respiratory_failure <- as.numeric(TotalPatients_ML$respiratory_failure)

respiratory_failure_table <- TotalPatients_ML[,c("respiratory_failure", "Label")]
respiratory_failure_table  <- na.omit(respiratory_failure_table)

model <- glm(respiratory_failure~Label, data= respiratory_failure_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "respiratory_failure.csv")

###ARDS
TotalPatients_ML$ARDS <- as.numeric(TotalPatients_ML$ARDS)

ARDS_table <- TotalPatients_ML[,c("ARDS", "Label")]
ARDS_table  <- na.omit(ARDS_table)

model <- glm(ARDS~Label, data= ARDS_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "ARDS.csv")

###sepsis
TotalPatients_ML$sepsis <- as.numeric(TotalPatients_ML$sepsis)

sepsis_table <- TotalPatients_ML[,c("sepsis", "Label")]
sepsis_table  <- na.omit(sepsis_table)

model <- glm(sepsis~Label, data= sepsis_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "sepsis.csv")

###septic_shock
TotalPatients_ML$septic_shock <- as.numeric(TotalPatients_ML$septic_shock)

septic_shock_table <- TotalPatients_ML[,c("septic_shock", "Label")]
septic_shock_table  <- na.omit(septic_shock_table)

model <- glm(septic_shock~Label, data= septic_shock_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "septic_shock.csv")


###secondary_infection
TotalPatients_ML$secondary_infection <- as.numeric(TotalPatients_ML$secondary_infection)

secondary_infection_table <- TotalPatients_ML[,c("secondary_infection", "Label")]
secondary_infection_table  <- na.omit(secondary_infection_table)

model <- glm(secondary_infection~Label, data= secondary_infection_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "secondary_infection.csv")

###heart_failure
TotalPatients_ML$heart_failure <- as.numeric(TotalPatients_ML$heart_failure)

heart_failure_table <- TotalPatients_ML[,c("heart_failure", "Label")]
heart_failure_table  <- na.omit(heart_failure_table)

model <- glm(heart_failure~Label, data= heart_failure_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "heart_failure.csv")


###kidney_injury
TotalPatients_ML$kidney_injury <- as.numeric(TotalPatients_ML$kidney_injury)

kidney_injury_table <- TotalPatients_ML[,c("kidney_injury", "Label")]
kidney_injury_table  <- na.omit(kidney_injury_table)

model <- glm(kidney_injury~Label, data= kidney_injury_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "kidney_injury.csv")


###liver_injury
TotalPatients_ML$liver_injury <- as.numeric(TotalPatients_ML$liver_injury)

liver_injury_table <- TotalPatients_ML[,c("liver_injury", "Label")]
liver_injury_table  <- na.omit(liver_injury_table)

model <- glm(liver_injury~Label, data= liver_injury_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "liver_injury.csv")


###coagulopathy
TotalPatients_ML$coagulopathy <- as.numeric(TotalPatients_ML$coagulopathy)

coagulopathy_table <- TotalPatients_ML[,c("coagulopathy", "Label")]
coagulopathy_table  <- na.omit(coagulopathy_table)

model <- glm(coagulopathy~Label, data= coagulopathy_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "coagulopathy.csv")

###hypoprotein
TotalPatients_ML$hypoprotein <- as.numeric(TotalPatients_ML$hypoprotein)

hypoprotein_table <- TotalPatients_ML[,c("hypoprotein", "Label")]
hypoprotein_table  <- na.omit(hypoprotein_table)

model <- glm(hypoprotein~Label, data= hypoprotein_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "hypoprotein.csv")


###icu_ad
TotalPatients_ML$icu_ad <- as.numeric(TotalPatients_ML$icu_ad)

icu_ad_table <- TotalPatients_ML[,c("icu_ad", "Label")]
icu_ad_table  <- na.omit(icu_ad_table)

model <- glm(icu_ad~Label, data= icu_ad_table)
summary(model)

x <- exp (cbind (OR  = coef(model), confint(model)))
y <- signif(x, digits = 4)
write.csv(y, "icu_ad.csv")