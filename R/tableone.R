library(tableone)

mydata <- Circlize_ml_new[, c(2:34)]

CreateTableOne(data=mydata)
dput(names(mydata))

catVars<-c("Death", "Hospital", "Gender",   
           "Hypertention", "Diabetes", "Coronary heart disease", "Liver disease", 
           "Tumor", "HBV", "CKD", "COPD", "Fever", "Temp(max) ≥ 39℃", 
           "Cough", "Dyspnea", "Sputum", "Fatigue", "Diarrhea", "Myalgia", 
           "Vomitting", "Consciousness")
nonvar<-c( "BUN", "DDimer", "No. comorbidities", "RR", "Mean arterial pressure", 
           "SpO2", "LBC", "ALB", "Age", 
           "PLT")
stratavar <- c("Hospital", "Death")

tab2<-CreateTableOne(data=mydata, factorVars=catVars, strata = stratavar)
d <- print(tab2,nonnormal = nonvar,showAllLevels = TRUE, noSpaces = TRUE)

write.csv(d, "d.csv")
