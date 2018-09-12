library(mice)
library(VIM)
library(lattice)
library(vcd)

bioage <- read.csv("bioage_V1.csv", header=T, stringsAsFactors = T
                   , na.strings = c("NA","NaN",NA, NaN, ""," ","  ","   ","    ","     ","      "))
head(bioage)  
bioage = bioage[,2:ncol(bioage)] # Remove unnamed
colnames(bioage)
dim(bioage)
categorical = colnames(bioage)[c(60:63,70,72:77)]
categorical
over1 = c('WAIST','HIP','CPK','DIRBIL','IRON','TIBC',
           'UIBC','AMYLASE','PDW','T3','CRP','HBA1C','WBC', "ESR")

# 1이상 변수 제거 = 14개
bioage[ ,over1] <- list(NULL)
colnames(bioage)
dim(bioage)

# 남/녀
bioage_m <- bioage[bioage$SEX == "1", ]
bioage_f <- bioage[bioage$SEX == "2", ]
dim(bioage_m)
dim(bioage_f)

# Correlation
cor(bioage_m[5:10], use = "pairwise.complete.obs")

#1. Body
body_m = bioage_m[,3:8]
body_f = bioage_f[,3:8]
colnames(body_m)

md.pattern(body_m)
md.pattern(body_f)
body_m_aggr = aggr(body_m, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                   labels=names(body_m), gap=3, 
                   ylab=c("Proportion of missingness","Missingness Pattern"))
body_f_aggr = aggr(body_f, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                   labels=names(body_m), gap=3, 
                   ylab=c("Proportion of missingness","Missingness Pattern"))

method_body = c("","pmm","pmm","pmm","pmm",'norm.predict')
body_m_imp = mice(body_m, m=5, meth=method_body, maxit = 40, seed=2525)
body_f_imp = mice(body_f, m=5, meth=method_body, maxit = 40, seed=2525)
densityplot(body_m_imp)
densityplot(body_f_imp)
plot(body_m_imp) 
plot(body_f_imp)

# Imputation
body_m_imp1 = complete(body_m_imp, 1)
body_m_imp2 = complete(body_m_imp, 2)
body_m_imp3 = complete(body_m_imp, 3)
body_m_imp4 = complete(body_m_imp, 4)
body_m_imp5 = complete(body_m_imp, 5)
body_f_imp1 = complete(body_f_imp, 1)
body_f_imp2 = complete(body_f_imp, 2)
body_f_imp3 = complete(body_f_imp, 3)
body_f_imp4 = complete(body_f_imp, 4)
body_f_imp5 = complete(body_f_imp, 5)

body_m_comp = (body_m_imp1 + body_m_imp2 + body_m_imp3 + body_m_imp4 + body_m_imp5)/5
bioage_m[,3:8] = body_m_comp
body_f_comp = (body_f_imp1 + body_f_imp2 + body_f_imp3 + body_f_imp4 + body_f_imp5)/5
bioage_f[,3:8] = body_f_comp

saveRDS(bioage_m, file="bioage_m_1.rds")
saveRDS(bioage_f, file="bioage_f_1.rds")
bioage_m <- readRDS("bioage_m_1.rds")
bioage_f <- readRDS("bioage_f_1.rds")

# 2. Lung
# Correlation = 0.79과 식을 고려해 FEV_FVC는 나중에 Imputation 후 다시 Imputation
lung_m = bioage_m[,c(3,9:11)]
lung_f = bioage_f[,c(3,9:11)]
colnames(lung_m)

cor(lung_m, use = "pairwise.complete.obs")
lung_m_imp = mice(lung_m, m=5, meth='norm.predict', maxit = 40, seed=2525)
lung_f_imp = mice(lung_f, m=5, meth='norm.predict', maxit = 40, seed=2525)

## FEV1_FVC분포가 유독 다름.
densityplot(lung_m_imp)
densityplot(lung_f_imp)


# Remove FEV1_FVC 
lung_1_m = bioage_m[,c(9:10)]
lung_1_f = bioage_f[,c(9:10)]
colnames(lung_1_m)
lung_1_m_imp = mice(lung_1_m, m=5, meth='norm.predict', maxit = 40, seed=2525)
lung_1_f_imp = mice(lung_1_f, m=5, meth='norm.predict', maxit = 40, seed=2525)
densityplot(lung_1_m_imp)
densityplot(lung_1_f_imp)
plot(lung_1_m_imp) 
plot(lung_1_f_imp)

# Imputation
lung_1_m_imp1 = complete(lung_1_m_imp, 1)
lung_1_m_imp2 = complete(lung_1_m_imp, 2)
lung_1_m_imp3 = complete(lung_1_m_imp, 3)
lung_1_m_imp4 = complete(lung_1_m_imp, 4)
lung_1_m_imp5 = complete(lung_1_m_imp, 5)
lung_1_f_imp1 = complete(lung_1_f_imp, 1)
lung_1_f_imp2 = complete(lung_1_f_imp, 2)
lung_1_f_imp3 = complete(lung_1_f_imp, 3)
lung_1_f_imp4 = complete(lung_1_f_imp, 4)
lung_1_f_imp5 = complete(lung_1_f_imp, 5)
lung_1_m_comp = (lung_1_m_imp1 + lung_1_m_imp2 + lung_1_m_imp3 + lung_1_m_imp4 + lung_1_m_imp5)/5
colnames(lung_1_m_comp)
colnames(bioage_m)
bioage_m[,c(9,10)] = lung_1_m_comp
lung_1_f_comp = (lung_1_f_imp1 + lung_1_f_imp2 + lung_1_f_imp3 + lung_1_f_imp4 + lung_1_f_imp5)/5
bioage_f[,c(9,10)] = lung_1_f_comp

saveRDS(bioage_m, file="bioage_m_2.rds")
saveRDS(bioage_f, file="bioage_f_2.rds")
bioage_m <- readRDS("bioage_m_2.rds")
bioage_f <- readRDS("bioage_f_2.rds")


# 키와 연관된 변수인 BMI, BMP, BFP로 FEV1FVC 예측
colnames(bioage_m)
lung_2_m = bioage_m[,c(3,6,7,8,11)]
lung_2_f = bioage_f[,c(3,6,7,8,11)]
colnames(lung_2_m)

lung_2_m_imp = mice(lung_2_m, m=5, meth='pmm', maxit = 40, seed=2525)
lung_2_f_imp = mice(lung_2_f, m=5, meth='pmm', maxit = 40, seed=2525)
densityplot(lung_2_m_imp)
densityplot(lung_2_f_imp)
plot(lung_2_m_imp) 
plot(lung_2_f_imp)

# Imputation
lung_2_m_imp1 = complete(lung_2_m_imp, 1)
lung_2_m_imp2 = complete(lung_2_m_imp, 2)
lung_2_m_imp3 = complete(lung_2_m_imp, 3)
lung_2_m_imp4 = complete(lung_2_m_imp, 4)
lung_2_m_imp5 = complete(lung_2_m_imp, 5)
lung_2_f_imp1 = complete(lung_2_f_imp, 1)
lung_2_f_imp2 = complete(lung_2_f_imp, 2)
lung_2_f_imp3 = complete(lung_2_f_imp, 3)
lung_2_f_imp4 = complete(lung_2_f_imp, 4)
lung_2_f_imp5 = complete(lung_2_f_imp, 5)

lung_2_m_comp = (lung_2_m_imp1 + lung_2_m_imp2 + lung_2_m_imp3 + lung_2_m_imp4 + lung_2_m_imp5)/5
bioage_m[,11] = lung_2_m_comp[,5]
lung_2_f_comp = (lung_2_f_imp1 + lung_2_f_imp2 + lung_2_f_imp3 + lung_2_f_imp4 + lung_2_f_imp5)/5
bioage_f[,11] = lung_2_f_comp[,5]


saveRDS(bioage_m, file="bioage_m_3.rds")
saveRDS(bioage_f, file="bioage_f_3.rds")
bioage_m <- readRDS("bioage_m_3.rds")
bioage_f <- readRDS("bioage_f_3.rds")
bioage


# 3. Blood

# 3-1. LDL = TC - HDL - (0.2TG) TG가 400이하에서 계산가능
colnames(bioage_m)
blood1_m = bioage_m[,c(28,29,30,31)]
blood1_f = bioage_f[,c(28,29,30,31)]
colnames(blood1_m)

# 400이하에서 missing value 3개 impute
sum(is.na(blood1_m[blood1_m[,3]<=400,]))
sum(is.na(blood1_m[,4]))
for (i in 1:nrow(bioage_m)){ 
  if (blood1_m[i,3]<=400){
    blood1_m[i,4] = blood1_m[i,1] - blood1_m[i,2] - 0.2*blood1_m[i,3]
  }
}
sum(is.na(blood1_f[blood1_f[,3]<=400,]))
sum(is.na(blood1_f[,4]))

bioage_m[,c(28,29,30,31)] = blood1_m
bioage_f[,c(28,29,30,31)] = blood1_f


saveRDS(bioage_m, file="bioage_m_4.rds")
saveRDS(bioage_f, file="bioage_f_4.rds")
bioage_m <- readRDS("bioage_m_4.rds")
bioage_f <- readRDS("bioage_f_4.rds")



# 3-1-1. TG >=400 이면 전부 NA
colnames(bioage_m)
blood1_m = bioage_m[,c(28,29,30,31)]
blood1_f = bioage_f[,c(28,29,30,31)]
colnames(blood1_m)

blood1_m_400 = blood1_m[blood1_m[,3]>400,]
dim(blood1_m_400)[1] == sum(is.na(blood1_m_400[,4])) # 434
blood1_f_400 = blood1_f[blood1_f[,3]>400,]
dim(blood1_f_400)[1] == sum(is.na(blood1_f_400[,4])) # 56


# 3-2. TIBC - IRON = UIBC
bioage2 <- read.csv("bioage_V1.csv", header=T, stringsAsFactors = T
                   , na.strings = c("NA","NaN",""," ","  ",
                                    "   ","    ","     ","      "))
colnames(bioage2)
blood2 = bioage2[,c(39,40,41)]
colnames(blood2)
md.pattern(blood2)

# 3-3. MCV = (HCT/RBC) * 10
colnames(bioage_m)
blood3_m = bioage_m[,c(33,34,36)]
blood3_f = bioage_f[,c(33,34,36)]
colnames(blood3_m)

md.pattern(blood3_m)
md.pattern(blood3_f)

RBC_m_i = which(is.na(blood3_m[,1]))
HCT_m_i = which(is.na(blood3_m[,2]))
MCV_m_i = which(is.na(blood3_m[,3]))

for (i in RBC_m_i){
  blood3_m[i,1] =blood3_m[i,2]/blood3_m[i,3]*10 
}

for (i in HCT_m_i){
  blood3_m[i,2] =blood3_m[i,1]*blood3_m[i,3]/10 
}

for (i in MCV_m_i){
  blood3_m[i,3] =blood3_m[i,2]*blood3_m[i,1]/10 
}

RBC_f_i = which(is.na(blood3_f[,1]))
HCT_f_i = which(is.na(blood3_f[,2]))
MCV_f_i = which(is.na(blood3_f[,3]))


for (i in RBC_f_i){
  blood3_f[i,1] =blood3_f[i,2]/blood3_f[i,3]*10 
}

for (i in HCT_f_i){
  blood3_f[i,2] =blood3_f[i,1]*blood3_f[i,3]/10 
}

for (i in MCV_f_i){
  blood3_f[i,3] =blood3_f[i,2]*blood3_f[i,1]/10 
}

bioage_m[,c(33,34,36)] = blood3_m
bioage_f[,c(33,34,36)] = blood3_f

saveRDS(bioage_m, file="bioage_m_5.rds")
saveRDS(bioage_f, file="bioage_f_5.rds")

bioage_m <- readRDS("bioage_m_5.rds")
bioage_f <- readRDS("bioage_f_5.rds")


# 3-4. 백혈구
colnames(bioage_m)
colnames(bioage_m[,40:44])
sum(is.na(bioage_m[,43]))
sum(is.na(bioage_m[,44]))

# 3-4-1. 백혈구끼리 Imputation --) pmm : Bad / mean : Good
blood4_m = bioage_m[,40:43] 
blood4_f = bioage_f[,40:43] 
head(blood4_m)
# blood4_m_imp = mice(blood4_m, m=5, meth='pmm', maxit = 40, seed=2000, visitSequence = "arabic")
blood4_m_imp = mice(blood4_m, m=5, meth='mean', maxit = 40, seed=2000, visitSequence = "arabic")
#blood4_f_imp = mice(blood4_f, m=5, meth='pmm', maxit = 40, seed=1000, visitSequence = "arabic")
blood4_f_imp = mice(blood4_f, m=5, meth='mean', maxit = 40, seed=1000, visitSequence = "arabic")
densityplot(blood4_m_imp)
densityplot(blood4_f_imp)

blood4_m_imp1 = complete(blood4_m_imp, 1)
blood4_m_imp2 = complete(blood4_m_imp, 2)
blood4_m_imp3 = complete(blood4_m_imp, 3)
blood4_m_imp4 = complete(blood4_m_imp, 4)
blood4_m_imp5 = complete(blood4_m_imp, 5)
blood4_f_imp1 = complete(blood4_f_imp, 1)
blood4_f_imp2 = complete(blood4_f_imp, 2)
blood4_f_imp3 = complete(blood4_f_imp, 3)
blood4_f_imp4 = complete(blood4_f_imp, 4)
blood4_f_imp5 = complete(blood4_f_imp, 5)
blood4_m_comp = (blood4_m_imp1 + blood4_m_imp2 + blood4_m_imp3 + blood4_m_imp4 + blood4_m_imp5)/5
bioage_m[,40:43] = blood4_m_comp
blood4_f_comp = (blood4_f_imp1 + blood4_f_imp2 + blood4_f_imp3 + blood4_f_imp4 + blood4_f_imp5)/5
bioage_f[,40:43] = blood4_f_comp

# 100 - sum = BASOPHIL
for (i in nrow(bioage_m)){
  bioage_m[i,44] = 100 - apply(FUN = sum, bioage_m[i,c(40:43)], MARGIN = 1)
}
# 100 - sum = BASOPHIL
for (i in nrow(bioage_f)){
  bioage_f[i,44] = 100 - apply(FUN = sum, bioage_f[i,c(40:43)], MARGIN = 1)
}

saveRDS(bioage_m, file="bioage_m_6.rds")
saveRDS(bioage_f, file="bioage_f_6.rds")

bioage_m <- readRDS("bioage_m_6.rds")
bioage_f <- readRDS("bioage_f_6.rds")
dim(bioage_m)[2] == dim(bioage_f)[2]

# 3-5. Blood Imputation.
colnames(bioage_m)
blood5_m = bioage_m[,c(3, 12:44)]
blood5_f = bioage_f[,c(3, 12:44)]
colnames(blood5_m)

md.pattern(blood5_m)
md.pattern(blood5_f)

# Missing pattern visualization
blood5_m_aggr = aggr(blood5_m, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                   labels=names(blood5_m), gap=3, 
                   ylab=c("Proportion of missingness","Missingness Pattern"))
blood5_f_aggr = aggr(blood5_f, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                   labels=names(blood5_m), gap=3, 
                   ylab=c("Proportion of missingness","Missingness Pattern"))

blood5_m_imp = mice(blood5_m, m=5, meth='pmm', maxit = 40, seed=2525)
blood5_f_imp = mice(blood5_f, m=5, meth='pmm', maxit = 40, seed=2525)

densityplot(blood5_m_imp)
densityplot(blood5_f_imp)

plot(blood5_m_imp) 
plot(blood5_f_imp)

# Imputation
blood5_m_imp1 = complete(blood5_m_imp, 1)
blood5_m_imp2 = complete(blood5_m_imp, 2)
blood5_m_imp3 = complete(blood5_m_imp, 3)
blood5_m_imp4 = complete(blood5_m_imp, 4)
blood5_m_imp5 = complete(blood5_m_imp, 5)
blood5_f_imp1 = complete(blood5_f_imp, 1)
blood5_f_imp2 = complete(blood5_f_imp, 2)
blood5_f_imp3 = complete(blood5_f_imp, 3)
blood5_f_imp4 = complete(blood5_f_imp, 4)
blood5_f_imp5 = complete(blood5_f_imp, 5)

blood5_m_comp = (blood5_m_imp1 + blood5_m_imp2 + blood5_m_imp3 + blood5_m_imp4 + blood5_m_imp5)/5

dim(blood5_m_comp[,2:ncol(blood5_m_comp)]) == dim(bioage_m[,12:44])
bioage_m[,12:44] = blood5_m_comp[,2:ncol(blood5_m_comp)]
blood5_f_comp = (blood5_f_imp1 + blood5_f_imp2 + blood5_f_imp3 + blood5_f_imp4 + blood5_f_imp5)/5
dim(blood5_f_comp[,2:ncol(blood5_f_comp)]) ==dim(bioage_f[,12:44])
bioage_f[,12:44] = blood5_f_comp[,2:ncol(blood5_f_comp)]

saveRDS(bioage_m, file="bioage_m_7.rds")
saveRDS(bioage_f, file="bioage_f_7.rds")

bioage_m <- readRDS("bioage_m_7.rds")
bioage_f <- readRDS("bioage_f_7.rds")
dim(bioage_m)[2] == dim(bioage_f)[2]

# blood5_f = bioage_f[,c(3, 12:44)]
# md.pattern(blood5_f)

# 4. thyroid gland = 2개
colnames(bioage_m)
gland_m = bioage_m[,c(3,45,46)]
gland_f = bioage_f[,c(3,45,46)]
colnames(gland_m)

cor(gland_m,use="pairwise.complete.obs")
cor(gland_m,use="complete.obs")


# 6. tumor maker
colnames(bioage_m)
tumor_m = bioage_m[,c(3,51:55)]
tumor_f = bioage_f[,c(3,51:55)]
colnames(tumor_m)
md.pattern(tumor_m)
md.pattern(tumor_f)

# Missing pattern visualization
tumor_m_aggr = aggr(tumor_m, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                     labels=names(tumor_m), gap=3, 
                     ylab=c("Proportion of missingness","Missingness Pattern"))
tumor_f_aggr = aggr(tumor_f, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                     labels=names(tumor_m), gap=3, 
                     ylab=c("Proportion of missingness","Missingness Pattern"))

# 6-1. Remove CA125 for man, PSA for woman
colnames(bioage_m)
bioage_m = bioage_m[,-53]
colnames(bioage_f)
bioage_f = bioage_f[,-55]

saveRDS(bioage_m, file="bioage_m_8.rds")
saveRDS(bioage_f, file="bioage_f_8.rds")
bioage_m <- readRDS("bioage_m_8.rds")
bioage_f <- readRDS("bioage_f_8.rds")

# 6-2. ALl Imputation
colnames(bioage_m)
tumor_m_2 = bioage_m[,c(3,51:54)]
colnames(tumor_m_2)

colnames(bioage_f)
tumor_f_2 = bioage_f[,c(3,51:54)]
colnames(tumor_f_2)

tumor_m_2_aggr = aggr(tumor_m_2, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                    labels=names(tumor_m_2), gap=3, 
                    ylab=c("Proportion of missingness","Missingness Pattern"))
tumor_f_2_aggr = aggr(tumor_f_2, col=mdc(1:2), numbers=TRUE, sortVars=TRUE,
                    labels=names(tumor_f_2), gap=3, 
                    ylab=c("Proportion of missingness","Missingness Pattern"))

# tumor_m_2_imp = mice(tumor_m_2, m=5, meth='pmm', maxit = 30, seed=2525)
tumor_m_2_imp = mice(tumor_m_2, m=5, meth='norm', maxit = 30, seed=2525)
tumor_f_2_imp = mice(tumor_f_2, m=5, meth='norm', maxit = 30, seed=2525)

densityplot(tumor_m_2_imp)
densityplot(tumor_f_2_imp)
plot(tumor_m_2_imp) 
plot(tumor_f_2_imp)

# Imputation
tumor_m_2_imp1 = complete(tumor_m_2_imp, 1)
tumor_m_2_imp2 = complete(tumor_m_2_imp, 2)
tumor_m_2_imp3 = complete(tumor_m_2_imp, 3)
tumor_m_2_imp4 = complete(tumor_m_2_imp, 4)
tumor_m_2_imp5 = complete(tumor_m_2_imp, 5)
tumor_f_2_imp1 = complete(tumor_f_2_imp, 1)
tumor_f_2_imp2 = complete(tumor_f_2_imp, 2)
tumor_f_2_imp3 = complete(tumor_f_2_imp, 3)
tumor_f_2_imp4 = complete(tumor_f_2_imp, 4)
tumor_f_2_imp5 = complete(tumor_f_2_imp, 5)

tumor_m_2_comp = (tumor_m_2_imp1 + tumor_m_2_imp2 + tumor_m_2_imp3 + tumor_m_2_imp4 + tumor_m_2_imp5)/5
dim(bioage_m[,c(3,51:54)]) == dim(tumor_m_2_comp)
bioage_m[,c(3,51:54)] = tumor_m_2_comp

tumor_f_2_comp = (tumor_f_2_imp1 + tumor_f_2_imp2 + tumor_f_2_imp3 + tumor_f_2_imp4 + tumor_f_2_imp5)/5
dim(bioage_f[,c(3,51:54)]) == dim(tumor_f_2_comp)
bioage_f[,c(3,51:54)] = tumor_f_2_comp

saveRDS(bioage_m, file="bioage_m_9.rds")
saveRDS(bioage_f, file="bioage_f_9.rds")
bioage_m <- readRDS("bioage_m_9.rds")
bioage_f <- readRDS("bioage_f_9.rds")


# 5. Immune serum : All variables are categorical

# 5-2. Erase "HBSAB"
catcorrm <- function(vars, dat) sapply(vars, function(y) sapply(vars, function(x) assocstats(table(dat[,x], dat[,y]))$cramer))
catcorrm(colnames(bioage_m)[47:50], bioage_m)

colnames(bioage_m)
bioage_m = bioage_m[,-48]
bioage_f = bioage_f[,-48]

saveRDS(bioage_m, file="bioage_m_10.rds")
saveRDS(bioage_f, file="bioage_f_10.rds")
bioage_m <- readRDS("bioage_m_10.rds")
bioage_f <- readRDS("bioage_f_10.rds")

# Impute the remains.
# Step1. Find Correlation matrix : The result is a matrix of Cramer's V's.
catcorrm <- function(vars, dat) sapply(vars, function(y) sapply(vars, function(x) assocstats(table(dat[,x], dat[,y]))$cramer))
catcorrm(colnames(bioage_m)[55:62], bioage_m)
catcorrm(colnames(bioage_f)[55:62], bioage_f)

# Step2. Imputation
colnames(bioage_m)
bioage_m_var = bioage_m[,3:ncol(bioage_m)]
bioage_f_var = bioage_f[,3:ncol(bioage_f)]

method = c("","","","","","","","","",
           "","","","","","","","","",
           "","","","","","","","","",
           "","","","","","","","","",
           "","","","","","","pmm","pmm","polr",
           "polr","polr","pmm","pmm","pmm","pmm","polr","pmm","polr",
           "polr","polr","polr","polr","polr","pmm")

md.pattern(bioage_m_var)
md.pattern(bioage_f_var)

# Missing pattern visualization
bioage_m_aggr = aggr(bioage_m_var, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                     labels=names(bioage_m_var), gap=3, 
                     ylab=c("Proportion of missingness","Missingness Pattern"))
bioage_f_aggr = aggr(bioage_f_var, col=mdc(1:2), numbers=TRUE, sortVars=TRUE, 
                     labels=names(bioage_f_Var), gap=3, 
                     ylab=c("Proportion of missingness","Missingness Pattern"))

bioage_m_imp = mice(bioage_m_var, m=1, meth=method, maxit = 20, seed=1000)
bioage_f_imp = mice(bioage_f_var, m=1, meth=method, maxit = 20, seed=2525)

densityplot(bioage_m_imp)
densityplot(bioage_f_imp)
plot(bioage_m_imp) 
plot(bioage_f_imp)


# Imputation
dim(bioage_m[,3:ncol(bioage_m)]) == dim(complete(bioage_m_imp, 1))
dim(bioage_f[,3:ncol(bioage_f)]) == dim(complete(bioage_f_imp, 1))

bioage_m[,3:ncol(bioage_m)] = complete(bioage_m_imp, 1)
bioage_f[,3:ncol(bioage_f)] = complete(bioage_f_imp, 1)

saveRDS(bioage_m, file="bioage_m_imp.rds")
saveRDS(bioage_f, file="bioage_f_imp.rds")

bioage_m <- readRDS("bioage_m_imp.rds")
bioage_f <- readRDS("bioage_f_imp.rds")

md.pattern(bioage_f)


write.csv(file = "bioage_m_imputationR.csv", x = bioage_m)
write.csv(file = "bioage_f_R.csv", x = bioage_f)

head(bioage_f)
colnames(bioage_m)
