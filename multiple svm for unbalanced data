#multiple svm for unbalanced data, predicting transanction Santander Bank competition 

setwd("Documents/")

library(e1071)
library(caret)

normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

x <- read.csv("train.csv")


pos<-x[x$target==1,]
neg<-x[x$target==0,]

neg<-neg[sample(1:nrow(neg),nrow(neg)),]

nn<-nrow(pos)*3

neg1<-neg[1:nn,]
neg2<-neg[nn+1:nn+nn+1,]
neg3<-neg[nn+nn+2:nn+nn+2+nn,]

traindata1<- rbind(pos,neg1)
traindata2<- rbind(pos,neg2)
traindata3<- rbind(pos,neg3)

b1<-lapply(traindata1[,-1], normalize)
b2<-lapply(traindata2[,-1], normalize)
b3<-lapply(traindata3[,-1], normalize)


b1<-as.data.frame(b1)
b2<-as.data.frame(b2)
b3<-as.data.frame(b3)


obj1 <- svm(target~., data = b1)

obj2 <-svm(target~., data = b2)

obj3 <- svm(target~., data = b3)

td=read.csv("test.csv")

td2<-lapply(td[,-1],normalize)

td2<-as.data.frame(td2)

pf1<- predict(obj1,td2)
pf2<- predict(obj2,td2)
pf3<- predict(obj3,td2)

mo<-mean(pf1,pf2,pf3)

pf<-data.frame(td$ID_code,mo)
colnames(pf)=c("ID_code", "target")

write.csv(pf,"mean1.csv")



