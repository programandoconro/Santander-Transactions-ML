library(keras)
#install_keras()
library(ROSE)
library(caret)
setwd("Documents/")

x <- read.csv("train.csv")


normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

pos=x[x$target==1,-1];neg=x[x$target==0,-1]

set.seed(123)

epos=sample(1:nrow(pos),floor(0.7*nrow(pos)))
post=pos[epos,]
poscv=pos[-epos,]
eneg=sample(1:nrow(neg),floor(0.7*nrow(neg)))
negt=neg[eneg,]
negcv=neg[-eneg,]

tr=rbind(post,negt)
tr=lapply(tr, normalize)
tr=as.data.frame(tr)

cv=rbind(poscv,negcv)
cv=lapply(tr, normalize)
cv=as.data.frame(cv)

#dn<- ROSE(target ~ . , data = tr, N=1000000,p=0.1)$data

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = ncol(tr[,-1]), activation = 'relu', input_shape = ncol(tr[,-1]), kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
  layer_dropout(rate = 0.0) %>% 
  layer_dense(units = 150, activation = 'relu', kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.95) %>%
  layer_dense(units = 1, activation = 'sigmoid')


   
history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr=0.001),
  metrics = c('accuracy')
)


model %>% fit(
  as.matrix(tr[,-1],dimname=NULL), as.numeric(tr$target), 
  epochs =3, 
  batch_size = 200,
  validation_split = 0.0
  )


r=predict(model,as.matrix(cv[,-1],dimname=NULL))

a=roc.curve(cv$target, r)
a


model2 <- keras_model_sequential() 

model2 %>% 
  layer_dense(units = ncol(tr[,-1]), activation = 'relu', input_shape = ncol(tr[,-1]), kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
  layer_dropout(rate = 0.0) %>% 
  layer_dense(units = 150, activation = 'relu', kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.95) %>%
  layer_dense(units = 1, activation = 'sigmoid')



history <- model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


model2 %>% fit(
  as.matrix(tr[,-1],dimname=NULL), as.numeric(tr$target), 
  epochs =50, 
  batch_size = 200,
  validation_split = 0.0)


r2=predict(model2,as.matrix(cv[,-1],dimname=NULL))

roc.curve(cv$target, r2)

rt=(r+r2)/2


roc.curve(cv$target, rt)




#td=read.csv("test.csv")

td1<-lapply(td[,-1],normalize)
td1=as.data.frame(td1)


pp=predict(model,as.matrix(td1,dimname=NULL))

pf2=data.frame(td$ID_code,pp)
colnames(pf2)=c("ID_code", "target")

write.csv(pf2,"cayena5.csv",row.names=F)


##########
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(tr[,-1])) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 36, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )


history <- fit(
  object           = model_keras, 
  x                = as.matrix(tr[,-1]), 
  y                = tr$target,
  batch_size       = 50, 
  epochs           = 3,
  validation_split = 0.00
)

r=predict(history,as.matrix(cv[,-1],dimname=NULL))

roc.curve(cv$target, r)


