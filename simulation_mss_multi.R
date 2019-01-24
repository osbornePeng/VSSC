## The mse and auc of dirty model and data shared lasso

rm(list = ls())
library("glmnet")
library("mvtnorm")
library("reshape2")
library("ggplot2")
source('~/R/VSSC/L1_Linf_LASSO_Solver.R')
source('~/R/ROC_curve.R')
fold <- 1
J <- 5
d <- 500
n <- 500
signal <- 0.5#c(2, 1, 0.5)
sparsity <- 100#c(100, 20, 10)
type <- list()
w <- matrix(0, d, J)
coe_w_share = matrix(0, fold, J)
coe_w_j = matrix(0, fold, J)
coe_error = matrix(0, fold, J)
dirty <- list()
auc1 <- mse1 <- list()
# number of cv in dirty model
nc <- 1
nratio <- 1

for (s in 1: (nc * nratio)) {
  auc1[[s]] <- mse1[[s]] <- matrix(0, fold, J)
}
mse2 <- auc2 <- matrix(0, fold, J)
lam <- matrix(0, nc * nratio, 2)

mse_all <- list()
auc_all <- list()

## With correlation
# mu_simu = rep(0, d);
# sigma_simu = diag(rep(1, d));
# for (i1 in 1: (d - 1)) {
#   for (i2 in (i1 + 1): d) {
#     sigma_simu[i1, i2]=0.5^(abs(i1 - i2))
#   }
# }
# sigma_simu <- sigma_simu + t(sigma_simu)
# diag(sigma_simu) <- rep(1, d)
##
ss <- 0 # SNP and sparse label
for (r1 in 1: length(signal)) {
  for (c1 in 1: length(sparsity)) {
    ss <- ss + 1
    mse <- array(0, dim = c(fold, J))
    for (p in 1: fold) {
      type <- list()
      w_share <- rnorm(d)
      sum_n <- 0
      
      for (j in 1: J) {
        type_j <- list()
        #n <- round((0.3+(j-1)*0.2)*d)
        
        ## No correlation
        trainset_x=matrix(rnorm(n * d), n, d)
        
        ## With correlation
        # trainset_x=rmvnorm(n, mu_simu,sigma_simu)
        ##
        
        w[, j] = rnorm(d)
        
        ## Consider the sparsity for group coefficient
        if (sparsity[c1] > 1) {
          repeat {
            sparsity_index2 = sample(c(0, 1), d, T, c(1-1/sparsity[c1], 1/sparsity[c1]))
            if (sum(sparsity_index2)>0) {
              break
            }
          }
          w[sparsity_index2 == 0, j] = 0
        }
        
        ## Adjust the variance between each part
        coe_w_share[p, j] = var(trainset_x %*% w_share)
        coe_w_j[p, j] = var(trainset_x %*% w[, j])
        # weight
        weight_j = sqrt(coe_w_share[p, j] / coe_w_j[p, j])
        w[, j] = sqrt(1) * weight_j %*% w[, j] #Adjust the w
        coe_w_j[p, j] = var(trainset_x %*% w[, j])
        #
        error=rnorm(trainset_x[, 1])
        #w_share=w_share*0;%%%%
        weight_error=c(sqrt(var(trainset_x %*% (w[, j] + w_share)) / var(error)))
        #
        trainset_y=(trainset_x %*% (w[, j] + w_share) + sqrt(signal[r1]) * error * weight_error)
        coe_error[p, j]=var(trainset_y - trainset_x %*% (w_share + w[, j]))
        
        type_j <- list(trainset_x = trainset_x, 
                       trainset_y = trainset_y, n = n)
        type[[j]] <- type_j
        
        sum_n <- sum_n + n
      }
      
      w_mean <- t(matrix(rep(colMeans(matrix(rep(w_share,J), d)+w), d), J))
      w_var <- apply(((matrix(rep(w_share,J), d)+w) - w_mean)^2,2,mean)
      
      ## Dirty model
      
      x <- array(0, dim = c(n, d, J))
      y <- array(0, dim = c(n, J))
      s <- 0
      
      for (j in 1:J) {
        x[, , j] <- type[[j]]$trainset_x
        y[, j] <- type[[j]]$trainset_y
      }
      K <- dim(x)[3] # Number of tasks
      d <- dim(x)[2] # Number of features
      n <- dim(x)[1] # Number of samples in each task
      
      if (ss > 0) {
        for (c in seq(0.01, 100, length.out = nc)) {
          lambda1 <- c*sqrt(K*log(d)/n)
          for (ratio in seq(1/K, 1, length.out = nratio)) {
            lambda2 <- lambda1 * ratio
            s <- s + 1
            
            B <- S <- array(0, dim = c(d, J))
            lambda11 <- t(rep(lambda1, d))
            lambda22 <- t(rep(lambda2, d))
            
            lam[s, 1] <- lambda1
            lam[s, 2] <- lambda2
            
            dirty <- proc.time()
            dirty_w <- L1_Linf_LASSO_Solver(y, x, lambda11, lambda22, B, S, 10^-9)
            t1 <- proc.time() - dirty
            dirty_w <- as.matrix(dirty_w[[1]] + dirty_w[[2]])
            mse1[[s]][p, ] <- colMeans((dirty_w - (matrix(rep(w_share,J), d)+w))^2)/w_var
            
            for (j in 1:J) {
              x1 <- rep(1, d)
              x1[which(w[, j]==0)] <- 0
              auc1[[s]][p, j] <- ROC_curve(dirty_w[, j], x1)
            }
          }
        }
      } else {
        # B <- S <- array(0, dim = c(d, J))
        # lambda11 <- t(rep(lam[num, 1], d))
        # lambda22 <- t(rep(lam[num, 2], d))
        # 
        # dirty_w <- L1_Linf_LASSO_Solver(y, x, lambda11, lambda22, B, S, 10^-9)
        # dirty_w <- as.matrix(dirty_w[[1]] + dirty_w[[2]])
        # mse1[p, ] <- colMeans((dirty_w - (matrix(rep(w_share,J), d)+w))^2)/w_var
        # 
        # for (j in 1:J) {
        #   x1 <- rep(1, d)
        #   x1[which(w[, j]==0)] <- 0
        #   auc1[p, j] <- ROC_curve(dirty_w[, j], x1)
        # }
      }
      
      
      
      
      ## Data shared lasso
      
      rlabel <- 1
      clabel <-1
      z1 <- array(0, c(sum_n, d))
      z2 <- array(0, c(sum_n, d * J))
      y<- rep(0, sum_n)
      
      # The Z matrix
      for (j in 1 : J) {
        
        z1[rlabel : (rlabel + type[[j]]$n-1), ] <-
          type[[j]]$trainset_x
        z2[rlabel : (rlabel + type[[j]]$n-1), clabel : (clabel+d-1)] <-
          type[[j]]$trainset_x
        y[rlabel : (rlabel + type[[j]]$n-1)] <-
          type[[j]]$trainset_y
        
        rlabel <- type[[j]]$n + rlabel
        clabel <- d + clabel
        
      }
      z2 <- z2*1/J^(0.5) #choosing r_g
      z <- cbind(z1,z2)
      DSL <- proc.time()
      cvfit = cv.glmnet(z, y)
      t2 <- proc.time() - DSL
      w_datashare <- as.matrix(coef(cvfit, s = "lambda.min"))
      w_datashare <- w_datashare[-1] # Get rid of the intercept
      w_datashare_share <- rep(w_datashare[1 : d], J+1)
      w_datashare <- matrix(w_datashare * 1/J^(0.5) + w_datashare_share, d)
      w_datashare <- w_datashare[, -1]
      
      mse2[p, ] <- apply((w_datashare-(matrix(rep(w_share,J), d)+w))^2,2,mean)/w_var
      
      for (j in 1:J) {
        x1 <- rep(1, d)
        x1[which(w[, j]==0)] <- 0
        auc2[p, j] <- ROC_curve(w_datashare[, j], x1)
      }
      
      
    }
    
    # Dirty model
    
    if (ss > 0) {
      choose <- colMeans((sapply(mse1, colMeans)))
      num <- which(choose == min(choose))
      
      mse11 <- mse1[[num[1]]]
      auc11 <- auc1[[num[1]]]
    } 
    
    #
    
    Algorithms <- rep('Dirty_model', fold)
    SNP_Sparsity <- rep(ss, fold)
    
    mse11 <- data.frame(mse11, Algorithms, SNP_Sparsity)
    auc11 <- data.frame(auc11, Algorithms, SNP_Sparsity)
    
    colnames(mse11)[1:J] <- colnames(auc11)[1:J] <- 1:J
    
    # Data shared Lasso
    
    Algorithms <- rep('DSL', fold)
    SNP_Sparsity <- rep(ss, fold)
    
    mse2 <- data.frame(mse2, Algorithms, SNP_Sparsity)
    auc2 <- data.frame(auc2, Algorithms, SNP_Sparsity)
    
    colnames(mse2)[1:J] <- colnames(auc2)[1:J] <- 1:J
    
    mse_all <- rbind(mse_all, mse11, mse2)
    auc_all <- rbind(auc_all, auc11, auc2)
    
    mse11 <- mse2 <- auc11 <- auc2 <- matrix(0, fold, J)
    
  }
}

mse_plot <- melt(mse_all, id.vars = c("Algorithms", "SNP_Sparsity"), measure.vars = c("1", "2", "3", "4", "5"), variable.name = "Task", value.name = "Mean_Squared_Error")
auc_plot <- melt(auc_all, id.vars = c("Algorithms", "SNP_Sparsity"), measure.vars = c("1", "2", "3", "4", "5"), variable.name = "Task", value.name = "AUC")

ggplot(mse_plot,aes(Algorithms,Mean_Squared_Error))+
  geom_boxplot(aes(fill=Task))+facet_wrap(~SNP_Sparsity,ncol=3)+
  theme(legend.position="bottom",
        axis.title = element_text(size=20),
        text = element_text(size=20))

write.table(mse_plot,"VSSC/mse_plot_poco.txt",quote = FALSE,row.names = FALSE,
            col.names = FALSE)
write.table(auc_plot,"VSSC/auc_plot_poco.txt",quote = FALSE,row.names = FALSE,
            col.names = FALSE)