# Model :  Y[k] = X[k] ( B[k] + S[k] ) + W[k] for all k=1,2,...,r

# Solves : min_{B,S} ||Y-X(B+S)||^2 + lambda1 ||B||_{1,\infty} + lambda2 ||S||_{1,1}

# Y[k]      : n vector
# X[k]      : nxp matrix
# B[k],C[k] : p vectors
# W[k]      : n vector





# B,S are the initial points
# epsilon is the stopping criterion (~10^-10)


L1_Linf_LASSO_Solver <- function(Y, X, lambda1, lambda2, B, S, epsilon)
{
  #
  K <- dim(X)[3] # Number of tasks
  p <- dim(X)[2] # Number of features
  n <- dim(X)[1] # Number of samples in each task
  
  
  
  #create temp var c, d, alph, BS_old#
  c <- matrix(0,p,K);
  d <- array(0,dim=c(p,p,K));
  alph <- matrix(0,p,K);
  sum_old <- 1e14;
  
  #initialization#
  for (j in 1:p)
    for (k in 1:K)
    {
      c[j,k]<-t(Y[,k]) %*% X[,j,k];
      for (i in 1:p)
        d[i,j,k]<-t(X[,i,k]) %*% X[,j,k];
    }
  
  #main loop#
  ITR = 0;
  repeat
  {
    ITR <- ITR +1;
    
    for (j in 1:p)
    {
      for (k in 1:K)
      {
        temp <- 0;
        for (i in 1:p)
          if (i!=j)
            temp <- temp + (B[i,k]+S[i,k])*d[i,j,k];
          alph[j,k] <- (c[j,k] - temp - B[j,k]*d[j,j,k])/d[j,j,k];
          
          if (abs(alph[j,k]) <= lambda2[1,j])
            S[j,k] <- 0
          else
            S[j,k] <- alph[j,k]-lambda2[1,j]*sign(alph[j,k])/sqrt(ITR);
      }
    }
    for (j in 1:p)
    {
      for (k in 1:K)
      {
        temp <- 0;
        for (i in 1:p)
          if (i!=j)
            temp <- temp + (B[i,k]+S[i,k])*d[i,j,k];
          alph[j,k] <- (c[j,k] - temp - S[j,k]*d[j,j,k])/d[j,j,k];
      }
      if (sum(abs(alph[j,])) <= lambda1[1,j]) 
      {
        B[j,]<- 0;
      }
      else
      {
        Ordalpha<-sort(abs(alph[j,]),decreasing=TRUE);
        mx<-0;
        alphasum<-0;
        mstar<-0;
        for (m in 1:K)
        {
          alphasum<-alphasum+Ordalpha[m];
          if((alphasum-lambda1[1,j])/m>mx)
          {
            mx=(alphasum-lambda1[1,j])/m;
            mstar=m;
          }
        }
        pivot<-Ordalpha[mstar];
        alphasum<-0;
        for (m in 1:mstar)
          alphasum<-alphasum+Ordalpha[m];
        for (k in 1:K)
        {
          if (abs(alph[j,k])<pivot)
            B[j,k] <- alph[j,k]
          else
            B[j,k] <- sign(alph[j,k])*(alphasum-lambda1[1,j]/sqrt(ITR))/mstar;		 		 
        }
      }
    }
    
    
    sum1 <- 0;
    for (k in 1:K)
      sum1 <- sum1 + norm(Y[,k]-X[,,k] %*% (B[,k]+S[,k]),'F')^2;
    for (j in 1:p)
    {
      sum1 <- sum1 + lambda1[1,j]*max(abs(B[j,]));
      sum1 <- sum1 + lambda2[1,j]*sum(abs(S[j,]));
    }
    print(abs(1-sum1/sum_old)/(n*p*K))
    if (abs(sum_old-sum1)<n*p*K*epsilon*sum_old) break;
    sum_old <- sum1;
    
  }
  print(ITR);
  L1_Linf_LASSO_Solver <- list(B,S);
}





