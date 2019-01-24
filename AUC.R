library(ggplot2)

folder <- character(4)
folder[1] <- "corr_group"
folder[2] <- "corr_group_pool"
folder[3] <- "noco_group"
folder[4] <- "noco_group_pool"

for (j in 3:3) {
  
  pic_name <- paste(folder[j], "_auc", ".png", sep = "")
  
  noco_mse_df <- 
    read.table(paste("VSSC/",folder[j],"/auc.txt",sep = ""))
  
  SNP_name<-read.table("VSSC/name.txt")
  colnames(noco_mse_df)[1] <- "SNP_Sparsity"
  colnames(noco_mse_df)[2] <- "Algorithms"
  colnames(noco_mse_df)[3] <- "Task"
  colnames(noco_mse_df)[4] <- "AUC"
  
  noco_mse_df$Algorithms[noco_mse_df$Algorithms==1] <- 'MSS'
  noco_mse_df$Algorithms[noco_mse_df$Algorithms==2] <- 'RSS'
  noco_mse_df$Algorithms[noco_mse_df$Algorithms==3] <- 'Lasso'
  
  for (i in 1:9) {
    noco_mse_df$SNP_Sparsity[noco_mse_df$SNP_Sparsity==i] <- 
      as.character(SNP_name$V1[i])
  }
  
  noco_mse_df$Task[noco_mse_df$Task==1]='1'
  noco_mse_df$Task[noco_mse_df$Task==2]='2'
  noco_mse_df$Task[noco_mse_df$Task==3]='3'
  noco_mse_df$Task[noco_mse_df$Task==4]='4'
  noco_mse_df$Task[noco_mse_df$Task==5]='5'
  
  ggplot(noco_mse_df,aes(Algorithms,AUC,fill=Task))+
    geom_boxplot()+facet_wrap(~SNP_Sparsity,ncol=3)+
    theme(legend.position="bottom",
          axis.title = element_text(size=20),
          text = element_text(size=20))
  
  ggsave(pic_name, plot = last_plot(), device = NULL, path = "VSSC/",
         scale = 1, width = 32, height = 24, units = "cm",
         dpi = 600, limitsize = TRUE)
}
