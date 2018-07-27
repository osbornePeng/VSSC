%% This code do the parameters estimation test for spile-slab share and spike-slab, Lasso etc separately for simulation.
%% Each group is learned separately for Spike-slab, Lasso and Ridge
%% this is for a linux server
%function cross_validation_of_w_share(trainset_x,trainset_y)
clc
clear
close all
path(path,'/home/daviddai/Penghao/glmnet_matlab/glmnet_matlab')%The path of glmnet
%%
% h1=figure('name','Within algorithm');
% h2=figure('name','Between algorithm');
% h3=figure('name','Error ratio');
% h4=figure('name','AUC');
% h5=figure('name','w_0');
% h6=figure('name','w_j');
% h7=figure('name','w_j spike-slab');
%%
fold=20;%%%%%%%%%%%%%%%%%%%%%%
J=3;%The genre
signal=[2,1,0.5];
sparsity=[100,20,10];%There are totally 1/sparsity items are nonzero
mse1=zeros(fold,J);%Spike-slab share
mse2=zeros(fold,J);
mse3=zeros(fold,J);
auc1=zeros(fold,J);
auc2=zeros(fold,J);
auc3=zeros(fold,J);
% mse2=zeros(fold,J);
% mse3=zeros(fold,J);
% mse4=zeros(fold,J);
% sizen=zeros(fold,J);
%figure
for r1=1:length(signal)
    for c1=1:length(sparsity)
        for p=1:fold
            %         trainset_x=randn(n,d);
            %         w=randn(d,1);
            %% simulation
            clear type
            d=1000;%The number of dimention
            %% share coefficient
            %weight_w_share=sqrt(1/sparsity(c1));
            w_share=randn(d,1);
            %
            %             if sparsity(c1)>1
            %                 sparsity_index1=cvpartition(d,'kfold',sparsity(c1));
            %
            %                 w_share(sparsity_index1.training(1))=0;
            %
            %             end
            %
            for j=1:J
                %% N samples %%
                %type(j).n=round(0.1*d*(2^(j-1)));%The number of samples
                %type(j).n=0.3*d;
                type(j).n=round((0.3+(j-1)*0.2)*d);               
               %% No correlation
 %               type(j).trainset_x=randn(type(j).n,d);
               %% With correlation
                                mu_simu=zeros(d,1);
                                sigma_simu=diag(ones(d,1));
                                for i=1:d-1
                                    sigma_simu=sigma_simu+diag(repmat(0.5^i,d-i,1),i);
                                    sigma_simu=sigma_simu+diag(repmat(0.5^i,d-i,1),-i);
                                end
                                type(j).trainset_x=mvnrnd(mu_simu,sigma_simu,type(j).n);            
                %%
                w(:,j)=randn(d,1);
                %% Consider the sparsity for group coefficient
                if sparsity(c1)>1
                    
                    sparsity_index2=cvpartition(d,'kfold',sparsity(c1));
                    
                    w(sparsity_index2.training(1),j)=0;
                end
                %%
                %type(j).trainset_y=full(type(j).trainset_x*(w(:,j)+w_share)+sqrt(signal(r1))*randn(size(type(j).trainset_x(:,1)))*std(type(j).trainset_x*(w(:,j)+w_share)));
                %type(j).trainset_y=full(type(j).trainset_x*(w(:,j)+w_share)+sqrt(signal(r1))*randn(size(type(j).trainset_x(:,1)))*(d*weight_w_share^2+d/sparsity(c1)));
                %type(j).trainset_x=int16(type(j).trainset_x);
                %% Adjust the variance between each part
                coe_w_share(p,j)=var(type(j).trainset_x*w_share);
                coe_w_j(p,j)=var(type(j).trainset_x*w(:,j));
                % weight
                weight_j=sqrt(coe_w_share(p,j)/coe_w_j(p,j));
                w(:,j)=sqrt(1)*weight_j*w(:,j);%Adjust the w
                coe_w_j(p,j)=var(type(j).trainset_x*w(:,j));
                %
                error=randn(size(type(j).trainset_x(:,1)));
                %w_share=w_share*0;%%%%
                weight_error=sqrt(var(type(j).trainset_x*(w(:,j)+w_share))/var(error));
                type(j).trainset_y=full(type(j).trainset_x*(w(:,j)+w_share)+sqrt(signal(r1))*error*weight_error);
                coe_error(p,j)=var(type(j).trainset_y-type(j).trainset_x*(w_share+w(:,j)));
            end
            
            %% learning
            %             for j=1:J
            %                 cp(j).cvinfor=cvpartition(length(type(j).trainset_x(:,1)),'KFold',fold);%defining a random partition for k-fold cross validation
            %             end
            % Preparing to training
            %for p=1:fold%%%%%%%%%%%%%
            for j=1:J
                % Trainset
                %                     type(j).sub_trainset_x=type(j).trainset_x(cp(j).cvinfor.training(p),:);%Trainset
                %                     type(j).sub_trainset_y=type(j).trainset_y(cp(j).cvinfor.training(p),:);
                %                     % Testset
                %                     type(j).testset_x=type(j).trainset_x(cp(j).cvinfor.test(p),:);%test set
                %                     type(j).testset_y=type(j).trainset_y(cp(j).cvinfor.test(p),:);
                %                     % This is for Spike-Slab Share
                %                     type2(j).trainset_x=type(j).sub_trainset_x;
                %                     type2(j).trainset_y=type(j).sub_trainset_y;
                %                     %
                %                     sizen(p,j)=cp(j).cvinfor.TestSize(p);%The size of the j th part of the p th trainset
                
                %% training for normal spike_slab
                [sigma_e,sigma_b,a_p,a_2,mu_2,s_b,l_bound_2]=spike_slab(type(j).trainset_x,type(j).trainset_y);
                ass_2(:,j)=a_2;
                muss_2(:,j)=mu_2;
                w2=a_2.*mu_2;
                % mse
                mse2(p,j)=mean((w2-(w_share+w(:,j))).^2)/var((w_share+w(:,j)),1);
                % AUC
                x1=ones(size(a_2));
                x1(w(:,j)==0)=0;
                auc2(p,j)=ROC_curve(a_2,x1);
                
                %% training for ridge regression
                %                 [mn_3,id_likelihood_or_lb,alpha,beta]=ridge_regression_EM(type(j).trainset_x,type(j).trainset_y);
                %                 w3=mn_3;
                %                 %mse
                %                 mse3(p,j)=mean((w3-(w_share+w(:,j))).^2)/var((w_share+w(:,j)),1);
                % glmnet
                options=struct('alpha',0);
                tic
                fitinfo=cvglmnet(type(j).trainset_x,type(j).trainset_y,[],options);
                w3=fitinfo.glmnet_fit.beta(:,(fitinfo.lambda==fitinfo.lambda_min));
                mse3(p,j)=mean((w3-(w_share+w(:,j))).^2)/var((w_share+w(:,j)),1);
                toc
                
                %% training for Lasso
                'Lasso...'
                tic
                fitinfo=cvglmnet(type(j).trainset_x,type(j).trainset_y);
                w4=fitinfo.glmnet_fit.beta(:,(fitinfo.lambda==fitinfo.lambda_min));
                mse4(p,j)=mean((w4-(w_share+w(:,j))).^2)/var((w_share+w(:,j)),1);
                toc
                % AUC
                x1=ones(size(w4));
                x1(w(:,j)==0)=0;
                auc3(p,j)=ROC_curve(w4,x1);
            end
            
            %% spike-slab share
            [a,mu,mu_0,l_bound]=spike_slab_share(type);
            for j=1:J
                w1=a(:,j).*mu(:,j)+mu_0;
                %mse
                mse1(p,j)=mean((w1-(w_share+w(:,j))).^2)/var((w_share+w(:,j)),1);
                %,ones(cp(j).cvinfor.TestSize(p),1)
                %
                x1=ones(size(a(:,j)));
                x1(w(:,j)==0)=0;
                auc1(p,j)=ROC_curve(a(:,j),x1);
            end
            error_of_w_share(p)=var(w_share-mu_0)/var(w_share);
        end
        
        %% ploting
        %         figure(h1)
        %         label1=repmat({'Multi-task','Spike-slab','Ridge','Lasso'},J,1);%Major group label
        %         label1=label1(:)';
        %         %label2=repmat({['G1, ',num2str(type(1).n)],['G2, ',num2str(type(2).n)],['G3, ',num2str(type(3).n)]},1,3);%Minor group label
        %         label2=repmat({'G1','G2','G3'},1,4);
        %         labelall={label1,label2};
        %         set(gcf,'outerposition',get(0,'screensize'));
        %         subplot(length(signal),length(sparsity),(r1-1)*length(sparsity)+c1);
        %         boxplot([mse1,mse2,mse3,mse4],labelall,'colors',repmat('rgb',1,3),'Factorseparator',[1],'Labelverbosity','minor');%'Labels',{'Ridge','VB','Spike-slab','OLS'},
        %         title(['Noise:',num2str(signal(r1)),', Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         grid on
        %         grid minor
        %         %% Overall mse
        %         % weight the mse for each algorithm by group number
        %         %samplen=[type(1).n;type(2).n;type(3).n];
        %         mse1_ave=mse1(:);%*samplen/sum(samplen);
        %         mse2_ave=mse2(:);%*samplen/sum(samplen);
        %         mse3_ave=mse3(:);%*samplen/sum(samplen);
        %         mse4_ave=mse4(:);
        %         %
        %         figure(h2)
        %         set(gcf,'outerposition',get(0,'screensize'));
        %         subplot(length(signal),length(sparsity),(r1-1)*length(sparsity)+c1);
        %         boxplot([mse1_ave(:),mse2_ave(:),mse3_ave(:),mse4_ave(:)],'Labels',{'Multi-task','Spike-slab','Ridge','Lasso'},'colors','rbm');
        %         %boxplot([mse1',mse2',mse3'],'Labels',{'Ridge','VB','Spike-slab'},'colors','rbm')%No OLS
        %         title(['Noise:',num2str(signal(r1)),', Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         %         ylabel(['Noise:',num2str(signal(r1))]);
        %         %         xlabel(['Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         grid on
        %         grid minor
        %         %% Error
        %         figure(h3)
        %         set(gcf,'outerposition',get(0,'screensize'));
        %         label1=repmat({'Share','j','Error'},3,1);%Major group label
        %         label1=label1(:)';
        %         label2=repmat({'G1','G2','G3'},1,3);%Minor group label
        %         labelall={label2,label1};
        %         set(gcf,'outerposition',get(0,'screensize'));
        %         subplot(length(signal),length(sparsity),(r1-1)*length(sparsity)+c1);
        %         boxplot([coe_w_share,coe_w_j,coe_error],labelall,'colors',repmat('rgb',1,3),'Factorseparator',[1],'Labelverbosity','minor');%'Labels',{'Ridge','VB','Spike-slab','OLS'},
        %         title(['Noise:',num2str(signal(r1)),', Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         grid on
        %         grid minor
        %         %% AUC
        %         figure(h4)
        %         label1=repmat({'Multi-task','Spike-slab'},J,1);%Major group label
        %         label1=label1(:)';
        %         label2=repmat({'G1','G2','G3'},1,2);%Minor group label
        %         labelall={label1,label2};
        %         set(gcf,'outerposition',get(0,'screensize'));
        %         subplot(length(signal),length(sparsity),(r1-1)*length(sparsity)+c1);
        %         boxplot([auc1,auc2],labelall,'colors',repmat('rgb',1,2),'Factorseparator',[1],'Labelverbosity','minor');%'Labels',{'Ridge','VB','Spike-slab','OLS'},
        %         title(['Noise:',num2str(signal(r1)),', Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         grid on
        %         grid minor
        %         %% w_share
        %         figure(h5)
        %         set(gcf,'outerposition',get(0,'screensize'));
        %         subplot(length(signal),length(sparsity),(r1-1)*length(sparsity)+c1);
        %         boxplot([error_of_w_share(:)],'Labels',{'w_share'},'colors','rbm');
        %         %boxplot([mse1',mse2',mse3'],'Labels',{'Ridge','VB','Spike-slab'},'colors','rbm')%No OLS
        %         title(['Noise:',num2str(signal(r1)),', Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         %         ylabel(['Noise:',num2str(signal(r1))]);
        %         %         xlabel(['Sparsity(none zero ratio):',num2str(1/sparsity(c1))]);
        %         grid on
        %         grid minor
        save(['test_',num2str(r1),'_',num2str(c1)]);
    end
end
%%
% figure(h6)
% set(gcf,'outerposition',get(0,'screensize'));
% for j=1:J
%     subplot(3,2,2*j-1);
%     plot(w(:,j),'linewidth',1)
%     ax=axis;%get the axis limit of the current figure
%     title(['w_',num2str(j),' true']);
%     grid on
%     grid minor
%     subplot(3,2,2*j);
%     evaluated=plot(a(:,j).*mu(:,j),'color','r','linewidth',1);
%     axis(ax);%use the axis get before
%     title(['w_',num2str(j),' evaluated']);
%     grid on
%     grid minor
% end
% %%
% figure(h7)
% set(gcf,'outerposition',get(0,'screensize'));
% for j=1:J
%     subplot(3,2,2*j-1);
%     plot(w(:,j),'linewidth',1)
%     ax=axis;%get the axis limit of the current figure
%     title(['w_',num2str(j),' true']);
%     grid on
%     grid minor
%     subplot(3,2,2*j);
%     evaluated=plot(ass_2(:,j).*muss_2(:,j),'color','r','linewidth',1);
%     axis(ax);%use the axis get before
%     title(['w_',num2str(j),' evaluated']);
%     grid on
%     grid minor
% end
%save('test.mat')