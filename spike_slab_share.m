% VSSC:A variational approach to detecting shared and specific components in multi-task learning
% This function solve a multi-task learning problem where the specific
% components is sparsity.
% The input variable is a structure array including 2 fileds : trainset_x
% and trainset_y.
% For each trainset_x is a n by m design matrix and trainset y is a n by 1
% vector.

function [a,mu,mu_0,l_bound]=spike_slab_share(type)
%clc
tic
%% simulating
% clc
% clear
% J=50;
% d=10;
% w_share=randn(d,1)*d;
% for j=1:J
%     type(j).n=100;
%     type(j).trainset_x=sparse(round(rand(type(j).n,d)-0.25));
%     %type(j).trainset_x=round(rand(type(j).n,d)-0.25);
%     %type(j).trainset_x=100*randn(type(j).n,d);
%     w(:,j)=randn(d,1)*10;
%      %w(1:floor(0.90*d),j)=0;
%     type(j).trainset_y=full(type(j).trainset_x*(w(:,j)+w_share)+0.1*std(type(j).trainset_x*(w(:,j)+w_share))*randn(size(type(j).trainset_x(:,1))));
%     %type(j).trainset_x=int16(type(j).trainset_x);
% end
%%
J=size(type,2);
i=0;
%% intercept, there are two parts
%m=size(type(1).trainset_x,2)+1;%The number of coloums plus the intercept

%% centering
m=size(type(1).trainset_x,2);%dimention

%% Initialization
% paramaters
l_bound_p1=zeros(1,J);
l_bound_p3=zeros(1,J);
mu_0_part1=zeros(m,J);
mu_0_part2=zeros(m,J);
sigma_e=100*ones(1,J);
xtx=zeros(m,J);
n=zeros(1,J);
xy=zeros(m,J);
%xyt=zeros(m,J);%temp
mu2=zeros(m,J);%temp
% Variables
a=0.01*ones(m,J);%initial the posterior probality of gamma
a_p=0.01*ones(1,J);%initial the prior probility of gamma
mu=zeros(m,J);%initial the posterior mean of \beta_{jl}
s=ones(m,J);%initial the posterior variance of \beta_{jl}
sigma_b=100*ones(1,J);%initial the prior variance of \beta_{jl}
%s_0=1000*ones(m,1);%initial the posterior variance of \beta_0
mu_0=zeros(m,1);%initial the posterior mean of \beta_0
sigma_b0=100;%initial the prior variance of \beta_0
%phi=zeros(m,m,J);%The format of phi, int or single
for j=1:J % There are J types of movies
    n(j)=size(type(j).trainset_x,1);%The number of rows in each j
    
    %% Centreing
    type(j).trainset_x=type(j).trainset_x-repmat(mean(type(j).trainset_x),n(j),1);
    type(j).trainset_y=type(j).trainset_y-mean(type(j).trainset_y);
    %% Intercept (why don't centering? Because for sparsity matrix)
    %type(j).trainset_x=[type(j).trainset_x,ones(n(j),1)];
    
    %% Initialization
    sigma_e(j)=var(type(j).trainset_y)/2;%initial the prior variance of y
    xtx(:,j)=sum(type(j).trainset_x.^2)';%This is the x_{jl}^T*x_{jl}
    type(j).yt=type(j).trainset_x*(a(:,j).*mu(:,j));% The \tilde{y}
    type(j).yt_0=type(j).trainset_x*mu_0;%The \tilde{y}_0
    %phi(:,:,j)=type(j).trainset_x'*type(j).trainset_x;%The product of design matrixs
    xy(:,j)=type(j).trainset_x'*type(j).trainset_y;%The product of x' and y
    mu_0_part1(:,j)=type(j).trainset_x'*type(j).yt-xy(:,j);%The second and therd part of mu_0
    sigma_b(j)=sigma_e(j)/2/(m*0.1);%initial the prior variance of \beta_{jl}
end
sigma_b0=sigma_e(j)/2/m;%initial the prior variance of \beta_0


max_iteration=100;
l_bound=zeros(max_iteration,1);

for ite=1:max_iteration
    i=i+1;%The step
    
    %% E-step
    s_0=-0.5./(xtx*(-0.5./sigma_e)'-0.5/sigma_b0);%The evaluation of s_0, consider it as a vector
    %% Upadte the mu_0
    for l=1:m
        for j=1:J
            type(j).ylt_0=type(j).yt_0-type(j).trainset_x(:,l).*mu_0(l);
            mu_0_part2(l,j)=type(j).ylt_0'*type(j).trainset_x(:,l);%Upadte the mu_0_part2
        end
        mu_0(l)=s_0(l)*(mu_0_part2(l,:)+mu_0_part1(l,:))*(-1./sigma_e)';
        for j=1:J
            type(j).yt_0=type(j).ylt_0+type(j).trainset_x(:,l).*mu_0(l);
        end
    end
    %% Update other variables in E-step
    for j=1:J
        s(:,j)=sigma_e(j)./(xtx(:,j)+sigma_e(j)/sigma_b(j));%The evaluattion of each s in j
        for l=1:m
            type(j).ylt=type(j).yt-a(l,j)*mu(l,j)*type(j).trainset_x(:,l);%update the \tilde(y)_{jl}
            mu(l,j)=(type(j).trainset_x(:,l)'*(type(j).trainset_y-type(j).ylt)...
                -type(j).trainset_x(:,l)'*type(j).yt_0)*s(l,j)/sigma_e(j);%The evaluation of mu(:,j)
            u=log(a_p(j)/(1-a_p(j)))+0.5*log(s(l,j)/sigma_b(j))+(mu(l,j)^2)/(2*s(l,j));%The evaluation of u
            a(l,j)=1/(1+exp(-u));%The evaluation of a_l
            %
            type(j).yt=type(j).ylt+a(l,j)*mu(l,j)*type(j).trainset_x(:,l);%update the \tilde{y}_j
        end
        mu_0_part1(:,j)=type(j).trainset_x'*type(j).yt-xy(:,j);%Upadte the mu_0_part1
    end
    
    %% M-step
    for j=1:J
        mu2(:,j)=mu(:,j).^2;%temp
        sigma_e(j)=(sum((type(j).trainset_y-type(j).yt).^2)+(a(:,j).*(s(:,j)+mu2(:,j))-(a(:,j).*mu(:,j)).^2)'*xtx(:,j)...
            +sum(type(j).yt_0.^2)+sum(s_0.*xtx(:,j))+2*(mu_0_part1(:,j))'*mu_0)/n(j);%The evaluation of \sigma_j
        sigma_b(j)=(a(:,j)'*(s(:,j)+mu2(:,j)))/(sum(a(:,j)));%The evaluation of \sigma_{\beta_j}
        a_p(j)=sum(a(:,j))/m;%The evaluation of the prior of a
    end
    sigma_b0=1/m*(mu_0'*mu_0+sum(s_0));%The evalutation of \sigma_{\beta_0}, the prior of s_0
    
    %% Lower bound
    for j=1:J
        l_bound_p1(j)=-0.5*n(j)*log(2*pi*sigma_e(j))-sum((type(j).trainset_y-type(j).yt).^2)/(2*sigma_e(j))...
            -0.5*((a(:,j).*(s(:,j)+mu2(:,j))-(a(:,j).*mu(:,j)).^2)'*xtx(:,j))/sigma_e(j)...
            +sum(a(:,j).*log((a_p(j)+(a_p(j)==0))./(a(:,j)+(a(:,j)==0))))...
            +sum((1-a(:,j)).*log((1-a_p(j)+(a_p(j)==1))./(1-a(:,j)+(a(:,j)==1))))...
            +0.5*sum(a(:,j).*(1+log(s(:,j)./sigma_b(j))-(mu2(:,j)+s(:,j))./sigma_b(j)));
        l_bound_p3(j)=(mu_0_part1(:,j))'*mu_0*(-1)./sigma_e(j)...
            +sum(type(j).yt_0.^2)*(-0.5/sigma_e(j));
    end
    l_bound_p2=sum(mu_0.^2)*(-0.5)/sigma_b0+sum(s_0.*(xtx*(-0.5./sigma_e)'-0.5/sigma_b0))...
        -0.5*m*log(2*pi*sigma_b0)+0.5*sum(log(s_0))+0.5*m*(1+log(2*pi));
    l_bound(i)=sum(l_bound_p1)+sum(l_bound_p3)+l_bound_p2;
%     a_prior1(i)=a_p(1);
%     a_prior2(i)=a_p(2);
%     a_prior3(i)=a_p(3);
    fprintf('Spike-slab share Iteration: %d, lower bound : %d\n',i,l_bound(i));
    if i>1
        if abs((l_bound(i)-l_bound(i-1))/l_bound(i-1))<1e-6
%             break
        end
    end
end
toc