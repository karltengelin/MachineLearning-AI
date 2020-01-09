load('A1_data.mat')
%% Task 4
lambda = 1.6;
figure
title('lambda = 1.6')
hold on
plot(n,t,'*b')
what1 = lasso_ccd(t,X,lambda);
y = X*what1;
plot(n,y,'*r')
y2 = Xinterp*what1;
plot(ninterp,y2,'r')
legend('original t-points','estimated t-points','interpolated curve')
xlabel('Time')
hold off

lambda = 0.1;
figure
title('lambda = 0.1')
hold on
plot(n,t,'*b')
what2 = lasso_ccd(t,X,lambda);
y = X*what2;
plot(n,y,'*r')
y2 = Xinterp*what2;
plot(ninterp,y2,'r')
legend('original t-points','estimated t-points','interpolated curve')
xlabel('Time')
hold off

lambda = 10;
figure
title('lambda = 10')
hold on
plot(n,t,'*b')
what3 = lasso_ccd(t,X,lambda);
y = X*what3;
plot(n,y,'*r')
y2 = Xinterp*what3;
plot(ninterp,y2,'r')
legend('original t-points','estimated t-points','interpolated curve')
xlabel('Time')
hold off

L_user = length(find(what1));
L_01 = length(find(what2));
L_10 = length(find(what3));

lambda_pos = [L_01 L_user L_10]

%% Task 5
K = 10;
lambda_min = 0.01;
lambda_max = 10;
N_lambda = 100;
lambda_grid = exp(linspace( log(lambda_min), log(lambda_max), N_lambda));

[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambda_grid,K);

y = X*wopt;

figure
hold on
plot(n,t,'*b')
plot(n,y,'*r')
plot(ninterp,Xinterp*wopt)
legend('original t-points','estimated t-points','interpolated curve')
xlabel('time')
hold off

figure
hold on
plot(lambda_grid,RMSEval,'-*')
plot(lambda_grid,RMSEest,'-*')
xline(lambdaopt,'--r');
legend('RMSEval','RMSEest','lamdaoptimal (=2.0092)')
xlabel('lambda')
hold off
%% Task 6
soundsc(Ttrain,fs);
K = 10;
lambda_min = 10^(-4);
lambda_max = 0.03;
N_lambda = 75;
lambda_grid = exp(linspace( log(lambda_min), log(lambda_max), N_lambda));
[Wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain,Xaudio,lambda_grid,K);
%%
figure
hold on
title('RMSE for different lambdas for estimation data and validation data')
xlabel('lambda')
plot(lambda_grid,RMSEval,'-*')
plot(lambda_grid,RMSEest,'-*')
xline(lambdaopt,'--r');
legend('RMSEval','RMSEest','lamdaoptimal (=0.0046)')
hold off
%% Task 7
[Yclean] = lasso_denoise(Ttest,X,0.0044);
save('denoised_audio','Yclean','fs');
%%
soundsc(Ttest,fs);
%%
soundsc(Yclean,fs);

