clc; close all; clear all;
% Model A, using Sturup data, both measurments and predictions
load tstu93.mat;    load tstu94.mat;    load tstu95.mat;  
load ptstu93.mat;    load ptstu94.mat;    load ptstu95.mat;  

noLags=50;
k = 1; % prediction horizon
test_init = 55; % to ignore intitail destroyed smaples while filtering test data
% The whole dataset
y_orig=[tstu93;tstu94;tstu95]; % The tempreture signal
u_orig=[ptstu93;ptstu94;ptstu95]; % The input signal
% k2 = find(~u) % to find the missing smaples
u_initial_zero_indx = 2184;
u_orig = u_orig(u_initial_zero_indx:end);
y_orig = y_orig(u_initial_zero_indx:length(u_orig)+u_initial_zero_indx-1);

mstart=24*450-u_initial_zero_indx -5;
vstart=mstart-1+10*7*24;
vEnd=vstart-1+2*7*24;
tstart= vstart+25*7*24;     tEnd=tstart-1+2*7*24;
ml=mstart:vstart;
vl=vstart:vEnd;
tl=tstart:tEnd;

%orginial datasets
my_orig = y_orig(ml);
vy_orig = y_orig(vl);
ty_orig = y_orig(tl);

% To scale the variance of prediction errors with variance of validation and test data 
var_val= var(vy_orig(1:3:end));
var_test= var(ty_orig(test_init:3:end)); % excluding 54 initial smaples which are ruined during filtering

% Let's visualize the data
figure
h = zeros(1,4);
h(1) = plot(y_orig,'Color','b');
hold on;
h(2) = plot(ml,my_orig,'Color','r');
h(3) = plot(vl,vy_orig,'Color','k'); 
h(4) = plot(tl,ty_orig,'Color','g');
hold off;
legend(h,'Whole dataset','Modelling dataset','Validation dataset','Test dataset'); 
%% Data investigation
% Let's check if there is any outliers by checking if ACF and TACF plots are different
% If so, remove them or imput them using median filter
figure
g=zeros(1,10);
acf( y_orig, noLags, 0.05,1 );
hold on
tacf(y_orig,noLags,0.02,0.05,1);title('TACF and ACF of modelling, validation, and test datasets')
hold off
% As can be seen, acf and tacf plots are quite similar to each other, so
% there is no significant effects by outliers

% Let's check if there needs any transform
figure 
lambda_max = bcNormPlot(y_orig,1);   
fprintf('The Box-Cox curve is maximized at %4.2f. This suggests that a log transform might be helpful.\n', lambda_max)

figure
normplot(y_orig) % the modelling and validation data have a normal distibution

%%
% Lets transform the data.First we need to make all the values positive so
% that log function does not crach
y=y_orig+15; % Adding a positive number to data to make all the datpoints positive
% figure
% plot(y)
% title('Translated data')
% original  = y;
y = log( y );
figure
plot(y)
title('Transfomed data')

%transdormed datasets
my= y(ml);
vy=y(vl);
ty=y(tl);
% Let's visualize the data
figure
h = zeros(1,4);
h(1) = plot(y,'Color','b');
hold on;
h(2) = plot(ml,my,'Color','r');
h(3) = plot(vl,vy,'Color','k'); 
h(4) = plot(tl,ty,'Color','g');
hold off;
legend(h,'Whole dataset','Modelling dataset','Validation dataset','Test dataset'); 

%% Detrending
% There is a linear trend for the modelling data, so let's differentiate it
my_diff = filter([1 -1],1,my);      
my_diff = my_diff(2:end);
figure
plot(my_diff)
title('Detrended modelling data')
%% Model development for the modelling data
plotACFnPACF( my_diff, noLags, 'Modelling data' );% clear periodicity at 24, showing 24 hours a day
%%
sday  = 24; % 24 hours as the periodicity
dayPoly  = [1 zeros(1,sday-1) -1];
my_diff = filter(dayPoly,1,my_diff);      
my_diff = my_diff(sday+1:end);
plotACFnPACF( my_diff, noLags, 'Differentiated modelling data' );
% Some significant components at 1,2 25, 26, 49,and 50. If we consider the
% first component, other significant componenets, which are somehow
% related to the process periodicity (24) and the lag 1, could be
% covered

%% Make a first model
% We begin modelling the AR-part with the first PACF lag
dataContainer = iddata( my_diff );
Am = [ 1 1];                           
Cm = [ 1 ];
polyContainer=idpoly( Am,[],Cm );
polyContainer.Structure.a.Free = Am;    
foundModel = pem( dataContainer,polyContainer );
present( foundModel );                  

ey = filter( foundModel.A, foundModel.C, my_diff );  ey = ey(length(foundModel.A):end );

plotACFnPACF( ey, noLags, 'Residual model 1' );
checkIfWhite( ey );

%%
% The residual is clearly not white. After includeing an AR-part, include
% also an MA-part. There are the strong 24-season and lag 3 in the ACF. 
% Lets add them too.
Cm = conv(dayPoly,[1 0 0 -1]);
Cm(2:end) = 0.5*Cm(2:end);              
polyContainer = idpoly( Am,[], Cm );

polyContainer.Structure.a.Free = Am;    
polyContainer.Structure.c.Free = Cm;    
foundModel = pem( dataContainer, polyContainer );
present( foundModel );

ey = filter( foundModel.A, foundModel.C, my_diff );  ey = ey(length(foundModel.A):end);
plotACFnPACF( ey, noLags, 'Residual, model 2' );
checkIfWhite( ey );% The modelling residual is not WN.

%% Make a third model, adding a further periodicity also for the AR-part.
% There seems to be some significant components at lags 3,9 and 18, lets add them...
Am = conv([1 1],[1 0 0 -1 zeros(1,5) -1 zeros(1,8) -1]);
polyContainer = idpoly( Am,[],Cm );
polyContainer.Structure.c.Free = Cm;
polyContainer.Structure.a.Free = Am;
foundModel = pem( dataContainer, polyContainer );
present(foundModel); 
% the 9th and 10th parameters are not significant, so we can omit them

ey = filter( foundModel.A, foundModel.C, my_diff );  ey = ey(length(foundModel.A):end);
plotACFnPACF( ey, noLags, 'Residual, model 3' );
checkIfWhite( ey );% The modelling residual is WN

%% Make a fourth model, removing the insignificant parameters from AR part.
Am = [1 -1 0 -1 -1 zeros(1,13) -1 -1];
polyContainer = idpoly( Am,[],Cm );
polyContainer.Structure.c.Free = Cm;
polyContainer.Structure.a.Free = Am;
foundModel = pem( dataContainer, polyContainer );
present(foundModel); % all the parameters are significant

ey = filter( foundModel.A, foundModel.C, my_diff );  ey = ey(length(foundModel.A):end);
plotACFnPACF( ey, noLags, 'Residual, model 4' );
checkIfWhite( ey );% The modelling residual is WN

%% Evaluation on validation data
data_y=[my;vy];
time = mstart-1:vEnd;
f1 = figure;
plot(time, data_y )
hold on
plot(vl, vy,'r')
hold off

figProp = get(f1);
line( [vstart vstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
legend('Modelling data', 'Validation data', 'Prediction starts','Location','NW')
title('Data to predict')

%% Part 1: Form the prediction using the true polynomials. 

C=foundModel.C;
A = conv(conv([1 -1],dayPoly), foundModel.A);          % Form the A polynomial taking the differentiation into account.

[F, G] = polydiv( C, A, k ) ;                    % Compute the G and F polynomials.
yhatk_A  = filter( G, C, data_y );                  % Form the predicted data.
yhatk_A_test  = filter( G, C, ty );                  % Form the predicted data.
% Compute the average group delay.
shiftK = round( mean( grpdelay(G, 1) ) );
fprintf('The average group delay is %i.\n', shiftK)


%% Plotting the predictions compared to original data
data_y_orig =exp( data_y )-15;
yhatk_A_orig =exp( yhatk_A )-15;
pred_st = vstart-mstart;
pred_end = vEnd-mstart;

time_test = tstart:tEnd;
yhatk_A_test_orig =exp( yhatk_A_test )-15;

f1 = figure
subplot(211) % validation data
plot(time(1:end-shiftK),[data_y_orig(1:end-shiftK) yhatk_A_orig(shiftK+1:end)] )
figProp = get(f1);
line( [vstart vstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
title( sprintf('Shifted %i-step prediction of the validation data using model A',k))
legend('Measured data', 'Predicted data', 'Prediction starts','Location','NE')
xlim([vstart-100 vEnd])% plotting the prediction for validation data
ylim([min(y_orig) max(y_orig)])

subplot(212) % test data
plot(time_test(54:end-shiftK),[ty_orig(54:end-shiftK) yhatk_A_test_orig(shiftK+54:end)] )% discard some ruined intial data from true observations
title( sprintf('Shifted %i-step prediction of the test data using model A',k))
legend('Measured test data', 'Predicted test data','Location','NE')

%% Form the residual. Is it behaving as expected? Recall, no shift here!
ey_A = data_y - yhatk_A;

ey_A = ey_A(pred_st:3:pred_end);

figure
acfEst = acf( ey_A, noLags, 0.05, 1 );
title( sprintf('Prediction residual, %i-step prediction', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ey_A );
checkIfNormal(acfEst(k+1:end), 'ACF');

%% Examining the prediction residuals of both validation and test data using model A
data_y_orig = exp(data_y) - 15;
yhatk_A_orig = exp(yhatk_A) - 15;

ey_A_orig = data_y_orig - yhatk_A_orig;
ey_A_orig = ey_A_orig(pred_st:3:pred_end); % only true observations

ey_A_test = ty_orig - yhatk_A_test_orig;
ey_A_test = ey_A_test(test_init:3:end); % ignoring 54 initail destroyed samples, only true observations

var_ey_A_test = var (ey_A_test)/ var_test;
var_ey_A_val = var (ey_A_orig) / var_val;

% plotting
figure
subplot(211) % validation data
plot(ey_A_orig)
title(sprintf('%i-step prediction residual using model A on validation data',k))

subplot(212) % test data
plot(ey_A_test)
title(sprintf('%i-step prediction residual using model A on test data',k))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model B, using Sturup prediction data as an external signal

mu_orig= u_orig(ml);
vu_orig= u_orig(vl);
tu_orig= u_orig(tl);

% Let's visualize the data first
figure
h = zeros(1,4);
h(1) = plot(u_orig,'Color','b');
hold on;
h(2) = plot(ml,mu_orig,'Color','r');
h(3) = plot(vl,vu_orig,'Color','k'); 
h(4) = plot(tl,tu_orig,'Color','g');
hold off;
legend(h,'Whole dataset','Modelling dataset','Validation dataset','Test dataset'); 
%% Data investigation
% Let's check if there is any missing data

% Let's check if there is any outliers by checking if pacf and TACF plots are different
% If so, remove them or imput them using median filter
figure
g=zeros(1,10);
acf( u_orig, noLags, 0.05,1 );
hold on
tacf(u_orig,noLags,0.02,0.05,1);title('TACF and ACF of modelling, validation, and test datasets')
hold off
% As can be seen, acf and tacf plots are quite similar to each other, so
% there is no outliers

% Let's check if there needs any transform
figure 
lambda_max = bcNormPlot(u_orig,1);   
fprintf('The Box-Cox curve is maximized at %4.2f. This suggests that a log transform might be helpful.\n', lambda_max)

figure
normplot(u_orig) % the modelling and validation data have a normal distibution

%%
% Lets transform the data.First we need to make all the values positive so
% that log function does not crach
u=u_orig+15; % Adding a positive number to data to make all the datpoints positive
% figure
% plot(u)
% title('Translated data')
u = log( u );
% figure
% plot(u)
% title('Transfomed data')

%transdormed datasets
mu= u(ml);
vu=u(vl);
tu=u(tl);
% Let's visualize the data
figure
h = zeros(1,4);
h(1) = plot(u,'Color','b');
hold on;
h(2) = plot(ml,mu,'Color','r');
h(3) = plot(vl,vu,'Color','k'); 
h(4) = plot(tl,tu,'Color','g');
hold off;
legend(h,'Whole dataset','Modelling dataset','Validation dataset','Test dataset'); 

%% Detrending
% There is a linear trend for the modelling data, so let's differentiate it
mu_diff = filter([1 -1],1,mu);      
mu_diff = mu_diff(2:end);
figure
plot(mu_diff)
title('Detrended modelling data')
%% Model development for the modelling data
plotACFnPACF( mu_diff, noLags, 'Modelling data' );% clear periodicity at 24, showing 24 hours a day
%%
sday  = 24; % 24 hours as the periodicity
dayPoly  = [1 zeros(1,sday-1) -1];
mu_diff = filter(dayPoly,1,mu_diff);      
mu_diff = mu_diff(sday+1:end);
plotACFnPACF( mu_diff, noLags, 'Differentiated modelling data' );
% Some significant components at 1,2 25, 26, 49,and 50. If we consider the
% first component, other significant componenets, which are somehow
% related to the process periodicity (24) and the lag 1, could be
% covered

%% Make a first model
% We begin modelling the AR-part with the first PACF lag
dataContainer = iddata( mu_diff );
Am = [ 1 1];                           
Cm = [ 1 ];
polyContainer=idpoly( Am,[],Cm );
polyContainer.Structure.a.Free = Am;    
foundModel_u = pem( dataContainer,polyContainer );
present( foundModel_u );                  

eu = filter( foundModel_u.A, foundModel_u.C, mu_diff );  eu = eu(length(foundModel_u.A):end );

plotACFnPACF( eu, noLags, 'Residual model 1' );
checkIfWhite( eu );

%%
% The residual is clearly not white. After includeing an AR-part, include
% also an MA-part. There are the strong 24-season and lag 3 in the ACF. 
% Lets add them too.
Cm = conv(dayPoly,[1 0 0 -1 ]);
Cm(2:end) = 0.5*Cm(2:end);              
polyContainer = idpoly( Am,[], Cm );

polyContainer.Structure.a.Free = Am;    
polyContainer.Structure.c.Free = Cm;    
foundModel_u = pem( dataContainer, polyContainer );
present( foundModel_u );

eu = filter( foundModel_u.A, foundModel_u.C, mu_diff );  eu = eu(length(foundModel_u.A):end);
plotACFnPACF( eu, noLags, 'Residual, model 2' );
checkIfWhite( eu );% The modelling residual is not WN.

%% Make a third model, adding a further periodicity also for the AR-part.
% There seems to be some significant components at lags 3, 6, 14, 15 and
% 18, lets add them all
Am = conv([1 1],[1 0 0 -1 0 0 -1 zeros(1,7) -1 -1 0 0 -1]);
polyContainer = idpoly( Am,[],Cm );
polyContainer.Structure.c.Free = Cm;
polyContainer.Structure.a.Free = Am;
foundModel_u = pem( dataContainer, polyContainer );
present(foundModel_u); 
% the 9th and 10th parameters are not significant, so we can omit them

eu = filter( foundModel_u.A, foundModel_u.C, mu_diff );  eu = eu(length(foundModel_u.A):end);
plotACFnPACF( eu, noLags, 'Residual, model 3' );
checkIfWhite( eu );% The modelling residual is still WN
%% Make a fourth model, removing the insignificant parameters from AR part.
Am = [1 -1 0 -1 -1 0 -1 -1 zeros(1,6) -1 0 0 0 -1 -1];
polyContainer = idpoly( Am,[],Cm );
polyContainer.Structure.c.Free = Cm;
polyContainer.Structure.a.Free = Am;
foundModel_u = pem( dataContainer, polyContainer );
present(foundModel_u); % all the parameters are significant

eu = filter( foundModel_u.A, foundModel_u.C, mu_diff );  eu = eu(length(foundModel_u.A):end);

plotACFnPACF( eu, noLags, 'Residual, model 4' );
checkIfWhite( eu );% The modelling residual is WN
%%
ey = filter( foundModel_u.A, foundModel_u.C, my_diff );  ey = ey(length(foundModel_u.A):end);

%%
Nextra = 1; % in case if there are some strange behaviour in cross-correlation of input and the observation

eu = eu(Nextra:end);
ey = ey(Nextra:end);

figure;
plot([eu ey])
figure;
[Cxy,lags] = xcorr( ey, eu, noLags, 'coeff' );
stem( lags, Cxy )
hold on
condInt = 2*ones(1,length(lags))./sqrt( length(ey) );
plot( lags, condInt,'r--' )
plot( lags, -condInt,'r--' )
hold off
xlabel('Lag')
ylabel('Amplitude')
title('Crosscorrelation between filtered in- and output')
% Suggests a delay of d=8 and  s=9 ( d+s = 17). There does not seem to be much of a
% decay, so lets try r=0.


%% Form a first model using the transfer model.
% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
estimateBJ( my_diff, mu_diff, [1], [1], [0 zeros(1,7) 1 zeros(1,8) 1], [1], 'BJ model 1', noLags );
% There seems to be strong dependencies for the first 3 AR lag, lets add the first one.


%% Form a second model using the transfer model.
% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
estimateBJ(my_diff, mu_diff, [1], [1 1], [0 zeros(1,7) 1 zeros(1,8) 1], [1], 'BJ model 2', noLags );
% There is some MA dependency at lag 3. Lets add it!.


%% Form a third model using the transfer model.
% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
estimateBJ( my_diff, mu_diff, [1 0 0 1], [1 1], [0 zeros(1,7) 1 zeros(1,8) 1], [1], 'BJ model 3', noLags );
% Some dependencies at 3, 6 and 18 and 21 AR lages, Lets add them!

%% Form a fourth model using the transfer model.
% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
foundModel = estimateBJ( my_diff, mu_diff, [1 0 0 1], [1 1 0 1 0 0 1 zeros(1,11) 1 0 0 1], [0 zeros(1,7) 1 zeros(1,8) 1], [1], 'BJ model 4', noLags );
% Yes, now it seems to be white!!! :-)

%% Check correlation of the resulting model.
% Ideally, the residual formed as tilde_et = yt - [ B / A2 ] xt should be
% uncorrelated with xt. Lets check! All seems fine! Compare the resulting
% model with how the signal was generated.
tilde_et = my_diff - filter( foundModel.B, foundModel.F, mu_diff );      

% Note that we now have to remove samples from u as well.
tilde_et  = tilde_et(length(foundModel.B):end );
filter_ut = mu_diff(length(foundModel.B):end );

figure
[Cxy,lags] = xcorr( filter_ut, tilde_et, noLags, 'coeff' );
stem( lags, Cxy )
hold on
condInt = 2*ones(1,length(lags))./sqrt( length(my_diff) );
plot( lags, condInt,'r--' )
plot( lags, -condInt,'r--' )
hold off
xlabel('Lag')
ylabel('Amplitude')
title('Crosscorrelation between input and residual without the influence of the input')
% There are some correlation between the input and the output

%% Let's try the model on some simulated data

%{
A1(z) = 1 - 0.6593 (+/- 0.02374) z^-1 - 0.3453 (+/- 0.06985) z^                   
          -3 + 0.333 (+/- 0.05231) z^-4 - 0.1566 (+/- 0.03863) z^-6                
          + 0.1718 (+/- 0.02829) z^-7 + 0.07532 (+/- 0.02331) z^                   
          -14 - 0.1439 (+/- 0.02494) z^-18 + 0.07265 (+/- 0.02537) z^-19           
                                                                                   
                                                                                   
C3(z) = 1 - 0.7065 (+/- 0.06486) z^-3 - 0.7556 (+/- 0.01751) z^                   
                                        -24 + 0.5083 (+/- 0.05277) z^-27
B(z) = -0.1117 (+/- 0.02226) z^-8 + 0.06431 (+/- 0.01933) z^-17             
                                                                              
C(z) = 1 - 0.8041 (+/- 0.01837) z^-3                                        


D(z) = 1 - 0.785 (+/- 0.01936) z^-1 - 0.06676 (+/- 0.02348) z^              
          -3 + 0.07718 (+/- 0.01735) z^-6 - 0.06091 (+/- 0.01705) z^-18       
                                            + 0.1027 (+/- 0.01703) z^-21      
%}
n=10000;    extraN=100;
A3=[1 -.6511 0 -.3426 .3387 0 -.1566 .1718 zeros(1,6) .07532 0 0 0 -.1439 .07265 ];
C3=[1 0 0 -.7065 zeros(1,20) -.7556 0 0 .5083 ];
w_sim=sqrt(2)*randn(n+extraN,1);
u_sim=filter(C3,A3,w_sim);
A1=[1 -.785 0 -.06676 0 0 .07718 zeros(1,11) .06091 0 0 .1027];
A2=1;
C1=[1 0 0 -.8041];
B= [zeros(1,7) -.1117 zeros(1,8) .06431];
e_sim=sqrt(1.5)*randn(n+extraN,1);
y_sim=filter(C1,A1,e_sim)+filter(B,A2,u_sim); %Create the output
u_sim=u_sim(extraN:end); %omit the initial samples
y_sim=y_sim(extraN:end);
figure 
lambda_max = bcNormPlot(u_sim,1);   
fprintf('The Box-Cox curve is maximized at %4.2f. This suggests that a linear-transform might be helpful.\n', lambda_max)

%% Estimate a first model for the input
foundModel_ARMA =estimateARMA( u_sim, [1 -1 0 -1 -1 0 -1 -1 zeros(1,6) -1 0 0 0 -1 -1 ], [1 0 0 -1 zeros(1,20) -1 0 0 -1], 'Input model 1', noLags );
% All the parameters are significant, and fairly close to their true values


%% Form the filtered signals and compute their cross-correlation.
w_t = filter( foundModel_ARMA.A, foundModel_ARMA.C, u_sim );   w_t = w_t(length(foundModel_ARMA.A):end );
eps_t = filter( foundModel_ARMA.A, foundModel_ARMA.C, y_sim );   eps_t = eps_t(length(foundModel_ARMA.A):end );

%%
M=50;
figure;
[Cxy,lags] = xcorr( eps_t,w_t, M, 'coeff' ); % crosscorr funtion gives me errors, while xcorr does not

stem(lags,Cxy);

hold on
condInt = 2/sqrt( n )*ones(1,2*M+1);
plot( lags, condInt,'r--' )
plot( lags, -condInt,'r--' )
hold off
xlabel('Lag')
ylabel('Amplitude')
title('Crosscorrelation function')
% Suggests a delay of d=7 and s=9. There seems not to be an decay, so lets try r=0.


% A1=[1 -.92];
% A2=[1 -1.test_init .436];
% C=1;
% B=[0 0 0.068 0.004];
%% Form a first model using the transfer model.
% The function call is estimateBJ( y, x, C1, A1, B, A2, titleStr, noLags )
foundModel_BJ= estimateBJ( y_sim, u_sim, [1 0 0 1], [1 1 0 1 0 0 1 zeros(1,11) 1 0 0 1], [0 zeros(1,6) 1 zeros(1,8) 1], [1], 'Simulated input model', noLags );

%% Check correlation of the resulting model.
% Ideally, the residual formed as tilde_et = yt - [ B / A2 ] xt should be
% uncorrelated with xt. Lets check! All seems fine! Compare the resulting
% model with how the signal was generated.

tilde_et = y_sim - filter( foundModel_BJ.B, foundModel_BJ.F, u_sim );      

% Note that we now have to remove samples from u as well.
tilde_et  = tilde_et(length(foundModel_BJ.B):end );
filter_ut = u_sim(length(foundModel_BJ.B):end ); 

figure
[Cxy,lags] = xcorr( filter_ut, tilde_et, noLags, 'coeff' );
stem( lags, Cxy )
hold on
condInt = 2*ones(1,length(lags))./sqrt( length(y_sim) );
plot( lags, condInt,'r--' )
plot( lags, -condInt,'r--' )
hold off                 
xlabel('Lag')
ylabel('Amplitude')
title('Crosscorrelation between input and residual without the influence of the input')
% showing that the input noise is somehow uncorrelated to the modeling residual
%% Lets predict the input first.
data_u = [mu;vu];
[Fu, Gu] = polydiv( foundModel_u.C, conv(conv([1 -1],dayPoly),foundModel_u.A), k );
uhatk = filter(Gu, foundModel_u.C, data_u);
thatk = filter(Gu, foundModel_u.C, tu);
 
% Compute the average group delay.
shiftK = round( mean( grpdelay(Gu, 1) ) );
fprintf('The average group delay is %i.\n', shiftK)
time = mstart-1:vEnd;

%% Plotting the predictions compared to original data for the input
data_u_orig =exp( data_u )-15;
uhatk_orig =exp( uhatk )-15;

%plotting
f1 = figure;
subplot(211) % validation data
plot(time(1:end-shiftK),[data_u_orig(1:end-shiftK) uhatk_orig(shiftK+1:end)] )
figProp = get(f1);
line( [vstart vstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
title( sprintf('Shifted %i-step predictions of validation data for the input',k))
legend('Measured data', 'Predicted data', 'Prediction starts','Location','NE')
xlim([vstart-100 vEnd])% plotting the prediction for validation data

subplot(212) % test data
plot(time_test(test_init:end-shiftK),[tu(test_init:end-shiftK) thatk(shiftK+test_init:end)] ) % ignoring the 54 intial destroyed smaples
legend('True test signal', 'Predicted test signal')
title( sprintf('Shifted %i-step predictions of test data for the input',k))

%% Examining the prediction error for the input
ehat = data_u - uhatk;
ehat = ehat(pred_st:end);
figure
acfEst = acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite(ehat);
checkIfNormal(acfEst(k+1:end), 'ACF');

%% Proceed to predict the data using the predicted input.
% Form the BJ prediction polynomials. In our notation, these are
%   A1 = foundModel.D
%   C1 = foundModel.C
%   A2 = foundModel.F
% 
% The KA, KB, and KC polynomials are formed as:
%   KA = conv( A1, A2 );
%   KB = conv( A1, B );
%   KC = conv( A2, C1 );
KA = conv(conv( conv([1 -1],dayPoly), foundModel.D), foundModel.F );
KB = conv(conv( conv([1 -1],dayPoly), foundModel.D), foundModel.B );
KC = conv( foundModel.F, foundModel.C );

% Form the ARMA prediction for y_t (note that this is not the same G
% polynomial as we computed above (that was for u_t, this is for y_t).
[Fy, Gy] = polydiv( foundModel.C, conv(conv([1 -1],dayPoly) ,foundModel.D), k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh, Ghh] = polydiv( conv(Fy, KB), KC, k );

% Form the predicted output signal using the predicted input signal.
% If the prediction horizon is bigger than the lag in the input, the measured input 
% can be used instead of the predicted input to calculate the output
% prediction

if k < 7 
    yhatk_B  = filter(Fhh, 1, uhatk) + filter(Ghh, KC, data_u) + filter(Gy, KC, data_y);
    yhatk_B_test  = filter(Fhh, 1, thatk) + filter(Ghh, KC, tu) + filter(Gy, KC, ty);
else
    yhatk_B  = filter(Fhh, 1, data_u) + filter(Ghh, KC, data_u) + filter(Gy, KC, data_y); 
    yhatk_B_test  = filter(Fhh, 1, tu) + filter(Ghh, KC, tu) + filter(Gy, KC, ty);

end
% Group delay    
shiftK = round( mean( grpdelay(Fhh, 1) ) );     %As we have three filters, this might be quite a poor estimate.

%% Plotting test prediction
yhatk_B_orig =exp( yhatk_B )-15;
f1 = figure;
subplot(211)  % validation data
plot(time(1:end-shiftK),[data_y_orig(1:end-shiftK) yhatk_B_orig(shiftK+1:end)] )
figProp = get(f1);
line( [vstart vstart], figProp.CurrentAxes.YLim, 'Color', 'red', 'LineStyle', ':' )
title( sprintf('Shifted %i-step predictions using model B on validation data',k))
legend('Measured data', 'Predicted data', 'Prediction starts','Location','SE')
xlim([vstart-100 vEnd])
ylim([min(data_y_orig) max(data_y_orig)])

subplot(212) % test data
plot(time_test(test_init:end-shiftK),[ty(test_init:end-shiftK) yhatk_B_test(shiftK+test_init:end)] ) % ignoring the 54 intial destroyed smaples
legend('True test signal', 'Predicted test signal')
title( sprintf('Shifted %i-step predictions using model B on test data',k))

%% Examining the prediction residual of true validation data
eyk_B = data_y - yhatk_B;
eyk_B = eyk_B(pred_st:3:pred_end);

figure
acfEst = acf( eyk_B, noLags, 0.05,1 );% It's is not an MA model, for: eyk_B_t+k|t = F(z) e_t+k + Fˆ(z) x_t+k 
title( sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( eyk_B );
checkIfNormal( acfEst(k+1:end), 'ACF' );

%% Form the residual
ey_B_orig = data_y_orig - yhatk_B_orig;
ey_B_orig = ey_B_orig(pred_st:3:pred_end); % only true measurements

yhatk_B_test_orig = exp( yhatk_B_test )-15;
ey_B_test = ty_orig - yhatk_B_test_orig;
ey_B_test = ey_B_test(test_init:3:end); % ignore 54 initail destroyed samples, only true measurements

var_ey_B_val = var (ey_B_orig)/ var_val; % scaled variance of prediction error on true measurements of validation data
var_ey_B_test = var (ey_B_test) / var_test; % scaled variance of prediction error on true measurements of test data

figure
subplot(211) % validation data
plot(ey_B_orig)
title(sprintf('%i-step prediction error for model B on validation data',k))

% Exmine model B on test data
subplot(212) % test data
plot(ey_B_test)
title(sprintf('%i-step prediction error for model B on test data',k))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model C, recursive model
% Estimate the unknown parameters using a Kalman filter and form the k-step prediction.
%{
 
B(z) = -0.1231 (+/- 0.02152) z^-8 + 0.07903 (+/- 0.02145) z^-17             

C(z) = 1 - 0.7355 (+/- 0.02067) z^-3                                        


D(z) = 1 - 0.8228 (+/- 0.01571) z^-1 + 0.1101 (+/- 0.0286) z^-6             
- 0.04629 (+/- 0.02605) z^-7  
%}

a1 = - 0.82;  a6 = 0.1103;   a7 =  - 0.046;  c3 = - 0.73;     b8 = -0.123;    b17 = 0.07886;
std_a1 = 0.0157;  std_a6 = 0.02861;   std_a7 =  0.02605;  std_c3 = 0.02061;     std_b8 = 0.02154;    std_b17 = 0.0215;

params_vec = [a1, a6, a7, c3, b8, b8*a1, b8*a6, b8*a7, b17, b17*a1, b17*a6, b17*a7 ];
std_vec= [std_a1, std_a6, a7, std_c3, std_b8, std_b8*std_a1, std_b8*std_a6,...
    std_b8*std_a7, std_b17, std_b17*std_a1, std_b17*std_a6, std_b17*std_a7];



p0 = 3;                                         % Number of unknowns in the A polynomial.
q0 = 1;                                         % Number of unknowns in the C polynomial.
s0 = 8;                                         % Number of unknowns in the B polynomial.

N = length(y);
A     = eye(p0+q0+s0);
Rw    = 1;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-6.*eye(p0+q0+s0);                                   % System noise covariance matrix, R_e
Rx_t1 = diag(std_vec.^2);                             % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = [zeros(p0+q0+s0 , 23), params_vec', zeros(p0+q0+s0 , N-24)];   % Estimated states in controlable form. Intial state, x_{1|0} = 0.
yhat_C  = zeros(N,1);                             % Estimated output.
yhatk_C = zeros(N,1);                             % Estimated k-step prediction.
for t=25:N-k                                     % We use t-3, so start at t=4. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C = [ -y(t-1), -y(t-6), -y(t-7), h_et(t-3),u(t-8), u(t-9),...
        u(t-14), u(t-15), u(t-17), u(t-18), u(t-23), u(t-24)]; % C_{t|t-1} % NOTE to form the 1-step prediction
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat_C(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = y(t)-yhat_C(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. 
    
    yk = zeros(1,k+1);  % to store predicted values in every iteration

    Ck = [ -y(t), -y(t-5), -y(t-6), h_et(t-2),u(t-7), u(t-8),...
        u(t-13), u(t-14), u(t-16), u(t-17), u(t-22), u(t-23) ];           % C_{t+1|t}
    yk(1) = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
   
    for k0=2:k
        if k0 <= 6 % while k0 is less than 6, we still have observations for y(t+k0-6)
            Ck = [ -yk(k0-1), -y(t+k0-6), -y(t+k0-7), h_et(t+k0-3),u(t+k0-8), u(t+k0-9),...
            u(t+k0-14), u(t+k0-15), u(t+k0-17), u(t+k0-18), u(t+k0-23), u(t+k0-24) ]; 
            yk(k0) = Ck*A^k*xt(:,t);
            
        % while k0 is 7, we do not have observations for y(t+k0-6),
        % but still have observations for y(t+k0-7)
        elseif k0 == 7
            Ck = [ -yk(k0-1), -yk(k0-6), -y(t+k0-7), h_et(t+k0-3),u(t+k0-8), u(t+k0-9),...
            u(t+k0-14), u(t+k0-15), u(t+k0-17), u(t+k0-18), u(t+k0-23), u(t+k0-24) ]; 
            yk(k0) = Ck*A^k*xt(:,t);   
            
        % when k0 is bigger than 7, we do not have observations for y(t+k0-6) and y(t+k0-7),
        % so we need to use the predicted value of 6 and 7 time-steps
        % before, respectivel, which were already saved in yk array
        else
            Ck = [ -yk(k0-1), -yk(k0-6), -yk(k0-7), h_et(t+k0-3),u(t+k0-8), u(t+k0-9),...
            u(t+k0-14), u(t+k0-15), u(t+k0-17), u(t+k0-18), u(t+k0-23), u(t+k0-24) ]; 
            yk(k0) = Ck*A^k*xt(:,t);                   
        end
        

    end
    
    if k> 1
        yhatk_C(t+k) = yk(k0);                            % Note that this should be stored at t+k.
    else 
        yhatk_C(t+k) = yk(1);                            % Note that this should be stored at t+k.
    end


    
end


%% Examine the estimated parameters.
figure

Q0 = params_vec;                       % These are the true parameters we seek.
xt_t = xt';
plot(xt_t)
for k0=1:length(Q0)
    line([1 N], [Q0(k0) Q0(k0)], 'Color','red','LineStyle',':')
end
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re, Rw))
xlabel('Time')
ylim([-1.5 1.5])
xlim([1 N-k])

%% Plotting the 1-step prediction using model C on validation and test data
yhat_C_orig =exp( yhat_C )-15;
figure
subplot(211)% on validation data
plot( [y_orig yhat_C_orig] )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([vstart vEnd])
title('1-step prediction for model C on validation data')


subplot(212)% on test data
plot( [y_orig yhat_C_orig] )
title('1-step prediction for model C on test data')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([tstart+test_init tEnd])% removing 54 initial destroyed samples
%% Examine 1-step prediction residual of true measurements for both test and validation data
ey_C_orig = y_orig-yhat_C_orig;
ey_C_val= ey_C_orig(vstart:3:vEnd);                             
ey_C_test= ey_C_orig(tstart+test_init:3:tEnd);% Ignore 54 initial values to let the filter converge first.

var_ey_C_val= var(ey_C_val) / var_val; % scaled variance of prediction error on true measurements of validation data
var_ey_C_test= var(ey_C_test) / var_test; % scaled variance of prediction error on true measurements of test data

%% Examine k-step prediction residual of true measurements of validation data
eyk_C = y-yhatk_C;
eyk_C = eyk_C(vstart:3:vEnd);                            

[acfEst, pacfEst] = plotACFnPACF( eyk_C, noLags, sprintf('%i-step prediction using the Kalman filter', k)  );

pacfEst = pacfEst(k+1:end);
checkIfNormal( pacfEst, 'PACF' );

acfEst = acfEst(k+1:end);
checkIfNormal( acfEst, 'ACF' );
%% Examine 1-step prediction residual of true measurements of validation data
ey_C = y-yhat_C;
ey_C = ey_C(vstart:3:vEnd);                            

[acfEst, pacfEst] = plotACFnPACF( eyk_C, noLags, '1-step prediction using model C' );

pacfEst = pacfEst(1+1:end);
checkIfNormal( pacfEst, 'PACF' );

acfEst = acfEst(1+1:end);
checkIfNormal( acfEst, 'ACF' );

%% Plotting the k-step prediction using model C on both validatioin and test data
yhatk_C_orig =exp( yhatk_C )-15;

figure
subplot(211) % validation data
plot( [y_orig(1:N-k) yhatk_C_orig(k+1:N)] )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([vstart vEnd])
title( sprintf('%i-step prediction using model C on validation data(shifted %i steps)', k, k) )

subplot(212)% test data
plot( [y_orig(1:N-k) yhatk_C_orig(k+1:N)] )
title( sprintf('%i-step prediction using model C on test data(shifted %i steps)', k, k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([tstart+54 tEnd])% removing some initial destroyed samples

%% Examine k-step prediction residual of both test and validation data
eyk_C_orig = y_orig-yhatk_C_orig;
eyk_C_val= eyk_C_orig(vstart:3:vEnd);                            
eyk_C_test= eyk_C_orig(tstart+test_init:3:tEnd); % Ignore 54 initial values to let the filter converge first.

figure
subplot(211)% validation data
plot([eyk_C_val ey_C_val])
legend('k-step prediction', '1-step prediction', 'Location','SW')
title('prediction residaul using model C on validation data')


subplot(212)% test data
plot([eyk_C_test ey_C_test])
title('prediction residaul using  model C on test data')
legend('k-step prediction', '1-step prediction', 'Location','SW')

var_eyk_C_val= var(eyk_C_val) / var_val; % scaled variance of prediction error on true measurements of validation data
var_eyk_C_test= var(eyk_C_test) / var_test; % scaled variance of prediction error on true measurements of test data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model C, SIMPLIFIED recursive model
% Estimate the unknown parameters using a Kalman filter and form the k-step prediction.
%{
 
B(z) = -0.1231 (+/- 0.02152) z^-8 + 0.07903 (+/- 0.02145) z^-17             

C(z) = 1 - 0.7355 (+/- 0.02067) z^-3                                        


D(z) = 1 - 0.8228 (+/- 0.01571) z^-1 + 0.1101 (+/- 0.0286) z^-6             
- 0.04629 (+/- 0.02605) z^-7  
%}

a1 = - 0.82;  a6 = 0.1103;   a7 =  - 0.046;  c3 = - 0.73;     b8 = -0.123;    b17 = 0.07886;
std_a1 = 0.0157;  std_a6 = 0.02861;   std_a7 =  0.02605;  std_c3 = 0.02061;     std_b8 = 0.02154;    std_b17 = 0.0215;

% since the siganls are at the same scales, the paramaters a7 and b17
% having low values can be excluded to build a simplified model
params_vec_simp = [a1, a6, c3, b8, b8*a1, b8*a6 ]; 
std_vec_simp = [std_a1, std_a6, std_c3, std_b8, std_b8*std_a1, std_b8*std_a6];

reg_dim = length(params_vec_simp);
N = length(y);
A     = eye(reg_dim);
Rw    = 1;                                      % Measurement noise covariance matrix, R_w
Re    = 1e-6.*eye(reg_dim);                                   % System noise covariance matrix, R_e
Rx_t1 = diag(std_vec_simp.^2);                             % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = [zeros(reg_dim , 13), params_vec_simp', zeros(reg_dim , N-14)];   % Estimated states in controlable form. Intial state, x_{1|0} = 0.
yhat_C_simp  = zeros(N,1);                             % Estimated output.
yhatk_C_simp = zeros(N,1);                             % Estimated k-step prediction.
for t=15:N-k                                     % We use t-3, so start at t=4. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}C = [ -y(t-1), -y(t-6), -y(t-7), h_et(t-3),u(t-8), u(t-9),...
    C = [ -y(t-1), -y(t-6), h_et(t-3),u(t-8), u(t-9),u(t-14)]; % C_{t|t-1} % NOTE to form the 1-step prediction
         
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat_C_simp(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = y(t)-yhat_C_simp(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 


    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re
    
    yk = zeros(1,k+1);    % to store predicted values in every iteration

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. 
    Ck = [  -y(t), -y(t-5) h_et(t-2),u(t-7), u(t-8),u(t-13) ];          
    yk(1) = Ck*xt(:,t); % putting the 1-step prediction in ths 1st element

    for k0=2:k
        if k0 <= 6 % while k0 is less than 6, we still have observations for y(t+k0-6)
        Ck = [ -yk(k0-1), -y(t+k0-6), h_et(t+k0-3),u(t+k0-8), u(t+k0-9),u(t+k0-14) ]; 
        yk(k0) = Ck*A^k*xt(:,t);
        
        % when k0 is bigger than 6, we do not have observations for y(t+k0-6),
        % so we need to use the predicted value of 6 time-steps before,
        % which was already saved in yk array
        else 
        Ck = [ -yk(k0-1), -yk(k0-6), h_et(t+k0-3),u(t+k0-8), u(t+k0-9),u(t+k0-14) ]; 
        yk(k0) = Ck*A^k*xt(:,t);                    
        end 
    end
    
    if k> 1
        yhatk_C_simp(t+k) = yk(k0);                            % Note that this should be stored at t+k.
    else 
        yhatk_C_simp(t+k) = yk(1);                            % Note that this should be stored at t+k.
    end
    
end

%% Examine the estimated parameters.
figure
Q0 = params_vec;                       % These are the true parameters we seek.
xt_t = xt';
plot(xt_t)
for k0=1:length(Q0)
    line([1 N], [Q0(k0) Q0(k0)], 'Color','red','LineStyle',':')
end
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re, Rw))
xlabel('Time')
ylim([-1.5 1.5])
xlim([1 N-k])

%% Plotting the 1-step prediction using model C on validation data
yhat_C_orig_simp =exp( yhat_C_simp )-15;% the 1-step prediction transformed back to have the same distribution as origanl data has
figure
subplot(211)% on validation data
plot( [y_orig yhat_C_orig_simp] )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([vstart vEnd])
title('1-step prediction using simplifided model C on validation data')


subplot(212)% on test data
plot( [y_orig yhat_C_orig_simp] )
title('1-step prediction using simplifided model C on test data')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([tstart+test_init tEnd])% removing 54 initial destroyed samples
%% Examine 1-step prediction residual of true measurements for both test and validation data
ey_C_orig_simp = y_orig-yhat_C_orig_simp;
ey_C_val_simp= ey_C_orig_simp(vstart:3:vEnd);                             
ey_C_test_simp= ey_C_orig_simp(tstart+test_init:3:tEnd);% Ignore 54 initial values to let the filter converge first.

var_ey_C_val_simp= var(ey_C_val_simp) / var_val; % scaled variance of prediction error on true measurements of validation data
var_ey_C_test_simp = var(ey_C_test_simp) / var_test; % scaled variance of prediction error on true measurements of test data

%% Examine k-step prediction residual of true measurements of validation data
yhatk_C_orig_simp =exp( yhatk_C_simp )-15; % the k-step prediction transformed back to have the same distribution as origanl data has
eyk_C_simp = y-yhatk_C_simp;
eyk_C_simp = eyk_C_simp(vstart:3:vEnd);                           

[acfEst ,pacfEst ] = plotACFnPACF( eyk_C_simp, noLags, sprintf('%i-step prediction using simplified model C', k)  );
pacfEst = pacfEst(k+1:end);
checkIfNormal( pacfEst, 'PACF' );

acfEst = acfEst(k+1:end);
checkIfNormal( acfEst, 'ACF' );
%% Examine 1-step prediction residual of true measurements of validation data
yhat_C_orig_simp =exp( yhat_C_simp )-15; % the k-step prediction transformed back to have the same distribution as origanl data has
ey_C_simp = y-yhat_C_orig_simp;
ey_C_simp = ey_C_simp(vstart:3:vEnd);                           

[acfEst ,pacfEst ] = plotACFnPACF( ey_C_simp, noLags, ' 1-step prediction using simplified model C' );
pacfEst = pacfEst(k+1:end);
checkIfNormal( pacfEst, 'PACF' );

acfEst = acfEst(k+1:end);
checkIfNormal( acfEst, 'ACF' );
%% Plotting the k-step prediction using model C on validatioin and test data
figure
subplot(211)% validation data
plot( [y_orig(1:N-k) yhatk_C_orig_simp(k+1:N)] )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
title( sprintf('%i-step prediction using simplifided model C on validation data(shifted %i steps)', k, k) )
xlim([vstart vEnd])

subplot(212)% test data
plot( [y_orig(1:N-k) yhatk_C_orig_simp(k+1:N)] )
title( sprintf('%i-step prediction using simplifided model C on test data(shifted %i steps)', k, k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([tstart+54 tEnd])% removing some initial destroyed samples

%% Examine k-step prediction residual of both test and validation data
eyk_C_orig_simp = y_orig-yhatk_C_orig_simp;
eyk_C_val_simp= eyk_C_orig_simp(vstart:3:vEnd);                            
eyk_C_test_simp= eyk_C_orig_simp(tstart+test_init:3:tEnd); % Ignore 54 initial values to let the filter converge first.

var_eyk_C_val_simp= var(eyk_C_val_simp) / var_val; % scaled variance of prediction error on true measurements of validation data
var_eyk_C_test_simp= var(eyk_C_test_simp) / var_test; % scaled variance of prediction error on true measurements of test data

figure
subplot(211)% validation data
plot([eyk_C_val_simp ey_C_val_simp])
legend('k-step prediction', '1-step prediction', 'Location','SW')
title('prediction residaul using simplifided model C on validation data')

subplot(212) % test data
plot([eyk_C_test_simp ey_C_test_simp])
title('prediction residaul using simplifided model C on test data')
legend('prediction', '1-step prediction', 'Location','SW')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Naive 1-step and 9-step predictors on validation and test data

% Naive 1-step and 9-step predictors on validation data
y_naive1 = y_orig(vstart-1:3:vEnd-1); % the prediction is the same as the true observation in validation data in one hour before
ey_naive_1 = y_orig(vstart:3:vEnd) - y_naive1; % 1-step prediction residual of the Naive model for true observations in validation data
var_naive1_val = var(ey_naive_1) / var_val;

y_naive9 = y_orig(vstart-24+9:3:vEnd-24+9); % the prediction is the same as the true observation in validation data a day before at the same time as the prediction targets, namely 9 hours ahead
ey_naive_9 = y_orig(vstart:3:vEnd) - y_naive9; % 9-step prediction residual of the Naive model for true observations in validation data
var_naive9_val = var(ey_naive_9) / var_val;

% Naive 1-step and 9-step predictors on test data
y_naive1_test = y_orig(tstart-1:3:tEnd-1); % Naive 1-step predictor
ey_naive_1_test = y_orig(tstart:3:tEnd) - y_naive1_test;
var_naive1_test = var(ey_naive_1_test) / var_test;

y_naive9_test = y_orig(tstart-24+9:3:tEnd-24+9); % Naive 9-step predictor
ey_naive_9_test = y_orig(tstart:3:tEnd) - y_naive9_test;
var_naive9_test = var(ey_naive_9_test) / var_test;

%% Plotting both the 1-step and 9-step prediction of the Naive models on both validation and test datasets
figure
subplot(211) % validation data
plot([y_orig(vstart:3:vEnd) y_naive1 y_naive9]) 
xlabel('Time')
legend('True data', 'Naive 1-step prediction', 'Naive 9-step prediction', 'Location','SW')
title( 'Naive 1-step and 9-step predictions of validation data' )

subplot(212) % test data
plot([y_orig(tstart:3:tEnd) y_naive1_test y_naive9_test]) 
xlabel('Time')
legend('True data', 'Naive 1-step prediction', 'Naive 9-step prediction', 'Location','SW')
title( 'Naive 1-step and 9-step predictions of test data' )

