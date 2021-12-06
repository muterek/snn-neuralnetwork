%% Projekt SNN - Zagadnienie aproksymacji funkcji (model regresyjny) (A)
close all 
clear all

% Wczytanie danych, centrowanie i normalizacja
testData = load('zestaw_apr_14_test.txt');
trainData = load('zestaw_apr_14_train.txt');

trainDataX = trainData(:,1);
trainDataY = trainData(:,2);
[trainDataX, mX, sX] = zscore(trainDataX);
[trainDataY, mY, sY] = zscore(trainDataY);
testDataX = testData(:,1);
testDataY = testData(:,2);
testDataX = (testDataX - mX)/sX;
testDataY = (testDataY - mY)/sY;

% Operacja min-max (skalowanie danych do wartoœci [-1,1])
trainDataX = -1+((trainDataX-min(trainDataX))*2/(max(trainDataX)-min(trainDataX)));
trainDataY = -1+((trainDataY-min(trainDataY))*2/(max(trainDataY)-min(trainDataY)));
testDataX = -1+((testDataX-min(testDataX))*2/(max(testDataX)-min(testDataX)));
testDataY = -1+((testDataY-min(testDataY))*2/(max(testDataY)-min(testDataY)));

% Wizualizacja danych trenuj¹cych i testowych
figure()
plot(trainDataX,trainDataY,'.')
hold on
plot(testDataX,testDataY,'.')
legend('train', 'test');
xlabel('X');
ylabel('Y');
title('Wizualizacja danych');
hold off

%% Wybór optymalnej iloœci neuronów ukrytych - MSE
maxHiddenUnits = 13; % max liczba neuronów ukrytych
iterations = 100; % liczba iteracji
mseTest = zeros(iterations,maxHiddenUnits); % maxierz b³êdów mse dla danych testowych
mseTrain = zeros(iterations,maxHiddenUnits); % macierz b³êdów mse dla danych trenuj¹cych
varHii = nan(iterations,maxHiddenUnits); % wariancja dŸwigni
mseMinTest = zeros(1,iterations); 
mseMinTestIdx = zeros(1,iterations);
mseMinTrain = zeros(1,iterations);
mseMinTrainIdx = zeros(1,iterations);

for j = 1:iterations
    rng('default')
    rng(j);
    for i = 1:maxHiddenUnits
        
        net = net1(trainDataX,trainDataY,i);

        % MSE
        ypredTest = net(testDataX')';
        ypredTrain = net(trainDataX')';
        
        mseTest(j,i) = mse(net,testDataY,ypredTest);
        mseTrain(j,i) = mse(net,trainDataY,ypredTrain);
        
        % virtual leave-one-out
        Z = calc_jacobian(net, trainDataX); %okreœlenie macierzy Jacobiego
        rankZ = rank(Z);
        if rankZ==min(size(Z,1),size(Z,2)) %jeœli macierz Z jest pe³nego rzêdu
            H = Z*inv(Z'*Z)*Z';
            
            hii = diag(H);
            shii = sum(hii);
            
            varHii(j,i) = var(hii);
            
            mHii = mean(hii);
            
%             figure();
%             subplot(1,2,1);
%             plot(trainDataX,trainDataY,'.');
%             hold on;
%             plot(trainDataX,ypredTrain);
%             legend('Dane','Aproksymacja');
%             xlabel('X');
%             ylabel('Y');
%             title(['Liczba neuronów ukrytych: ', num2str(i)]);
%             hold off;
%             
%             subplot(1,2,2);
%             plot(trainDataX,hii);
%             hold on;
%             plot([-1 1],[mHii mHii],'k');
%             xlabel('X');
%             ylabel('h_i_i');
%             legend('DŸwignia h_i_i','œrednia hii')
            
        end
        
        Z = [];
        H = [];
        
    end
    [mseMinTest(j), mseMinTestIdx(j)] = min(mseTest(j,:));
    [mseMinTrain(j), mseMinTrainIdx(j)] = min(mseTrain(j,:));
    
end

modeMse = mode(mseMinTestIdx);
[minVarHii, minVarHiiIdx] = min(varHii);
stdTrain = std(mseTrain,1);
stdTest = std(mseTest,1);

figure();
plot(1:maxHiddenUnits,sum(mseTest,1)./iterations,'-');
hold on;
plot(1:maxHiddenUnits,sum(mseTrain,1)./iterations,'-');
hold on;
[mTe,mTeI] = min(sum(mseTest,1)./iterations);
[mTr,mTrI] = min(sum(mseTrain,1)./iterations);
plot(mTeI, mTe,'b*');
hold on;
plot(mTrI,mTr,'r*');
legend('mse test', 'mse train','min mse test','min mse train');
xlabel('liczba neuronów ukrytych');
ylabel('MSE');
title('B³¹d œredniokwadratowy');
hold off; 

rows = any(isnan(varHii),2);
varHii(rows,:) = [];

figure();
plot(1:maxHiddenUnits,sum(varHii,1)./iterations,'-');
xlabel('liczba neuronów ukrytych');
ylabel('var(hii)');
title('Wykres wariancji dŸwigni od liczby neuronów ukrytych');

%% Symulacja 50 sieci neuronowych, Ep i u
hiddUnits = 4;
net = [];
ypredTest = zeros(length(testData),50);
ypredTrain = zeros(length(trainData),50);
Ep = nan(1,50);
u = nan(1,50);

for i = 1:50
    rng('default')
    rng(i);
    net{i} = net1(trainDataX,trainDataY,hiddUnits);
    ypredTest(:,i) = net{i}(testDataX');
    ypredTrain(:,i) = net{i}(trainDataX');
    
    Z = calc_jacobian(net{i}, trainDataX); %okreœlenie macierzy Jacobiego
    rankZ = rank(Z);
    if rankZ==min(size(Z,1),size(Z,2)) %jeœli macierz Z jest pe³nego rzêdu
        H = Z*inv(Z'*Z)*Z';
        
        hii = diag(H);
        Ri = trainDataY-ypredTrain(:,i);
        
        Ep(i) = sqrt(1/length(trainData)*sum(((Ri./(1-hii)).^2))); %estymator b³êdu generalizacji
        u(i) = 1/length(trainData)*sum(sqrt(length(trainData)/size(Z,2)*hii)); %indeks stopnia nadmiernego dopasowania
    end
    
    H = [];
    
end
[minEp, minEpIdx] = min(Ep);
figure();
plot(Ep,u,'.')
xlabel('Ep')
ylabel('u')
hold on
plot(minEp, u(minEpIdx),'r.')
title(['Liczba neuronów w warstwie ukrytej: ' num2str(hiddUnits)])
%% Wybór najlepszej sieci
[minEp, minEpIdx] = min(Ep);

yTest = net{minEpIdx}(testDataX');
yTrain = net{minEpIdx}(trainDataX');

mseTest1 = mse(net{minEpIdx},testDataY,yTest');
mseTrain1 = mse(net{minEpIdx},trainDataY,yTrain');

weights = getwb(net{minEpIdx});
Iw = cell2mat(net{minEpIdx}.IW); % wagi wejœciowe
b1  = cell2mat(net{minEpIdx}.b(1)); % bias 1 (neuronów w warstwie ukrytej)
Lw = cell2mat(net{minEpIdx}.Lw); % wagi neuronów ukrytych do neuronu wyjœciowego
b2 = cell2mat(net{minEpIdx}.b(2)); % bias 2 (neuronu wyjœciowego)

view(net{minEpIdx})

Z = calc_jacobian(net{minEpIdx}, trainDataX); %okreœlenie macierzy Jacobiego
rankZ = rank(Z);
H = Z*inv(Z'*Z)*Z';

hii = diag(H);
figure(); 
histogram(hii,10); 
title('Histogram dŸwigni dla najlepszej sieci');
mHii = mean(hii);
q = size(Z,2);
u = 1/length(trainData)*sum(sqrt(length(trainData)/q*hii));

s = sqrt(1/(length(trainData)-q)*sum((trainDataY-yTrain').^2)); 

t = 2.00215;
T = t.*s.*sqrt(hii)';

figure();
subplot(1,2,1);
plot(trainDataX,yTrain)
hold on
plot(trainDataX,yTrain+T,'r')
hold on
plot(trainDataX,yTrain-T,'r')
hold on
plot(trainDataX,trainDataY,'.');
xlabel('X')
ylabel('Y')
title('Wykres funkcji wraz z przedzia³ami ufnoœci');
hold off;
subplot(1,2,2);
plot(trainDataX,hii);
hold on;
plot([-1 1],[mHii mHii],'k')
legend('DŸwignia h_i_i','œrednia h_i_i');
xlabel('X')
ylabel('hii')
title('Wykres dŸwigni');
hold off;

figure();
plot(testDataX,yTest);
hold on;
plot(testDataX, testDataY,'.');
legend('aproksymacja','dane testowe');
hold off;
