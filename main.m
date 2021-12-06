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

% Operacja min-max (skalowanie danych do warto�ci [-1,1])
trainDataX = -1+((trainDataX-min(trainDataX))*2/(max(trainDataX)-min(trainDataX)));
trainDataY = -1+((trainDataY-min(trainDataY))*2/(max(trainDataY)-min(trainDataY)));
testDataX = -1+((testDataX-min(testDataX))*2/(max(testDataX)-min(testDataX)));
testDataY = -1+((testDataY-min(testDataY))*2/(max(testDataY)-min(testDataY)));

% Wizualizacja danych trenuj�cych i testowych
figure()
plot(trainDataX,trainDataY,'.')
hold on
plot(testDataX,testDataY,'.')
legend('train', 'test');
xlabel('X');
ylabel('Y');
title('Wizualizacja danych');
hold off

%% Wyb�r optymalnej ilo�ci neuron�w ukrytych - MSE
maxHiddenUnits = 13; % max liczba neuron�w ukrytych
iterations = 100; % liczba iteracji
mseTest = zeros(iterations,maxHiddenUnits); % maxierz b��d�w mse dla danych testowych
mseTrain = zeros(iterations,maxHiddenUnits); % macierz b��d�w mse dla danych trenuj�cych
varHii = nan(iterations,maxHiddenUnits); % wariancja d�wigni
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
        Z = calc_jacobian(net, trainDataX); %okre�lenie macierzy Jacobiego
        rankZ = rank(Z);
        if rankZ==min(size(Z,1),size(Z,2)) %je�li macierz Z jest pe�nego rz�du
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
%             title(['Liczba neuron�w ukrytych: ', num2str(i)]);
%             hold off;
%             
%             subplot(1,2,2);
%             plot(trainDataX,hii);
%             hold on;
%             plot([-1 1],[mHii mHii],'k');
%             xlabel('X');
%             ylabel('h_i_i');
%             legend('D�wignia h_i_i','�rednia hii')
            
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
xlabel('liczba neuron�w ukrytych');
ylabel('MSE');
title('B��d �redniokwadratowy');
hold off; 

rows = any(isnan(varHii),2);
varHii(rows,:) = [];

figure();
plot(1:maxHiddenUnits,sum(varHii,1)./iterations,'-');
xlabel('liczba neuron�w ukrytych');
ylabel('var(hii)');
title('Wykres wariancji d�wigni od liczby neuron�w ukrytych');

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
    
    Z = calc_jacobian(net{i}, trainDataX); %okre�lenie macierzy Jacobiego
    rankZ = rank(Z);
    if rankZ==min(size(Z,1),size(Z,2)) %je�li macierz Z jest pe�nego rz�du
        H = Z*inv(Z'*Z)*Z';
        
        hii = diag(H);
        Ri = trainDataY-ypredTrain(:,i);
        
        Ep(i) = sqrt(1/length(trainData)*sum(((Ri./(1-hii)).^2))); %estymator b��du generalizacji
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
title(['Liczba neuron�w w warstwie ukrytej: ' num2str(hiddUnits)])
%% Wyb�r najlepszej sieci
[minEp, minEpIdx] = min(Ep);

yTest = net{minEpIdx}(testDataX');
yTrain = net{minEpIdx}(trainDataX');

mseTest1 = mse(net{minEpIdx},testDataY,yTest');
mseTrain1 = mse(net{minEpIdx},trainDataY,yTrain');

weights = getwb(net{minEpIdx});
Iw = cell2mat(net{minEpIdx}.IW); % wagi wej�ciowe
b1  = cell2mat(net{minEpIdx}.b(1)); % bias 1 (neuron�w w warstwie ukrytej)
Lw = cell2mat(net{minEpIdx}.Lw); % wagi neuron�w ukrytych do neuronu wyj�ciowego
b2 = cell2mat(net{minEpIdx}.b(2)); % bias 2 (neuronu wyj�ciowego)

view(net{minEpIdx})

Z = calc_jacobian(net{minEpIdx}, trainDataX); %okre�lenie macierzy Jacobiego
rankZ = rank(Z);
H = Z*inv(Z'*Z)*Z';

hii = diag(H);
figure(); 
histogram(hii,10); 
title('Histogram d�wigni dla najlepszej sieci');
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
title('Wykres funkcji wraz z przedzia�ami ufno�ci');
hold off;
subplot(1,2,2);
plot(trainDataX,hii);
hold on;
plot([-1 1],[mHii mHii],'k')
legend('D�wignia h_i_i','�rednia h_i_i');
xlabel('X')
ylabel('hii')
title('Wykres d�wigni');
hold off;

figure();
plot(testDataX,yTest);
hold on;
plot(testDataX, testDataY,'.');
legend('aproksymacja','dane testowe');
hold off;
