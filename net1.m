function net = net1(trainDataX,trainDataY,h_u)

% Input:
% - trainDataX - dane trenuj�ce x
% - trainDataY - dane trenuj�ce y
% - h_l - liczba neuron�w ukrytych

% Output:
% - net - sie� neuronowa (Perceptron dwuwarstwowy z jedn� warstw� neuron�w 
% ukrytych i jednym neuronem wyj�ciowym

rand('state',sum(100*clock));           %inicjalizacja generatora liczb 
                                        %pseudolosowych

net = feedforwardnet;
net.numInputs = 1;
net.numLayers = 2;
net.layers{1}.transferFcn = 'tansig'; % Funkcja aktywacji neuron�w ukrytych - tansig
net.layers{1}.dimensions = h_u;
net.layers{2}.transferFcn = 'purelin'; % Funkcja aktywacji neuronu wyj�ciowego - liniowa (purelin)
net.layers{2}.dimensions = 1;
net.inputConnect = [1 ;0];
net.layerConnect = [0 0; 1 0];
net.outputConnect = [0 1];

% Inicjalizacja wag i warto�ci sta�ych (ang. bias): inicjalizacja domy�lna 
% wg. Nguyena-Widrowa (initnw) lub liczby losowe z przedzia�u (-0.15, 0.15)

% net.inputWeights{1}.initFcn = 'initnw';

net.layers{1}.initFcn = 'initnw';
net.layers{2}.initFcn = 'initnw';
init(net);

% Algorytm uczenia
% Gradientowy wstecznej propagacji b��du BP (traingd) oko�o 100 epok, 
% nast�pnie algorytm LevenbergaMarquardta (trainlm) oko�o 200 epok

% ----------------------------- traingd -------------------------------
net.trainFcn = 'traingd';
net.trainParam.epochs = 100; % liczba epok
net.trainParam.goal = 0.0001; % kryerium stopu

net.trainParam.showWindow = false;
net = train(net, trainDataX.', trainDataY.');

% ----------------------------- trainlm -------------------------------

net.trainFcn = 'trainlm';
net.trainParam.epochs = 200;
net.trainParam.goal = 0.0001;

net.trainParam.showWindow = false;
net = train(net, trainDataX.', trainDataY.');

end

