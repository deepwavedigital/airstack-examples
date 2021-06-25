% Copyright 2021, Deepwave Digital, Inc.
% SPDX-License-Identifier: BSD-3-Clause

% This application is to be used as a starting point for creating a deep neural network
% using MATLAB to be deployed on the AIR-T. An extremely simple network is created
% that calculates the average of the instantaneous power for an input buffer. The output
% of the application is a saved uff file that may be optimized using NVIDIA's TensorRT
% tool to create a plan file. That plan file is deployable on the AIR-T.

% Users may modify the layers in the neural_network function for the network that is best
% suited for that application. Note that any layer that is used must be supported by
% TensorRT so that it can be deployed on the AIR-T.

clear all; clc;

% Dummy input to train net and test results
N = 4096;   % Vector Length
M = 1;  %  Number of trials

Xtrain = rand(M, N);
Ytrain = rand(M, 1);  % dummy variable

% Calculate power
Nsq = Xtrain.^2;

%  Normalization constant
norm_factor = 1/(length(Nsq)/2);

% Power calculation
av_pwr = sum(Nsq.*norm_factor);

% Define neural network
mainpath = [
featureInputLayer(N, 'Name', 'input_buffer')
multiplicationLayer(2, 'Name', 'mul_1')
fullyConnectedLayer(1, 'Name', 'FC')
regressionLayer('Name', 'output')]

% Connect the layers
lgraph = layerGraph(mainpath);

% Connect the input buffer to the 2nd input of the multiplication layer in order
% to perform a squared operation
lgraph = connectLayers(lgraph, 'input_buffer', 'mul_1/in2');

%  Optonally plot the neural net
%  plot(lgraph)

%  The neural network cannot be modified until after it is trained
%  We perform a dummy training session so we can modify network later
minBatchsize = M;
maxEpochs = 1;
options =  trainingOptions('adam', 'ExecutionEnvironment', 'CPU',...
    'GradientThreshold', 1,...
    'MaxEpochs', maxEpochs,...
    'SequenceLength', 'longest', ...
    'Shuffle', 'never',...   
    'Verbose', 1,...
    'Plots', 'training-progress');

net = trainNetwork(Xtrain, Ytrain, lgraph, options);

%%  After training, we can assigned fixed values to weights and biases.
modNet = net.saveobj;
modNet.Layers(3).Weights = single(norm_factor.*ones(1,N));
modNet.Layers(3).Bias = single(0);
modNet = net.loadobj(modNet);

% Compare net calculated power to MATLAB calculated power
disp(['Calculated average power = ' ,  num2str(av_pwr)]);
nnet_pwr = predict(modNet, Xtrain);
disp(['Average Power as determined by neural net  = ',  num2str(nnet_pwr)]);

%  Both values match, modified neural net is calculating power correctly
%  Export modified neural net in an ONNX file
exportONNXNetwork(modNet, 'avg_pow_net.onnx')
disp


%% Optional:  Viewing outputs of specific layers - the activations() function
% activations(modNet,Xtrain,'input_buffer') 
% activations(modNet,Xtrain,'mul_1') 
% activations(modNet,Xtrain,'FC') %output of FC layer
% activations(modNet,Xtrain,'const_multiplier') %output of constant multiple layer
    