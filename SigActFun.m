function H = SigActFun(Tr_data,InputWeight,Bias)

%%%%%%%% Feedforward neural network using sigmoidal activation function
V=Tr_data*InputWeight'; ind=ones(1,size(Tr_data,1));
BiasMatrix=Bias(ind,:);      
V=V+BiasMatrix;
H = 1./(1+exp(-V));