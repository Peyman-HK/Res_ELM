function [Tr_Time,Tr_Acc, Beta_ELM] = Trad_ELM(H, Tr_cls)


T = Tr_cls;

%% Preprocessing 

tr_y = [];
label = sort(unique(T));
Num_out = size(label,1);
for j = 1:length(label)
    tr_y = [tr_y T==label(j)];
end
T=tr_y;

%% Calculate output weights OutputWeight (beta_i)
tic;
Beta_ELM = pinv(H) * T;
% Beta_ELM = inv(H'*H)*H' * T;

Tr_Time=toc;      

%% Calculate the training accuracy
Y  = H * Beta_ELM;                         
clear H;

%%%%%%%%%% Calculate training accuracy 

Miss_Tr=0;
for i = 1 : size(T, 1)
    [x, label_index_expected]=max(T(i,:));
    [x, label_index_actual]=max(Y(i,:));
    output(i)=label(label_index_actual);
    if label_index_actual~=label_index_expected
        Miss_Tr=Miss_Tr+1;
    end
end
Tr_Acc=1-Miss_Tr/size(T,1);

% Tr_Acc = sqrt(mse(T-Y)); 

% save('elm_model', 'Num_Inp', 'Num_out', 'InputWeight',...
%      'Num_Bias_H', 'OutputWeight', 'label');
