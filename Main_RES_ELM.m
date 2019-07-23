% Residual error based_ELM Model
clc , close all , clear all
global H
load wbcd.mat;
Data_All    = [x y];
num_points  = size(Data_All,1);
Factor      = round(0.9*num_points);
Sp_Col  = .10:0.05:0.85;
Num_Hid = 50:15:700;
lbl_val     = sort(unique(Data_All(:,end)));
if lbl_val(1,1)==0
    lbl_val=lbl_val+1;
    Data_All(:,end)=Data_All(:,end)+1;
end

for N_Hid = 1:size(Num_Hid,2)
    Perm           =       randperm(num_points)';
    Tr_data        =       Data_All(Perm(1:Factor),1:end-1);
    Tr_cls         =       Data_All(Perm(1:Factor),end);
    Te_data        =       Data_All(Perm(Factor+1:end),1:end-1);
    Te_cls         =       Data_All(Perm(Factor+1:end),end);
    N              =       size(Tr_data,1);
    tr_y = [];
    label = sort(unique(Tr_cls));
    Num_out = size(label,1);
    for j = 1:length(label)
        tr_y = [tr_y Tr_cls==label(j)];
    end
    Tr_cls_indic = tr_y;
    
    %% Calculate weights & biases
    Num_Tr  = size(Tr_data,1);
    Num_Inp = size(Tr_data,2);
    
    InputWeight = rand(round(Num_Hid(N_Hid)),Num_Inp)*2-1;
    Bias = rand(1,round(Num_Hid(N_Hid)))*2-1;
    H     = SigActFun(Tr_data,InputWeight,Bias);
    H_test      = SigActFun(Te_data,InputWeight,Bias);
    
    %% ELM Sigmoid
    tic;
    [~, Tr_Acc_ELM, Beta_ELM] = Trad_ELM(H, Tr_cls);
    Tr_time_ELM = toc;
    Perf_ELM_Tr(N_Hid) = Tr_Acc_ELM;
    Time_ELM_Tr(N_Hid) = Tr_time_ELM;
    
    tic;
    Te_ELM       = H_test * Beta_ELM;
    Te_time_ELM  = toc;
    [y_val y_te2]  = max(Te_ELM, [],2);
    Te_acc_ELM     = sum(y_te2 == Te_cls)/length(Te_ELM);  % Accuracy
    Perf_ELM_Te(N_Hid)  = Te_acc_ELM;
    Time_ELM_Te(N_Hid)  = Te_time_ELM;
    
    %% CWP_ELM Sig
    tic;
    [Tr_Acc_CWP_ELM, Ind_CWP, Beta_CWP_ELM] = CWP_ELM(H, Tr_cls, Sp_Col);
    Tr_time_CWPELM = toc;
    Tr_Time_CWP(N_Hid)   = Tr_time_CWPELM;
    
    % Test stage RES-ELM
    for ii = 1 : size(Sp_Col,2)
        tic;
        H_prune_te                 = H_test(:,Ind_CWP{ii});
        Te_CWP_ELM                 = H_prune_te * Beta_CWP_ELM{ii};
        [~, y_te2]                 = max(Te_CWP_ELM, [],2);
        Te_Acc_CWP                = sum(y_te2 == Te_cls)/length(Te_CWP_ELM);
        Perf_CWP_Te(N_Hid, ii)       = Te_Acc_CWP;
        Time_CWP_Te(N_Hid, ii)      = toc;
    end
    
    %% RS_ELM Sig
    tic;
    [Tr_Acc_RS_ELM, Ind_RS, Beta_RS_Col] = RS_ELM(H, Tr_cls, Sp_Col);
    Tr_time_RS_ELM = toc;
    Tr_Time_CWP(N_Hid)   = Tr_time_RS_ELM;
    
    % Test stage RS-ELM
    for ii = 1 : size(Sp_Col,2)
        tic;
        H_prune_te                 = H_test(:,Ind_RS{1,ii});
        Te_RS_ELM                 = H_prune_te * Beta_RS_Col{1,ii};
        [~, y_te2]                 = max(Te_RS_ELM, [],2);
        Te_Acc_RS                = sum(y_te2 == Te_cls)/length(Te_RS_ELM);
        Perf_RS_Te(N_Hid, ii)       = Te_Acc_RS;
        Time_RS_Te(N_Hid, ii)       = toc;
    end
end

clc
%% RES-ELM performance
MAX_RES_ELM   = max(max(max(max(Perf_CWP_Te))))
Mean_RES_ELM  = max(mean(Perf_CWP_Te))

%% Randomly Sparsed-ELM (RS-ELM) performance
MAX_RS_ELM = max(max(max(Perf_RS_Te)))
Mean_RS_ELM  = max(mean(Perf_RS_Te))

%% ELM performance
MAX_ELM = max(max(max(Perf_ELM_Te)))
Mean_ELM  = max(mean(Perf_ELM_Te))

[Mean_ELM Mean_RS_ELM Mean_RES_ELM]

