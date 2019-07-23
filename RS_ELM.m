function [Tr_Acc_RS_ELM, ind_RS_col, Beta_RS_Col, Round_Prune_RS] = RS_ELM(H, Tr_cls, Sp_Col)

T       = Tr_cls;
Num_Tr  = size(Tr_cls,1);
Num_Inp = size(Tr_cls,2);

%% Preprocessing
tr_y = [];
label = sort(unique(T));
Num_out = size(label,1);
for j = 1:length(label)
    tr_y = [tr_y T==label(j)];
end
T=tr_y;
%% Column-wise
warning off

total_col = 1:size(H,2);


%% Random-wise pruning
for N_Sp = 1:size(Sp_Col,2)
    size_Rel_col        =  round(Sp_Col(N_Sp)*size(H,2));
    Rand_col{:,N_Sp}    =  randperm(size(H,2));
    ind_RS_col{:,N_Sp}  =  Rand_col{:,N_Sp}(:,1:size_Rel_col);
    H_prune = H(:,ind_RS_col{:,N_Sp});
    Beta_RS_Col{:,N_Sp}   = pinv(H_prune)*T;
    
    Eval_cols_prune =  pinv(H_prune)*H_prune;
    Col_Eval_prune = [];
    Identity = eye(size(Eval_cols_prune,1));
    for i = 1 : size(Eval_cols_prune,1)
        temp  = sum (abs(Eval_cols_prune(:,i) - Identity(:,i)));
        Col_Eval_prune = [temp; Col_Eval_prune];
    end
    Round_Prune_RS(N_Sp) = sum(Col_Eval_prune);
    
    %% Calculate the training accuracy
    Y      = H_prune * Beta_RS_Col{:,N_Sp};
    %%%%%%%%%% Calculate training & testing classification accuracy
    Miss_Tr=0;
    for i = 1 : size(T, 1)
        [x, label_index_expected]=max(T(i,:));
        [x, label_index_actual]=max(Y(i,:));
        output(i)=label(label_index_actual);
        if label_index_actual~=label_index_expected
            Miss_Tr=Miss_Tr+1;
        end
    end
    Tr_Acc_RS_ELM(1,N_Sp) = 1-Miss_Tr/Num_Tr;
end
% diag_Eval =  abs(Eval_cols-eye(size(Eval_cols,1)));
% diag_Eval =   abs(diag(Eval_cols)-1);
% [val_diag, ind_diag] = sort(diag_Eval); %
