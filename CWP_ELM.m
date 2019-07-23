function [Tr_Acc_CWP_ELM, ind_CWP_col, Beta_CWP_Col, Round_H_Col, Round_Prune_Col] = CWP_ELM(H, Tr_cls, Sp_Col)

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
H_Pseudo = pinv(H);
Eval_cols =  H_Pseudo*H;
Col_Eval = [];
Identity = eye(size(Eval_cols,1));
for i = 1 : size(Eval_cols,1)
    temp  = sum (abs(Eval_cols(:,i) - Identity(:,i)));
    Col_Eval = [temp; Col_Eval];
end
[val_Col, ind_Col] = sort(Col_Eval);   %
Round_H_Col = sum(Col_Eval);
%% Column-wise pruning 
for N_Sp = 1:size(Sp_Col,2)
    
        
        Columns(N_Sp) = round(Sp_Col(N_Sp)*size(Col_Eval,1));
        ind_CWP_col{N_Sp}  = ind_Col(1:Columns(1,N_Sp)',:);
        H_prune = H(:,ind_CWP_col{N_Sp});
        %     H_Pseudo_prune = H_Pseudo(ind_CWP_col{:,N_Sp},:);
        %     T_prune   = T(ind_BDP_col{:,N_Sp},:);
        %     Beta_CWP_Col{:,N_Sp}   = pinv(H_prune)*T;
        Beta_CWP_Col{N_Sp}   = inv(H_prune'*H_prune)*H_prune'*T;
             
        Eval_cols_prune =  pinv(H_prune)*H_prune;  
        Col_Eval_prune = [];  
        Identity = eye(size(Eval_cols_prune,1)); 
        for i = 1 : size(Eval_cols_prune,1) 
            temp  = sum (abs(Eval_cols_prune(:,i) - Identity(:,i))); 
            Col_Eval_prune = [temp; Col_Eval_prune]; 
        end 
        Round_Prune_Col(N_Sp) = sum(Col_Eval_prune); 
        %% Calculate the training accuracy  
        Y      = H_prune * Beta_CWP_Col{N_Sp}; 
        %%%%%%%%%% Calculate training & testing classification accuracy 
        Miss_Tr=0; 
        for i = 1 : size(T, 1) 
            [x, label_index_expected] =  max(T(i,:)); 
            [x, label_index_actual]   =  max(Y(i,:)); 
            output(i)=label(label_index_actual); 
            if label_index_actual~=label_index_expected 
                Miss_Tr=Miss_Tr+1; 
            end 
        end 
        Tr_Acc_CWP_ELM(N_Sp) = 1-Miss_Tr/Num_Tr; 
     
end 
% diag_Eval =  abs(Eval_cols-eye(size(Eval_cols,1))); 
% diag_Eval =   abs(diag(Eval_cols)-1); 
% [val_diag, ind_diag] = sort(diag_Eval);   % 

