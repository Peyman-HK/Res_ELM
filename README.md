# Matlab code for Res_ELM
Residual based Extreme Learning Machine

This Matlab code is related to the paper "Multimodal Sparse Classifier for Adolescent Brain Age Prediction"

Author(s)
Peyman Hosseinzadeh Kassani ; Alexej Gossmann ; Yu-Ping Wang
Published in: IEEE Journal of Biomedical and Health Informatics, June 2019, DOI: 10.1109/JBHI.2019.2925710
https://ieeexplore.ieee.org/document/8750866


![main flow](/../main/images/Res_ELM Flowchart.jpeg?raw=true "Residual based ELM pipeline")


You run Main_RES_ELM.m to get results on WBCD data (wbcd.mat). Residual based sparse classifier takes advantage of the residuals of an extreme learning machine (ELM) and rank neuouns that can be used to remove or sparsify the data. 

For the sake of comparision with traditional ELM, you run Trad_ELM.m. 
