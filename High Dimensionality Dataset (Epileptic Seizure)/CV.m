%Panteleimon Manouselis AEM:9249
%Function created for Regression (Third) and Classification (Fourth) Exercise of Ypologistiki Noimosini
%% 5-fold cross Validation
function [error]=CV(init_fis,trnData,chkData,parti_data,Num_Feat,i,idx)
tic
error=zeros(5,1);
for k = 1:5
    %k=1:5 because of 5-fold cross Validation
    fprintf('\n Fold number %d\n', k);
    
    trn_index = parti_data.training(k);
    tst_index = parti_data.test(k);
    %             Identify the training indices in the #i fold of a
    %             partition of 12758 observations for 5-fold cross-validation
    %             Returns a vector of ones and zeros the size of the original
    %             trnData(:,end). If the index is 1 then the specific data
    %             point of trnData was used in cross validation as a training
    %             data If the index is 0 then the specific data point of
    %             trnData was used in cross validation as a test data
    
    
    % Spliting the trnData into training data and validation data used in Cross
    % Validation fold #i
    validation_data_input = trnData(tst_index, idx(1:Num_Feat(i)));
    validation_data_output = trnData(tst_index, end);
    validation_data=[validation_data_input validation_data_output];
    
    training_data_input = trnData(trn_index, idx(1:Num_Feat(i)));
    training_data_output = trnData(trn_index, end);
    training_data=[training_data_input training_data_output];
    
    %Calculating the FIS then finding the model output and the error
    [trnFis,trnError,~,valFis,valError]=anfis(training_data,init_fis,[60 0 0.01 0.9 1.1],[0 0 0 0],validation_data);
    %60 epoxes. Parapanw epoxes analonoun apeiro xrono
    
    y_hat = evalfis(valFis,chkData(:, idx(1:Num_Feat(i))));
    %Gia y_hat kai error xrisimopoioume check Data. Sto cross-Validation
    %eixame xrisimopoihsei mono trnData
    
    % Calculate the error
    error(k) = sum((y_hat - chkData(:, end)) .^ 2);
end
toc
end