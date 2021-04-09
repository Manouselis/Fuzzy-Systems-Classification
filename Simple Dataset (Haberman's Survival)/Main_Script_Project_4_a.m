%Panteleimon Manouselis AEM:9249
%Script created for Classification (Fourth) Exercise of Ypologistiki Noimosini
%%
tic
format compact
clear
clc
warning('off','all');
%suppresing warning outputs

%% Load data
data=load('haberman.data');
%3 eisodoi, 1 eksodos. 306 SET dedomenwn
% Attribute Information:
%
% 1. Age of patient at time of operation (numerical)
% 2. Patient's year of operation (year - 1900, numerical)
% 3. Number of positive axillary nodes detected (numerical)
% 4. Survival status (class attribute)
% -- 1 = the patient survived 5 years or longer
% -- 2 = the patient died within 5 year
%% Split data
percentage_ones=[15;0;0];
while(max(percentage_ones)-min(percentage_ones)>5)
    
    [trnData,chkData,tstData]=split_scale(data,1);
    %Splitting and Normalization
    
    %% Checking if the frequency of the samples with the same class is split equally in trnData,chkData,tstData
    %For trnData
    
    num_of_ones=zeros(size(trnData,1),1);
    for i=1:size(trnData,1)
        if(trnData(i,4)==1)
            num_of_ones(i)=1;
        end
    end
    total_num_of_ones=sum(num_of_ones);
    percentage_ones_trn=100*total_num_of_ones/size(trnData,1)
    
    %For tstData
    num_of_ones=zeros(size(tstData,1),1);
    for i=1:size(tstData,1)
        if(tstData(i,4)==1)
            num_of_ones(i)=1;
        end
    end
    total_num_of_ones=sum(num_of_ones);
    percentage_ones_tst=100*total_num_of_ones/size(tstData,1)
    
    %For chkData
    num_of_ones=zeros(size(chkData,1),1);
    for i=1:size(chkData,1)
        if(chkData(i,4)==1)
            num_of_ones(i)=1;
        end
    end
    total_num_of_ones=sum(num_of_ones);
    percentage_ones_chk=100*total_num_of_ones/size(chkData,1)
    
    percentage_ones=[percentage_ones_trn;percentage_ones_chk;percentage_ones_tst];
    if((max(percentage_ones)-min(percentage_ones)<5))
        fprintf('Frequency of the samples with the same class is split equally in trnData,chkData,tstData\n');
        break;
    end
    fprintf('Frequency of the samples with the same class is NOT split equally in trnData,chkData,tstData. Thus we repeat the data split\n');
end


%% TSK Models(4)

%% TSK 1
opt = genfisOptions('SubtractiveClustering');
opt.ClusterInfluenceRange=0.15; %prwti akraia timi. Efoson exoume poli mikro influence range tha dimiourgithoun polloi kanones
fis1=genfis(trnData(:,(1:end-1)), trnData(:,end), opt);
% MFPlotter(fis1,3)
% plotmf(fis1,'output',1)
for i= 1:length(fis1.output(1).mf)
    fis1.output(1).mf(i).type = 'constant';
end
[trnFis1,trnError1,~,valFis1,valError1]=anfis(trnData,fis1,[100 0 0.01 0.9 1.1],[0 0 0 0],chkData);


%% TSK 2
opt2= genfisOptions('SubtractiveClustering');
opt2.ClusterInfluenceRange=0.85; %deuteri akraia timi. Efoson exoume poli megalo influence range tha dimiourgithoun ligoi kanones
fis2=genfis(trnData(:,(1:end-1)), trnData(:,end), opt2);
for i= 1:length(fis2.output(1).mf)
    fis2.output(1).mf(i).type = 'constant';
end
[trnFis2,trnError2,~,valFis2,valError2]=anfis(trnData,fis2,[100 0 0.01 0.9 1.1],[0 0 0 0],chkData);


%% TSK 3
%Building FIS
% Radius is very small (0.15) in order to create a lot of rules.
fis3=newfis('fis3','FISType','sugeno');

class1=data(find(data(:,4)==1),:); %gia to TSK 3 prepei na xwrisoume ta dedomena me vasi tis klaseis tous
class2=data(find(data(:,4)==2),:);
class=[class1' class2']'; %dedomena katanemimena me tetoia seira wste prwta na einai h klasi 1 kai meta h klasi 2

boolean=trnData(:,4)==1; %We find which lines of trnData have class 1
boolean2=trnData(:,4)==2; %We find which lines of trnData have class 2 (final column is 2 )
[cluster_centers1,sigma1]=subclust(trnData(boolean,:),0.15); %we find the cluser center for only the training data that belong to class 1
[cluster_centers2,sigma2]=subclust(trnData(boolean2,:),0.15); %we find the cluser center for only the training data that belong to class 2
%[centers,sigma] = subclust(___) returns the sigma values specifying the range of influence of a cluster center in each of the data dimensions.
%Cluster sigma values indicate the range of influence of the computed cluster centers in each data dimension.

%cluster_centers1(:,1) periexei ta cluster centers tis stilis 1 twn trnData
% pou exoyn eksodo 1


%ADDING THE INPUtS OF THE NEW FIS
fis3=addInput(fis3,'Name','Age1');
fis3=addInput(fis3,'Name','YearOfOperation1');
fis3=addInput(fis3,'Name','AuxiliaryNodes1');

% fis3=addInput(fis3,'Name','Age2');
% fis3=addInput(fis3,'Name','YearOfOperation2');
% fis3=addInput(fis3,'Name','AuxiliaryNodes2');


%ADDING THE OUTPUT OF THE NEW FIS
fis3=addOutput(fis3,'Name','output1');


%Adding the input membership functions
for i=1:size(cluster_centers1,1)
    fis3=addMF(fis3,'Age1','gaussmf',[sigma1(1) cluster_centers1(i,1)]);
    %For gaussmf :Membership function parameters, specified as the vector [σ c], where σ is the standard deviation and c is the mean.
    fis3=addMF(fis3,'YearOfOperation1','gaussmf',[sigma1(2) cluster_centers1(i,2)]);
    fis3=addMF(fis3,'AuxiliaryNodes1','gaussmf',[sigma1(3) cluster_centers1(i,3)]);
end
for i=1:size(cluster_centers2,1)
    fis3=addMF(fis3,'Age1','gaussmf',[sigma2(1) cluster_centers2(i,1)]);
    fis3=addMF(fis3,'YearOfOperation1','gaussmf',[sigma2(2) cluster_centers2(i,2)]);
    fis3=addMF(fis3,'AuxiliaryNodes1','gaussmf',[sigma2(3) cluster_centers2(i,3)]);
end
%Notice that each membership function type is gaussmf (Gaussian type membership function) and the parameters of each membership function are [a b],
%where a represents the spread coefficient of the Gaussian curve and b represents the center of the Gaussian curve.
%MF1 captures the position and influence of the first cluster for the input variable 1 (C(1,1)=b, S(1)=a )


%Adding the output membership function
for i=1:size(cluster_centers1,1)
    fis3=addMF(fis3,'output1','constant',1);
    %h eksodos einai mia stathera 1 (pio panw diaxwrisame ta dedomena mas ara twra
    %exoyme mono auta poy einai 1 to class).
end
for i=1:size(cluster_centers2,1)
    fis3=addMF(fis3,'output1','constant',2);
end


%Adding the Rules
for i=1:size(cluster_centers1,1)
    fis3=addRule(fis3,[i i i i 1 1]);
    %"Age1==mfi & YearOfOperation1==mfi & AuxiliaryNodes1==mfi => output1=mfi
    %(1)" the last 2 ones represent the weights
end
for j=1:size(cluster_centers2,1)
    fis3=addRule(fis3,[i+j i+j i+j i+j 1 1]);
    %"Age1==mf(i+j) & YearOfOperation1==mf(i+j) & AuxiliaryNodes1==mf(i+j) =>
    %output1=mf(i+j)
    %(1)" the last 2 ones represent the weights
end

%Training the fis

% a=reshape(trnData(:,1),[],2);
% b=reshape(trnData(:,2),[],2);
% c=reshape(trnData(:,3),[],2);
% d=reshape(trnData(:,4),[],2);
% trnDataNew=[a b c d]
%
% a=reshape(chkData(:,1),[],2);
% b=reshape(chkData(:,2),[],2);
% c=reshape(chkData(:,3),[],2);
% d=reshape(chkData(:,4),[],2);
% chkDataNew=[a b c d]


[trnFis3,trnError3,~,valFis3,valError3]=anfis(trnData,fis3,[100 0 0.01 0.9 1.1],[0 0 0 0],chkData);

% aka=trnData(find(trnData(:,4)==1),:)
% aka2=trnData(find(trnData(:,4)==2),:)
% sz=size(aka2,1);
% aka2((sz+1:size(aka,1)),1:3)=nan
% trnData=[aka(:,1:3) aka2(:,1:3) aka(:,end) aka2(:,end)]

%% TSK 4
%Building FIS
% Radius is very large (0.85) in order to create few rules.
fis4=newfis('fis4','FISType','sugeno');

boolean=trnData(:,4)==1; %We find which lines of trnData have class 1
boolean2=trnData(:,4)==2; %We find which lines of trnData have class 2 (final column is 2 )
[cluster_centers1,sigma1]=subclust(trnData(boolean,:),0.85); %we find the cluser center for only the training data that belong to class 1
[cluster_centers2,sigma2]=subclust(trnData(boolean2,:),0.85); %we find the cluser center for only the training data that belong to class 2
%[centers,sigma] = subclust(___) returns the sigma values specifying the range of influence of a cluster center in each of the data dimensions.
%Cluster sigma values indicate the range of influence of the computed cluster centers in each data dimension.

%cluster_centers1(:,1) periexei ta cluster centers tis stilis 1 twn trnData
% pou exoyn eksodo 1


%ADDING THE INPUtS OF THE NEW FIS
fis4=addInput(fis4,'Name','Age');
fis4=addInput(fis4,'Name','YearOfOperation');
fis4=addInput(fis4,'Name','AuxiliaryNodes');

%ADDING THE OUTPUT OF THE NEW FIS
fis4=addOutput(fis4,'Name','output');


%Adding the input membership functions
for i=1:size(cluster_centers1,1)
    fis4=addMF(fis4,'Age','gaussmf',[sigma1(1) cluster_centers1(i,1)]);
    %For gaussmf :Membership function parameters, specified as the vector [σ c], where σ is the standard deviation and c is the mean.
    fis4=addMF(fis4,'YearOfOperation','gaussmf',[sigma1(2) cluster_centers1(i,2)]);
    fis4=addMF(fis4,'AuxiliaryNodes','gaussmf',[sigma1(3) cluster_centers1(i,3)]);
end
for i=1:size(cluster_centers2,1)
    fis4=addMF(fis4,'Age','gaussmf',[sigma2(1) cluster_centers2(i,1)]);
    fis4=addMF(fis4,'YearOfOperation','gaussmf',[sigma2(2) cluster_centers2(i,2)]);
    fis4=addMF(fis4,'AuxiliaryNodes','gaussmf',[sigma2(3) cluster_centers2(i,3)]);
end

%Adding the output membership function
for i=1:size(cluster_centers1,1)
    fis4=addMF(fis4,'output','constant',1);
end
for i=1:size(cluster_centers2,1)
    fis4=addMF(fis4,'output','constant',2);
end


%Adding the Rules
for i=1:size(cluster_centers1,1)
    fis4=addRule(fis4,[i i i i 1 1]);
end
for j=1:size(cluster_centers2,1)
    fis4=addRule(fis4,[i+j i+j i+j i+j 1 1]);
end


[trnFis4,trnError4,~,valFis4,valError4]=anfis(trnData,fis4,[100 0 0.01 0.9 1.1],[0 0 0 0],chkData);

%% Aksiologisi Montelwn
% Error Matrices FOR ALL TSK
y_hat=zeros(length(tstData),4);

y_hat(:,1)=evalfis(valFis1,tstData(:,(1:end-1)));
y_hat(:,2)=evalfis(valFis2,tstData(:,(1:end-1)));
y_hat(:,3)=evalfis(valFis3,tstData(:,(1:end-1)));
y_hat(:,4)=evalfis(valFis4,tstData(:,(1:end-1)));

%Kanoume swsto round tis times poy h entoli round(y_hat) den tha kataferne
%na kanei swsta
for i=1:length(tstData)
    for j=1:4
        if (y_hat(i,j)<1)
            y_hat(i,j)=1;
        elseif (y_hat(i,j)>2)
            y_hat(i,j)=2;
        end
    end
end

y_hat=round(y_hat);
y_actual=tstData(:,end);
EM=zeros(2,2,4); % dimensions because we create all the error matrices and put them into one array
for j=1:4
    for i=1:length(y_hat)
        if(y_hat(i,j)==y_actual(i)) %stoixeia kirias diagwniou gia to error matrix j
            if(y_hat(i,j)==1) %Kaname prediction 1 kai ontws htan 1 gia to error matrix j
                EM(1,1,j)=EM(1,1,j)+1;
            else %Kaname prediction 2 kai ontws htan 2 gia to error matrix j
                EM(2,2,j)=EM(2,2,j)+1;
            end
        else % Kaname lathos prediction gia to error matrix j
            if(y_hat(i,j)==1) %Kaname prediction 1 kai htan 2 to pragmatiko
                EM(1,2,j)=EM(1,2,j)+1;
            else %Kaname prediction 2 kai htan 1 to pragmatiko
                EM(2,1,j)=EM(2,1,j)+1;
            end
        end
    end
end

% Overall Accuracy FOR ALL ERROR MATRICES
OA_TSK=zeros(4,1);
for j=1:4
    OA_TSK(j)=(1/length(y_hat))*(EM(1,1,j)+EM(2,2,j));
end

% Producer's Accuracy - User's Accuracy
for j=1:4
    for i=1:2
        x_r(i,j)=(EM(i,1,j)+EM(i,2,j)); % Plithos ton stoixeiwn pou taksinomithikan stin klasi i gia tsk j
        x_c(i,j)=(EM(1,i,j)+EM(2,i,j)); %Plithos ton stoixeiwn pou anikoyn stin klasi i gia tsk j
    end
end

% Akrivia Paragwgou
for j=1:4
    PA(1,j)=EM(1,1,j)/x_c(1,j);
    PA(2,j)=EM(2,2,j)/x_c(2,j);
end

% Akriveia Xristi
for j=1:4
    UA(1,j)=EM(1,1,j)/x_r(1,j);
    UA(2,j)=EM(2,2,j)/x_r(2,j);
end

% K hat
for j=1:4
    k_hat(j)=((length(y_hat))*(EM(1,1,j)+EM(2,2,j))-(x_r(1,j)*x_c(1,j)+x_r(2,j)*x_c(2,j)))/((length(y_hat))^2-(x_r(1,j)*x_c(1,j)+x_r(2,j)*x_c(2,j)));
end

%Error Matrix gia ola ta TSK
EM

%Overall accuracy gia ola ta TSK
OA_TSK

%Akriveia Paragwgou gia ola ta TSK
PA

%Akriveia xristi gia ola ta TSK
UA

%K hat gia ola ta TSK
k_hat

%Number of rules for each FIS
fprintf('Number of rules for each FIS:\n');
length(valFis1.Rules)
length(valFis2.Rules)
length(valFis3.Rules)
length(valFis4.Rules)
for i=1:4
[EM(:,:,i),order(:,i)] = confusionmat(y_actual,y_hat(:,i));
figure
cm = confusionchart(EM(:,:,i),order(:,i));
title('Error Matrix')
%WHERE EM is the error matrix
EM(:,:,i)=EM(:,:,i)';
end


%% Plots
%% Fuzzy set after validation (zitoumeno 1)
MFPlotter(valFis1,size(chkData,2)-1)
MFPlotter(valFis2,size(chkData,2)-1)
MFPlotter(valFis3,size(chkData,2)-1)
MFPlotter(valFis4,size(chkData,2)-1)

%% Learning curves (zitoumeno 2)
LCPlotter(trnError1,valError1);
title('ANFIS Classification Part A FIS 1','Interpreter','Latex');
LCPlotter(trnError2,valError2);
title('ANFIS Classification Part A FIS 2','Interpreter','Latex');
LCPlotter(trnError3,valError3);
title('ANFIS Classification Part A FIS 3','Interpreter','Latex');
LCPlotter(trnError4,valError4);
title('ANFIS Classification Part A FIS 4','Interpreter','Latex');

%% Prediction Error
%We compare evaluated output of FIS (where input is the test data ) with
%the actual testData output
figure
stem(tstData(:,end)-y_hat(:,1),'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for first FIS','Interpreter','Latex');

figure
stem(tstData(:,end)-y_hat(:,2),'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for Second FIS','Interpreter','Latex');


figure
stem(tstData(:,end)-y_hat(:,3),'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for Third FIS','Interpreter','Latex');


figure
stem(tstData(:,end)-y_hat(:,4),'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for Fourth FIS','Interpreter','Latex');