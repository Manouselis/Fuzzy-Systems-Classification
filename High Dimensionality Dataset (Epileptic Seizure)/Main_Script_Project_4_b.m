%Panteleimon Manouselis AEM:9249
%Script created for Classification (Fourth) Exercise of Ypologistiki Noimosini
%%
format compact
clear
clc
warning('off','all');
%suppresing warning outputs

%% Load data
data=csvread('data.csv',1,1);
%11500 SET dedomenwn

% The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178
%
% y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}:
%
% 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open
%
% 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed
%
% 3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area
%
% 2 - They recorder the EEG from the area where the tumor was located
%
% 1 - Recording of seizure activity
%
% All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure. Our motivation for creating this version of the data was to simplify access to the data via the creation of a .csv version of it. Although there are 5 classes most authors have done binary classification, namely class 1 (Epileptic seizure) against the rest.
%% Split data
[trnData,chkData,tstData]=split_scale(data,1);
%Splitting and Normalization

Num_Feat = [3 7 13 21]; % number of features

radii = [0.2 0.4 0.6 0.8 0.9]; % values for radii


%% GRID SEARCH & 5-fold cross validation

[idx, weights] = relieff(data(:, 1:end - 1), data(:, end), 100);
%Similar to ReliefF, RReliefF also penalizes the predictors that give different values to neighbors with the same response values, and rewards predictors that give different values to neighbors with different response values.
%However, RReliefF uses intermediate weights to compute the final predictor weights.
bar(weights(idx))
xlabel('Predictor rank')
ylabel('Predictor importance weight')
% katatasoume tis stiles se seira simantikotitas me vasi to varos tous

for i = 1 : length(Num_Feat)
    
    for j = 1 : length(radii)
        %dokimazoume olous tous sindiasmous diaforetikwn arithmon apo
        %features kai diaforetikon arithmon apo radii
        
        parti_data = cvpartition(trnData(:, end), 'KFold', 5);
        %Epilegetai h  stili apo to trnData mas kai diaxwrizetai
        %se dio sinola dedomenon (80% kai 20% tou arxikou sinolou
        %dedomenwn). Ayto ginetai 5 fores (5 set diaxorismenwn dedomenwn)
        
        %% Creating the initial FIS which later changes because of cross Validation
        
        %Building FIS
        init_fis=newfis('init_fis','FISType','sugeno');
        
        boolean_result=zeros(length(trnData),5,'logical'); % 5 because 5 are the number of Classes.
        
        for k=1:5
            boolean_result(:,k)=trnData(:,end)==k; %We find which lines of trnData have class 1 which have class 2 which have ... class 5
        end
        
        %We select the data from the trnData that we will use to build the
        %cluster center
        trnData_sel=[trnData(boolean_result(:,1),idx(1:Num_Feat(i))) trnData(boolean_result(:,1),end)];
        [cluster_center1,sigma1]=subclust(trnData_sel,radii(j));
        %[centers,sigma] = subclust(___) returns the sigma values specifying the range of influence of a cluster center in each of the data dimensions.
        %Cluster sigma values indicate the range of influence of the computed cluster centers in each data dimension.
        %cluster_centers1(:,1) periexei ta cluster centers tis stilis 1 twn trnData pou exoyn eksodo 1
        
        trnData_sel=[trnData(boolean_result(:,2),idx(1:Num_Feat(i))) trnData(boolean_result(:,2),end)];
        [cluster_center2,sigma2]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,3),idx(1:Num_Feat(i))) trnData(boolean_result(:,3),end)];
        [cluster_center3,sigma3]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,4),idx(1:Num_Feat(i))) trnData(boolean_result(:,4),end)];
        [cluster_center4,sigma4]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,5),idx(1:Num_Feat(i))) trnData(boolean_result(:,5),end)];
        [cluster_center5,sigma5]=subclust(trnData_sel,radii(j));
        
        
        if (length(cluster_center4) >150 ||length(cluster_center5) > 200)
            continue;
        end
        
        %if there are more than 500 or 600 cluster centers then the computational time is
        %very large an thus we choose not to create the FIS be (we will
        %have an explosion of rules)
        
        
        
        %ADDING THE INPUTS OF THE NEW FIS
        names_in={'in1','in2','in3','in4','in5','in6','in7','in8','in9','in10','in11','in12','in13','in14','in15','in16','in17','in18','in19','in20','in21','in22','in23','in24','in25','in26','in27','in28','in29','in30','in31','in32','in33','in34','in35','in36','in37','in38','in39','in40'};
        for k=1:Num_Feat(i)
            init_fis=addInput(init_fis,'Name',names_in{k});
        end
        
        
        %ADDING THE OUTPUT OF THE NEW FIS
        init_fis=addOutput(init_fis,'Name','output');
        
        for k=1:Num_Feat(i)
            for l=1:size(cluster_center1,1)
                init_fis=addMF(init_fis,names_in{k},'gaussmf',[sigma1(k) cluster_center1(l,k)]);
                %For gaussmf :Membership function parameters, specified as the vector [σ c], where σ is the standard deviation and c is the mean.
            end
            
            for l=1:size(cluster_center2,1)
                init_fis=addMF(init_fis,names_in{k},'gaussmf',[sigma2(k) cluster_center2(l,k)]);
            end
            
            for l=1:size(cluster_center3,1)
                init_fis=addMF(init_fis,names_in{k},'gaussmf',[sigma3(k) cluster_center3(l,k)]);
            end
            
            for l=1:size(cluster_center4,1)
                init_fis=addMF(init_fis,names_in{k},'gaussmf',[sigma4(k) cluster_center4(l,k)]);
            end
            
            for l=1:size(cluster_center5,1)
                init_fis=addMF(init_fis,names_in{k},'gaussmf',[sigma5(k) cluster_center5(l,k)]);
            end
        end
        
        
        %Adding the output membership function
        for l=1:size(cluster_center1,1)
            init_fis=addMF(init_fis,'output','constant',1);
        end
        for l=1:size(cluster_center2,1)
            init_fis=addMF(init_fis,'output','constant',2);
        end
        for l=1:size(cluster_center3,1)
            init_fis=addMF(init_fis,'output','constant',3);
        end
        for l=1:size(cluster_center4,1)
            init_fis=addMF(init_fis,'output','constant',4);
        end
        for l=1:size(cluster_center5,1)
            init_fis=addMF(init_fis,'output','constant',5);
        end
        
        
        %Adding the Rules
        array_of_rules=zeros(1,Num_Feat(i)+1+2);
        idex_mem=0;
        for l=1:size(cluster_center1,1)
            array_of_rules(1:end-2)=l;
            array_of_rules(end-1:end)=1;
            init_fis=addRule(init_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center2,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            init_fis=addRule(init_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center3,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            init_fis=addRule(init_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center4,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            init_fis=addRule(init_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center5,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            init_fis=addRule(init_fis,array_of_rules);
        end
        
        Rule_Grid(i, j) = length(init_fis.rule);
        if (Rule_Grid(i, j) == 1 || Rule_Grid(i,j) > 300)
            continue;
        end
        % if only one rule exists we cannot create FIS
        %if there are more than 800 rules then the computational time is
        %very large an thus we choose not to create the FIS
        
        %% 5-fold cross Validation
        fprintf('\n Number of features %d\n', Num_Feat(i));
        fprintf('\n Radii is equal to %d\n', radii(j));
        
        %%%%%%
        %Cross validation happens here
        error=CV(init_fis,trnData,chkData,parti_data,Num_Feat,i,idx);
        %%%%%%
        
        Tot_CV_error=sum(error);
        Average_CV_error=Tot_CV_error/5;
        Mean_error(i,j)=Average_CV_error/length(chkData(:,end));
    end
end
%Rule_Grid
%% Plotter (Erotima 2)

for i=1:length(Num_Feat)
    for j=1:length(radii)
        if(Mean_error(i,j)==0)
            Mean_error(i,j)=nan;
            %removing errors that are zero since they are fake (in reality
            %they are not zero)
        end
    end
end
[mini,inde1]=min(Mean_error,[],'omitnan');%returns a vector with the indices of the smallest values of each column of A( the NaN are ommited)
[mini,inde2]=min(mini);%we find the indice of the smallest value

figure
stem3(radii,Num_Feat,Mean_error,'filled')
hold on
stem3(radii(inde2),Num_Feat(inde1(inde2)),Mean_error(inde1(inde2),inde2),'r','filled'); %we plot the smallest value found differently than the other values
%for that we use the indices we have found above
grid on;
xlabel('Radius','Interpreter','Latex');
ylabel('Number of Features','Interpreter','Latex');
zlabel('Mean Error','Interpreter','Latex');

% Diagramma opou apeikonizetai to sfalma se sxesi me tin aktina gia kathe
% arithmo xaraktiristikwn
figure
plot(radii,Mean_error(1,:),'-or',radii,Mean_error(2,:),'--ob',radii,Mean_error(3,:),':om',radii,Mean_error(4,:),'-.og')
legend(['Number of Features is ', num2str(Num_Feat(1))],['Number of Features is ',num2str(Num_Feat(2))],['Number of Features is ',num2str(Num_Feat(3))],['Number of Features is ',num2str(Num_Feat(4))],'Interpeter','Latex')
xlabel('Radius','Interpreter','Latex');
ylabel('Mean Error','Interpreter','Latex');

% Diagramma opou apeikonizetai to sfalma se sxesi me ton arithmo
% xaraktiristikwn gia kathe aktina
figure
plot(Num_Feat,Mean_error(:,1),'-or',Num_Feat,Mean_error(:,2),'--ob',Num_Feat,Mean_error(:,3),':om',Num_Feat,Mean_error(:,4),'-.og',Num_Feat,Mean_error(:,5),'-.oc')
legend(['Radii is ', num2str(radii(1))],['Radii is ',num2str(radii(2))],['Radii is ',num2str(radii(3))],['Radii is ',num2str(radii(4))],['Radii is ',num2str(radii(5))],'Interpeter','Latex')
xlabel('Number of Features','Interpreter','Latex');
ylabel('Mean Error','Interpreter','Latex');

% Diagramma opou apeikonizetai to sfalma se sxesi me ton arithmo ton
% kanonwn
figure
stem(Rule_Grid,Mean_error,'filled')

xlabel('Number of Rules','Interpreter','Latex');
ylabel('Mean Error','Interpreter','Latex');
xlim([0 25]);

%% Final TSK Model Training
%Building FINAL FIS 
%Minimum Error when we have 13 features and 0.4 radius
        end_fis=newfis('end_fis','FISType','sugeno');
        
        boolean_result=zeros(length(trnData),5,'logical'); % 5 because 5 are the number of Classes.
        
        for k=1:5
            boolean_result(:,k)=trnData(:,end)==k; %We find which lines of trnData have class 1 which have class 2 which have ... class 5
        end
        j=2; %radius is 0.4
        trnData_sel=[trnData(boolean_result(:,1),idx(1:13)) trnData(boolean_result(:,1),end)];
        [cluster_center1,sigma1]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,2),idx(1:13)) trnData(boolean_result(:,2),end)];
        [cluster_center2,sigma2]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,3),idx(1:13)) trnData(boolean_result(:,3),end)];
        [cluster_center3,sigma3]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,4),idx(1:13)) trnData(boolean_result(:,4),end)];
        [cluster_center4,sigma4]=subclust(trnData_sel,radii(j));
        
        trnData_sel=[trnData(boolean_result(:,5),idx(1:13)) trnData(boolean_result(:,5),end)];
        [cluster_center5,sigma5]=subclust(trnData_sel,radii(j));
        
        
        %ADDING THE INPUTS OF THE NEW FIS
        names_in={'in1','in2','in3','in4','in5','in6','in7','in8','in9','in10','in11','in12','in13','in14','in15','in16','in17','in18','in19','in20','in21','in22','in23','in24','in25','in26','in27','in28','in29','in30','in31','in32','in33','in34','in35','in36','in37','in38','in39','in40'};
        for k=1:13%Num of Feat
            end_fis=addInput(end_fis,'Name',names_in{k});
        end
        
        %ADDING THE OUTPUT OF THE NEW FIS
        end_fis=addOutput(end_fis,'Name','output');
        
        for k=1:13
            for l=1:size(cluster_center1,1)
                end_fis=addMF(end_fis,names_in{k},'gaussmf',[sigma1(k) cluster_center1(l,k)]);
                %For gaussmf :Membership function parameters, specified as the vector [σ c], where σ is the standard deviation and c is the mean.
            end
            
            for l=1:size(cluster_center2,1)
                end_fis=addMF(end_fis,names_in{k},'gaussmf',[sigma2(k) cluster_center2(l,k)]);
            end
            
            for l=1:size(cluster_center3,1)
                end_fis=addMF(end_fis,names_in{k},'gaussmf',[sigma3(k) cluster_center3(l,k)]);
            end
            
            for l=1:size(cluster_center4,1)
                end_fis=addMF(end_fis,names_in{k},'gaussmf',[sigma4(k) cluster_center4(l,k)]);
            end
            
            for l=1:size(cluster_center5,1)
                end_fis=addMF(end_fis,names_in{k},'gaussmf',[sigma5(k) cluster_center5(l,k)]);
            end
        end
        
        
        %Adding the output membership function
        for l=1:size(cluster_center1,1)
            end_fis=addMF(end_fis,'output','constant',1);
        end
        for l=1:size(cluster_center2,1)
            end_fis=addMF(end_fis,'output','constant',2);
        end
        for l=1:size(cluster_center3,1)
            end_fis=addMF(end_fis,'output','constant',3);
        end
        for l=1:size(cluster_center4,1)
            end_fis=addMF(end_fis,'output','constant',4);
        end
        for l=1:size(cluster_center5,1)
            end_fis=addMF(end_fis,'output','constant',5);
        end
        
        
        %Adding the Rules
        array_of_rules=zeros(1,13+1+2);
        idex_mem=0;
        for l=1:size(cluster_center1,1)
            array_of_rules(1:end-2)=l;
            array_of_rules(end-1:end)=1;
            end_fis=addRule(end_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center2,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            end_fis=addRule(end_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center3,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            end_fis=addRule(end_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center4,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            end_fis=addRule(end_fis,array_of_rules);
        end
        idex_mem=l+idex_mem;
        for l=1:size(cluster_center5,1)
            array_of_rules(1:end-2)=l+idex_mem;
            array_of_rules(end-1:end)=1;
            end_fis=addRule(end_fis,array_of_rules);
        end

final_trnData=[trnData(:,idx(1:13)) trnData(:,end)];
final_chkData=[chkData(:,idx(1:13)) chkData(:,end)];
[trnFisfin,trnErrorfin,~,valFisfin,valErrorfin]=anfis(final_trnData,end_fis,[100 0 0.01 0.9 1.1],[0 0 0 0],final_chkData);
%Training our model for 100 epochs. Xrisimopoioumai tis veltistes times ton
%parametrwn. Gia auto oi stiles tou trnData kai chkData pou tha
%xrisimopoieithoun epilegontai me vasi to idx(1:13) dedomenou oti theloume
%13 xaraktiristika (13 stiles )

%% Predictions of trained model and real values (zitoumeno 1 erotima 3)
y_hat=evalfis(valFisfin,tstData(:,idx(1:13)));
%The below ten lines are for fixing the output (we want discrete values in
%the range of [1,5] not continuous values
for i=1:length(tstData)
    if (y_hat(i)<1)
        y_hat(i)=1;
    elseif (y_hat(i)>5)
        y_hat(i)=5;
    end
end
y_hat=round(y_hat);
y_actual=tstData(:,end);
error=y_actual-y_hat;

figure
stem(y_hat)
hold on
stem(y_actual)
hold off
grid on;
legend('Predicted Output','Actual Output','Interpeter','Latex')
xlabel('Sample Size','Interpreter','Latex');
ylabel('Magnitude of Output','Interpreter','Latex');

figure
stem(error,'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for final FIS','Interpreter','Latex');

%% Learning Curves (zitoumeno 2 erotima 3)
LCPlotter(trnErrorfin,valErrorfin)
title('');
%% Fuzzy set initial and afterwards figure (zitoumeno 3 erotima 3)
dimension=size(final_chkData,2)-1;
rand_input_fis=randperm(dimension,5);
%returns a row vector containing k unique integers selected randomly from 1 to n.

%Before the training
figure
for i=1:5
    subplot(2,3,i)
    hold on
    plotmf(end_fis,'input',rand_input_fis(i))
    ylabel('Degree of membership before training the final model', 'Interpreter', 'latex')
    title(['Membership function for number ',num2str(rand_input_fis(i)), ' input of FIS'],  'Interpreter', 'latex')
end

%After the training
figure
for i=1:5
    subplot(2,3,i)
    hold on
    plotmf(valFisfin,'input',rand_input_fis(i))
    ylabel('Degree of membership after training the final model', 'Interpreter', 'latex')
    title(['Membership function for number ',num2str(rand_input_fis(i)), ' input of FIS'],  'Interpreter', 'latex')
end

%% Aksiologisi Montelou(zitoumeno 4 erotima 3)
[EM,order] = confusionmat(y_actual,y_hat);
figure
cm = confusionchart(EM,order);
title('Error Matrix')
%WHERE EM is the error matrix

EM=EM'; %the confusion matrix command produces an error matrix that is anastrofos to the ERROR Matrix given in the ekfonisi of the ergasia


% Overall Accuracy FOR ERROR MATRIX
OA_final=0;
for k=1:size(EM,1)
    OA_final=OA_final+EM(k,k);
end
OA_final=(1/length(y_hat))*OA_final;

% Producer's Accuracy - User's Accuracy
x_r=zeros(1,size(EM,1));
x_c=zeros(1,size(EM,1));
for i=1:size(EM,1)
    for j=1:size(EM,2)
        x_r(i)=EM(i,j)+x_r(i); % Plithos ton stoixeiwn pou taksinomithikan stin klasi i
        x_c(i)=EM(j,i)+x_c(i); %Plithos ton stoixeiwn pou anikoyn stin klasi i
    end
end

% Akrivia Paragwgou
PA=zeros(1,size(EM,1));
for i=1:size(EM,1)
    PA(i)=EM(i,i)/x_c(i);
end

% Akriveia Xristi
UA=zeros(1,size(EM,1));
for i=1:size(EM,1)
    UA(i)=EM(i,i)/x_r(i);
end

% K hat
a=0;
b=0;
for i=1:length(EM)
    a=a+EM(i,i);
end
a=length(y_hat)*a;

for i=1:length(EM)
    b=b+x_r(i)*x_c(i);
end

k_hat=(a-b)/((length(y_hat).^2)-b);

%Error Matrix
EM

%Overall accuracy
OA_final

%Akriveia Paragwgou 
PA

%Akriveia xristi
UA

%K hat gia ola ta TSK
k_hat

%Number of rules for each FIS
fprintf('Number of rules for trained Final FIS:\n');
length(valFisfin.Rules)