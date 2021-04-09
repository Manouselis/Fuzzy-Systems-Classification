%Panteleimon Manouselis AEM:9249
%Function created for Regression (Third)  Exercise of Ypologistiki Noimosini
%% Function Plots the FIS membership Functions
function []=MFPlotter(fis,dimension)
figure
plotdim=sqrt(dimension);
plotdim1=ceil(plotdim);
plotdim2=ceil(dimension/plotdim1);
% n=plotdim1*plotdim2
for i=1:dimension
    subplot(plotdim1,plotdim2,i)
    hold on
    plotmf(fis,'input',i)
    ylabel('Degree of membership', 'Interpreter', 'latex')
    title(['Membership function for number ',num2str(i), ' input of FIS'],  'Interpreter', 'latex')
end
end