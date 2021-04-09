%Panteleimon Manouselis AEM:9249
%Function created for Regression (Third) Exercise of Ypologistiki Noimosini
%% Function Plots the Learning Curve
function []=LCPlotter(trnError,valError)
    figure;
    plot([trnError valError],'LineWidth',2);
    grid on;
    xlabel('Number of Iterations','Interpreter','Latex');
    ylabel('Error','Interpreter','Latex');
    legend('Training Error','Validation Error');
    title('ANFIS Hybrid Training - Validation');
end