data = readtable('football_encoded.csv');
head(data);

%% ============================================================
%  STEP 1 : Train-Test Split (80% Train, 20% Test)
% =============================================================

cv = cvpartition(height(data),'HoldOut',0.20);

trainData = data(training(cv),:);
testData  = data(test(cv),:);


%% ============================================================
%  STEP 2 : Define Target Variable
% =============================================================

target = "match_outcome";


%% ============================================================
%  STEP 3 : Define Numerical Features (Only These Will Be Scaled)
% =============================================================

numericVars = {
'home_elo'
'away_elo'
'elo_diff'

'home_avg_overall'
'home_maoverall'
'home_avg_attack'
'home_avg_defense'
'home_avg_pace'
'home_avg_shooting'
'home_avg_passing'

'away_avg_overall'
'away_maoverall'
'away_avg_attack'
'away_avg_defense'
'away_avg_pace'
'away_avg_shooting'
'away_avg_passing'

'overall_diff'
'attack_diff'
'defense_diff'

'home_form_scored'
'home_form_conceded'
'home_form_win_rate'

'away_form_scored'
'away_form_conceded'
'away_form_win_rate'
};


%% ============================================================
%  STEP 4 : Calculate Mean and Standard Deviation
%           (ONLY FROM TRAINING DATA)
% =============================================================

mu = mean(trainData{:,numericVars});
sigma = std(trainData{:,numericVars});

% Avoid division by zero
sigma(sigma==0) = 1;


%% ============================================================
%  STEP 5 : Scale Training Data
% =============================================================

trainData{:,numericVars} = ...
    (trainData{:,numericVars} - mu) ./ sigma;


%% ============================================================
%  STEP 6 : Scale Test Data
%           (Use SAME mu and sigma from Training Data)
% =============================================================

testData{:,numericVars} = ...
    (testData{:,numericVars} - mu) ./ sigma;


%% ============================================================
%  STEP 7 : Separate Features (X) and Target (Y)
% =============================================================

XTrain = removevars(trainData,target);
YTrain = trainData.(target);

XTest = removevars(testData,target);
YTest = testData.(target);


%% ============================================================
%  STEP 8 : Save Scaling Parameters
%           (Needed During Deployment)
% =============================================================

save('scaler.mat','mu','sigma','numericVars');


%% ============================================================
%  DATA IS NOW READY FOR MODEL TRAINING
% =============================================================

disp('Training and Testing data prepared successfully!');
head(trainData);

%% ============================================================
% STEP 9 (REVISED) : Logistic Regression with Cost Matrix
% ============================================================
% Same cost matrix used in Decision Tree and Random Forest
costMatrix = [0 1 1;
              2 0 2;
              1 1 0];

t = templateLinear('Learner','logistic');
logisticModel = fitcecoc(XTrain, YTrain, 'Learners', t, 'Cost', costMatrix);

%% ============================================================
% STEP 10 : Predict on Test Data
% ============================================================
YPred = predict(logisticModel, XTest);

%% ============================================================
% STEP 11 : Test Accuracy
% ============================================================
accuracy = mean(YPred == YTest);
fprintf('Logistic Regression Accuracy = %.2f%%\n', accuracy*100);

%% ============================================================
% STEP 11B : Training Accuracy (to check overfitting)
% ============================================================
YPredTrain = predict(logisticModel, XTrain);
accuracyTrain = mean(YPredTrain == YTrain);
fprintf('Logistic Regression TRAINING Accuracy = %.2f%%\n', accuracyTrain*100);

%% ============================================================
% STEP 12 : Confusion Matrix
% ============================================================
confMat = confusionmat(YTest, YPred);
disp('Logistic Regression Confusion Matrix');
disp(confMat);

%% ============================================================
% STEP 13 : Confusion Chart
% ============================================================
figure;
confusionchart(YTest, YPred);
title('Logistic Regression Confusion Matrix (with Cost Matrix)');

%% ============================================================
% STEP 13B : Per-Class Recall
% ============================================================
classes = unique(YTest);
for i = 1:length(classes)
    c = classes(i);
    recall = sum(YPred == c & YTest == c) / sum(YTest == c);
    fprintf('Recall for class %s = %.2f%%\n', string(c), recall*100);
end

tabulate(data.match_outcome)
tabulate(YTrain)


%% ============================================================
% STEP 14 (REVISED) : Decision Tree with Cost Matrix
% ============================================================
% Cost matrix rows = actual class, columns = predicted class
% Format: cost(i,j) = cost of predicting class j when actual is class i
% Classes order: [0 (Away), 1 (Draw), 2 (Home)]
costMatrix = [0 1 1;    % actual Away: cost of predicting Draw=1, Home=1
              2 0 2;    % actual Draw: cost of predicting Away=2, Home=2 (reduced from 5 to 2)
              1 1 0];   % actual Home: cost of predicting Away=1, Draw=1

treeModel = fitctree(XTrain, YTrain, ...
    'MaxNumSplits', 20, ...
    'MinLeafSize', 50, ...
    'Cost', costMatrix);

%% ============================================================
% STEP 15 : Predict on Test Data
% ============================================================
YPredTree = predict(treeModel, XTest);

%% ============================================================
% STEP 16 : Test Accuracy
% ============================================================
accuracyTree = mean(YPredTree == YTest);
fprintf('Decision Tree Accuracy = %.2f%%\n', accuracyTree*100);

%% ============================================================
% STEP 17 : Training Accuracy
% ============================================================
YPredTrainTree = predict(treeModel, XTrain);
accuracyTrainTree = mean(YPredTrainTree == YTrain);
fprintf('Decision Tree TRAINING Accuracy = %.2f%%\n', accuracyTrainTree*100);

%% ============================================================
% STEP 18 : Confusion Matrix
% ============================================================
confMatTree = confusionmat(YTest, YPredTree);
disp('Decision Tree Confusion Matrix');
disp(confMatTree);

%% ============================================================
% STEP 19 : Per-Class Recall
% ============================================================
classes = unique(YTest);
for i = 1:length(classes)
    c = classes(i);
    recall = sum(YPredTree == c & YTest == c) / sum(YTest == c);
    fprintf('Recall for class %s = %.2f%%\n', string(c), recall*100);
end









%% ============================================================
% STEP 20 (REVISED) : Random Forest with Cost Matrix
% ============================================================
% Same cost matrix used in Decision Tree - penalize missing Draw 2x
costMatrix = [0 1 1;
              2 0 2;
              1 1 0];

randomForestModel = fitcensemble(XTrain, YTrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 50, 'MinLeafSize', 20), ...
    'Cost', costMatrix);

%% ============================================================
% STEP 21 : Predict on Test Data
% ============================================================
YPredRF = predict(randomForestModel, XTest);

%% ============================================================
% STEP 22 : Test Accuracy
% ============================================================
accuracyRF = mean(YPredRF == YTest);
fprintf('Random Forest Accuracy = %.2f%%\n', accuracyRF*100);

%% ============================================================
% STEP 23 : Training Accuracy (to check overfitting)
% ============================================================
YPredTrainRF = predict(randomForestModel, XTrain);
accuracyTrainRF = mean(YPredTrainRF == YTrain);
fprintf('Random Forest TRAINING Accuracy = %.2f%%\n', accuracyTrainRF*100);

%% ============================================================
% STEP 24 : Confusion Matrix
% ============================================================
confMatRF = confusionmat(YTest, YPredRF);
disp('Random Forest Confusion Matrix');
disp(confMatRF);

%% ============================================================
% STEP 25 : Confusion Chart
% ============================================================
figure;
confusionchart(YTest, YPredRF);
title('Random Forest Confusion Matrix (with Cost Matrix)');

%% ============================================================
% STEP 26 : Per-Class Recall
% ============================================================
classes = unique(YTest);
for i = 1:length(classes)
    c = classes(i);
    recall = sum(YPredRF == c & YTest == c) / sum(YTest == c);
    fprintf('Recall for class %s = %.2f%%\n', string(c), recall*100);
end

%% ============================================================
% STEP 27 : Feature Importance
% ============================================================
imp = predictorImportance(randomForestModel);
figure;
bar(imp);
title('Random Forest Feature Importance (with Cost Matrix)');
xlabel('Feature Index');
ylabel('Importance');
xticklabels(XTrain.Properties.VariableNames);
xtickangle(90);

save('randomForestModel.mat', 'randomForestModel');

%% ============================================================
% STEP 28 : Build a "Latest Team Stats" Lookup Table
% ============================================================
rawData = readtable('football_prediction.csv');

if ~isdatetime(rawData.date)
    rawData.date = datetime(rawData.date, 'InputFormat', 'yyyy-MM-dd');
end

allTeams = unique([rawData.home_team; rawData.away_team]);
teamStats = table();

for i = 1:length(allTeams)
    team = allTeams(i);

    homeMatches = rawData(strcmp(rawData.home_team, team), :);
    awayMatches = rawData(strcmp(rawData.away_team, team), :);

    latestHomeDate = NaT;
    latestAwayDate = NaT;
    if ~isempty(homeMatches)
        latestHomeDate = max(homeMatches.date);
    end
    if ~isempty(awayMatches)
        latestAwayDate = max(awayMatches.date);
    end

    if isnat(latestHomeDate) && isnat(latestAwayDate)
        continue;
    end

    if isnat(latestAwayDate) || (~isnat(latestHomeDate) && latestHomeDate >= latestAwayDate)
        row = homeMatches(homeMatches.date == latestHomeDate, :);
        row = row(1,:);
        stats = table( ...
            row.home_elo, row.home_avg_overall, row.home_maoverall, ...
            row.home_avg_attack, row.home_avg_defense, row.home_avg_pace, ...
            row.home_avg_shooting, row.home_avg_passing, ...
            row.home_form_scored, row.home_form_conceded, row.home_form_win_rate, ...
            row.date, ...
            'VariableNames', {'elo','avg_overall','maoverall','avg_attack', ...
            'avg_defense','avg_pace','avg_shooting','avg_passing', ...
            'form_scored','form_conceded','form_win_rate','last_match_date'});
    else
        row = awayMatches(awayMatches.date == latestAwayDate, :);
        row = row(1,:);
        stats = table( ...
            row.away_elo, row.away_avg_overall, row.away_maoverall, ...
            row.away_avg_attack, row.away_avg_defense, row.away_avg_pace, ...
            row.away_avg_shooting, row.away_avg_passing, ...
            row.away_form_scored, row.away_form_conceded, row.away_form_win_rate, ...
            row.date, ...
            'VariableNames', {'elo','avg_overall','maoverall','avg_attack', ...
            'avg_defense','avg_pace','avg_shooting','avg_passing', ...
            'form_scored','form_conceded','form_win_rate','last_match_date'});
    end

    stats.team = team;
    teamStats = [teamStats; stats];
end

teamStats = movevars(teamStats, 'team', 'Before', 1);

save('teamStats.mat', 'teamStats');
disp('Team stats lookup table built successfully!');
head(teamStats)