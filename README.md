% Get the CSV file path from user input
csvFilePath = input('Masukkan lokasi file CSV: ', 's');

% loading data
data = csvread(csvFilePath, 1, 0); % Assuming the first row is the header

% dropping tables and renaming columns
data(:, [2, 3]) = []; % Drop columns 2 and 3
data = [data(:, 1), data(:, end)]; % Rename columns

x = data(:, 1:end-1);
y = data(:, end);

% preparing data for training and testing
rng(42); % Set random seed for reproducibility
[trainIdx, testIdx] = datasample(1:size(x, 1), round(0.7*size(x, 1)), 'Replace', false);
x_train = x(trainIdx, :);
y_train = y(trainIdx);
x_test = x(testIdx, :);
y_test = y(testIdx);

% building model
model = TreeBagger(100, x_train, y_train, 'Method', 'classification', 'NumPredictorsToSample', 'all', 'Options', statset('UseParallel', true));
y_hat = predict(model, x_test);

% Convert predicted labels to numeric
y_hat_num = str2double(y_hat);

% metrics
accuracy = sum(y_hat_num == y_test) / numel(y_test);
fprintf('**** ACCURACY_SCORE **** \n\n %.4f \n', accuracy);
fprintf('**** CLASSIFICATION_REPORT **** \n\n');
disp(classification_report(y_test, y_hat_num));

% Plotting results
figure;
% Plotting true labels...
scatter(x_test(y_test == 0, 1), x_test(y_test == 0, 2), 'o', 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'black', 'DisplayName', 'Label 0');
hold on;
scatter(x_test(y_test == 1, 1), x_test(y_test == 1, 2), 's', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black', 'DisplayName', 'Label 1');

% Plotting predicted labels
scatter(x_test(y_hat_num == 0, 1), x_test(y_hat_num == 0, 2), 'x', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'DisplayName', 'Predicted Label 0');
scatter(x_test(y_hat_num == 1, 1), x_test(y_hat_num == 1, 2), '^', 'MarkerEdgeColor', 'purple', 'LineWidth', 1.5, 'DisplayName', 'Predicted Label 1');

xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter Plot of True vs Predicted Labels');
legend('Location', 'best');
grid on;
hold off;

% Notification when the program is finished
disp('Program selesai.');
