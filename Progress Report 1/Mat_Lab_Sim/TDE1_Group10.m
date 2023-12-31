
%Group_10 Design Simulation

%In this section, the program calculates time delays and accuracy for each microphone:
%Call the findSignalDelay function for each microphone's recorded signal.
%Calculate the accuracy of the calculated delays by comparing them to reference delays.
%Display the results for each microphone's delay and accuracy.
%You can modify signal parameters or noise levels for testing


% Call the function to find the delay for signal_m4
TDOA_m4 = findSignalDelay(signal_m4, signal_m1, fs);
actual_TDOA_m4 = delay_m4 - delay_m1;

% Calculate accuracy for signal_m4
accuracy_m4 = 100 * (1 - abs((actual_TDOA_m4 - TDOA_m4) / (actual_TDOA_m4)));

% Display the result for signal_m4
fprintf('Microphone 4 TDOA:\n');
fprintf('  Calculated Delay: %.7f seconds\n', TDOA_m4);
fprintf('  Reference Delay:  %.7f seconds\n', (actual_TDOA_m4));
fprintf('  Accuracy: %.2f%%\n', accuracy_m4);

fprintf('\n'); % Add a newline separator

% Call the function to find the delay for signal_m3
TDOA_m3 = findSignalDelay(signal_m3, signal_m1, fs);
actual_TDOA_m3 = delay_m3 - delay_m1;

% Calculate accuracy for signal_m3
accuracy_m3 = 100 * (1 - abs(((actual_TDOA_m3) - TDOA_m3) / (actual_TDOA_m3)));

% Display the result for signal_m4
fprintf('Microphone 3 TDOA:\n');
fprintf('  Calculated Delay: %.7f seconds\n', TDOA_m3);
fprintf('  Reference Delay:  %.7f seconds\n', (actual_TDOA_m3));
fprintf('  Accuracy: %.2f%%\n', accuracy_m3);

fprintf('\n'); % Add a newline separator

% Call the function to find the delay for signal_m2
TDOA_m2 = findSignalDelay(signal_m2, signal_m1, fs);
actual_TDOA_m2 = delay_m2 - delay_m1;

% Calculate accuracy for signal_m2
accuracy_m2 = 100 * (1 - abs(((actual_TDOA_m2) - TDOA_m2) / (actual_TDOA_m2)));

% Display the result for signal_m2
fprintf('Microphone 2 TDOA:\n');
fprintf('  Calculated Delay: %.7f seconds\n', TDOA_m2);
fprintf('  Reference Delay:  %.7f seconds\n', (actual_TDOA_m2));
fprintf('  Accuracy: %.2f%%\n', accuracy_m2);

fprintf('\n'); % Add a newline separator

function delayInSeconds = findSignalDelay(signal1, signal2, samplingRate)
    % Inputs:
    %   - signal1: Reference signal
    %   - signal2: Signal with delay and noise
    %   - samplingRate: Sampling rate of the signals (in Hz)

    % Normalize the signals
    signal1 = signal1 / norm(signal1);
    signal2 = signal2 / norm(signal2);

    % Calculate the cross-correlation
    crossCorrelation = xcorr(signal1, signal2);

    % Find the delay corresponding to the maximum correlation
    [~, maxIndex] = max(crossCorrelation);

    % Calculate the delay in samples (accounting for zero-based indexing)
    delaySamples = maxIndex - numel(signal1) + 1;

    % Convert delay from samples to seconds
    delayInSeconds = delaySamples / samplingRate;
end


