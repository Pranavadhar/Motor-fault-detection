audioFilePath = 'healthy_rotor_noload.mp3';
[y, Fs] = audioread(audioFilePath);
N = length(y);
time = (0:N-1) / Fs;
subplot(2,2,1); plot(time,y);
title('Noisy Signal');
xlabel('Time (s)');
ylabel('Amplitude');
rmsValue = rms(y);
disp(['RMS Value: ', num2str(rmsValue)]);

Y = fft(y);
frequencies = (0:N-1) * (Fs/N);
subplot(2,2,2);
plot(frequencies, abs(Y));
title('FFT of Noisy Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Compute the Wiener filter to estimate the clean signal 'x'
window_size = 100; % Adjust the window size as needed
x_hat = wiener_filter(y, window_size);
% % Plot the clean signal estimate
subplot(2,2,3);
plot(time, x_hat);
title('Wiener Filter Estimate of Clean Signal');
xlabel('Time (s)');
ylabel('Amplitude');
rmsValue = rms(x_hat);
disp(['RMS Value: ', num2str(rmsValue)]);

% % Compute the FFT of the clean signal estimate
N = length(x_hat);
X_hat = fft(x_hat);

% Plot the magnitude spectrum of the estimated signal
subplot(2,2,4);
plot(frequencies, abs(X_hat));
title('FFT of Clean Signal Estimate');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Calculate mean, variance, and kurtosis of the clean signal estimate
meanValue = mean(x_hat);
varianceValue = var(x_hat);
kurtosisValue = kurtosis(x_hat);

disp(['Mean: ', num2str(meanValue)]);
disp(['Variance: ', num2str(varianceValue)]);
disp(['Kurtosis: ', num2str(kurtosisValue)]);

% Initialize arrays to store the features
num_segments = 100;
segment_length = floor(length(x_hat) / num_segments);

mean_values = zeros(1, num_segments);
variance_values = zeros(1, num_segments);
kurtosis_values = zeros(1, num_segments);

% Loop over segments
for i = 1:num_segments
    % Define the current segment
    start_idx = (i - 1) * segment_length + 1;
    end_idx = min(i * segment_length, length(x_hat));
    
    % Extract the segment
    segment = x_hat(start_idx:end_idx);
    
    % Calculate features for the segment
    mean_values(i) = mean(segment);
    variance_values(i) = var(segment);
    
    % Calculate kurtosis without using kurt function
    n = length(segment);
    m4 = sum((segment - mean_values(i)).^4) / n;
    m2 = sum((segment - mean_values(i)).^2) / n;
    kurtosis_values(i) = (m4 / m2^2) - 3;
end

% Create a table with the calculated features
segmentedFeatureData = table(mean_values', variance_values', kurtosis_values', 'VariableNames', {'Mean', 'Variance', 'Kurtosis'});

% Write the table to an Excel file
outputFilePathSegments = 'segmentedFeatureOutput_healthy_noload.xlsx';
writetable(segmentedFeatureData, outputFilePathSegments);
disp(['Segmented data exported to ', outputFilePathSegments]);

% Wiener filter function
function x_hat = wiener_filter(y, window_size)
    x_hat = zeros(size(y));
    epsilon = 1e-10; % Small constant to avoid division by zero
    for i = 1:length(y) - window_size
        % Estimate the power spectral density of the signal
        Pyy = abs(fft(y(i:i+window_size-1))).^2;
        
        % Estimate the power spectral density of the noise (assuming noise is stationary)
        Pnn = mean(Pyy);
        
        % Compute the Wiener filter gain
        alpha = Pyy ./ (Pyy + Pnn + epsilon);
        
        % Apply the Wiener filter
        x_hat(i:i+window_size-1) = alpha .* y(i:i+window_size-1);
    end
end
