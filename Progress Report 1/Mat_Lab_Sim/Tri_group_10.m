
%Group_10 Design Simulation

%This section estimates the position of the sound source using multilateration:
%Store the actual sound source position.
%Calculate distances from time delays.
%Perform multilateration to estimate the sound source's position.
%Display the actual and estimated sound source positions.
%Calculate accuracy by comparing the estimated position to the actual position.
%The accuracy percentage indicates the quality of the estimation.
%Can vary the rounding of the accuracy value to see mm and nm differences

Actual_sound_source_position = Sound_source_position;

% Calculate distances from time delays (as before)
% Distance_m1_ss_1 = delayInSeconds_m1 * speed_of_sound;
% Distance_m2_ss_1 = delayInSeconds_m2 * speed_of_sound;
% Distance_m3_ss_1 = delayInSeconds_m3 * speed_of_sound;
% Distance_m4_ss_1 = delayInSeconds_m4 * speed_of_sound;

% Uncomment the lines below to run the triangulation tests with the actual
% TDOAs, not the calculated versions.
TDOA_m2 = actual_TDOA_m2;
TDOA_m3 = actual_TDOA_m3;
TDOA_m4 = actual_TDOA_m4;


% Calculate differences in distances between reference mic (mic 1) and
% other mics
d_1_2 = TDOA_m2 * speed_of_sound;
fprintf('Calculated d_1_2: %.3f%\n', d_1_2);
fprintf('\n');
fprintf('Actual d_1_2: %.3f%\n', Distance_m2_ss - Distance_m1_ss);
fprintf('\n');
d_1_3 = TDOA_m3 * speed_of_sound;
fprintf('Calculated d_1_3: %.3f%\n', d_1_3);
fprintf('\n');
fprintf('Actual d_1_3: %.3f%\n', Distance_m3_ss - Distance_m1_ss);
fprintf('\n');
d_1_4 = TDOA_m4 * speed_of_sound;
fprintf('Calculated d_1_4: %.3f%\n', d_1_4);
fprintf('\n');
fprintf('Actual d_1_4: %.3f%\n', Distance_m4_ss - Distance_m1_ss);

% Find points of intersection between hyperbolae tracking the difference in
% distance
x_1 = 0;
y_1 = grid_height;
x_2 = grid_width;
y_2 = 0;
diff_1 = d_1_2;
diff_2 = d_1_4;

% sign goes in front of the difference in distance and decides which hyperbolic curve to use out of the 2 that could be drawn
% Explanation: If the TDOA is positive, then the sound source is closer to
% the reference mic. If the difference in the distances between the sound
% sources and the 2 mics in question is positive, then the sound source is
% further away from the ref mic, when the hyperbola equation is in the form
% diff_distance = distance_from_ref_mic - distance_from_other_mic.
% Therefore the sign of the distance_diff needs to be switched if the TDOA
% is negative.
% sign_1 = TDOA_m2/(abs(TDOA_m2)); 
% sign_2 = TDOA_m4/(abs(TDOA_m4));
fun = @(x)paramfun(x,diff_1,diff_2,x_1,x_2,y_1,y_2);

x0 = [0 0];
x_2_4 = fsolve(fun,x0);

% Calculate the intersection of parobolas from mic 2 & ref mic, and mic 3 &
% ref mic
x_1 = 0;
y_1 = grid_height;
x_2 = grid_width;
y_2 = grid_height;
diff_1 = d_1_2;
diff_2 = d_1_3;

% sign_1 = TDOA_m2/(abs(TDOA_m2)); 
% sign_2 = TDOA_m3/(abs(TDOA_m3));
fun = @(x)paramfun(x,diff_1,diff_2,x_1,x_2,y_1,y_2);

x_2_3 = fsolve(fun,x0);

% Calculate the intersection of parobolas from mic 3 & ref mic, and mic 4 &
% ref mic
x_1 = grid_width;
y_1 = grid_height;
x_2 = grid_width;
y_2 = 0;
diff_1 = d_1_3;
diff_2 = d_1_4;

% sign_1 = TDOA_m3/(abs(TDOA_m3)); 
% sign_2 = TDOA_m4/(abs(TDOA_m4));
fun = @(x)paramfun(x,diff_1,diff_2,x_1,x_2,y_1,y_2);

x_3_4 = fsolve(fun,x0);

% Calculate average between 3 intersection points, excluding any points that
% are outside of the grid
if (x_2_3(1)<0) || (x_2_3(1)>grid_width) || (x_2_3(2)<0) || (x_2_3(2)>grid_height)
    x_ave = (x_2_4 + x_3_4)/2;
elseif (x_2_4(1)<0) || (x_2_4(1)>grid_width) || (x_2_4(2)<0) || (x_2_4(2)>grid_height)
    x_ave = (x_2_3 + x_3_4)/2;
elseif (x_3_4(1)<0) || (x_3_4(1)>grid_width) || (x_3_4(2)<0) || (x_3_4(2)>grid_height)
    x_ave = (x_2_3 + x_2_4)/2;
else
    x_ave = (x_2_3 + x_2_4 + x_3_4)/3;
end


% Multilateration to estimate the sound source's position - old method
% A = 2 * [Microphone2_position - Microphone1_position;
%          Microphone3_position - Microphone1_position;
%          Microphone4_position - Microphone1_position];
% b = [Distance_m1_ss_1^2 - Distance_m2_ss_1^2 + norm(Microphone2_position)^2 - norm(Microphone1_position)^2;
%      Distance_m1_ss_1^2 - Distance_m3_ss_1^2 + norm(Microphone3_position)^2 - norm(Microphone1_position)^2;
%      Distance_m1_ss_1^2 - Distance_m4_ss_1^2 + norm(Microphone4_position)^2 - norm(Microphone1_position)^2];
% predicted_sound_source_position = A \ b; % Solve the system of equations


% Display the output
fprintf('Actual Sound Source Position: (%.3f, %.3f)\n', Actual_sound_source_position);
fprintf('Predicted Sound Source Position from ref mic and mics 2 and 4: (%.3f, %.3f)\n', x_2_4);
fprintf('Predicted Sound Source Position from ref mic and mics 2 and 3: (%.3f, %.3f)\n', x_2_3);
fprintf('Predicted Sound Source Position from ref mic and mics 3 and 4: (%.3f, %.3f)\n', x_3_4);
fprintf('Predicted Sound Source Position average from 3 readings above: (%.3f, %.3f)\n', x_ave);

%  figure;

hold on;
plot(Microphone1_position(1), Microphone1_position(2), 'b^');
plot(Microphone2_position(1), Microphone2_position(2), 'b^');
plot(Microphone3_position(1), Microphone3_position(2), 'b^');
plot(Microphone4_position(1), Microphone4_position(2), 'b^');

plot(Sound_source_position(1),Sound_source_position(2), 'r*');

% Uncomment lines below to plot the points of intersection of all the
% hyperbolae tracked from each mic to the reference mic.
% plot(x_2_4(1),x_2_4(2),'k.');
% plot(x_2_3(1),x_2_3(2),'m.');
% plot(x_3_4(1),x_3_4(2),'g.');

plot(x_ave(1),x_ave(2),'k.');
grid on;
legend('Mic_1','Mic_2','Mic_3','Mic_4','Sound Source','Predicted Position');
xlabel("x in metres");
ylabel("y in metres");
hold off

d_max = norm([grid_width grid_height]);

% Calculate the Euclidean distance between actual and predicted positions
error_2_4 = norm(Actual_sound_source_position - x_2_4);
error_2_3 = norm(Actual_sound_source_position - x_2_3);
error_3_4 = norm(Actual_sound_source_position - x_3_4);
error_ave = norm(Actual_sound_source_position - x_ave);

% % Calculate the accuracy as a percentage
acc_2_4 = (1 - error_2_4 / d_max) * 100;
acc_2_3 = (1 - error_2_3 / d_max) * 100;
acc_3_4 = (1 - error_3_4 / d_max) * 100;
acc_ave = (1 - error_ave / d_max) * 100;

fprintf('Accuracy of ref mic and mics 2 & 4: %.3f%%\n', acc_2_4);
fprintf('Accuracy of ref mic and mics 2 & 3: %.3f%%\n', acc_2_3);
fprintf('Accuracy of ref mic and mics 3 & 4: %.3f%%\n', acc_3_4);
fprintf('Accuracy of average position: %.3f%%\n', acc_ave);
fprintf('Distance from actual sound source: %.3fm\n', error_ave);

function F = paramfun(x,diff_1,diff_2,x_1,x_2,y_1,y_2)
F = [ sqrt(x(1)^2 + x(2)^2) - sqrt((x(1)-x_1)^2 + (x(2)-y_1)^2) + diff_1
      sqrt(x(1)^2 + x(2)^2) - sqrt((x(1)-x_2)^2 + (x(2)-y_2)^2) + diff_2];
end






