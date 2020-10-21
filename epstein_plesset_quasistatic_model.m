% Based on Epstein and Plesset (1950)
% assumes density of bubble is linear with pressure
% assumes saturation concentration is linear with pressure
% assumes pressure drops linearly
% p's, Rc, and c's  are from slide 7 in Huikuan's presentation on 26 Mar, 2018
% diffusivity data estimated from diffusivity of CO2 in glycerol

% System parameters
D = 1e-1; % diffusivity [nm^2/ns]
ci = 6150; % initial concentration of CO2 in solution [mol/m^3]
cbi = 5000; % initial concentration of CO2 in bubble [mol/m^3]
p0 = 460; % initial pressure [bar]
pf = 1; % final pressure [bar]
tf = 1e8; % final time [ns]
Rc = 3.5; % initial radius [nm]
% derived parameters
b = ((p0-pf)/p0)/tf;
% ode parameters
tspan = [0, tf]; % span of time [ns]

[t, R] = ode45(@(t,R)epstein_plesset( D, ci, cbi, b, t, R), tspan,...
    Rc);

figure()
plot(t,R)
xlabel('time [ns]')
ylabel('Radius [nm]')
title('Bubble Growth over Time')

%% Use estimates for 20 bar of co2 in glycerol
% System parameters
D = 1e-1; % diffusivity [nm^2/ns]
ci = 0.04; % initial concentration of CO2 in solution [%w/w]
cbi = 0.08; % initial concentration of CO2 in bubble [mol/m^3]
p0 = 460; % initial pressure [bar]
pf = 1; % final pressure [bar]
tf = 1e8; % final time [ns]
Rc = 3.5; % initial radius [nm]
% derived parameters
b = ((p0-pf)/p0)/tf;
% ode parameters
tspan = [0, tf]; % span of time [ns]

[t, R] = ode45(@(t,R)epstein_plesset( D, ci, cbi, b, t, R), tspan,...
    Rc);

figure()
plot(t,R)
xlabel('time [ns]')
ylabel('Radius [nm]')
title('Bubble Growth over Time')

%% Couple pressure and density
%% Use estimates for 20 bar of co2 in glycerol
% System parameters
D = 1e-1; % diffusivity [nm^2/ns]
ci = 6150; % initial concentration of CO2 in solution [mol/m^3]
cbi = 5000; % initial concentration of CO2 in bubble [mol/m^3]
p0 = 460; % initial pressure [bar]
pf = 1; % final pressure [bar]
tf = 1e8; % final time [ns]
Rc = 3.5; % initial radius [nm]
% derived parameters
b = ((p0-pf)/p0)/tf;
% ode parameters
tspan = [0, tf]; % span of time [ns]

[t, R] = ode45(@(t,R)epstein_plesset_coupled( D, ci, cbi, b, t, R), tspan,...
    Rc);

figure()
plot(t,R)
xlabel('time [ns]')
ylabel('Radius [nm]')
title('Bubble Growth over Time')

figure()
loglog(t,R)
xlabel('time [ns]')
ylabel('Radius [nm]')
title('Bubble Growth over Time')