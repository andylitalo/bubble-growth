% Based on Epstein and Plesset (1950)
% assumes constant pressure and temperature
% p's, Rc, and c's  are from slide 7 in Huikuan's presentation on 26 Mar, 2018
% diffusivity data estimated from diffusivity of CO2 in glycerol

% the estimate is highly sensitive to the density of the critical nucleus
% "wb," the diffusivity "D" (non-monotonically), which are not very
% accurately measured/estimated, so this model is not very reliable

clear; close 'all'; clc;
cd 'C:\Users\Andy.DESKTOP-CFRG05F\OneDrive - California Institute of Technology\Documents\Research\Kornfield\Calculations\bubble_growth_models\'
% System parameters
D = 1e-1; % diffusivity [nm^2/ns] from Di Caprio et al (2016) Fl Ph Equil Fig 6
p0 = 100e5; % initial pressure [Pa]
pf = 1e5; % final pressure [Pa]
wb = 0.18; % weight fraction of gas in critical nucleus of bubble
tf = 1e8; % final time [ns]
Rc = 2.285; % initial radius [nm]

% ode parameters
tspan = [0, tf]; % span of time [ns]

[t, R] = ode45(@(t,R)epstein_plesset_static( D, p0, pf, wb, t, R), tspan,...
    Rc);

figure()
plot(t/1e6,R/1000, 'LineWidth', 2)
set(gca, 'FontSize', 16)
xlabel('time [ms]', 'FontSize', 20)
ylabel('Radius [\mum]', 'FontSize', 20)
title('Bubble Growth over Time', 'FontSize', 24)

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