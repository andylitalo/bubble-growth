function dRdt = epstein_plesset_static( D, p0, pf, wb, t, R)
%EPSTEIN_PLESSET_STATIC gives the Epstein-Plesset equation for a system in
%which the pressure is suddenly quenched.

% data points from Dow for VORANOL 360
% pressure in psia
p_data = [198.1, 405.6, 606.1, 806.8, 893.9];
% weight fraction
w_data = [0.0372, 0.0821, 0.1351, 0.1993, 0.2336];

% conversions
Pa2psi = 14.5e-5;
pi_psi = Pa2psi*p0;
pf_psi = Pa2psi*pf;

% interpolate weight fractions with cubic spline
a = spline(p_data, w_data, [pi_psi, pf_psi]);
wi = a(1);
wf = a(2);

% epstein-plesset equation
dRdt = D.*(wi-wf)./(wb).*(1./R + 1./sqrt(pi.*D.*(t+1e-12)));
end

