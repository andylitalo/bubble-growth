function dRdt = epstein_plesset_coupled( D, ci, cbi, b, t, R)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% decreasing bubble density with ideal gas law (assumes no mass gain)
% dRdt = D.*ci.*(b.*t)./(cbi.*(1-b.*t)).*(1./R + 1./sqrt(pi.*D.*(t+0.00001)));
% constant density of bubble
% dRdt = D.*ci.*(b.*t)./(cbi).*(1./R + 1./sqrt(pi.*D.*(t+0.00001)));

% coupled to changing pressure/density; assumes saturation concentration
% linear in pressure
dRdt = R*b./(3*(1-b.*t)) + D.*(ci.*b.*t)./(cbi.*(1-b.*t)).*(1./R+...
    1./sqrt(pi.*D.*(t+1e-6)));
end

