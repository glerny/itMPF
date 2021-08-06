%% INPUT PARAMETERS
% x: vector of abscissa (time axis)
% a: 1x3 Fitting parameters
%    a(1) - centroid time (peak maxima)
%    a(2) - standard deviation

function [f, g] = gaussPeak(axe, a)
f = exp(-(axe-a(1)).^2/(2*a(2)^2));

if nargout > 1 % gradient required
    g = [((axe-a(1))/(a(2)^2)).* f, ...
        (((axe-a(1)).^2)/(a(2)^3)).* f];
end
end