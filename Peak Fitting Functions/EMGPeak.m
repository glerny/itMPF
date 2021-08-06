%% INPUT PARAMETERS
% x: vector of abscissa (time axis)
% a: 1x3 Fitting parameters
%    a(1) - centroid time (peak maxima)
%    a(2) - standard deviation of the Gaussian component
%    a(3) - time constant of the exponential function
%% REFERENCES
% 1. Naish, P.J., Hartwell, S. Exponentially Modified Gaussian functions—A
% good model for chromatographic peaks in isocratic HPLC?. Chromatographia
% 26, 285–296 (1988). https://doi.org/10.1007/BF02268168

function dataOut = EMGPeak(x, a)

if a(3) > 0
    dataOut = exp(0.5*(a(2)/a(3))^2 - (x-a(1))/a(3)).*...
        (erf(1/sqrt(2)*(a(1)/a(2)+a(2)/a(3))) + ...
        erf(1/sqrt(2).*((x-a(1))/a(2)-a(2)/a(3))));
else
    dataOut = exp(0.5*(a(2)/a(3))^2 - (x-a(1))/a(3)).*...
        (erfc(1/sqrt(2)*(a(1)/a(2)+a(2)/a(3))) + ...
        erfc(1/sqrt(2).*((x-a(1))/a(2)-a(2)/a(3))));
end

if sum( ~isfinite( dataOut ) ) ~= 0 % the distribution is very close to a Gaussian, replace with a Gaussian
    dataOut = exp(-(x-a(1)).^2/(2*a(2)^2));

end
