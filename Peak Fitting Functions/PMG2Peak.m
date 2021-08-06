%% INPUT PARAMETERS
% x: vector of abscissa (time axis)
% a: 1x3 Fitting parameters
%    a(1) - centroid time (peak maxima)
%    a(2) - 1st component of the standard deviation
%    a(3) - 2nd component of the standard deviation
%    a(4) - 3rd component of the standard deviation

%% REFERENCES
% 1. Nikitas, P., Pappa-Louisi, A., Papageorgiou, A., On the equations
% describing chromatographic peaks and the problem of the deconvolution of
% overlapped peaks, Journal of Chromatography A, 9121, 13-29 (2001). 
% 2. Baeza-Baeza, J.J., Ortiz-Bolsico, C., García-Álvarez-Coque, M.C. New
% approaches based on modified Gaussian models for the prediction of
% chromatographic peaks. Analytica Chimica Acta 758, 36–44(2013).
% https://doi.org/10.1016/j.aca.2012.10.035

function dataOut = PMG2Peak(axe, a)

s = a(2) + a(3)*(axe-a(1)) + a(4)*(axe-a(1)).^2;
dataOut = a(2)./s.*exp(-(axe-a(1)).^2./(2*s.^2));

end

