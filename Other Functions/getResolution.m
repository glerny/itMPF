function Resolution = getResolution(X)

[myLin, myCol] = size(X);

for ii = 1:myLin
    Resolution(:, ii) = abs((X(ii, 1) - X(:,1))./(2*(X(ii, 2) + X(:,2))));
end

for ii = 1:myLin
    Resolution(ii, ii) = NaN;
end
end

