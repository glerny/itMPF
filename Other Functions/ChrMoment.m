function M = ChrMoment(XY)

X = XY(:,1);
Y = XY(:,2);
try
    M(1) = trapz(X, Y);
    M(2) = trapz(X, X.*Y)/M(1);
    M(3) = trapz(X, (X - M(2)).^2.*Y)/M(1);
    M(4) = trapz(X, (X - M(2)).^3.*Y)/M(1);
    M(5) = max(Y) - min(Y);
    
catch
    M(1) = NaN;
    M(2) = NaN;
    M(3) = NaN;
    M(4) = NaN;
    M(5) = NaN;
end