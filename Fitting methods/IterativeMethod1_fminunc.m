%% DESCRIPTION
%% INPUT PARAMETERS
%% EXAMPLES
%% REFERENCES
%% COPYRIGHT
% Copyright BSD 3-Clause License Copyright 2021 G. Erny
% (guillaume@fe.up.pt), FEUP, Porto, Portugal
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Model, FittedChannels, Stats, myModel, Options] = IterativeMethod1_fminunc(AxisX, Data, Options)

%TODO: catch rank deficiency warnings
tic
%% 1. Initialisation and validation of variables
narginchk(2, 3)
opts = optimset('Display','off');

[m, n] = size(AxisX);
if m == 1 && n ~= 1
    AxisX = AxisX';
    m = n; n = 1;
elseif m == 1 && n == 1
    error('AxisX')
elseif m ~= 1 && n ~= 1
    error('AxisX')
elseif m == 1
    error('squared data')
end

[M, N] = size(Data);

if M == N
    error('squared data')
end

if N == m
    Data = Data';
    nbrChannels = M;
    nbrPts = N;
    
else
    nbrChannels = N;
    nbrPts = M;
end

if nargin == 2
    Options = {};
end

if ~isfield(Options, 'maxPeaks'), Options.maxPeaks = 20; end
if ~isfield(Options, 'Function'), Options.Function = 'Gauss'; end
if ~isfield(Options, 'LoopMe'), Options.LoopMe = 5; end
if ~isfield(Options, 'RecursiveLoop'), Options.RecursiveLoop = 0.95; end
if ~isfield(Options, 'InitialFactor'), Options.InitialFactor = [1 0.75 0.5 0.25]; end
if ~isfield(Options, 'MinResolution'), Options.MinResolution = 0; end
if ~isfield(Options, 'Penalisation')
    Options.Penalisation = true;
    Options.PenalisationWeight = 4;
end
if ~isfield(Options, 'Constrained')
    Options.Constrained.SharedParameters = "Partial"; % "Full", "Partial" or "None"
    Options.Constrained.Limits = [4 0];
end
if ~isfield(Options, 'PointsPerPeaks'), Options.PointsPerPeaks = 10; end
if ~isfield(Options, 'MinMax'), Options.MinMax = 0.05; end
if ~isfield(Options, 'Robust'), Options.Robust = false; end

%TODO: Check the Options validity

MinVar = (mean(AxisX(2:end, 1) - AxisX(1:end-1, 1)));
Data(~isfinite(Data)) = 0;
FittedModel = zeros(size(Data));

%% 2. Initialisation of the recursive loop
if Options.Constrained.SharedParameters == "Full"
    % All fitting parameters (a2, a2,...) are equals in all
    % 1. Find 1st Local maximum and peak variance
    tData = mean(Data, 2);
    
    switch Options.Function
        case "Gauss"
            addFitPar = 0;  % additional fitting parameters other than center
            % and variance
        case "PMG1"
            addFitPar = 1;
            
        case "EMG"
            addFitPar = 1;
            
        case "PMG2"
            addFitPar = 2;
    end
    nPeak = 1;
    
    % variance
    [Val4Max, Id4Max] = max(tData);
    
    % finding the peak width at half height
    is = find(tData(1:Id4Max) <= Val4Max/2, 1, 'last');
    if isempty(is), is = 1; end
    ie = find(tData(Id4Max:end) <= Val4Max/2, 1, 'first');
    if isempty(ie)
        ie = length(AxisX);
    else
        ie = min(ie + Id4Max, length(AxisX));
    end
    a(1) = (AxisX(ie, 1) - AxisX(is, 1))/2.355;
    if a(1) < MinVar, a(1) = 2*MinVar; end
    
    % center
    a(addFitPar+2) =  AxisX(Id4Max, 1);
    
    [f_ini, FittedModel] = myFirstFit(a);
    
    [a, fval, exitflag] = fminunc(@myFirstFit, a, opts);
    count = 1;
    while exitflag == 0
        [a, fval, exitflag] = fminunc(@myFirstFit, a, opts);
        if count > Options.LoopMe, break; else, count = count +1; end
    end
    [f_ini, FittedModel] = myFirstFit(a);
    
    a_ini = a;
    while 1
        
        if nPeak >= Options.maxPeaks
            break
        end
        
        a_test = {}; f_test = [];
        for iF = 1:length(Options.InitialFactor)
            for iP = 1:length(Options.PointsPerPeaks)
                [a_test{end+1}, f_test(end+1)] = addAPeak_Full(a, Options.PointsPerPeaks(iP), Options.InitialFactor(iF), Options.LoopMe, MinVar);
            end
        end
        nPeak = nPeak+1;
        [~, Imin] = min(f_test);
        a = a_test{Imin};
        [f, FittedModel, ~, Ratio] = myFirstFit(a);
        
        affp = a(1:addFitPar+1); % fixed parameters
        avfp = a(addFitPar+2:end); % variable parameters,e.g. center
        Resolution = getResolution([avfp(1, :)' ones(size(avfp(1, :)'))*affp(1)]);
        if f <= Options.RecursiveLoop*f_ini ...
                && ~any(Resolution < Options.MinResolution, 'all')...
                && min(max(Ratio(:, 2:end), [], 1)) >= Options.MinMax*max(Ratio(:, 2:end), [], "all")
            f_ini = f;
            a_ini = a;
        else
            if Options.Robust && nPeak < Options.maxPeaks
                
                a_test = {}; f_test = [];
                for iF = 1:length(Options.InitialFactor)
                    for iP = 1:length(Options.PointsPerPeaks)
                        [a_test{end+1}, f_test(end+1)] = addAPeak_Full(a, Options.PointsPerPeaks(iP), Options.InitialFactor(iF), Options.LoopMe, MinVar);
                    end
                end
                nPeak = nPeak+1;
                [~, Imin] = min(f_test);
                a = a_test{Imin};
                [f, FittedModel, ~, Ratio] = myFirstFit(a);
                
                affp = a(1:addFitPar+1); % fixed parameters
                avfp = a(addFitPar+2:end); % variable parameters,e.g. center
                Resolution = getResolution([avfp(1, :)' ones(size(avfp(1, :)'))*affp(1)]);
                if f <= Options.RecursiveLoop*f_ini ...
                        && ~any(Resolution < Options.MinResolution, 'all')...
                        && min(max(Ratio(:, 2:end), [], 1)) >= Options.MinMax*max(Ratio(:, 2:end), [], "all")
                    f_ini = f;
                    a_ini = a;
                else
                    nPeak = nPeak -2;
                    break
                end
                
            else
                nPeak = nPeak -1;
                break
            end
        end
    end
    
    % reorganised a in matrix "form". First column baseline fit, the each
    % column a peaks
    
    [f_final, FittedModel, myModel, Ratio] = myFirstFit(a_ini);
    a = zeros(addFitPar+1 ,nPeak);
    affp = a_ini(1:addFitPar+1);
    avfp = a_ini(addFitPar+2:end);
    a(1, :) = avfp(1, :);
    a(2:addFitPar+2, :) = affp'*ones(1, nPeak);
    a_ini = a;
else
    %%
    if Options.Constrained.SharedParameters == "Partial"
        % allow the fitting parameters to adjust independently and continue
        % addind % peaks if possible
        if isfield(Options.Constrained, 'Limits')
            Limits = Options.Constrained.Limits;
        else
            Limits = [];
        end
        
    elseif Options.Constrained.SharedParameters == "None"
        Limits = [];
        
    else
        error('')
    end
    
    switch Options.Function
        case "Gauss"
            if isempty(Limits)
                Limits(1) = 0;
            elseif Limits(1) <= 1
                Limits(1) = 0;
            end
            Limits = Limits(1);
            a(2, 1) =  0;
            
        case "PMG1"
            if isempty(Limits)
                Limits(2) = 0;
            elseif length(Limits) == 1
                Limits(2) = 0;
            end
            Limits = Limits(1:2);
            Limits(Limits <= 1) = 0;
            a(3, 1) =  0;
            
        case "EMG"
            if isempty(Limits)
                Limits(2) = 0;
            elseif length(Limits) == 1
                Limits(2) = 0;
            end
            Limits = Limits(1:2);
            Limits(Limits <= 1) = 0;
            a(3, 1) =  0;
            
        case "PMG2"
            if isempty(Limits)
                Limits(3) = 0;
            elseif length(Limits) < 3
                Limits(3) = 0;
            end
            Limits = Limits(1:3);
            Limits(Limits <= 1) = 0;
            a(4, 1) =  0;
    end
    nPeak = 1;
    
    % variance
    tData = mean(Data, 2);
    [Val4Max, Id4Max] = max(tData);
    
    % finding the peak width at half height
    is = find(tData(1:Id4Max) <= Val4Max/2, 1, 'last');
    if isempty(is), is = 1; end
    ie = find(tData(Id4Max:end) <= Val4Max/2, 1, 'first');
    if isempty(ie)
        ie = length(AxisX);
    else
        ie = min(ie + Id4Max, length(AxisX));
    end
    a(1, 1) =  AxisX(Id4Max, 1);
    a(2, 1) = (AxisX(ie, 1) - AxisX(is, 1))/2.355;
    if a(2, 1) < MinVar, a(2, 1) = MinVar; end
    
    
    [f_ini, FittedModel] = mySecondFit(a);
    
    
    [a, fval, exitflag] = fminunc(@mySecondFit, a, opts);
    count = 1;
    while exitflag == 0
        [a, fval, exitflag] = fminunc(@mySecondFit, a, opts);
        if count > Options.LoopMe, break; else, count = count +1; end
    end
    [f_ini, FittedModel] = mySecondFit(a);
    
    a_ini = a;
    while 1
        
        if nPeak >= Options.maxPeaks
            break
        end
        
        a_test = {}; f_test = [];
        for iF = 1:length(Options.InitialFactor)
            for iP = 1:length(Options.PointsPerPeaks)
                [a_test{end+1}, f_test(end+1)] = addAPeak_Partial(a, Options.PointsPerPeaks(iP), Options.InitialFactor(iF), Options.LoopMe, MinVar);
            end
        end
        nPeak = nPeak+1;
                
        [~, Imin] = min(f_test);
        a = a_test{Imin};
        [f, FittedModel, Mdl, Ratio] = mySecondFit(a);
        
        Resolution = getResolution([a(1, :)', a(2, :)']);
        if f <= Options.RecursiveLoop*f_ini ...
                && ~any(Resolution < Options.MinResolution, 'all') ...
                && min(max(Ratio(:, 2:end), [], 1)) >= Options.MinMax*max(Ratio(:, 2:end), [], "all")
            f_ini = f;
            a_ini = a;
        else
            if Options.Robust && nPeak < Options.maxPeaks
                
                a_test = {}; f_test = [];
                for iF = 1:length(Options.InitialFactor)
                    for iP = 1:length(Options.PointsPerPeaks)
                        [a_test{end+1}, f_test(end+1)] = addAPeak_Partial(a, Options.PointsPerPeaks(iP), Options.InitialFactor(iF), Options.LoopMe, MinVar);
                    end
                end
                nPeak = nPeak+1;
                
                [~, Imin] = min(f_test);
                a = a_test{Imin};
                [f, FittedModel, Mdl, Ratio] = mySecondFit(a);
                
                Resolution = getResolution([a(1, :)', a(2, :)']);
                if f <= Options.RecursiveLoop*f_ini ...
                        && ~any(Resolution < Options.MinResolution, 'all')...
                        && min(max(Ratio(:, 2:end), [], 1)) >= Options.MinMax*max(Ratio(:, 2:end), [], "all")
                    f_ini = f;
                    a_ini = a;
                else
                    nPeak = nPeak -2;
                    break
                end
                
            else
                nPeak = nPeak -1;
                break
            end
        end
    end
    
    [f_final, FittedProfile, myModel, Ratio] = mySecondFit(a_ini);
end

Stats.Options = Options;
Stats.TOC = toc;
Stats.NbrPeaks = nPeak;
if nPeak >= Options.maxPeaks
    Stats.EndConditions = "Maximum number of peaks reached";
elseif f > Options.RecursiveLoop*f_ini
    Stats.EndConditions = "Convergence reached";
else
    Stats.EndConditions = "Minimum resolution or minmax condition reached";
end
Stats.RMSE = f_final;

Model.Peaks.Function = Options.Function;
Model.Peaks.FittingParameters = a_ini;
Model.Peaks.FittingIntensities = Ratio(:, 2:end);
Model.Peaks.BackgroundIntensities = Ratio(:,1);

for ii = 1:nPeak+1
    for jj = 1:nbrChannels
        FittedChannels(:, ii, jj) = Ratio(jj, ii)*myModel(:, ii);
    end
end

    function [f, FittedProfile, Model, Ratio] = myFirstFit(x)
        % Constant fitting parameters
        % Introduction: created FittedProfiles and reorganised x
        fixFitPar = x(1:addFitPar+1);
        varFitPar = x(addFitPar+2:end);
        nP = length(varFitPar);
        
        Model = zeros(size(Data, 1), nP+1);
        Model(:,1) = 1;
        
        if any(varFitPar < 0, 'all') ...
                || fixFitPar(1) < MinVar
            Model = inf(size(Model));
            
        else
            for f_ii = 1:nP
                
                if varFitPar(1, f_ii) < AxisX(1) - 2*fixFitPar(1)  ...
                        || varFitPar(1, f_ii) > AxisX(end) + 2*fixFitPar(1)
                    f = 3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
                    Ratio = nan(nbrChannels, nP+1);
                    FittedProfile = nan(nbrPts, nbrChannels);
                    return
                    
                else
                    switch Options.Function
                        case "Gauss"
                            dataOut = gaussPeak(AxisX, [varFitPar(1, f_ii), fixFitPar]);
                            
                        case "PMG1"
                            dataOut = PMG1Peak(AxisX, [varFitPar(1, f_ii), fixFitPar]);
                            
                        case "EMG"
                            dataOut = EMGPeak(AxisX, [varFitPar(1, f_ii), fixFitPar]);
                            
                        case "PMG2"
                            dataOut = PMG2Peak(AxisX, [varFitPar(1, f_ii), fixFitPar]);
                            
                    end
                    Model(:, f_ii+1) = dataOut;
                end
            end
        end
        
        
        if any(~isfinite(Model), 'all')
            f = 3*mean(sum((Data - zeros(size(Data))).^2, 'all'));
            Ratio = nan(nbrChannels, nP+1);
            FittedProfile = nan(nbrPts, nbrChannels);
            
        else
            Ratio = Data'/Model';
            Ratiop = Ratio;
            if Options.Penalisation
                Ratiop(Ratiop < 0) = Options.PenalisationWeight*Ratiop(Ratiop < 0);
            end
            FittedProfile = Model*Ratio';
            
            f = sqrt(mean((Data -  Model*Ratiop').^2, 'all'));
            if ~isfinite(f)
                f = 3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
            end
        end
        
    end

    function [a, fMe] = addAPeak_Full(a_ini, PpP, MultiMe, MaxLoop, MVar)
        
        a = a_ini;
        [fMe, FtMdl] = myFirstFit(a);
        Res = smoothdata(Data - FtMdl, 'movmean', PpP);
        Res = mean(Res, 2);
        [Val, Id] = max(Res);
        a(end+1) = AxisX(Id, 1);
        a(1) = max(MultiMe*a(1), MVar);
        
        [a, ~, flag] = fminunc(@myFirstFit, a, opts);
        cnt = 1;
        while flag == 0
            [a, ~, flag] = fminunc(@myFirstFit, a, opts);
            if cnt > MaxLoop, break; else, cnt = cnt +1; end
        end
        fMe = myFirstFit(a);
    end

    function [a, fMe] = addAPeak_Partial(a_ini, PpP, MultiMe, MaxLoop, MVar)
        
        a = a_ini;
        [fMe, FtMdl] = mySecondFit(a);
        Res = smoothdata(Data - FtMdl, 'movmean', PpP);
        Res = mean(Res, 2);
        [Val, Id] = max(Res);
        a(1, end+1) = AxisX(Id, 1);
        a(2, end) = mean(a(2, 1:end-1));
        a(2, :) = max(MultiMe*a(2, :), MVar);
        while 1
            LM = true;
            if any(Limits ~= 0)
                il = find(Limits ~= 0)+1;
                for ic = 1:length(il)
                    if any(abs(a(il(ic), :))/mean(abs(a(il(ic), :))) > Limits(il(ic)-1)) ...
                            || any(abs(a(il(ic), :))/mean(abs(a(il(ic), :))) < 1/Limits(il(ic)-1))
                        tgt = find(abs(a(il(ic), :))/mean(abs(a(il(ic), :))) > Limits(il(ic)-1) ...
                            | abs(a(il(ic), :))/mean(abs(a(il(ic), :))) < 1/Limits(il(ic)-1));
                        
                        ix = (1:size(a, 2))' ~= tgt(1);
                        a(il(ic), tgt) = mean(a(il(ic), ix));
                        LM = false;
                    end
                end
            end
            if LM, break; end
        end
        
        try
            [a, ~, flag] = fminunc(@mySecondFit, a, opts);
        catch
            disp('pp')
        end
        cnt = 1;
        while flag == 0
            [a, ~, flag] = fminunc(@mySecondFit, a, opts);
            if cnt > MaxLoop, break; else, cnt = cnt +1; end
        end
        fMe = mySecondFit(a);
    end

    function [f, FittedProfile, Model, Ratio] =  mySecondFit(x)
        % Constant fitting parameters
        
        nP = size(x, 2);
        Model = zeros(size(Data, 1), nP+1);
        Model(:,1) = 1;
        
        if any(x(1:2, :) < 0, 'all') ...
                || any(x(2, :) < MinVar, 'all')
            Model = inf(size(Model));
            
        elseif any(Limits ~= 0)
            id = find(Limits ~= 0)+1;
            for iLm = 1:length(id)
                if any(abs(x(id(iLm), :))/mean(abs(x(id(iLm), :))) > Limits(id(iLm)-1)) ...
                        || any(abs(x(id(iLm), :))/mean(abs(x(id(iLm), :))) < 1/Limits(id(iLm)-1))
                    Model = inf(size(Model));
                end
            end
        end
        
        
        if any(~isfinite(Model), 'all')
            f = 3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
            Ratio = nan(nbrChannels, nP+1);
            FittedProfile = nan(nbrPts, nbrChannels);
            
        else
            for f_ii = 1:nP
                
                if x(1, f_ii) < AxisX(1) - 2*x(2, f_ii)  ...
                        || x(1, f_ii) > AxisX(end) + 2*x(2, f_ii)
                    f = 3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
                    Ratio = nan(nbrChannels, nP+1);
                    FittedProfile = nan(nbrPts, nbrChannels);
                    return
                    
                else
                    
                    switch Options.Function
                        case "Gauss"
                            dataOut = gaussPeak(AxisX, x(1:2, f_ii));
                            
                        case "PMG1"
                            dataOut = PMG1Peak(AxisX, x(1:3, f_ii));
                            
                        case "EMG"
                            dataOut = EMGPeak(AxisX, x(1:3, f_ii));
                            
                        case "PMG2"
                            dataOut = PMG2Peak(AxisX, x(1:4, f_ii));
                            
                    end
                    Model(:, f_ii+1) = dataOut;
                end
            end
            
            
            
            Ratio = Data'/Model';
            Ratiop = Ratio;
            if Options.Penalisation
                Ratiop(Ratiop < 0) = Options.PenalisationWeight*Ratiop(Ratiop < 0);
            end
            FittedProfile = Model*Ratio';
            
            f = sqrt(mean((Data -  Model*Ratiop').^2, 'all'));
            if ~isfinite(f)
                f = 3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
            end
        end
        
    end
end