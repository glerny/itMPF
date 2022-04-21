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

function [Model, FittedChannels, Stats, myModel, Options, Interval] = IterativeMethod2_fminsearch(AxisX, Data, ModelSpectra, Options)

%TODO: catch rank deficiency warnings
tic
%% 1. Initialisation and validation of variables
narginchk(3, 4)
opts = optimset('Display','off');

[M, N, O] = size(Data);

nbrChannels = N;
nbrPts = M;
Data = reshape(Data,  M, N*O);

if nargin <= 3
    Options = {};
end


if ~isfield(Options, 'maxPeaks'), Options.maxPeaks = 20; end
if ~isfield(Options, 'LoopMe'), Options.LoopMe = 5; end
if ~isfield(Options, 'RecursiveLoop'), Options.RecursiveLoop = 0.95; end
if ~isfield(Options, 'InitialFactor'), Options.InitialFactor = [1 0.75 0.5 0.25]; end
if ~isfield(Options, 'MinResolution'), Options.MinResolution = 0; end
if ~isfield(Options, 'Penalisation')
    Options.Penalisation = true;
    Options.PenalisationWeight = 1.5;
end
if ~isfield(Options, 'Constrained')
    Options.Constrained.SharedParameters = "Full"; % "Full", "Partial" or "None"
    Options.Constrained.Limits = [4 0];
end
if ~isfield(Options, 'PointsPerPeaks'), Options.PointsPerPeaks = [10 25]; end
if ~isfield(Options, 'MinMax'), Options.MinMax = 0.05; end
if ~isfield(Options, 'Robust'), Options.Robust = false; end
if ~isfield(Options, 'Limits'), Options.Limits = [0 inf]; end

%TODO: Check the Options validity
Dms = mean(AxisX(2:end, 1) - AxisX(1:end-1, 1));
StartVar = (Dms*Options.PointsPerPeaks(1)/4)^2;  %Peakwith at base (4 sigma) => PtsPerPeaks/4
MinVar   = max(StartVar/10, Options.Limits(1));
MaxVar   = min(StartVar*10, Options.Limits(2));
Interval = [MinVar MaxVar];
Data(~isfinite(Data)) = 0;
FittedModel = zeros(size(Data));

%% 2. Initialisation of the recursive loop
if Options.Constrained.SharedParameters == "Full"
    % All fitting parameters (a2, a2,...) are equals in all
    % 1. Find 1st Local maximum and peak variance
    
    % averaged over all channels smoothed (to remove spikes) profile
    nPeak = 0;
    a = [];
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
        
        Resolution = getResolution([a(2:end)', sqrt(a(1))*ones(size(a(2:end)'))]);
        if f <= Options.RecursiveLoop*f_ini ...
                && (~any(Resolution < Options.MinResolution, 'all')) ...
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
                
                Resolution = getResolution([a(2:end)', sqrt(a(1))*ones(size(a(2:end)'))]);
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
    
elseif Options.Constrained.SharedParameters == "None"
    nPeak = 0;
    
    % variance
    a = [];
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
        [f, FittedModel, ~, Ratio] = mySecondFit(a);
        
        Resolution = getResolution([a(1, :)', sqrt(a(2, :))']);
        if f <= Options.RecursiveLoop*f_ini ...
                && (~any(Resolution < Options.MinResolution, 'all')) ...
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
                [f, FittedModel, ~, Ratio] = mySecondFit(a);
                
                Resolution = getResolution([a(2:end)', sqrt(a(1))*ones(size(a(2:end)'))]);
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
    
    [f_final, FittedModel, myModel, Ratio] = mySecondFit(a_ini);
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
Stats.RMSE = f_final/(M+N+O);
Stats.RSSE = f_final;

if ~isempty(a_ini)
    if Options.Constrained.SharedParameters == "Full"
        Model.Peaks.FittingParameters = [a_ini(2:end)', a_ini(1)*ones(size(a_ini(2:end)'))];
    elseif Options.Constrained.SharedParameters == "None"
        Model.Peaks.FittingParameters = a_ini';
    end
    Model.Peaks.FittingIntensities = reshape(Ratio(:, 2:end),  N, O, nPeak);
    Model.Peaks.BackgroundIntensities = reshape(Ratio(:, 1),  N, O);
    
    for ii = 1:nPeak+1
        for jj = 1:nbrChannels
            FittedChannels(:, ii, jj) = Ratio(jj, ii)*myModel(:, ii);
        end
    end
else
    Model.Peaks.FittingParameters = nan;
    Model.Peaks.FittingIntensities = nan;
    Model.Peaks.BackgroundIntensities = reshape(Ratio(:, 1),  N, O);
    FittedChannels = [];
    
end

    function [a, fMe] = addAPeak_Full(a_ini, PpP, MultiMe, MaxLoop, MVar)
        
        a = a_ini;
        [fMe, FtMdl] = myFirstFit(a);
        Res = smoothdata(Data - FtMdl, 'movmean', PpP);
        Res = mean(Res, 2);
        [Val, Id] = max(Res);
        if isempty(a)
            a(1) = (MinVar + MaxVar)/2;
            a(2) =  AxisX(Id, 1);
            
        else
            a(end+1) = AxisX(Id, 1);
            a(1) = max(MultiMe*a(1), MVar);
        end
        
        [a, ~, flag] = fminsearch(@myFirstFit, a, opts);
        cnt = 1;
        while flag == 0
            [a, ~, flag] = fminsearch(@myFirstFit, a, opts);
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
        if isempty(a)
            
            a(1, 1) = AxisX(Id, 1);
            a(2, 1) = (MinVar + MaxVar)/2;
        else
            
            a(1, end+1) = AxisX(Id, 1);
            a(2, end) = mean(a(2, 1:end-1));
        end
        
        a(2, :) = max(MultiMe*a(2, :), MVar);
        
        [a, ~, flag] = fminsearch(@mySecondFit, a, opts);
        cnt = 1;
        while flag == 0
            [a, ~, flag] = fminsearch(@mySecondFit, a, opts);
            if cnt > MaxLoop, break; else, cnt = cnt +1; end
        end
        fMe = mySecondFit(a);
    end

    function [f, FittedProfile, Model, Ratio] = myFirstFit(x)
        % Constant fitting parameters
        % Introduction: created FittedProfiles and reorganised x
        
        
        %% Constant fitting parameters
        warning off
        % TODO: catch warning rank deficienty
        
        
        if isempty(x)
            nP = 0;
        else
            nP = length(x) - 1;
        end 
        
        Model = zeros(size(Data, 1), nP+1);
        Model(:,1) = 1;
        stop = false;
        if ~isempty(x)
            if any(x < 0, 'all') ...
                    || x(1) < MinVar ...
                    || x(1) > MaxVar
                
                Model = inf(size(Model));
            end
        end
        
        if any(~isfinite(Model), 'all')
            stop = true;
            Ratio = nan(nbrChannels, nP+1);
            FittedProfile = nan(nbrPts, nbrChannels);
        else
            for f_ii = 1:nP
                
                if x(f_ii+1) < AxisX(1) - 2*sqrt(x(1))  ...
                        || x(f_ii+1) > AxisX(end) + 2*sqrt(x(1))
                    stop = true;
                    Ratio = nan(nbrChannels, nP+1);
                    FittedProfile = nan(nbrPts, nbrChannels);
                    break
                    
                else
                    cModel = ModelSpectra;
                    cModel(:, 1) = cModel(:,1)*sqrt(x(1)) + x(f_ii+1) ;
                    Model(:, f_ii+1) = interp1(cModel(:,1), cModel(:,2), AxisX);
                end
            end
        end
        Model(isnan(Model)) = 0;
        
        if stop
            f = inf;
            Ratio = nan(nbrChannels, nP+1);
            FittedProfile = nan(nbrPts, nbrChannels);
            
        else
            Ratio = Data'/Model';
            Ratiop = Ratio;
            if Options.Penalisation
                Ratiop(Ratiop < 0) = Options.PenalisationWeight*Ratiop(Ratiop < 0);
            end
            FittedProfile = Model*Ratio';
            f = sqrt(sum((Data -  Model*Ratiop').^2, 'all'));
        end
        
        warning on
        
    end

    function [f, FittedProfile, Model, Ratio] =  mySecondFit(x)
        % Constant fitting parameters
        
        % Constant fitting parameters
        % Introduction: created FittedProfiles and reorganised x
        
        
        %% Constant fitting parameters
        warning off
        % TODO: catch warning rank deficienty
        if isempty(x)
            nP = 0;
        else
            nP = size(x, 2);
        end 
        
        Model = zeros(size(Data, 1), nP+1);
        Model(:,1) = 1;
        stop = false;
        
        if ~isempty(x)
            if any(x < 0, 'all') ...
                    || any(x(2, :) < MinVar, 'all') ...
                    ||  any(x(2, :) > MaxVar, 'all')
                
                Model = inf(size(Model));
            end
        end
        
        if any(~isfinite(Model), 'all')
            stop = true;
            Ratio = nan(nbrChannels, nP+1);
            FittedProfile = nan(nbrPts, nbrChannels);
        else
            for f_ii = 1:nP
                
                if x(1, f_ii) < AxisX(1) - 2*sqrt(x(2, f_ii))  ...
                        || x(1, f_ii) > AxisX(end) + 2*sqrt(x(2, f_ii))
                    stop = true;
                    Ratio = nan(nbrChannels, nP+1);
                    FittedProfile = nan(nbrPts, nbrChannels);
                    break
                    
                else
                    cModel = ModelSpectra;
                    cModel(:, 1) = cModel(:,1)*sqrt(x(2, f_ii)) + x(1, f_ii) ;
                    Model(:, f_ii+1) = interp1(cModel(:,1), cModel(:,2), AxisX);
                end
            end
        end
        Model(isnan(Model)) = 0;
        
        if stop
            f = inf;
            Ratio = nan(nbrChannels, nP+1);
            FittedProfile = nan(nbrPts, nbrChannels);
            
        else
            Ratio = Data'/Model';
            Ratiop = Ratio;
            if Options.Penalisation
                Ratiop(Ratiop < 0) = Options.PenalisationWeight*Ratiop(Ratiop < 0);
            end
            FittedProfile = Model*Ratio';
            f = sqrt(sum((Data -  Model*Ratiop').^2, 'all'));
        end
        
        warning on
    end
end