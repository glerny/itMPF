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

function [Model, FittedChannels, Stats, myModel, Options] = IterativeMethod1_fminsearch(AxisX, Data, Options)

%TODO: catch rank deficiency warnings
tic
%% 1. Initialisation and validation of variables
narginchk(2, 3)
opts = optimset('Display','off');



[M, N] = size(Data);
nbrChannels = N;
nbrPts = M;

if nargin == 2
    Options = {};
end

if ~isfield(Options, 'maxPeaks'), Options.maxPeaks = 20; end
if ~isfield(Options, 'Function'), Options.Function = 'PMG1'; end
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
            elseif Limits(1) < 1
                Limits(1) = 0;
            end
            Limits = Limits(1);
            b(2, 1) =  0;
            
        case "PMG1"
            if isempty(Limits)
                Limits(2) = 0;
            elseif length(Limits) == 1
                Limits(2) = 0;
            end
            Limits = Limits(1:2);
            Limits(Limits <= 1) = 0;
            b(3, 1) =  0;
            
        case "EMG"
            if isempty(Limits)
                Limits(2) = 0;
            elseif length(Limits) == 1
                Limits(2) = 0;
            end
            Limits = Limits(1:2);
            Limits(Limits <= 1) = 0;
            b(3, 1) =  0;
            
        case "PMG2"
            if isempty(Limits)
                Limits(3) = 0;
            elseif length(Limits) < 3
                Limits(3) = 0;
            end
            Limits = Limits(1:3);
            Limits(Limits <= 1) = 0;
            b(4, 1) =  0;
    end
    nPeak = 0;
    
    % variance
    tData = mean(Data, 2);
    [Val4Max, Id4Max] = max(tData);
    
    a = [];
    
    [f_ini, FittedModel] = mySecondFit(a);
    
    a_ini = [];
    while 1
        
        if nPeak >= Options.maxPeaks
            break
        end
        
        a_test = {}; f_test = []; count = 1;
        
        for iF = 1:length(Options.InitialFactor)
            for iPP = 1:length(Options.PointsPerPeaks)
                [a_test{count}, f_test(count)] = addAPeak_Partial(a, Options.PointsPerPeaks(iPP), Options.InitialFactor(iF), Options.LoopMe, MinVar);
                count = count + 1;
            end
        end
        nPeak = nPeak+1;
        [~, Imin] = min(f_test);
        a = a_test{Imin};
        [f, FittedModel, Mdl, Ratio] = mySecondFit(a);
        
        
        Resolution = getResolution([a(1, :)', a(2, :)']);
        if nPeak == 1, Resolution = inf; end
        if f <= Options.RecursiveLoop*f_ini && nPeak == 1
            f_ini = f;
            a_ini = a;
            
        elseif f <= Options.RecursiveLoop*f_ini ...
                && ~any(Resolution < Options.MinResolution, 'all') ...
                && min(max(Ratio(:, 3:end), [], 1)) >= Options.MinMax*max(Ratio(:, 3:end), [], "all")
            f_ini = f;
            a_ini = a;
            
        else
            if Options.Robust && nPeak < Options.maxPeaks
                
                a_test = {}; f_test = []; count = 1;
                
                for iF = 1:length(Options.InitialFactor)
                    for iPP = 1:length(Options.PointsPerPeaks)
                        [a_test{count}, f_test(count)] = addAPeak_Partial(a, Options.PointsPerPeaks(iPP), Options.InitialFactor(iF), Options.LoopMe, MinVar);
                        count = count + 1;
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
Stats.RMSE = sqrt(f_final/sum(size(Data)));

Model.Peaks.Function = Options.Function;
Model.Peaks.FittingParameters = a_ini;
Model.Peaks.FittingIntensities = Ratio(:, 3:end);

for ii = 1:nPeak+2
    for jj = 1:nbrChannels
        FittedChannels(:, ii, jj) = Ratio(jj, ii)*myModel(:, ii);
    end
end


    function [a, fMe] = addAPeak_Partial(a_ini, PpP, MultiMe, MaxLoop, MVar)
        
        a = a_ini;
        [fMe, FtMdl] = mySecondFit(a);
        Res = smoothdata(Data - FtMdl, 'movmean', PpP);
        Res = mean(Res, 2);
        [Val, Id] = max(Res);
        a(:, end+1) = b;
        a(1, end) = AxisX(Id, 1);
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
        
        [a, ~, flag] = fminsearch(@mySecondFit, a, opts);
        cnt = 1;
        while flag == 0
            [a, ~, flag] = fminsearch(@mySecondFit, a, opts);
            if cnt > MaxLoop, break; else, cnt = cnt +1; end
        end
        fMe = mySecondFit(a);
    end

    function [f, FittedProfile, Model, Ratio] =  mySecondFit(x)
        % Constant fitting parameters
        if isempty(x)
            nP = 0;
        else
            nP = size(x, 2);
        end
        
        Model = zeros(size(Data, 1), nP+2);
        Model(:,1) = 1;
        Model(:,2) = polyval([1, 0], AxisX);
        
        if ~isempty(x)
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
                f = inf; %3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
                Ratio = nan(nbrChannels, nP+1);
                FittedProfile = nan(nbrPts, nbrChannels);
                
            else
                for f_ii = 1:nP
                    
                    if x(1, f_ii) < AxisX(1) - 2*x(2, f_ii)  ...
                            || x(1, f_ii) > AxisX(end) + 2*x(2, f_ii)
                        f = inf;
                        Ratio = nan(nbrChannels, nP+2);
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
                        Model(:, f_ii+2) = dataOut;
                    end
                end
            end
        end
        
        if any(~isfinite(Model), 'all')
            f = inf;%3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
            Ratio = nan(nbrChannels, nP+2);
            FittedProfile = nan(nbrPts, nbrChannels);
        else
            Ratio = Data'/Model';
            Ratiop = Ratio;
            if Options.Penalisation
                Ratiop(Ratiop < 0) = Options.PenalisationWeight*Ratiop(Ratiop < 0);
            end
            FittedProfile = Model*Ratio';
            
            f = sum((Data -  Model*Ratiop').^2, 'all');
            if ~isfinite(f)
                f = inf;%3*sqrt(mean((Data - zeros(size(Data))).^2, 'all'));
            end
        end
        
    end
end