# Predict parameter evolution in MultiSAR


""" 
    PredLocalMultiSAR_SV

Compute predictions with MultiSAR model and evaluate Log Predictive Scores (LPS)

""" 
function PredLocalMultiSAR_SV(nPredPerIter, yTest, zₜall_orig, pFit, season, statTrans, 
    θₜpost, Hₜpost, ϕpost, σ²ₙpost, μpost, σₑₜpost, ϕ̃post, μ̃post, σ̄²ₙpost, forecastHorizons)

    p, nIter = size(θₜpost)
    activeLags = FindActiveLagsMultiSAR(pFit, season)
    LPS = zeros(length(forecastHorizons))
    MAE = NaN*ones(length(forecastHorizons))
    forecastHorizons = forecastHorizons[forecastHorizons .<= length(yTest)]
    maxHorizon = maximum(forecastHorizons)
    yPreds = NaN*ones(maxHorizon, nIter*nPredPerIter)
    yPred = zeros(maximum(forecastHorizons))
    @showprogress for i = 1:nIter
        for j = 1:nPredPerIter
            zₜall = zₜall_orig
            h = Hₜpost[:,i]
            θ = θₜpost[:,i]
            μ = μpost[:,i]
            ϕ = ϕpost[:,i]
            σₙ = sqrt.(σ²ₙpost[:,i])
            h̃ = log(σₑₜpost[i]^2)
            μ̃ = μ̃post[i]
            ϕ̃ = ϕ̃post[i]
            σ̄²ₙ = σ̄²ₙpost[i]
            count = 0
            for hor = 1:maxHorizon
                h = μ .+ ϕ .* (h .- μ) .+ rand.(σₙ.*LogisticBeta(1/2,1/2))
                θ = θ .+ exp.(h/2) .* randn(p)
                h̃ = μ̃ + ϕ̃*(h̃ - μ̃) + sqrt(σ̄²ₙ)*randn() # StochVol
                zₜ = zₜall[activeLags]
                ϕ̃reg = MultiSARtoReg(θ, pFit, season, activeLags; ztrans = statTrans)
                predDist = Normal(zₜ ⋅ ϕ̃reg, exp(h̃/2))
                yPred = rand(predDist)
                yPreds[hor, j + (i-1)*nPredPerIter] = yPred
                if hor ∈ forecastHorizons 
                    count = count + 1
                    LPS[count] = LPS[count] + pdf(predDist, yTest[hor]) # LPS
                end
                zₜall = [yPred; zₜall[1:(end-1)]]
            end
        end
    end
    LPS[1:length(forecastHorizons)] .= log.(LPS[1:length(forecastHorizons)] / 
        (nIter*nPredPerIter))
    LPS[length(forecastHorizons)+1:end] .= NaN
    medianPred = median(yPreds, dims = 2)
    MAE[1:length(forecastHorizons)] .= abs.(medianPred[forecastHorizons] .- 
        yTest[forecastHorizons])
    return yPreds, LPS, MAE
end

""" 
    PredMultiSAR_SV

Compute predictions with static MultiSAR model and evaluate Log Predictive Scores (LPS)

""" 
function PredMultiSAR_SV(nPredPerIter, yTest, zₜall_orig, pFit, season, statTrans, θₜpost, 
   σₑₜpost, ϕ̃post, μ̃post, σ̄²ₙpost, forecastHorizons)

    p, nIter = size(θₜpost)
    activeLags = FindActiveLagsMultiSAR(pFit, season)
    LPS = zeros(length(forecastHorizons))
    MAE = NaN*ones(length(forecastHorizons))
    forecastHorizons = forecastHorizons[forecastHorizons .<= length(yTest)]
    maxHorizon = maximum(forecastHorizons)
    yPreds = NaN*ones(maxHorizon, nIter*nPredPerIter)
    yPred = zeros(maximum(forecastHorizons))
    @showprogress for i = 1:nIter
        for j = 1:nPredPerIter
            zₜall = zₜall_orig
            θ = θₜpost[:,i]
            h̃ = log(σₑₜpost[i]^2)
            μ̃ = μ̃post[i]
            ϕ̃ = ϕ̃post[i]
            σ̄²ₙ = σ̄²ₙpost[i]
            count = 0
            ϕ̃reg = MultiSARtoReg(θ, pFit, season, activeLags; ztrans = statTrans)
            for hor = 1:maxHorizon
                h̃ = μ̃ + ϕ̃*(h̃ - μ̃) + sqrt(σ̄²ₙ)*randn() # StochVol
                zₜ = zₜall[activeLags]
                predDist = Normal(zₜ ⋅ ϕ̃reg, exp(h̃/2))
                yPred = rand(predDist)
                yPreds[hor, j + (i-1)*nPredPerIter] = yPred
                if hor ∈ forecastHorizons 
                    count = count + 1
                    LPS[count] = LPS[count] + pdf(predDist, yTest[hor]) # LPS
                end
                zₜall = [yPred; zₜall[1:(end-1)]]
            end
        end
    end
    LPS[1:length(forecastHorizons)] .= log.(LPS[1:length(forecastHorizons)] / 
        (nIter*nPredPerIter))
    LPS[length(forecastHorizons)+1:end] .= NaN
    medianPred = median(yPreds, dims = 2)
    MAE[1:length(forecastHorizons)] .= abs.(medianPred[forecastHorizons] .- 
        yTest[forecastHorizons])
    return yPreds, LPS, MAE
end
