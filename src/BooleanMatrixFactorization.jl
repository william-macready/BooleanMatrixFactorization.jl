__precompile__()

module BooleanMatrixFactorization

using Parameters
using Distributions
using StatsFuns

export BeliefPropagationOptions, ProblemSpecification, makeData, makeMask,
    inference, experiment


"""
    BeliefPropagationOptions

Pararmeters define the operation of belief propagation.
"""
@with_kw immutable BeliefPropagationOptions
    learningRate = 0.5     # damping parameter
    maxIter = 200          # max number of message updates
    verbose = true
    tol = 1e-6         # convergence criterion on message diffs
    minSum = true          # true for minSum, false for sumProd
    @assert tol > 0.0
    @assert learningRate > 0.0
    @assert maxIter > 0
end


"""
    ProblemSpecification

Structure representing te parameters defining a factorization problem
"""
@with_kw immutable ProblemSpecification
    xor = false                                # is problem xor or and
    K = 1                                      # desired rank
    PX1 = 0.5                                  # prior probability of X[i,k] = 1
    PY1 = 0.5                                  # prior probability of Y[j,k] = 1
    PO1GivenZ1 = 0.99                          # noise model, P(O[i,j]=1|Z[i,j]=1)
    PO0GivenZ0 = 0.99                          # noise model, P(O[i,j]=0|Z[i,j]=0)
    Obs = Array{Bool}(0,0)                     # observation matrix
    mask = Array{Bool}(0,0)                    # only use observation O[i,j] where mask[i,j]==true
    numObs = count(mask)                       # number of observations in O
    posObs = Obs[mask]                         # positive observations
    negObs = Vector{Bool}(.!(posObs))          # negative observations
    logOddsX = log(PX1) - log(1-PX1)           # log odds for X[i,j]=1
    logOddsY = log(PY1) - log(1-PY1)           # log odds for X[i,j]=1
    co0 = log(1-PO1GivenZ1) - log(PO0GivenZ0)  # log(P(O[i,j]=0|Z[i,j]=1)/P(O[i,j]=0|Z[i,j]=0))
    co1 = log(PO1GivenZ1) - log(1-PO0GivenZ0)  # log(P(O[i,j]=1|Z[i,j]=1)/P(O[i,j]=1|Z[i,j]=0))
    @assert 0 < K
    @assert 0.0 ≤ PX1 ≤ 1.0
    @assert 0.0 ≤ PY1 ≤ 1.0
    @assert 0.0 ≤ PO1GivenZ1 ≤ 1.0
    @assert 0.0 ≤ PO0GivenZ0 ≤ 1.0
    @assert size(Obs) == size(mask)
end


"""
    setData(probSPec::ProblemSpec,O,m=trues(O))

Set the data and mask in probSpec to O and m.
"""
function setData(probSpec::ProblemSpecification, O, m=trues(O), K=probSpec.K)
    size(O) ≠ size(m) && error("Observation and mask matrices must be the same size")
    (any(K .> size(O)) || (K<0)) && error("require 0 < K ≤ size(O)")
    posObs = O[m]
    ProblemSpecification(probSpec; Obs=O, mask=m, K=K, posObs=posObs, negObs=Vector{Bool}(.!(posObs)))
end


"""
    X,Y,Z,O = makeData(M,N,probSpec)

Generate test data X,Y,Z,O such that O is a noisy version of Z = X*Y'. Z is M×N
"""
function makeData(M, N, probSpec::ProblemSpecification)
    X = Matrix{Bool}(rand(M, probSpec.K) .< probSpec.PX1)
    Y = Matrix{Bool}(rand(M, probSpec.K) .< probSpec.PY1)
    Z = Matrix{Bool}(probSpec.xor ? ((X * Y' % 2) .== 1) :  (X * Y' .> 0))
    # apply the noise model
    B0 = Bernoulli(1.0-probSpec.PO0GivenZ0) # Prob(O=1|Z=0)
    B1 = Bernoulli(probSpec.PO1GivenZ1)     # Prob(O=1|Z=1)
    O = similar(Z)
    for (i,z) ∈ enumerate(Z)
        O[i] = z ? rand(B1) : rand(B0)
    end
    X, Y, Z, O
end


"""
    makeMask(M,N, pObserved=1.0, permMatrix=false)

Generate a random mask of observations within an M×N observation matrix
"""
makeMask(M, N, pObserved=1.0) = Matrix{Bool}(rand(M,N) .< pObserved)
makeMask(O::Matrix{Bool}, pObserved=1.0) = makeMask(size(O,1),size(O,2),pObserved)

phi(x::Float64) = max(0, x)
psi(x::Float64) = log1pexp.(x)


function getAdjList(A, bip_connections = false)# A is m x n, nbM: list of nonzeros in each row
    (M, N) = findn(A);  m,n = size(A)
    nbM = [Int64[] for i=1:m];  nbN = [Int64[] for i=1:n]
    for z=1:length(N)
        push!(nbN[N[z]], z);  push!(nbM[M[z]], z)
    end
    if bip_connections
        nbMN = [Int64[] for i=1:m];  nbNM = [Int64[] for i=1:n]
        for z=1:length(N)
            push!(nbNM[N[z]], M[z]);  push!(nbMN[M[z]], N[z])
        end
        nbM, nbN, nbMN, nbNM
    else
        nbM, nbN
    end
end

"""
    updateMinSum!(outX,outY,inZ,outZ,newInX,newInY,ps::ProblemSpecification)
"""
function updateMinSum!(outX::Array{Float64, 2}, outY::Array{Float64, 2}, inZ::Array{Float64, 2},
                       outZ::Array{Float64, 2}, newInX::Array{Float64, 2}, newInY::Array{Float64, 2},
                       probSpec::ProblemSpecification)
    inZ[:,:] = min.(outX + outY, outX, outY)
    inZpos = phi.(deepcopy(inZ))
    inZ_max, inZ_maxind = findmax(inZ, 2)   # max over k
    inZ[inZ_maxind[:]] = -Inf
    inZ_max_sec = max.(-maximum(inZ, 2), 0) # new
    inZ_max = phi.(-inZ_max)
    sumval = sum(inZpos, 2)
    sumval[probSpec.posObs,1] += probSpec.co1
    sumval[probSpec.negObs,1] += probSpec.co0
    inZ_maxind_k = map(x -> ind2sub(size(outX), x)[2], inZ_maxind)
    tmp_inZ_max = deepcopy(inZ_max)
    inZpos = -inZpos .+ sumval
    for k = 1:size(outZ,2)
        self_maxind = (inZ_maxind_k .== k)
        tmp_inZ_max[self_maxind] = inZ_max_sec[self_maxind]
        outZ[:, k] = min.( tmp_inZ_max[:], inZpos[:,k]) #sumval_edge[:] - inZpos[posObs,K])
        tmp_inZ_max[self_maxind] = inZ_max[self_maxind]
    end
    newInX[:,:] = phi.(outZ + outY) - phi.(outY)
    newInY[:,:] = phi.(outZ + outX) - phi.(outX)
end

"""
    updateMinsumNoiseless(outX, outY, inZ, outZ, newInX, newInY, probSpec)
"""
function updateMinSumNoiseLess!(outX::Array{Float64, 2}, outY::Array{Float64, 2}, inZ::Array{Float64, 2},
                                outZ::Array{Float64, 2}, newInX::Array{Float64, 2}, newInY::Array{Float64, 2},
                                prodSpec::ProblemSpecification)
    inZ[:,:] = min(outX[probSpec.posObs,:]+outY[probSpec.posObs,:], outX[probSpec.posObs,:], outY[probSpec.posObs,:])
    inZ_max, inZ_maxind = findmax(inZ, 2)
    inZ[inZ_maxind[:]] = -Inf
    inZ_max_sec = phi(-maximum(inZ, 2))
    inZ_max = phi(-inZ_max)
    inZ_maxind_k = map(x -> ind2sub(size(outZ), x)[2], inZ_maxind)
    tmp_inZ_max = deepcopy(inZ_max)
    for k = 1:size(outZ,2)
        self_maxind = (inZ_maxind_k .== k)
        tmp_inZ_max[self_maxind] = inZ_max_sec[self_maxind]
        outZ[:, k] = tmp_inZ_max[:]
        tmp_inZ_max[self_maxind] = inZ_max[self_maxind]
    end
    newInY[:,:] = -phi(outX)
    newInX[:,:] = -phi(outY)
    newInX[probSpec.posObs,:] += phi(outZ + outY[probSpec.posObs,:])
    newInY[probSpec.posObs,:] += phi(outZ + outX[probSpec.posObs,:])
end

"""
    updateSumProd!(outX, outY, inZ, outZ, newInX, newInY, probSpec::ProblemSpecification)
"""
function updateSumProd!(outX::Array{Float64, 2}, outY::Array{Float64, 2}, inZ::Array{Float64, 2},
                        outZ::Array{Float64, 2}, newInX::Array{Float64, 2}, newInY::Array{Float64, 2},
                        probSpec::ProblemSpecification)
    inZ[:,:] = psi(outX + outY - log(1 + exp(outX) + exp(outY)))
    sumvalLog = sum(inZ, 2)
    inZ[:,:] = -inZ .+ sumvalLog
    outZ[:,:] = exp(inZ) - 1
    inZ[probSpec.posObs,:] += probSpec.co1
    inZ[probSpec.negObs,:] += probSpec.co0
    outZ[probSpec.posObs,:] *= exp(probSpec.co1)
    outZ[probSpec.negObs,:] *= exp(probSpec.co0)
    outZ[:,:] = inZ - log(1 + outZ)
    newInX[:,:]= psi(outZ + outY) - psi(outY)
    newInY[:,:] = psi(outZ + outX) - psi(outX)
end

"""
    updateMargs!()
"""
function updateMargs!(margX::Array{Float64, 2}, margY::Array{Float64, 2}, outX::Array{Float64, 2},
                      outY::Array{Float64, 2}, inX::Array{Float64, 2}, inY::Array{Float64, 2},
                      nbM::Vector{Vector{Int64}}, nbN::Vector{Vector{Int64}}, probSpec::ProblemSpecification)
    for M=1:length(nbM)
        margX[M,:] = sum(inX[nbM[M],:], 1) .+ probSpec.logOddsX
        outX[nbM[M], :] = -inX[nbM[M],:] .+ margX[M,:]'
    end
    for N=1:length(nbN)
        margY[N,:] = sum(inY[nbN[N],:], 1) .+ probSpec.logOddsY
        outY[nbN[N], :] = -inY[nbN[N],:] .+ margY[N,:]'
    end
end

"""
    inference(O,mask,)
"""
function inference(probSpec::ProblemSpecification, bpOpts::BeliefPropagationOptions)
    O, mask, K = probSpec.Obs, probSpec.mask, probSpec.K
    M, N = size(O);  nObs = probSpec.numObs
    nObs = countnz(mask)   # number of observations in O
    nbM, nbN = getAdjList(mask)
    posObs = O[mask];  negObs = Vector{Bool}(.!(posObs))
    margX = zeros(M, K);  margY = zeros(N, K)
    localX = probSpec.logOddsX + log.(rand(M,K))
    localY = probSpec.logOddsY + log.(rand(N,K))
    inX = zeros(nObs, K); outX = log.(rand(nObs, K))
    newInX = zeros(nObs, K);  newInY = zeros(nObs, K)
    inY = zeros(nObs, K);  outY = log.(rand(nObs, K))
    inZ = zeros(nObs, K);  outZ = zeros(nObs, K)
    iters = 1;  diffMsg = Inf
    while (diffMsg > bpOpts.tol) && iters ≤ bpOpts.maxIter
        if bpOpts.minSum
            updateMinSum!(outX, outY, inZ, outZ, newInX, newInY, probSpec)
        else
            updateSumProd!(outX, outY, inZ, outZ, newInX, newInY, probSpec)
        end
        diffMsg = maximum(abs,newInX - inX)
        inX .= (1. - bpOpts.learningRate)*inX .+ bpOpts.learningRate*newInX
        inY .= (1. - bpOpts.learningRate)*inY .+ bpOpts.learningRate*newInY
        updateMargs!(margX, margY, outX, outY, inX, inY, nbM, nbN, probSpec)
        bpOpts.verbose && println("iter $iters, diff:$diffMsg")
        iters += 1
    end
    X = Matrix{Bool}(margX .> 0);  Y = Matrix{Bool}(margY .> 0)
    X, Y, Matrix{Bool}(probSpec.xor ? (X * Y' % 2) .== 1 : X * Y' .> 0)
end


"""
    compare(A,B,mask)

Compare two Boolean matrices A and B by reporting the fraction of elements
differing both across all elements, and across those flagged as true within mask.
"""
function compare(A, B, mask=trues(A))
    any(size(A)≠size(B)) && error("Matrices must be the same size")
    d1, d2, obs = countnz(A)/length(A), countnz(B)/length(B), countnz(mask)/length(mask)
    println("densities: $d1,$d2    observations: $obs")
    dist = countnz(A .!== B) / length(A)
    distObs = countnz(A[mask] .!== B[mask]) / sum(mask)
    println("diff (all elements): $dist,  diff (on observations): $distObs")
    dist, distObs
end


"""
    experiment(M,N)

Run an experiment generating random test data, running BP and comparing inference
to  test data.
"""
function experiment(M,N,k=1)
    srand(1234)
    probSpec = ProblemSpecification(K=k)
    X, Y, Z, O = makeData(M, N, probSpec)
    pObserved = 0.99
    mask = makeMask(O, pObserved)
    probSpec = setData(probSpec, O, mask, k)
    bpOpts = BeliefPropagationOptions()
    Xh,Yh,Zh = inference(probSpec, bpOpts)
    println("comparing Zh and O");  compare(Zh, O, mask)
    println("comparing X and Xh");  compare(X, Xh)
    print("X: ");  show(STDOUT, "text/plain", X);  println("")
    print("Xh: ");  show(STDOUT, "text/plain", Xh);  println("")
    println("comparing Y and Yh");  compare(Y, Yh)
    print("Y: ");  show(STDOUT, "text/plain", Y);  println("")
    print("Yh: ");  show(STDOUT, "text/plain", Yh);  println("")
    println("comparing Z and Zh");  compare(Z, Zh, mask)
    print("Z: ");  show(STDOUT, "text/plain", Z);  println("")
    print("Zh: ");  show(STDOUT, "text/plain", Zh);  println("")
    O, Z, Zh, mask
end

end # module BooleanMatrixFactorization
