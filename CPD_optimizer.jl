using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra

# In this you will consturct a CPD optimizer of an order 3 Array.

struct CPDObj
  Factors
end

dim(cpd::CPDObj, i::Int) = size(cpd.Factors[i])[2]
CPrank(cpd::CPDObj) = size(cpd.Factors[1])[1]
## This function should take two matrices of 
## A(rank x I) and B (rank x J) 
function Khatri_Rao_Product(A::Matrix, B::Matrix)
  elt = eltype(A)
  ## Loop over indices of rank I and J and save to new matrix C
  C = zeros(elt, rank, I, J)

  return C
end

## reconstructs the order 3 CPD
function reconstruct(cpd::CPDObj)
  I,J,K = dim(cpd,1), dim(cpd,2), dim(cpd,3)
  AB = Khatri_Rao_Product(cpd.Factors[1], cpd.Factors[2])
  AB = reshape(AB, (CPrank(cpd), I * J))
  T = AB' * cpd.Factors[3]
  return reshape(T, (I, J, K))
end


I, J, K = 2,3,4
actual_cp_rank = 5
cpd = CPDObj([
  randn(actual_cp_rank,I), 
  randn(actual_cp_rank,J), 
  randn(actual_cp_rank,K)]);
## This will be our target tensor so we know the rank!
T = randn(I,J,K)
T = reconstruct(cpd)

### This algorithm we are going to use the invert the KRP optimize the problem
## Mode is a number 1 2 or 3
function Compute_CPD_Update(cpd::CPDObj, mode::Int)
  I,J,K = dim(cpd, 1), dim(cpd, 2), dim(cpd, 3)
  R = CPrank(cpd)
  updated_factors = copy(cpd.Factors)
 if mode == 1
    BC = Khatri_Rao_Product(cpd.Factors[2], cpd.Factors[3])
    BC = reshape(BC, (R, J * K))
    Tm = reshape(T, (I, J * K))
    u,s,v = svd(BC)
    Astar = Tm * (u * diagm(1 ./ s) * v')'
    updated_factors[1] = reshape(Astar', (R, I))
  elseif mode == 2
    
  elseif mode == 3

  else
    throw("Error mode invalid")
  end
  return CPDObj(updated_factors)
end


function random_cpd(rank, I, J, K)
  return CPDObj([
  randn(rank, I),
  randn(rank, J),
  randn(rank, K)
  ])
end

guess_rank = 3
cpguess = random_cpd(guess_rank, I, J, K)

updated_CPD = Compute_CPD_Update(cpguess, 1)

@show norm(T - reconstruct(cpguess)) / norm(T)
@show norm(T - reconstruct(updated_CPD)) / norm(T)
## contraction sequences, whats the best way to contract tensors
## What do we optimzer for, flops vs transposes...



#### ITensorCPD hans on with ising model 
using ITensorCPD, ITensors

#check = ITensorCPD.FitCheck(1e-3, 100, norm(T))
iT = itensor(T, Index.((I,J,K)))
CPopt = ITensorCPD.decompose(iT, 3; check = ITensorCPD.NoCheck(10), verbose=true);

norm(iT - ITensorCPD.reconstruct(CPopt)) / norm(iT)

check = ITensorCPD.FitCheck(1e-10, 1000, norm(iT))
CPopt = ITensorCPD.decompose(iT, 5; check, verbose=true);


#### contracting CPD based tensor networks with non-CPD networks.
using ITensorNetworks

X1, X2 = Index.((10, 20))
L = random_itensor(Float64, Index(I), X1)
M = random_itensor(Float64, Index(J), X1, X2)
N = random_itensor(Float64, Index(K), X2)

TN = ITensorNetwork([L,M,N])

cpd = ITensorCPD.random_CPD(TN, 5)
check = ITensorCPD.FitCheck(1e-3, 100, norm(contract(TN)))
CPOpt = ITensorCPD.als_optimize(TN, cpd; check, verbose=true);
