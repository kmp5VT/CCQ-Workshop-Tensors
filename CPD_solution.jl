using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra

R, I, J = 10,5,2
A = rand(R,I)
B = rand(R,J)
## This function should take two matrices of 
## A(rank x I) and B (rank x J) 
function Khatri_Rao_Product(A::Matrix, B::Matrix)
  elt = eltype(A)
  @assert eltype(B) == elt
  rank,I = size(A)
  _, J = size(B)
  
  C = zeros(elt, rank, I, J)
  for r in 1:rank
    for i in 1:I
      # a = A[r,i]
      for j in 1:J
        C[r,i,j] = A[r,i] * B[r,j]
      end
    end
  end

  return C
end

Ceasy = [A[r,i] * B[r,j] for r in 1:R, i in 1:I, j in 1:J]
C = Khatri_Rao_Product(A,B)

norm(C - Ceasy)

# In this you will consturct a CPD optimizer of an order 3 Array.

## contraction sequences, whats the best way to contract tensors
## What do we optimzer for, flops vs transposes...



#### ITensorCPD hans on with ising model 


#### contracting CPD based tensor networks with non-CPD networks.