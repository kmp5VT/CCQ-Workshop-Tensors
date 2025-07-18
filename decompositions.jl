using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra
using Plots
using BenchmarkTools
plotly()

A = randn(5000,5000)
A = A * A'
B = randn(2000,5000)

u,s,v = svd(A)

Asvd = u * diagm(s) * v'
norm(A - Asvd) / norm(A)

Msvd = nothing
@btime begin
  u,s,v = svd(A)
  Msvd = B * (u * diagm( 1 ./ s) * v')
end;

@btime Msolve = (A \ B')';
@btime Mqr = (qr(A) \ B')';
@btime Mlu = (lu(A) \ B')';
@btime Mcd = (cholesky(A; check=false) \ B')';

@btime Mqr = (qr(A, ColumnNorm()) \ B')';
@btime Mlu = (lu(A, RowMaximum()) \ B')';
@btime Mcd = (cholesky(A, RowMaximum(); check=false) \ B')';
@btime Mcd = (cholesky(A, NoPivot(); check=false) \ B')';
#Msvd - Msolve

A = randn(5000,50)
A = A * A'
B = randn(2000,5000)

u,s,v = svd(A)

Asvd = u * diagm(s) * v'
norm(A - Asvd) / norm(A)

Msvd = nothing
@btime begin
  u,s,v = svd(A)
  Msvd = B * (u * diagm( 1 ./ s) * v')
end;

@btime Msolve = (A \ B')';
@btime Mqr = (qr(A) \ B')';
@btime Mlu = (lu(A) \ B')';
@btime Mcd = (cholesky(A; check=false) \ B')';

@btime Mqr = (qr(A, ColumnNorm()) \ B')';
@btime Mlu = (lu(A, RowMaximum()) \ B')';
@btime Mcd = (cholesky(A, RowMaximum(); check=false) \ B')';
@btime Mcd = (cholesky(A, NoPivot(); check=false) \ B')';

cholesky(A, Val(true); check=false)