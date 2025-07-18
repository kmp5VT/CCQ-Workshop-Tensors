###########
## This is an introduction into tensors and tensor based solutions
###########

using Pkg
Pkg.activate(".")

using LinearAlgebra, Plots

rangex = 0:0.1*π:2π
s = [sin(x) for x in rangex]

plot(rangex, s)

rangey = π:0.1*π:3π
sc = [sin(x ) * cos(y) for x in rangex, y in rangey]

surface(sc)

u,s,v = svd(sc)

plot(s)

rangex = -3:0.1:3
rangey = 0:0.1:2

A = 1
x0 = 0
y0 = 0
σx = σy = 1
gauss = [A * exp(-(sinh(x) - x0)^2 / (2 * σx^2) - (y - y0)^2 / (2 * σy^2)) for x in rangex, y in rangey]

surface(sc)
savefig("ps.html")
u,s,v = svd(sc)

surface(u * diagm(s))

m = gauss * sc'

plotly()
surface(m)

########################
## Matrix Ranks  ######
#######################

A = randn(5,10)

u,s,v = svd(A)
s

B = randn(5, 50)

C = A' * B
u,s,v = svd(C)
s
plot(s)


#######################
## Tensor Ranks #######
######################

A = randn(5, 10, 20)
A1 = reshape(A, (5, 200))
u1,s1,v1 = svd(A1)

A2 = reshape(permutedims(A, (2,1,3)) , (10, 100))
u2, s2, v1 = svd(A2)

A3 = reshape(permutedims(A, (3,1,2)), (20, 50))
u3, s3, v3 = svd(A3)

p1 = plot()
for (s, label) in zip([s1,s2,s3], ("5", "10", "20"))
  p1 = plot!(s ./ s[1]; label)
end
@show(p1)

# 5, 10 and 20 are the Kruskal ranks!

### We can make a system with reduced kruskal rank
L = randn(5,5)
M = randn(5,10,5)
N = randn(5, 20)

B = reshape(reshape(L * reshape(M, (5, 50)), (50, 5)) * N, (5,10,20))

B1 = reshape(B, (5, 200))
u1,s1,v1 = svd(B1)

B2 = reshape(permutedims(B, (2,1,3)) , (10, 100))
u2, s2, v1 = svd(B2)

B3 = reshape(permutedims(B, (3,1,2)), (20, 50))
u3, s3, v3 = svd(B3)

p2 = plot()
for (s, label) in zip([s1,s2,s3], ("5", "10", "20"))
  p2 = plot!(s  ./ s[1]; label)
end
plot(p1, p2;)
#### This is an Tensor Train and you can see the structure forces the 3rd mode to have a kruskal rank that is smaller than the dimension 

K = randn(5,5,5)
L = svd(randn(5, 5)).U
M = svd(randn(10,5)).U
N = svd(randn(5,20)).Vt

LK = reshape(L * reshape(K, (5,25)), (5,5,5))
LKN = reshape(reshape(LK, (25,5)) * N, (5,5,20))
T = M * reshape(permutedims(LKN, (2,3,1)), 5, 100)
T = permutedims(reshape(T, (10,20,5)), (3,1,2))

T1 = reshape(T, (5, 200))
u1,s1,v1 = svd(T1)

T2 = reshape(permutedims(T, (2,1,3)) , (10, 100))
u2, s2, v1 = svd(T2)

T3 = reshape(permutedims(T, (3,1,2)), (20, 50))
u3, s3, v3 = svd(T3)

p3 = plot()
for (s, label) in zip([s1,s2,s3], ("5", "10", "20"))
  p3 = plot!(s  ./ s[1]; label)
end
p3

