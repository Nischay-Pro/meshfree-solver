using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

root = 0  
globalarr = Array{Int64}(undef, 2*size)
localarr = Array{Int64}(undef, 2)

if rank == root
    globalarr = [i for i in 1:2*size]
end

MPI.Scatter!(globalarr, localarr, 2, root, comm)

# if root == rank
#     MPI.Scatter!(globalarr, nothing, 2, root, comm)
# else
#     MPI.Scatter!(nothing, localarr, 2, root, comm)        
# end


print("$rank: Local array received - $localarr. Global array is $globalarr\n")

@. localarr = 2 * localarr

MPI.Gather!(localarr, globalarr, 2, root, comm)

print("$rank: Local array sent - $localarr. Global array is $globalarr\n")

#=
0: Local array received - [1, 2]. Global array is [1, 2, 3, 4, 5, 6, 7, 8]
1: Local array received - [3, 4]. Global array is [0, 0, 0, 0, 0, 0, 0, 0]
3: Local array received - [7, 8]. Global array is [0, 0, 0, 0, 0, 0, 0, 0]
2: Local array received - [5, 6]. Global array is [0, 0, 0, 0, 0, 0, 0, 0]
1: Local array sent - [6, 8]. Global array is [0, 0, 0, 0, 0, 0, 0, 0]
3: Local array sent - [14, 16]. Global array is [0, 0, 0, 0, 0, 0, 0, 0]
2: Local array sent - [10, 12]. Global array is [0, 0, 0, 0, 0, 0, 0, 0]
0: Local array sent - [2, 4]. Global array is [2, 4, 6, 8, 10, 12, 14, 16]
=#