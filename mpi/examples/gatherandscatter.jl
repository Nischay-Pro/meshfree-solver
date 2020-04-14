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

print("$rank: Local array received - $localarr. Global array is $globalarr\n")
