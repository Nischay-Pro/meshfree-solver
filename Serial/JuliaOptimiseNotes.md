Notes-
julia --track-allocation=user
run stuff
julia> using Profile
julia> Profile.clear_malloc_data()
run same stuff
close

julia
using Coverage
analyze_malloc(".")
Coverage.clean_folder(".")
