echo "Distributed"
cd src

/path/to/julia/executable -O3 runner_local.jl x points /path/to/grid

# Sample
# julia --check-bounds=no -O3 runner_local.jl 4 48738 /opt/grids/quadtree/part_partitions/point_40K_4
