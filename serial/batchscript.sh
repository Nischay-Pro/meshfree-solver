shopt -s expand_aliases
echo "Patience"
cd src
julia --check-bounds=no -O3 adjoint_runner.jl /opt/grids/quadtree/part/partGrid40K 10
#julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid40K 10
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid40K 10 > residue.out &
# julia --check-bounds=no -O3 runner.jl /opt/grids/legacy/part/partGrid9600 10
# julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid190K
# julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid300K
# julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid800K 10
# julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid2.5M
# julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid10M

# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid40K 500 > file1.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid190K 100 > file2.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid300K 100 > file3.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid800K 100 > file4.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid2.5M 50 > file5.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid10M 20 > file6.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid15M 10 > file7.out &
# nohup julia --check-bounds=no -O3 runner.jl /opt/grids/quadtree/part/partGrid25M 10 > file8.out &

