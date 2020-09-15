# Installation

## For PAW-ATM 2020
All results and benchmarks were computed using the the 'distributed' folder in the `master` branch.  
The main repository for the history of the code can be found in [this repo](https://github.com/TestSubjector/HybridMeshfreeSolver/tree/hpc).

### 40M Grid File
We have provided a 40 Million fine grid `partGrid40M_unpartitioned` which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1iqPZOxj0UBDS3u6mv-CAHdOgbhXKAIYN).
Reviewers are requested to use the [mfpre](https://github.com/Nischay-Pro/mfpre) partitioner with `--quadtree` flag along with their specified number of partitions they want.

## Dependencies
1. Julia 1.5.1

### Julia Libraries
ClusterManagers  
Distributed  
DistributedArrays  
StaticArrays  
Printf  
TimerOutputs  
  
### Configuration of the test case.
1. Run the partitioner on the grid file provided in Google Drive.
2. A folder called `point` will be generated in the directory of the partitioner.
3. Using `mv` shift the `point` to a desired location and save its path.
4. Go to the `distributed` folder in the `master` branch
4. Rename `config.json.example` file to `config.json` with the following configuration.

```
"cfl": 0.2
"max_iters": 1000
"type": "quadtree"
"vl_const": 50
"mach": 0.85
"aoa": 1
```
5. In batchscript.sh  
* If using `Local manager with CPU affinity`instead of `SLURM`, change `runner_local.jl` to `runner.jl`
* Set `/path/to/julia/executable` to required Julia executable
* Set `x` to number of processes on which the code will run.
* Set `points` to the total number of points, which is `39381464` for `partGrid40M_unpartitioned`
* Set `/path/to/grid` to the `point` folder

### Execution
1. `chmod +x batchscript.sh` to make the execution script an executable
2. Run `batchscript.sh`
