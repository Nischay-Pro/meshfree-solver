#include <metis.h>
#include <iostream>
#include <iomanip> 
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <set>
#include <map>
#include <climits>
#include <cassert>
using namespace std;

typedef struct point {
    idx_t id;
    double x, y;
    int left, right;
    double pr, rho, u1,u2;
    idx_t flag1, flag2;
} point;

class Graph{
public:
    idx_t nvtxs;
    vector<point> ptVec;
    vector<idx_t> xadjVec;
    vector<idx_t> adjncyVec;
    
    // read file and build graph
    Graph(){
        ifstream infile("point-input");
	assert (infile.is_open());
        string line;
        nvtxs = 0;
        xadjVec.push_back(adjncyVec.size());
        while(getline(infile, line)){
            nvtxs++;
            istringstream iss(line);
            point temp ;
            iss >> temp.id >> temp.x >> temp.y 
                >> temp.left >> temp.right 
                >> temp.flag1 >> temp.flag2;
            temp.u1 = temp.u2 = temp.pr = temp.rho = INT_MAX;
            ptVec.push_back(temp);
            int neighborId;
            int neighborCount;
            iss >> neighborCount;
            while(iss >> neighborId){
                adjncyVec.push_back(neighborId-1);
            }
            xadjVec.push_back(adjncyVec.size());
        }
        
        ifstream infileIP("input.in");
        string line1;
        while(getline(infileIP, line1)){
            string input1;
            int    input2;
            istringstream iss1(line1);
            iss1 >> input1 >> input2;
            if(input1 == "files"){
                while(input2--){
                    string filename;
                    getline(infileIP, filename);
                    if(filename == "initConditions.dat"){
                        ifstream infile1("initConditions.dat");
                        int curr = 0;
                        string init;
                        while(getline(infile1, init)){
                            istringstream iss;
                            iss.precision(20);
                            iss.str(init);
                            int id;
                            iss >> id >> ptVec[curr].u1 >> ptVec[curr].u2 >> ptVec[curr].rho >> ptVec[curr].pr;
                            curr++;
                        }
                    }
                }
            }
        } 
    }

    // generates filename in string format from partition number
    string getfileName(int partID, int nParts){
        string s;
        stringstream out;
        out << partID;
        s = out.str();
        if(nParts == 1)
            return "partGrid";
        else {
            if(partID > -1 && partID < 10)
                return "partGrid0" + s;
            else 
                return "partGrid0" + s;
        }
    }

    // the main metis partitioning and output processing function
    void partition(int n){

        /*  partitioning routine  */
        idx_t nParts    = n; // number of parts to partition the graph
        idx_t nWeights  = 1; // number of balancing constraints, should be at least 1
        idx_t part[nvtxs]; // stores the partition vector of the graph
        idx_t objval; 
        idx_t *xadj = &xadjVec[0]; // Indexes of starting points in adjacent array
        idx_t *adjncy = &adjncyVec[0]; // Adjacent vertices in consecutive index order
        if(n==1){
            for(int i=0; i<nvtxs; i++){
                part[i] = 0;
            }
        } else {
            int out = METIS_PartGraphKway(&nvtxs, &nWeights, xadj, adjncy,
                        NULL, NULL, NULL, &nParts, NULL,
                        NULL, NULL, &objval, part);
            std::cout << "output code: " << out << std::endl;
            std::cout << "total edge cuts: " << objval << std::endl;
        }
        
        /*  output processing routine  */
        // creates sets of ghost nodes for each partition
        set<int> ghosts[nParts];
        int totalPoints[nParts] = {0};
        for(int i = 0; i < nvtxs; i++){
            totalPoints[part[i]]++;
            int partn = part[i];
            for(idx_t j = xadjVec[i]; j < xadjVec[i+1]; j++){
                int nbr = adjncyVec[(int)(j)];
                if(part[nbr] != partn){
                    if(ghosts[part[i]].find(nbr) == ghosts[part[i]].end()){
                        ghosts[part[i]].insert(nbr);
                    }
                }
            }
        }
        
        // creates mapping of local and input numberings for each partition
        map<int, int> inputToLoc[nParts];
        int currLocalIndex[nParts];
        for(int i=0; i<nParts; i++)
            currLocalIndex[i] = 1;
        for(int i = 0; i < nvtxs; i++){
            inputToLoc[ part[i] ][ ptVec[i].id ] = currLocalIndex[ part[i] ];
            currLocalIndex[ part[i] ]++;
        }
        for(int i=0; i < nParts; i++) {
            set<int>::iterator itr;
            for(itr = ghosts[i].begin(); itr!=ghosts[i].end(); itr++){
                inputToLoc[i][ ptVec[*itr].id ] = currLocalIndex[i];
                currLocalIndex[i]++;
            }
        }

        // creates mapping of local and global numberings over all partitions
        int currGlobalNum[nParts];
        int globNumCounter = 1;
        map<int, int> inputToGlob;
        for(int i=0; i<nParts; i++){
            currGlobalNum[i] = globNumCounter;
            globNumCounter += totalPoints[i];
        }
        for(int i = 0; i < nvtxs; i++){
            inputToGlob[i]= currGlobalNum[ part[i] ];
            currGlobalNum[ part[i] ]++;
        }

        /*  writing output routine  */
        // create files and write basic info
        for(int i=0; i<nParts; i++){
            string filename = getfileName(i, nParts);
            ofstream outfile (filename.c_str());
            outfile << totalPoints[i]+ghosts[i].size() << " "<< totalPoints[i] << " " << ghosts[i].size() << endl;
        }

        // writes the output to files in the format : 
        // max_points local_points ghost_points
        // local_id input_id x_cord y_cord flag_1 flag_2 num_Connectivity connectivity[]
        
        // local nodes
        for(int i = 0; i < nvtxs; i++){
            int partnId = part[i];
            string filename = getfileName(partnId, nParts);
            fstream write;
            write.open(filename.c_str(), fstream::app);
	        write << fixed;
            write << inputToLoc[partnId][ptVec[i].id] << " " << inputToGlob[i] 
                << setprecision(20)  << " " << ptVec[i].x << " " << ptVec[i].y 
		        << " " << inputToLoc[partnId][ptVec[i].left] << " " << inputToLoc[partnId][ptVec[i].right]
                << " " << ptVec[i].flag1 << " " << ptVec[i].flag2 << " ";
            write << xadjVec[i+1]-xadjVec[i] << " ";
            for(idx_t j=xadjVec[i]; j<xadjVec[i+1]; j++){
                write << inputToLoc[partnId][adjncyVec[j]+1] << " ";
            }
            if(ptVec[i].rho != INT_MAX) write << ptVec[i].rho << " ";
            if(ptVec[i].u1 != INT_MAX) write << ptVec[i].u1 << " ";
            if(ptVec[i].u2 != INT_MAX) write << ptVec[i].u2 << " ";
            if(ptVec[i].pr != INT_MAX) write << ptVec[i].pr << " ";
            
            currGlobalNum[partnId]++;   
            write << endl;
            write.close(); 
        }
        // ghost nodes
        for(int i=0; i < nParts; i++) {
            string filename = getfileName(i, nParts);
            fstream write;
            write.open(filename.c_str(), fstream::app);
            set<int>::iterator itr;
            write << fixed << setprecision(20);
            for(itr = ghosts[i].begin(); itr!=ghosts[i].end(); itr++){
                write << inputToLoc[i][ptVec[*itr].id] << " " << inputToGlob[*itr] 
                    << " " << ptVec[*itr].x << " " << ptVec[*itr].y << " ";
                    if(ptVec[i].rho != INT_MAX) write << ptVec[i].rho << " ";
                    if(ptVec[i].u1 != INT_MAX) write << ptVec[i].u1 << " ";
                    if(ptVec[i].u2 != INT_MAX) write << ptVec[i].u2 << " ";
                    if(ptVec[i].pr != INT_MAX) write << ptVec[i].pr << " ";
                    write << endl;
            }
            write.close(); 
        }
    }
};



int main(int argc, char *argv[]){
    int numPart;
    if (argc == 1) {
	numPart = 1;
    }
    else{
	numPart = atoi(argv[1]);
    }
    Graph g;
    g.partition(numPart);
    
    return 0;
}
