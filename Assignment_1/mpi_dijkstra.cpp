/*
 * This is code skeleton for COMP5112 assignment1
 * Compile: mpic++ -o mpi_dijkstra mpi_dijkstra.cpp
 * Run: mpiexec -n <number of processes> mpi_dijkstra <input file>, you will find the output in 'output.txt' file
 */


#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <cstring>
#include <algorithm>
#include "mpi.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;


/*
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and one matrix dimension convert(2D->1D) function
 */
namespace utils {
    int N; //number of vertices
    int *mat; // the adjacency matrix

    /*
     * convert 2-dimension coordinate to 1-dimension
     */
    int convert_dimension_2D_1D(int x, int y) {
        return x * N + y;
    }

    int read_file(string filename) {
        std::ifstream inputf(filename, std::ifstream::in);
        inputf >> N;
        assert(N < (1024 * 1024 *
                    20)); // input matrix should be smaller than 20MB * 20MB (400MB, we don't have two much memory for multi-processors)
        mat = (int *) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                inputf >> mat[convert_dimension_2D_1D(i, j)];
            }

        return 0;
    }

    string format_path(int i, int *pred) {
        string out("");
        int current_vertex = i;
        while (current_vertex != 0) {
            string s = std::to_string(current_vertex);
            std::reverse(s.begin(), s.end());
            out = out + s + ">-";
            current_vertex = pred[current_vertex];
        }
        out = out + std::to_string(0);
        std::reverse(out.begin(), out.end());
        return out;
    }

    int print_result(int *dist, int *pred) {
        std::ofstream outputf("output.txt", std::ofstream::out);
        outputf << dist[0];
        for (int i = 1; i < N; i++) {
            outputf << " " << dist[i];
        }
        for (int i = 0; i < N; i++) {
            outputf << "\n";
            if (dist[i] >= 1000000) {
                outputf << "NO PATH";
            } else {
                outputf << format_path(i, pred);
            }
        }
        outputf << endl;
        return 0;
    }
}//namespace utils
// you may add some helper functions here.

int find_local_minimum(int *dist, bool *visit, int loc_n) {
    int min = INT_MAX;
    int u = -1;
    for (int i = 0; i < loc_n; i++) {
        if (!visit[i]) {
            if (dist[i] < min) {
                min = dist[i];
                u = i;
            }
        }
    }
    return u;
}


void dijkstra(int my_rank, int N, int p, MPI_Comm comm, int *mat, int *all_dist, int *all_pred) {

    //------your code starts from here------
    int loc_N; // I need a local copy for N
    int loc_n; //how many vertices I need to process.
    int *loc_mat; //local matrix
    int *loc_dist; //local distance
    int *loc_pred; //local predecessor
    bool *loc_visit; //local visit record array
    //step 1: broadcast N
    if (my_rank == 0) {
        loc_N = N;
    }
    MPI_Bcast (&loc_N, 1, MPI_INT, 0, comm);



    //step 2: find loc_n
    loc_n = loc_N / p;
    //step 3: allocate local memory
    loc_mat = (int *) malloc(sizeof(int) * loc_N *loc_N);
    loc_dist = (int *) malloc(sizeof(int) * loc_n);
    loc_pred = (int *) malloc(sizeof(int) * loc_n);
    loc_visit = (bool *) malloc(loc_n * sizeof(bool));

    //step 4: broadcast matrix mat
    if (my_rank == 0) {
        memcpy (loc_mat, mat, loc_N * loc_N * sizeof (int));
    }
    MPI_Bcast (loc_mat, loc_N * loc_N, MPI_INT, 0, comm);

    //step 4: dijkstra algorithm
    // initial loc_dist, loc_pred, loc_vist

    for (int loc_i = 0; loc_i < loc_n; loc_i++) {
        int u = my_rank * loc_n + loc_i;
        loc_dist[loc_i] = loc_mat[utils::convert_dimension_2D_1D(0, u)];
        loc_pred[loc_i] = 0;
        loc_visit[loc_i] = false;
    }


    if (my_rank == 0) {
        loc_visit[0] = true;
    }


    for (int i = 1; i < loc_N; i++) {
        // find the global minimum
        int loc_u = find_local_minimum(loc_dist, loc_visit, loc_n);
        int loc_min[2], glo_min[2];
        if (loc_u == -1){
            loc_min[0] = INT_MAX;
            loc_min[1] = my_rank * loc_n + loc_u;
        }
        else{
            loc_min[0] = loc_dist[loc_u];
            loc_min[1] = my_rank * loc_n + loc_u;
        }
        MPI_Allreduce(loc_min, glo_min, 1, MPI_2INT, MPI_MINLOC, comm);
        // 

        int visit_pro = glo_min[1] / loc_n;

        if (my_rank == visit_pro) {
            int u = glo_min[1] % loc_n;
            loc_visit[u] = true;
        }

        for (int v= 0; v < loc_n; v++) {

            if (!loc_visit[v]) {
                int new_dist = glo_min[0] + loc_mat[utils::convert_dimension_2D_1D(glo_min[1], my_rank * loc_n + v)];
                if (new_dist < loc_dist[v]) {
                    loc_dist[v] = new_dist;
                    loc_pred[v] = glo_min[1];
                }
            }            
        }

    }
    
    //step 5: retrieve results back
    //Hint: use MPI_Gather(or MPI_Gatherv) function
    
    MPI_Gather (loc_dist, loc_n, MPI_INT, all_dist, loc_n, MPI_INT, 0, comm);
    MPI_Gather (loc_pred, loc_n, MPI_INT, all_pred, loc_n, MPI_INT, 0, comm);

    //step 6: remember to free memory
    free(loc_mat);
    free(loc_dist);
    free(loc_pred);
    free(loc_visit);

    //------end of your code------
}

int main(int argc, char **argv) {
    assert(argc > 1 && "Input file was not found!");
    string filename = argv[1];
    assert(utils::read_file(filename) == 0);

    //`all_dist` stores the distances and `all_pred` stores the predecessors
    int *all_dist;
    int *all_pred;
    all_dist = (int *) calloc(utils::N, sizeof(int));
    all_pred = (int *) calloc(utils::N, sizeof(int));

    //MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm comm;

    int p;//number of processors
    int my_rank;//my global rank

    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    dijkstra(my_rank, utils::N, p, comm, utils::mat, all_dist, all_pred);

    if (my_rank == 0)
        utils::print_result(all_dist, all_pred);
    MPI_Finalize();

    free(utils::mat);
    free(all_dist);
    free(all_pred);

    return 0;
}