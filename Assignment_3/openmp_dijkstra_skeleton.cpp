/* Name: Qiao Baochen       
 * ID: 20419075 
 * Email: bqiao@connect.ust.hk
 */


/*
 * This is code skeleton for COMP5112-17Spring assignment3
 * Compile: g++ -std=c++11 -fopenmp -o openmp_dijkstra openmp_dijkstra_skeleton.cpp
 * Run: ./openmp_dijkstra -n <number of threads> -i <input file>,
 * you will find the output in 'output.txt' file
 */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <cstring>
#include <algorithm>
#include <sys/time.h>
#include <time.h>
#include <getopt.h>

#include <omp.h>

using std::string;
using std::cout;
using std::endl;
using std::vector;


/*
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and one matrix dimension convert(2D->1D) function
 */
namespace utils {
    int num_threads; //number of thread
    int N; //number of vertices
    int *mat; // the adjacency matrix

    string filename; // input file name
    string outputfile; //output file name, default: 'output.txt'

    void print_usage() {
        cout << "Usage:\n" << "\topenmp_dijkstra -n <number of threads> -i <input file>" << endl;
        exit(0);
    }

    int parse_args(int argc, char **argv) {
        filename = "";
        outputfile = "output.txt";
        num_threads = 0;

        int opt;
        if (argc < 2) {
            print_usage();
        }
        while ((opt = getopt(argc, argv, "n:i:o:h")) != EOF) {
            switch (opt) {
                case 'n':
                    num_threads = atoi(optarg);
                    break;
                case 'i':
                    filename = optarg;
                    break;
                case 'o':
                    outputfile = optarg;
                    break;
                case 'h':
                case '?':
                default:
                    print_usage();
            }
        }
        if (filename.length() == 0 || num_threads == 0)
            print_usage();
        return 0;
    }

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
        std::ofstream outputf(outputfile, std::ofstream::out);
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

//------You may add helper functions and global variables here------

void dijkstra(int N, int p, int *mat, int *all_dist, int *all_pred) {
    //------your code starts from here------
    //std::cout << "n " << mat[2] << std::endl;

    //init parameter we need to share
    bool *all_visit;
    int u; //global_min index
    all_visit = (bool *) malloc(N * sizeof(bool));


    //init parameter to 0
    for (int i = 0; i < utils::N; ++i) {
        all_dist[i] = mat[utils::convert_dimension_2D_1D (0, i)];
        all_pred[i] = 0;
        all_visit[i] = false;
    }

    all_visit[0] = true;

    #pragma omp parallel num_threads(p)
    {
        for (int i = 1; i < utils::N; i++) {
            // find min_val
            int u = -1;
            int min_val = INT_MAX;
            #pragma omp parallel for reduction(min: min_val)
                for (int i = 1; i < utils::N; i++){
                    if (!all_visit[i]) {
                        if (all_dist[i] < min_val) {
                            min_val = all_dist[i];
                            u = i;
                        }
                    }
            }
            #pragma omp barrier          

            all_visit[u] = true;

            #pragma omp parallel for num_threads(p)
                for (int v = 1; v < utils::N; v++){
                    if (!all_visit[v]) {
                        int new_dist = all_dist[u] + utils::mat[utils::convert_dimension_2D_1D(u, v)];
                        if (new_dist < all_dist[v]) {
                            all_dist[v] = new_dist;
                            all_pred[v] = u;
                        }
                    }
            }
        }                    
    }

    free(all_visit);
    //------end of your code------
}

int main(int argc, char **argv) {
    assert(utils::parse_args(argc, argv) == 0);
    assert(utils::read_file(utils::filename) == 0);

    assert(utils::num_threads <= utils::N);
    //`all_dist` stores the distances and `all_pred` stores the predecessors
    int *all_dist;
    int *all_pred;
    all_dist = (int *) calloc(utils::N, sizeof(int));
    all_pred = (int *) calloc(utils::N, sizeof(int));

    //time counter
    timeval start_wall_time_t, end_wall_time_t;
    float ms_wall;

    //start timer
    gettimeofday(&start_wall_time_t, nullptr);

    dijkstra(utils::N, utils::num_threads, utils::mat, all_dist, all_pred);

    //end timer
    gettimeofday(&end_wall_time_t, nullptr);
    ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
               + end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

    std::cerr << "Time(ms): " << ms_wall << endl;

    utils::print_result(all_dist, all_pred);

    free(utils::mat);
    free(all_dist);
    free(all_pred);

    return 0;
}
