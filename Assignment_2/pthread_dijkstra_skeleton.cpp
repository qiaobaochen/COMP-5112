/* Name: QIAO BAOCHEN
 * ID: 20419075
 * Email: bqiao@connect.ust.hk
 */

/*
 * This is code skeleton for COMP5112-17Spring assignment2
 * Compile: g++ -std=c++11 -lpthread -o pthread_dijkstra pthread_dijkstra.cpp
 * Run: ./pthread_dijkstra -n <number of threads> -i <input file>,
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

#include <pthread.h>

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
        cout << "Usage:\n" << "\tpthread_dijkstra -n <number of threads> -i <input file>" << endl;
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

//Hint: use pthread condition variable or pthread barrier to do the synchronization
//------You may add helper functions and global variables here------

// find_local_minimum can find each thread's minimum and return node index;
int find_local_minimum(int *dist, bool *visit, int my_start, int my_end) {
    int min = INT_MAX;
    int u = -1;
    for (int i = my_start; i < my_end; i++) {
        if (!visit[i]) {
            if (dist[i] < min) {
                min = dist[i];
                u = i;
            }
        }
    }
    return u;
}

// from global min find the minmimum dist, return index of node
int find_global_minimum(int *global_min, int *global_index, int p){
    int min = INT_MAX;
    int u = -1;
    for (int i = 0; i < p; i++){
        if(global_min[i] < min){
            min = global_min[i];
            u = global_index[i];
        }
    }
    return u;
}

// dijkstra
void* paralle_dijkstra(void* t_parameter);

// the struct we need to pass to each thread.
struct thread_parameter
{
    int N;
    int p;
    long thread_number;
    int *mat;
    int *all_dist;
    int *all_pred;
    int *global_min;
    int *global_index;
    bool *all_visit;
    pthread_barrier_t *barrier_loc_min;
    pthread_barrier_t *barrier;
};


void dijkstra(int N, int p, int *mat, int *all_dist, int *all_pred) {
    //------your code starts from here------
    //create pthread

    pthread_t *thread = (pthread_t*) calloc (p, sizeof(pthread_t));
    struct thread_parameter **paramater = (struct thread_parameter**) calloc(p, sizeof(struct thread_parameter*));
    
    pthread_barrier_t barrier_loc_min;                         // barrier of find loc_min
    pthread_barrier_init(&barrier_loc_min, nullptr, p);

    pthread_barrier_t barrier;                                 // barrier of find gol_min
    pthread_barrier_init(&barrier, nullptr, p);
    
    //init parameter we need to share
    bool *all_visit;
    int *global_min;
    int *global_index;
    all_visit = (bool *) malloc(N * sizeof(bool));
    global_min = (int *) malloc(p * sizeof(int));
    global_index = (int *) malloc(p * sizeof(int));


    // init paramater
    for (int i = 0; i < utils::N; ++i) {
        all_dist[i] = mat[utils::convert_dimension_2D_1D (0, i)];
        all_pred[i] = 0;
        all_visit[i] = false;
    }

    int rc; // rc stand for the variable of create return
    long t; // thread number

    for (t = 0; t < p; t++){
        // init the parameter we need to pass to every thread
        paramater[t] = (struct thread_parameter*) malloc (sizeof(struct thread_parameter));
        paramater[t] -> N = N;
        paramater[t] -> p = p;
        paramater[t] -> thread_number = t;
        paramater[t] -> mat = mat;
        paramater[t] -> global_min = global_min;
        paramater[t] -> global_index = global_index;
        paramater[t] -> all_visit = all_visit;
        paramater[t] -> all_dist = all_dist;
        paramater[t] -> all_pred = all_pred;
        paramater[t] -> barrier_loc_min = &barrier_loc_min;               // also can be defined as global-variable
        paramater[t] -> barrier = &barrier;                               // also can be defined as global-variable

        //std::cout << "thread_number" << t << std::endl;
        rc = pthread_create(&thread[t], nullptr, paralle_dijkstra, (void *)paramater[t]);
        if (rc){
          std::cout <<"ERROR; return code from pthread_create() is %d\n"<< std::endl;
          exit(-1);
       }
    }
    
    for (t =0; t < p; t++){
        pthread_join(thread[t], nullptr);
        free(paramater[t]);
    }


    // destory the barriers and memory
    pthread_barrier_destroy (&barrier);
    pthread_barrier_destroy (&barrier_loc_min);
    free(all_visit);
    free(thread);
    free(paramater);
    free(global_min);
    free(global_index);
}

// paralle_dijkstra

void* paralle_dijkstra(void* t_parameter) {
    // init the parameter 
    struct thread_parameter * paramater = (struct thread_parameter *) t_parameter;
    int N = paramater -> N;
    int p = paramater ->p;
    int my_rank = paramater ->thread_number;
    int *mat = paramater ->mat;
    int *global_min = paramater ->global_min;
    int *global_index = paramater ->global_index;
    bool *all_visit = paramater ->all_visit;
    int *all_dist = paramater ->all_dist;
    int *all_pred = paramater ->all_pred;

    pthread_barrier_t *barrier_loc_min = paramater ->barrier_loc_min;           // also can be defined as global-variable
    pthread_barrier_t *barrier = paramater ->barrier;                           // also can be defined as global-variable

    // init loc varaibles
    int loc_n = N / p;
    int my_start = N / p * my_rank;
    int my_end = my_start + loc_n;
    if (my_rank == p - 1){
        loc_n = N - my_start;
        my_end = N; 
    }
    
    all_visit[0] = true;
    for (int i = 1; i < N; i++){
        // find local_minimum
        int x = find_local_minimum(all_dist, all_visit, my_start, my_end);

        if(x == -1){
            global_min[my_rank] = INT_MAX;
            global_index[my_rank] = x;
        }
        else{
            global_min[my_rank] = all_dist[x];
            global_index[my_rank] = x;
        }
        // sychronizations
        pthread_barrier_wait(barrier_loc_min);
        
        // find global_min
        int u = find_global_minimum(global_min, global_index, p);

        // sychronizations
        pthread_barrier_wait(barrier);
        
        all_visit[u] = true;
        for (int v = my_start; v < my_end; v++) {
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
