/*
 * This is a serial version of dijkstra algorithm
 * Compile: g++ -o serial_dijkstra serial_dijkstra.cpp
 * Run: ./serial_dijkstra <input file>, you will find the output in 'output.txt' file
 */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <climits>
#include <algorithm>

using std::string;
using std::cout;
using std::endl;

/*
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and one matrix dimension convert(2D->1D) function
 */
namespace utils {
    int N; //number of vertices
    int *mat; // the adjacency matrix

    /*
     * translate 2-dimension coordinate to 1-dimension
     */
    int convert_dimension_2D_1D(int x, int y) {
        return x * N + y;
    }

    int read_file(string filename) {
        std::ifstream inputf(filename, std::ifstream::in);
        inputf >> N;
        assert(N < (1024 * 1024 * 20)); // input matrix should be smaller than 20MB * 20MB (400MB, we don't have two much memory for multi-processors)
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

int find_global_minimum(int *dist, bool *visit) {
    int min = INT_MAX;
    int u = -1;
    for (int i = 0; i < utils::N; i++) {
        if (!visit[i]) {
            if (dist[i] < min) {
                min = dist[i];
                u = i;
            }
        }
    }
    return u;
}

int dijkstra(int *dist, int *pred, bool *visit) {
    for (int i = 0; i < utils::N; i++) {
        dist[i] = utils::mat[utils::convert_dimension_2D_1D(0, i)];
        pred[i] = 0;
        visit[i] = false;
    }

    visit[0] = true;

    for (int i = 1; i < utils::N; i++) {
        int u = find_global_minimum(dist, visit);
        visit[u] = true;
        for (int v = 1; v < utils::N; v++) {
            if (!visit[v]) {
                int new_dist = dist[u] + utils::mat[utils::convert_dimension_2D_1D(u, v)];
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pred[v] = u;
                }
            }
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    assert(argc > 1 && "Input file was not found!");
    string filename = argv[1];
    assert(utils::read_file(filename) == 0);

    int *dist;
    int *pred;
    bool *visit;

    dist = (int *) malloc(sizeof(int) * utils::N);
    pred = (int *) malloc(sizeof(int) * utils::N);
    visit = (bool *) malloc(utils::N * sizeof(bool));

    int ret = dijkstra(dist, pred, visit);
    assert(ret == 0);
    utils::print_result(dist, pred);

    free(dist);
    free(pred);
    free(visit);
    free(utils::mat);
    return 0;
}