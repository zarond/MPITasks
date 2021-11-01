#include <mpi.h>
#include <random>
//#include <chrono>
#include <iostream>

typedef double T;

void generate_random(T* Data, unsigned int n) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(T(-10), T(10));

    for (unsigned int i = 0; i < n; ++i) {
        Data[i] = dis(gen);
    }
}

void generate_random_char(char* Data, unsigned int n) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> dis(-127,127);

    for (unsigned int i = 0; i < n; ++i) {
        Data[i] = dis(gen);
    }
}

void two_process_mpi(char* Vec1, const int n, const int m) {
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Status status;
    if (rank < 2 && world >= 2)
        for (int i = 0; i < m; ++i) {
            if ((rank + i) % 2 == 0) {
                MPI_Send(Vec1, n, MPI_BYTE, (rank + 1) % 2, rank, MPI_COMM_WORLD);
            }
            else {
                Vec1[i%n] += char(1) + char(rank);
                MPI_Recv(Vec1, n, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
        }
    MPI_Barrier;
    return;
}

#define eps T(0.00001)
#define REPEATS 10

int main(int argc, char** argv) {
    int rank;
    int world;
    const int root = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Status status;

#define MPI_WTIME_IS_GLOBAL = true;

    int N = 5000;
    int M = 100;
    //int cores = world - 1;
    bool silent = false;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) M = std::atoi(argv[2]);
    if (argc >= 4) silent = true;
    if (!silent && rank == root) {
        std::cout << "vector bytes N: " << N << std::endl;
        std::cout << "number of messages M: " << M << std::endl;
        std::cout << "number of communicators: " << world << std::endl;
    }

    char* Vec1 = nullptr;
    double start = MPI_Wtime();
    double end = start;
    double diff = end - start;
    double diff1 = diff;
    double diff2 = diff;


    Vec1 = new char[N];
    if (rank == root) {
        generate_random_char(Vec1, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        two_process_mpi(Vec1, N, M);
        end = MPI_Wtime();

        diff += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    if (!silent && rank == root) {
        std::clog << "time(us) mpi: \t" << diff << std::endl;
    }
    else if (rank == root) {
        std::cout << N << " " << M << " " << world << " ";
        std::cout << diff << /*" " << diff1 << " " << diff2 <<*/ std::endl;
    }
    if (Vec1 != nullptr) delete[] Vec1;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
