#include <mpi.h>
#include <random>
//#include <chrono>
#include <limits>
#include <iostream>

typedef double T;
#define MINLIMIT -std::numeric_limits<T>::max()

void generate_random(T* Data, unsigned int n) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(T(-10), T(10));

    for (unsigned int i = 0; i < n; ++i) {
        Data[i] = dis(gen);
    }
}

T extremum(const T* Mat, const int n, bool mode = false) { // if mode==false then min, else max
    if (n == 0) return ((mode)? MINLIMIT:-MINLIMIT);
    T extr = Mat[0];
    for (int i = 1; i < n; ++i) {
        T v = Mat[i];
        if ((extr > v) xor mode)
            extr = v;
    }
    return extr;
}

T extremum_mpi(const T* Mat, const int n, bool mode = false) {//scatter+gather
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Status status;

    T Aextremum = MINLIMIT;
    int* sendcounts = nullptr;
    int* displs = nullptr;

    if (rank == root) {
        sendcounts = new int[world];
        displs = new int[world];
        int steps = (n / world);
        int add = n % world;
        for (int c = 0; c < world; ++c) {
            int b = c * steps + ((c < add) ? c : add);
            int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
            if (c == world - 1) e = n;
            sendcounts[c] = (e - b);
            displs[c] = b;
        }
    }

    int steps = (n / world);
    int add = n % world;
    int b = rank * steps + ((rank < add) ? rank : add);
    int e = (rank + 1) * steps + ((rank + 1 < add) ? rank + 1 : add);
    if (rank == world - 1) e = n;

    T* localArray = new T[(e - b)];
    T* localExtremums = nullptr;
    if (rank == root)  localExtremums = new T[world];

    MPI_Scatterv(Mat, sendcounts, displs, MPI_DOUBLE, localArray, (e - b), MPI_DOUBLE, root, MPI_COMM_WORLD);
    T localExtr = extremum(localArray, (e - b), mode);
    MPI_Gather(&localExtr, 1, MPI_DOUBLE, localExtremums, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank == root) {
        Aextremum = extremum(localExtremums, std::min(world,n), mode);
    }
    delete[] localArray;
    if (rank == root) { delete[] localExtremums; delete[] sendcounts; delete[] displs; }
    return Aextremum;
}

T extremum_mpi2(const T* Mat, const int n, bool mode = false) { //Isend+recv+gather
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status* statuses = new MPI_Status[world];
    MPI_Status status;

    T Aextremum = MINLIMIT;

    if (rank == root) {
        int steps = (n / world);
        int add = n % world;
        for (int c = 1; c < world; ++c) {
            int b = c * steps + ((c < add) ? c : add);
            int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
            if (c == world - 1) e = n;
            MPI_Isend(Mat + b, e - b, MPI_DOUBLE, c, c, MPI_COMM_WORLD, &requests[c - 1]); // or non blocking send
        }
        int c = 0;
        int b = c * steps + ((c < add) ? c : add);
        int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
        if (c == world - 1) e = n;
        T localExtr = extremum(Mat, e - b, mode);
        Aextremum = localExtr;
        MPI_Waitall(world - 1, requests, statuses);
    }

    if (rank != root) {
        int steps = (n / world);
        int add = n % world;
        int b = rank * steps + ((rank < add) ? rank : add);
        int e = (rank + 1) * steps + ((rank + 1 < add) ? rank + 1 : add);
        if (rank == world - 1) e = n;

        T* localArray = new T[(e - b)];
        MPI_Recv(localArray, e - b, MPI_DOUBLE, root, rank, MPI_COMM_WORLD, &status);
        T localExtr = extremum(localArray, (e - b), mode);
        MPI_Send(&localExtr, 1, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
        delete[] localArray;
    }
    T* localExtremums = nullptr;
    if (rank == root) {
        localExtremums = new T[world];
        localExtremums[0] = Aextremum;
        for (int c = 1; c < world; ++c) { MPI_Recv(&localExtremums[c], 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); }
        Aextremum = extremum(localExtremums, std::min(world, n), mode);
    }
    if (rank == root) { delete[] localExtremums;}
    delete[] requests;
    delete[] statuses;
    return Aextremum;
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
    //int cores = world;// -1;
    bool silent = false;
    bool mode = true;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) silent = true;
    if (!silent && rank == root) {
        std::cout << "vector N: " << N << std::endl;
        std::cout << "number of communicators: " << world << std::endl;
    }

    T* Mat = nullptr;
    //Mat = new T[N * N];
    //generate_random(Mat, N * N);
    double start = MPI_Wtime();
    double end = start;
    double diff = end - start;
    double diff1 = diff;
    double diff2 = diff;
    T DP0, DP1, DP2;// , DP1;

    if (rank == root) {
        Mat = new T[N];
        generate_random(Mat, N);
    }

    if (rank == root) {
        start = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i)
            DP0 = extremum(Mat, N, mode);
        end = MPI_Wtime();
        diff = (end - start) / REPEATS;
    }


    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP1 = extremum_mpi(Mat, N, mode);
        end = MPI_Wtime();

        diff1 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff1 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP2 = extremum_mpi2(Mat, N, mode);
        end = MPI_Wtime();

        diff2 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff2 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    if (!silent && rank == root) {
        std::clog << "time(us): \t" << diff << std::endl;
        std::clog << "time(us) mpi scatter + gather: \t" << diff1 << std::endl;
        std::clog << "time(us) mpi send + receive: \t" << diff2 << std::endl;

        if (std::abs(DP0 - DP1) <= eps && std::abs(DP0 - DP2) <= eps)
            std::cout << "extremum found OK: " << DP1 << std::endl;
        else
            std::cout << "Error: " << DP1 << "; Should be: " << DP0 << std::endl;
    }
    else if (rank == root) {
        if (std::abs(DP0 - DP1) <= eps && std::abs(DP0 - DP2) <= eps) {
            std::cout << N << " " << world << " ";
            std::cout << diff << " " << diff1 << " " << diff2 << std::endl;
        }
        else
            std::cout << 0 << " " << 0 << std::endl;
    }
    if (Mat != nullptr)
        delete[] Mat;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
