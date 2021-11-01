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

T maxmin(const T* Mat, const int n, const int m) {
    T max = MINLIMIT;
    for (int i = 0; i < n; ++i) {
        T min = Mat[i * m];
        for (int j = 1; j < m; ++j) {
            T v = Mat[i * m + j];
            if (min > v)
                min = v;
        }
        if (max < min) max = min;
    }
    return max;
}

T maxmin_mpi(const T* Mat, const int n) {
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status* statuses = new MPI_Status[world];
    MPI_Status status;
    T Amax = MINLIMIT;
    int cores = world;// -1;

    if (rank == root) {
        int steps = (n / cores);
        int add = n % cores;
        for (int c = 1/*0*/; c < cores; ++c) {
            int b = c * steps + ((c<add)? c:add);
            int e = (c + 1) * steps + ((c + 1  < add) ? c + 1 : add);
            if (c == cores - 1) e = n;
            int indx = b * n;
            MPI_Isend(Mat + indx, n*(e - b), MPI_DOUBLE, c, c, MPI_COMM_WORLD, &requests[c-1]); // or non blocking send
            //MPI_Send(Mat + indx, n * (e - b), MPI_DOUBLE, c + 1, c + 1, MPI_COMM_WORLD);
        }
        
        int c = 0;
        int b = c * steps + ((c < add) ? c : add);
        int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
        if (c == cores - 1) e = n;
        int m = e - b; // m - number of rows in submatrix
        T localMax = maxmin(Mat, m, n);
        Amax = localMax;
        MPI_Waitall(cores - 1/*cores*/, requests, statuses);
    }

    if (rank != root) {
        int steps = (n / cores);
        int add = n % cores;
        int c = rank;// -1;
        int b = c * steps + ((c < add) ? c : add);
        int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
        if (c == cores - 1) e = n;
        int m = e - b; // m - number of rows in submatrix
        
        T* localMat = new T[n * m];
        MPI_Recv(localMat, n * m, MPI_DOUBLE, root, rank, MPI_COMM_WORLD, &status);
        T localMax = maxmin(localMat, m, n);
        MPI_Send(&localMax, 1, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
        delete[] localMat;
    }

    if (rank == root) {
        T val = T(0);
        for (int c = 1; c < cores; ++c) {
            MPI_Recv(&val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (Amax < val) Amax = val;
        }
    }

    delete[] requests;
    delete[] statuses;
    return Amax;
}

T maxmin_mpi2(const T* Mat, const int n) { //use MPI_Scatter and MPI_Gather
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status status;

    T Amax = MINLIMIT;
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
            sendcounts[c] = n * (e - b);
            displs[c] = b * n;
        }
    }

    int steps = (n / world);
    int add = n % world;
    int b = rank * steps + ((rank < add) ? rank : add);
    int e = (rank + 1) * steps + ((rank + 1 < add) ? rank + 1 : add);
    if (rank == world - 1) e = n;

    T* localMat = new T[n * (e - b)];
    T* localMaxes = nullptr;
    if (rank == root)  localMaxes = new T[world];

    MPI_Scatterv(Mat, sendcounts, displs, MPI_DOUBLE, localMat, n * (e - b), MPI_DOUBLE, root, MPI_COMM_WORLD);
    T localMax = maxmin(localMat, (e - b), n);
    MPI_Gather(&localMax, 1, MPI_DOUBLE, localMaxes, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank == root) {
        for (int c = 0; c < world; ++c) {
            T val = localMaxes[c];
            if (Amax < val) Amax = val;
        }
    }
    delete[] localMat;
    if (rank == root) { delete[] localMaxes; delete[] sendcounts; delete[] displs; }
    delete[] requests;
    return Amax;
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
    int cores = world - 1;
    bool silent = false;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) silent = true;
    if (!silent && rank == root) {
        std::cout << "matrix N: " << N << std::endl;
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
        Mat = new T[N * N];
        generate_random(Mat, N * N);
    }

    if (rank == root) {
        start = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i)
            DP0 = maxmin(Mat, N, N);
        end = MPI_Wtime();
        diff = (end - start) / REPEATS;
    }

    if (cores <= 0) {
        std::clog << "time(us): \t" << diff << std::endl;
        std::cout << "Please, run MPI with N > 1" << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP1 = maxmin_mpi(Mat, N);
        end = MPI_Wtime();

        diff1 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff1 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP2 = maxmin_mpi2(Mat, N);
        end = MPI_Wtime();

        diff2 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff2 /= REPEATS;

    if (!silent && rank == root) {
        std::clog << "time(us): \t" << diff << std::endl;
        std::clog << "time(us) mpi send + receive: \t" << diff1 << std::endl;
        std::clog << "time(us) mpi scatter + gather: \t" << diff2 << std::endl;

        if (std::abs(DP0 - DP1) <= eps && std::abs(DP0 - DP2) <= eps)
            std::cout << "maxmin found OK: " << DP1 << std::endl;
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
    if (Mat!=nullptr)
        delete[] Mat;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
