#include <mpi.h>
#include <random>
//#include <chrono>
#include <iostream>
#include <limits>

typedef double T;
#define MINLIMIT -std::numeric_limits<T>::max()
#define MIN_EPS std::numeric_limits<T>::min()

T function(T x) {
    //T Val = x * x + std::cos(x);
    T Val = std::exp(x);
    return Val;
}

T integral(T(*func)(T), const T a, const T b, const int n) {
    T h = (b - a) / n;
    T Val = T(0);

    for (int i = 0; i < n; ++i) {
        T x = a + i * h;
        Val += func(x) * h;
    }
    return Val;
}

T integral_mpi(T(*func)(T), const T x1, const T x2, const int n) {
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status status;
    T sum = T(0);
    int cores = world;// -1;

    /*if (rank != root)*/ {
        int steps = (n / cores);
        int add = n % cores;
        int c = rank;// -1;
        int b = c * steps + ((c < add) ? c : add);
        int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
        if (c == cores - 1) e = n;
        int m = e - b; // m - number of rows in submatrix

        T h = (x2 - x1) / n;

        T z1 = x1 + b * h;
        T z2 = x1 + e * h;

        T localI = integral(function, z1, z2, m);
        if (rank != root)
            MPI_Send(&localI, 1, MPI_DOUBLE, root, 0, MPI_COMM_WORLD);
        else
            sum = localI;
    }

    if (rank == root) {
        T val = T(0);
        for (int c = 0; c < cores - 1; ++c) {
            MPI_Recv(&val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            sum += val;
        }
    }

    delete[] requests;
    return sum;
}

T integral_mpi2(T(*func)(T), const T x1, const T x2, const int n) { // use MPI_Scatter and MPI_Gather
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status status;
    T sum = T(0);
    T* localSums = nullptr; 
    if (rank == root) localSums = new T[world];

    int steps = (n / world);
    int add = n % world;
    int b = rank * steps + ((rank < add) ? rank : add);
    int e = (rank + 1) * steps + ((rank + 1 < add) ? rank + 1 : add);
    if (rank == world - 1) e = n;
    int m = e - b; 

    T h = (x2 - x1) / n;
    T z1 = x1 + b * h;
    T z2 = x1 + e * h;

    T localI = integral(function, z1, z2, m);

    MPI_Gather(&localI, 1, MPI_DOUBLE, localSums, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank == root) {
        T val = T(0);
        for (int c = 0; c < world; ++c) {
            T val = localSums[c];
            sum += val;
        }
    }

    delete[] requests;
    if (rank == root) delete[] localSums;
    return sum;
}


#define eps T(0.0001)
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

    T a = T(0);
    T b = T(10);

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) silent = true;
    if (!silent && rank == root) {
        std::cout << "N: " << N << std::endl;
        std::cout << "number of communicators: " << world << std::endl;
    }

    double start = MPI_Wtime();
    double end = start;
    double diff = end - start;
    double diff1 = diff;
    double diff2 = diff;
    T DP0, DP1, DP2;// , DP1;

    if (rank == root) {
        start = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i)
            DP0 = integral(function, a + MIN_EPS, b, N);
        end = MPI_Wtime();
        diff = (end - start) / REPEATS;
    }

    if (cores <= 0) {
        std::clog << "time(us): \t\t" << diff << std::endl;
        std::cout << "Please, run MPI with N > 1" << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {
        start = MPI_Wtime();
        DP1 = integral_mpi(function, a + MIN_EPS, b, N);
        end = MPI_Wtime();

        diff1 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff1 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {
        start = MPI_Wtime();
        DP2 = integral_mpi2(function, a + MIN_EPS, b, N);
        end = MPI_Wtime();

        diff2 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff2 /= REPEATS;

    if (!silent && rank == root) {
        std::clog << "time(us): \t" << diff << std::endl;
        std::clog << "time(us) mpi send + receive: \t" << diff1 << std::endl;
        std::clog << "time(us) mpi gather: \t" << diff2 << std::endl;

        if (std::abs(DP0 - DP1) <= eps && std::abs(DP0 - DP2) <= eps)
            std::cout << "integral found OK: " << DP1 << std::endl;
        else
            std::cout << "Error: " << DP1 << "," << DP2 << " ; Should be: " << DP0 << std::endl;
    }
    else if (rank == root) {
        if (std::abs(DP0 - DP1) <= eps && std::abs(DP0 - DP2) <= eps) {
            std::cout << N << " " << world << " ";
            std::cout << diff << " " << diff1 << " " << diff2 << std::endl;
        }
        else
            std::cout << 0 << " " << 0 << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
