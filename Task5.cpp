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

T dot_product(const T* Vec1, const T* Vec2, const int n) {
    T Val = T(0);

    for (int i = 0; i < n; ++i) {
        Val += Vec1[i] * Vec2[i];
    }
    return Val;
}

T dot_product_mpi(const T* Vec1, const T* Vec2, const int n) { //scatter + gather
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Status status;

    T AllSum = T(0);
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

    T* localVec1 = new T[(e - b)];
    T* localVec2 = new T[(e - b)];
    T* localSums = nullptr;
    if (rank == root)  localSums = new T[world];

    MPI_Scatterv(Vec1, sendcounts, displs, MPI_DOUBLE, localVec1, (e - b), MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Scatterv(Vec2, sendcounts, displs, MPI_DOUBLE, localVec2, (e - b), MPI_DOUBLE, root, MPI_COMM_WORLD);
    T localSum = dot_product(localVec1, localVec2, (e - b));
    MPI_Gather(&localSum, 1, MPI_DOUBLE, localSums, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank == root) {
        for (int i = 0; i < world; ++i) {
            AllSum += localSums[i];
        }
    }
    delete[] localVec1;
    delete[] localVec2;
    if (rank == root) { delete[] localSums; delete[] sendcounts; delete[] displs; }
    return AllSum;
}

T dot_product_mpi2(const T* Vec1, const T* Vec2, const int n) { //Isend + recv
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests1 = new MPI_Request[world];
    MPI_Request* requests2 = new MPI_Request[world];
    MPI_Status* statuses = new MPI_Status[world];
    MPI_Status status;

    T AllSum = T(0);

    if (rank == root) {
        int steps = (n / world);
        int add = n % world;
        for (int c = 1; c < world; ++c) {
            int b = c * steps + ((c < add) ? c : add);
            int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
            if (c == world - 1) e = n;
            MPI_Isend(Vec1 + b, e - b, MPI_DOUBLE, c, c, MPI_COMM_WORLD, &requests1[c - 1]); // or non blocking send
            MPI_Isend(Vec2 + b, e - b, MPI_DOUBLE, c, c, MPI_COMM_WORLD, &requests2[c - 1]); // or non blocking send
        }
        int c = 0;
        int b = c * steps + ((c < add) ? c : add);
        int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
        if (c == world - 1) e = n;
        T localSum = dot_product(Vec1, Vec2, (e - b));
        AllSum = localSum;
        MPI_Waitall(world - 1/*cores*/, requests1, statuses);
        MPI_Waitall(world - 1/*cores*/, requests2, statuses);
    }

    if (rank != root) {
        int steps = (n / world);
        int add = n % world;
        int b = rank * steps + ((rank < add) ? rank : add);
        int e = (rank + 1) * steps + ((rank + 1 < add) ? rank + 1 : add);
        if (rank == world - 1) e = n;

        T* localVec1 = new T[(e - b)];
        T* localVec2 = new T[(e - b)];
        MPI_Recv(localVec1, e - b, MPI_DOUBLE, root, rank, MPI_COMM_WORLD, &status);
        MPI_Recv(localVec2, e - b, MPI_DOUBLE, root, rank, MPI_COMM_WORLD, &status);
        T localSum = dot_product(localVec1, localVec2, (e - b));
        MPI_Send(&localSum, 1, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
        delete[] localVec1;
        delete[] localVec2;
    }

    if (rank == root) {
        T val = T(0);
        for (int i = 1; i < world; ++i) {
            MPI_Recv(&val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            AllSum += val;
        }
    }
    delete[] requests1, requests2;
    delete[] statuses;
    return AllSum;
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
    //int cores = world - 1;
    bool silent = false;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) silent = true;
    if (!silent && rank == root) {
        std::cout << "vector N: " << N << std::endl;
        std::cout << "number of communicators: " << world << std::endl;
    }

    T* Vec1 = nullptr;
    T* Vec2 = nullptr;
    //Mat = new T[N * N];
    //generate_random(Mat, N * N);
    double start = MPI_Wtime();
    double end = start;
    double diff = end - start;
    double diff1 = diff;
    double diff2 = diff;
    T DP0, DP1, DP2;// , DP1;

    if (rank == root) {
        Vec1 = new T[N];
        Vec2 = new T[N];
        generate_random(Vec1, N);
        generate_random(Vec2, N);
    }

    if (rank == root) {
        start = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i)
            DP0 = dot_product(Vec1, Vec2, N);
        end = MPI_Wtime();
        diff = (end - start) / REPEATS;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP1 = dot_product_mpi(Vec1, Vec2, N);
        end = MPI_Wtime();

        diff1 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff1 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP2 = dot_product_mpi2(Vec1, Vec2, N);
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
            std::cout << "dot product found OK: " << DP1 << std::endl;
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
    if (Vec1 != nullptr) delete[] Vec1;
    if (Vec2 != nullptr) delete[] Vec2;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
