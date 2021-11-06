#include <mpi.h>
#include <random>
#include <iostream>

typedef double T;

MPI_Datatype dt_row;
MPI_Datatype dt_column;
MPI_Datatype dt_sub_rows;
MPI_Datatype dt_sub_columns;
MPI_Datatype dt_sub_mat;

MPI_Comm decart;
MPI_Comm RowComm, ColComm;

bool MatMult(T* A, T* B, T* C, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm) {
    if (Am != Bn) return false;
    int n = An, m = Am;
    T* column = new T[m];
    for (int j = 0; j < Bm; ++j) {
        for (int c = 0; c < m; ++c) column[c] = B[c * Bm + j];
        for (int i = 0; i < n; ++i) {
            T S = 0;
            for (int k = 0; k < m; ++k) {
                S += A[i * m + k] * column[k];
            }
            C[i * Bm + j] = S;
        }
    }
    delete[] column;
    return true;
}

bool MatMultMPI(T* A, T* B, T* C, unsigned int An, unsigned int Am, unsigned int Bn, unsigned int Bm, int M, int Dim){
    if (Am != Bn) return false;
    int rank;
    int world;
    int world_decart;
    const int root = 0;
    MPI_Comm_rank(decart, &rank);
    MPI_Comm_size(decart, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status* statuses = new MPI_Status[world];
    MPI_Status status;
    int coords[2];
    MPI_Cart_coords(decart,rank, 2, coords);

    int steps = (An / Dim);
    int add = An % Dim;

    int n = An, m = Am;

    if (rank == root){
        for (int i = 0; i < Dim;++i) {
            for (int j = 0; j < Dim; ++j) {
                if (i == 0 && j == 0) continue;
                int crds[2] = {j,i};

                int b_x = crds[0] * steps + ((crds[0] < add) ? crds[0] : add);
                int e_x = (crds[0] + 1) * steps + ((crds[0] + 1 < add) ? crds[0] + 1 : add);
                if (crds[0] == Dim - 1) e_x = An;
                int b_y = crds[1] * steps + ((crds[1] < add) ? crds[1] : add);
                int e_y = (crds[1] + 1) * steps + ((crds[1] + 1 < add) ? crds[1] + 1 : add);
                if (crds[1] == Dim - 1) e_y = An;

                int SendTo;
                MPI_Cart_rank(decart, crds, &SendTo);

                MPI_Isend(A + b_y * n, 1, dt_sub_rows, SendTo, SendTo, decart, & requests[SendTo - 1]);
            }
        }
        MPI_Waitall(world - 1, requests, statuses);     
        for (int i = 0; i < Dim; ++i) {
            for (int j = 0; j < Dim; ++j) {
                if (i == 0 && j == 0) continue;
                int crds[2] = { i,j };

                int b_x = crds[0] * steps + ((crds[0] < add) ? crds[0] : add);
                int e_x = (crds[0] + 1) * steps + ((crds[0] + 1 < add) ? crds[0] + 1 : add);
                if (crds[0] == Dim - 1) e_x = An;
                int b_y = crds[1] * steps + ((crds[1] < add) ? crds[1] : add);
                int e_y = (crds[1] + 1) * steps + ((crds[1] + 1 < add) ? crds[1] + 1 : add);
                if (crds[1] == Dim - 1) e_y = An;

                int SendTo;
                MPI_Cart_rank(decart, crds, &SendTo);

                MPI_Isend(B + b_x, 1, dt_sub_columns, SendTo, SendTo, decart, & requests[SendTo - 1]);
            }
        }
        MPI_Waitall(world - 1, requests, statuses);
    }

    int b_x = coords[0] * steps + ((coords[0] < add) ? coords[0] : add);
    int e_x = (coords[0] + 1) * steps + ((coords[0] + 1 < add) ? coords[0] + 1 : add);
    if (coords[0] == Dim - 1) e_x = An;
    int b_y = coords[1] * steps + ((coords[1] < add) ? coords[1] : add);
    int e_y = (coords[1] + 1) * steps + ((coords[1] + 1 < add) ? coords[1] + 1 : add);
    if (coords[1] == Dim - 1) e_y = An;


    if (rank != root) {
        MPI_Recv(A + b_y * n, 1, dt_sub_rows, root, rank, decart, &status);
        MPI_Recv(B + b_x, 1, dt_sub_columns, root, rank, decart, &status);
    }

    T* column = new T[m];
    for (int j = b_x; j < e_x; ++j) {
        for (int c = 0; c < m; ++c) column[c] = B[c * Bm + j];
        for (int i = b_y; i < e_y; ++i) {
            T S = 0;
            for (int k = 0; k < m; ++k) {
                S += A[i * m + k] * column[k];
            }
            C[i * Bm + j] = S;
        }
    }
    delete[] column;

    if (rank == root)
        for (int k = 0; k < Dim * Dim - 1;++k) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, decart, &status);
            int RecvFrom = status.MPI_SOURCE;
            int crds[2];
            MPI_Cart_coords(decart, RecvFrom, 2, crds);

            b_x = crds[0] * steps + ((crds[0] < add) ? crds[0] : add);
            e_x = (crds[0] + 1) * steps + ((crds[0] + 1 < add) ? crds[0] + 1 : add);
            if (crds[0] == Dim - 1) e_x = An;
            b_y = crds[1] * steps + ((crds[1] < add) ? crds[1] : add);
            e_y = (crds[1] + 1) * steps + ((crds[1] + 1 < add) ? crds[1] + 1 : add);
            if (crds[1] == Dim - 1) e_y = An;

            MPI_Recv(C + b_y * n + b_x, 1, dt_sub_mat, RecvFrom, MPI_ANY_TAG, decart, &status);
        }
    else {
        MPI_Send(C + b_y * n + b_x, 1, dt_sub_mat, root, rank, decart);
    }

    return true;
}

void generate_random(T* Data, unsigned int n) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(T(-10), T(10));

    for (unsigned int i = 0; i < n; ++i) {
        Data[i] = dis(gen);
    }
}

#define eps T(0.00001)
bool compare(T* A, T* B, unsigned int n) {
    bool res = true;
    for (int i = 0; i < n; ++i) {
        res = res && (std::abs((A[i] / B[i]) - T(1)) < eps || std::abs(A[i] - B[i]) < eps);
    }
    if (n < 100 && res == false) for (int i = 0; i < n; ++i) printf("%f, %f; ", A[i], B[i]);
    return res;
}

#define REPEATS 1

int main(int argc, char** argv) {
    int rank;
    int world;
    const int root = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Status status;

#define MPI_WTIME_IS_GLOBAL = true;

    int N = 128;
    bool silent = false;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) silent = true;

    if (!silent && false) {
        int  namelen;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(processor_name, &namelen);
        fprintf(stderr, "Process %d on %s\n", rank, processor_name);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!silent && rank == root) {
        std::cout << "matrix N: " << N << std::endl;
        std::cout << "number of communicators: " << world << std::endl;
    }
    
    int dim_blocks = 0;
    for (int i = 1; i * i <= world; ++i) {
        dim_blocks = i;
    }
    int M = (N + dim_blocks - 1) / dim_blocks;

    if (!silent && rank == root) {
        std::cout << "sub M: " << M << std::endl;
        std::cout << "Dim: " << dim_blocks << std::endl;
    }

    if (M * dim_blocks != N) {
        if (rank == root)
            std::cout << "Error, please choose N and number of processes such that it divides matrix to equal blocks. " << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[2] = { dim_blocks,dim_blocks };
    int dims_p[2] = { 1,1 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, dims_p, 0, &decart);

    MPI_Type_contiguous(N, MPI_DOUBLE, &dt_row); // row on Matrix N x N
    MPI_Type_vector(N, 1, N, MPI_DOUBLE, &dt_column); // column on Matrix N x N
    //MPI_Type_vector(N, 1, N+1, MPI_DOUBLE, &dt_diag); // diag on Matrix N x N
    //MPI_Type_contiguous(4,MPI_INT,&dt_job);
    int all[2] = { N, N };
    int starts[2] = { 0, 0};
    int sizes_rows[2] = {M,N}; int sizes_colimns[2] = { N, M }; int sizes_smat[2] = { M,M };
    MPI_Type_create_subarray(2, all, sizes_rows, starts, MPI_ORDER_C, MPI_DOUBLE, &dt_sub_rows);
    MPI_Type_create_subarray(2, all, sizes_colimns, starts, MPI_ORDER_C, MPI_DOUBLE, &dt_sub_columns);
    MPI_Type_create_subarray(2, all, sizes_smat, starts, MPI_ORDER_C, MPI_DOUBLE, &dt_sub_mat);

    MPI_Type_commit(&dt_row);
    MPI_Type_commit(&dt_column);
    //MPI_Type_commit(&dt_diag);
    //MPI_Type_commit(&dt_job);
    MPI_Type_commit(&dt_sub_rows);
    MPI_Type_commit(&dt_sub_columns);
    MPI_Type_commit(&dt_sub_mat);

    T* MatA = nullptr;
    T* MatB = nullptr;
    T* MatC0 = nullptr;
    T* MatC = nullptr;
    double start = MPI_Wtime();
    double end = start;
    double diff = end - start;
    double diff1 = diff;
    double diff2 = diff;
    T DP0, DP1, DP2;// , DP1;

    MatA = new T[N * N];
    MatB = new T[N * N];
    MatC = new T[N * N];

    if (rank == root) {
        MatC0 = new T[N * N];
        generate_random(MatA, N * N);
        generate_random(MatB, N * N);
    }

    if (rank == root) {
        start = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i)
            MatMult(MatA, MatB, MatC0, N, N, N, N);
        end = MPI_Wtime();
        diff = (end - start) / REPEATS;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        DP1 = MatMultMPI(MatA, MatB, MatC, N, N, N, N, M, dim_blocks);
        end = MPI_Wtime();

        diff1 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff1 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);

    if (!silent && rank == root) {
        std::clog << "time(us): \t" << diff << std::endl;
        std::clog << "time(us) mpi: \t" << diff1 << std::endl;

        if (compare(MatC, MatC0, N*N))
            std::cout << "matrixmult found OK: " << std::endl;
        else
            std::cout << "Error " << std::endl;
    }
    else if (rank == root) {
        if (compare(MatC, MatC0, N * N)) {
            std::cout << N << " " << world << " ";
            std::cout << diff << " " << diff1 << std::endl;
        }
        else
            std::cout << 0 << " " << 0 << std::endl;
    }
    if (MatA != nullptr)
        delete[] MatA;
    if (MatB != nullptr)
        delete[] MatB;
    if (MatC != nullptr)
        delete[] MatC;
    if (MatC0 != nullptr)
        delete[] MatC0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&dt_row);
    MPI_Type_free(&dt_column);
    //MPI_Type_free(&dt_diag);
    //MPI_Type_free(&dt_job);
    MPI_Type_free(&dt_sub_rows);
    MPI_Type_free(&dt_sub_columns);
    MPI_Type_free(&dt_sub_mat);
    MPI_Finalize();
}
