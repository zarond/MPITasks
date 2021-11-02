#include <mpi.h>
#include <random>
//#include <chrono>
#include <iostream>

typedef double T;
#define eps T(0.00001)

struct planetInfo{
    T mass;
    T pos[3];
    T vel[3];
};

MPI_Datatype dt_planet;

void generate_random_planetInfo(planetInfo* Data, int n, int rank = -1) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis1(T(0), T(100));
    std::uniform_real_distribution<T> dis2(T(-1000), T(1000));
    std::uniform_real_distribution<T> dis3(T(-10), T(10));

    for (int i = 0; i < n; ++i) {
        Data[i].mass = (rank == -1) ? dis1(gen) : T(rank + 1);// ;
        Data[i].pos[0] = dis2(gen); Data[i].pos[1] = dis2(gen); Data[i].pos[2] = dis2(gen);
        Data[i].vel[0] = dis3(gen); Data[i].vel[1] = dis3(gen); Data[i].vel[2] = dis3(gen);
    }
}

void set_zeros(planetInfo* Data, int n) {
    for (int i = 0; i < n; ++i) {
        Data[i].mass = T(0);
        Data[i].pos[0] = T(0); Data[i].pos[1] = T(0); Data[i].pos[2] = T(0);
        Data[i].vel[0] = T(0); Data[i].vel[1] = T(0); Data[i].vel[2] = T(0);
    }
}

bool check_data(planetInfo* Data, int n, int world) {
    bool res = true;
    int v = 1;
    for (int i = 0; i < n; ++i) {
        res = res && (std::abs(Data[i].mass)>eps);
    }
    return res;
}

struct int2 {
    int b;
    int e;
};

inline int2 indices(int N, int workers, int current) {
    int2 indices;
    int steps = (N / workers);
    int add = N % workers;
    indices.b = current * steps + ((current < add) ? current : add);
    indices.e = (current + 1) * steps + ((current + 1 < add) ? current + 1 : add);
    if (current == workers - 1) indices.e = N;
    if (current == workers) { indices.b = N; indices.e = N; } 
    return indices;
}

void send_data_method1(planetInfo* all_planets, int N) {
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status* statuses = new MPI_Status[world];
    MPI_Status status;
    int workers = world;

    int2 thisPlanets = indices(N, workers, rank);
    int m = thisPlanets.e - thisPlanets.b;

    planetInfo* planet = &all_planets[thisPlanets.b];

    for (int i = 1; i < workers; ++i) {
        int sendTo = (rank + i) % world;
        int receiveFrom = (world + rank - i) % world;

        int2 otherPlanets = indices(N, workers, receiveFrom);
        int m_other = otherPlanets.e - otherPlanets.b;

        MPI_Isend(planet, m, dt_planet, sendTo, rank, MPI_COMM_WORLD, &requests[i-1]);
        MPI_Recv(&all_planets[otherPlanets.b], m_other, dt_planet, receiveFrom, receiveFrom, MPI_COMM_WORLD, &status);
    }
    MPI_Waitall(workers - 1, requests, statuses);
    delete[] requests;
    delete[] statuses;
}

void send_data_method2(planetInfo* all_planets, int N) { 
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    //MPI_Request* requests = new MPI_Request[world];
    MPI_Status status;

    int2 thisPlanets = indices(N, world, rank);
    int m = thisPlanets.e - thisPlanets.b;

    planetInfo* planet = &all_planets[thisPlanets.b];

    if (world % 2 == 1) { // odd
        for (int i = 0; i < world; ++i) {
            int sendTo = (world + i - rank) % world;
            int receiveFrom = sendTo;

            int2 otherPlanets = indices(N,world, receiveFrom);
            int m_other = otherPlanets.e - otherPlanets.b;

            if (rank - sendTo != 0)
                MPI_Sendrecv(planet, m, dt_planet, sendTo, rank, &all_planets[otherPlanets.b], m_other, dt_planet, receiveFrom, receiveFrom, MPI_COMM_WORLD, &status);
        }
    }
    else { // even
        for (int i = 0; i < world - 1; ++i) {
            int idle = (i * world / 2) % (world - 1);
            int sendTo;// = (world + i - rank) % world;

            if (rank == world - 1)
                sendTo = idle;
            else {
                if (rank == idle)
                    sendTo = world - 1;
                else
                    sendTo = (world - 1 + i - rank) % (world - 1);
            }
            int receiveFrom = sendTo;

            int2 otherPlanets = indices(N, world, receiveFrom);
            int m_other = otherPlanets.e - otherPlanets.b;

            if (rank - sendTo != 0)
                MPI_Sendrecv(planet, m, dt_planet, sendTo, rank, &all_planets[otherPlanets.b], m_other, dt_planet, receiveFrom, receiveFrom, MPI_COMM_WORLD, &status);
        }
    }

}

void send_data_method3(planetInfo* all_planets, int N) {
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status* statuses = new MPI_Status[world];
    MPI_Status status;

    for (int i = 1; i < world; i *= 2) {
        int type = (rank / i) % 2;
        int sendTo = (type == 0) ? rank + i : rank - i;
        int receiveFrom = sendTo;

        int2 HaveInfoAbout = { rank - (rank % i), std::min(rank - (rank % i) + i, world) };
        int2 receiveAbout = { receiveFrom - (receiveFrom % i), std::min(receiveFrom - (receiveFrom % i) + i, world) };
        
        int count = 0;
        if (rank == world - 1) {
            for (int j = world - i; j < world - 1; ++j) {
                if ((j / i) % 2 == 0) {
                    int2 sendInfoAbout1 = indices(N, world, HaveInfoAbout.b);
                    int2 sendInfoAbout2 = indices(N, world, HaveInfoAbout.e);
                    int m_1 = sendInfoAbout2.b - sendInfoAbout1.b;
                    MPI_Isend(&all_planets[sendInfoAbout1.b], m_1, dt_planet, j, rank, MPI_COMM_WORLD, &requests[count]);
                    ++count;
                }
            }
        }

        if (sendTo < world) {
            int2 sendInfoAbout1 = indices(N, world, HaveInfoAbout.b);
            int2 sendInfoAbout2 = indices(N, world, HaveInfoAbout.e);
            int m_1 = sendInfoAbout2.b - sendInfoAbout1.b;

            int2 receiveAboutPlanets1 = indices(N, world, receiveAbout.b);
            int2 receiveAboutPlanets2 = indices(N, world, receiveAbout.e);
            int m_2 = receiveAboutPlanets2.b - receiveAboutPlanets1.b;

            MPI_Sendrecv(&all_planets[sendInfoAbout1.b], m_1, dt_planet, sendTo, rank, &all_planets[receiveAboutPlanets1.b], m_2, dt_planet, receiveFrom, receiveFrom, MPI_COMM_WORLD, &status);
        }
        else {
            receiveFrom = world - 1;
            receiveAbout = { receiveFrom - (receiveFrom % i), std::min(receiveFrom - (receiveFrom % i) + i, world) };
            int2 receiveAboutPlanets1 = indices(N, world, receiveAbout.b);
            int2 receiveAboutPlanets2 = indices(N, world, receiveAbout.e);
            int m_2 = receiveAboutPlanets2.b - receiveAboutPlanets1.b;
            if (receiveFrom != rank)
                MPI_Recv(&all_planets[receiveAboutPlanets1.b], m_2, dt_planet, receiveFrom, receiveFrom, MPI_COMM_WORLD, &status);
        }
        MPI_Waitall(count, requests, statuses);
    }

    delete[] requests;
    delete[] statuses;
}

void send_data_method_allgather(planetInfo* all_planets, int N) {
    int rank;
    int world;
    const int root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Request* requests = new MPI_Request[world];
    MPI_Status status;

    int2 thisPlanets = indices(N, world, rank);
    int m = thisPlanets.e - thisPlanets.b;

    planetInfo* planet = &all_planets[thisPlanets.b];
    
    int* sendcounts = nullptr;
    int* displs = nullptr;

    sendcounts = new int[world];
    displs = new int[world];
    int steps = (N / world);
    int add = N % world;
    for (int c = 0; c < world; ++c) {
        int b = c * steps + ((c < add) ? c : add);
        int e = (c + 1) * steps + ((c + 1 < add) ? c + 1 : add);
        if (c == world - 1) e = N;
        sendcounts[c] = (e - b);
        displs[c] = b;
    }

    MPI_Allgatherv(MPI_IN_PLACE, m, dt_planet, all_planets, sendcounts, displs, dt_planet, MPI_COMM_WORLD);
    

    delete[] sendcounts; delete[] displs;
    delete[] requests;
}

#define REPEATS 10

int main(int argc, char** argv) {
    int rank;
    int world;
    const int root = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Status status;

    // make datatype
    //MPI_Datatype dt_planet;
    MPI_Type_contiguous(7, MPI_DOUBLE, &dt_planet);
    MPI_Type_commit(&dt_planet);

#define MPI_WTIME_IS_GLOBAL = true;

    int N = 1000;
    int workers = world;// -1;
    bool silent = false;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) silent = true;
    if (!silent && rank == root) {
        std::cout << "planets N: " << N << std::endl;
        std::cout << "number of workers: " << workers << std::endl;
    }

    double start = MPI_Wtime();
    double end = start;
    double diff = end - start;
    double diff1 = diff;
    double diff2 = diff;
    double diff3 = diff;
    double diff4 = diff;
    double diff5 = diff;
    T DP1 = 0, DP2 = 0;// , DP1;
    bool check = true;

    planetInfo* all_planets = new planetInfo[N];
    int2 thisPlanets = indices(N, workers, rank);
    int m = thisPlanets.e - thisPlanets.b;
    set_zeros(all_planets, N);
    generate_random_planetInfo(&all_planets[thisPlanets.b], m, rank);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        send_data_method1(all_planets,N);
        end = MPI_Wtime();

        diff1 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff1 /= REPEATS;
    check = check && check_data(all_planets, N, world);

    set_zeros(all_planets, N);
    generate_random_planetInfo(&all_planets[thisPlanets.b], m, rank);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();
        send_data_method2(all_planets, N);
        end = MPI_Wtime();

        diff2 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff2 /= REPEATS;
    check = check && check_data(all_planets, N, world);

    set_zeros(all_planets, N);
    generate_random_planetInfo(&all_planets[thisPlanets.b], m, rank);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();;
        send_data_method3(all_planets, N);
        end = MPI_Wtime();;

        diff3 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff3 /= REPEATS;
    check = check && check_data(all_planets, N, world);
    /*
    set_zeros(all_planets, N);
    generate_random_planetInfo(&all_planets[thisPlanets.b], m, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int y = 0; y < REPEATS; ++y) {

        start = MPI_Wtime();;
        send_data_method4(all_planets, N);
        end = MPI_Wtime();;

        diff4 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff4 /= REPEATS;
    check = check && check_data(all_planets, N, world);

    
    if (N < 100) {
        std::cout << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << all_planets[i].mass << " ";
        }
        std::cout << std::endl;
    }
    

    set_zeros(all_planets, N);
    generate_random_planetInfo(&all_planets[thisPlanets.b], m, rank);
*/
    MPI_Barrier(MPI_COMM_WORLD);

    for (int y = 0; y < REPEATS; ++y) {
        start = MPI_Wtime();
        send_data_method_allgather(all_planets, N);
        end = MPI_Wtime();

        diff5 += (end - start);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    diff5 /= REPEATS;

    MPI_Barrier(MPI_COMM_WORLD);
    check = check && check_data(all_planets, N, world);

    if (!silent) {
        if (rank == root) {
            std::clog << "time(us) mpi first method: \t" << diff1 << std::endl;
            std::clog << "time(us) mpi second method: \t" << diff2 << std::endl;
            std::clog << "time(us) mpi third method: \t" << diff3 << std::endl;
            //std::clog << "time(us) mpi fourth method: \t" << diff4 << std::endl;
            std::clog << "time(us) mpi allgather: \t" << diff5 << std::endl;
        }

        if (check){
            if (rank == root) std::cout << "all-to-all success " << std::endl;
        }
        else{
            std::cout << "Error copy data: " << std::endl;
            std::cout << rank << " : ";
            if (N < 1000) for (int i = 0; i < N; ++i) std::cout << all_planets[i].mass << " ";
            std::cout << std::endl;
        }
    }
    else {
        if (check) {
            if (rank == root){
                std::cout << N << " " << workers << " ";
                std::cout << diff1 << " " << diff2 << " " << diff3 << " " /* << diff4 << " "*/ << diff5 << std::endl;
            }
        }
        else
            std::cout << 0 << " " << 0 << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&dt_planet);
    MPI_Finalize();
}
