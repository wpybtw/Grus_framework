#ifndef TIMER_CUH
#define TIMER_CUH

// #include "common.cuh"
#include <stdlib.h>
#include <sys/time.h>

class Timer
{
private:
    //chrono::time_point<chrono::system_clock> A, B;
    timeval StartingTime;

public:
    void Start()
    {
        //A = chrono::system_clock::now();
        gettimeofday(&StartingTime, NULL);
    }
    float Finish()
    {
        //B = std::chrono::system_clock::now();
        //chrono::duration<double> elapsed_seconds = B - A;
        //time_t finish_time = std::chrono::system_clock::to_time_t(B);
        //cout << "title" << elapsed_seconds.count()*1000;
        timeval PausingTime, ElapsedTime;
        gettimeofday(&PausingTime, NULL);
        timersub(&PausingTime, &StartingTime, &ElapsedTime);
        float d = ElapsedTime.tv_sec * 1000.0 + ElapsedTime.tv_usec / 1000.0;
        Start();
        return d;
    }
};

#endif //	TIMER_HPP
