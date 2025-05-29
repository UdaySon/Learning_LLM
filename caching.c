#include <windows.h>
#include <stdio.h>

#define NUMBER 50000  // 32 KB

int timed_function(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

void flush_cache() {
    volatile int dummy[100000];
    for (int i = 0; i < 100000; i++) dummy[i] = 0xFF; // dummy random data to force usage of cache
}

int main() 
{
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    int arr[NUMBER];
	
    for (int i = 0; i < NUMBER; ++i) arr[i] = i;
    flush_cache();  // Simulate cold cache
    flush_cache();  // Simulate cold cache
	
    QueryPerformanceCounter(&start);
    volatile int res1 = timed_function(arr, NUMBER);
    QueryPerformanceCounter(&end);
	
    double time_us = (double)(end.QuadPart - start.QuadPart) * 1e6 / freq.QuadPart;
    printf("Cold (simulated) cache time: %.2f us\n", time_us);

    QueryPerformanceCounter(&start);
    volatile int res2 = timed_function(arr, NUMBER);
    QueryPerformanceCounter(&end);
	
    time_us = (double)(end.QuadPart - start.QuadPart) * 1e6 / freq.QuadPart;
    printf("Hot (cached) time: %.2f us\n", time_us);

    return 0;
}
