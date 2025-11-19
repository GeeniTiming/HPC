#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <mutex>

const uint threadnum = 50;
uint g_count = 0;
std::mutex g_count_mutex;

void thread_func(int threadID)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::unique_lock<std::mutex> lock(g_count_mutex);
    g_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    lock.unlock();
}
int main(int argc, char const *argv[])
{
    std::vector<std::thread> threads;
    for (int i = 0; i < threadnum; i++)
    {
        threads.emplace_back(std::thread(thread_func, i));
    }
    for (auto & thread : threads)
    {
        thread.join();
    }
    std::cout << "g_count = " << g_count << std::endl;
    return 0;
}
