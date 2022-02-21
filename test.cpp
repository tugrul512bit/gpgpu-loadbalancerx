//============================================================================
// Name        : test.cpp
// Author      : Tugrul
//============================================================================

#include <iostream>
using namespace std;

#include "LoadBalancerX.h"
int main() {

	std::vector<std::string> output(20);

	LoadBalanceLib::LoadBalancerX<int> lb;


	for(int i=0;i<20;i++)
	{
		lb.addWork(LoadBalanceLib::GrainOfWork<int>([&,i](int gpu){

			// simulating different GPUs (high-end = less sleep)
			std::this_thread::sleep_for(std::chrono::milliseconds(gpu+3));
			output[i]=std::string("work:")+std::to_string(i)+std::string(" gpu:")+std::to_string(gpu);
		}));
	}
	lb.addDevice(LoadBalanceLib::ComputeDevice<int>(0));
	lb.addDevice(LoadBalanceLib::ComputeDevice<int>(1));
	lb.addDevice(LoadBalanceLib::ComputeDevice<int>(2));
	lb.addDevice(LoadBalanceLib::ComputeDevice<int>(3));
	lb.addDevice(LoadBalanceLib::ComputeDevice<int>(4));
	lb.addDevice(LoadBalanceLib::ComputeDevice<int>(5));

	for(int i=0;i<10;i++)
	{
		lb.run();
	}

	std::cout<<"performance ratios:"<<std::endl;
	auto performances = lb.getRelativePerformancesOfDevices();
	for(int i=0;i<performances.size();i++)
	{
		std::cout<<performances[i]<<"% ";
	}
	std::cout<<std::endl;

	for(int i=0;i<output.size();i++)
	{
		std::cout<<output[i]<<std::endl;
	}
	return 0;
}
