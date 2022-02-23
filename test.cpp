//============================================================================
// Name        : test.cpp
// Author      : Tugrul
//============================================================================

#include <iostream>
#include <map>
using namespace std;

#include "LoadBalancerX.h"
int main() {

	// number of chunks in a divide&conquer algorithm
	const int grains = 1000;
	const int pixelsPerGrain=5;

	// simulating pixel buffer in host for a GPGPU task
	std::vector<float> input(grains*pixelsPerGrain);
	std::vector<float> output(grains*pixelsPerGrain);

	// simulate an image data
	for(int i=0;i<grains*pixelsPerGrain;i++)
	{
		input[i]=i&255;
	}

	// necessary device state information for all types of devices
	class DeviceState
	{
	public:
		int gpuId;
	};

	// necessary grain state information
	class GrainState
	{
	public:
		GrainState():whichGpuComputedMeLastTime(-1){}
		int whichGpuComputedMeLastTime;

		// just simulating a GPU's video-memory buffer
		std::map<int,std::vector<float>> cudaInputDevice;
		std::map<int,std::vector<float>> cudaOutputDevice;
	};

	// load balancer to distribute grains between devices fairly depending on their performance
	LoadBalanceLib::LoadBalancerX<DeviceState, GrainState> lb;


	for(int i=0;i<grains;i++)
	{
		lb.addWork(LoadBalanceLib::GrainOfWork<DeviceState, GrainState>(
				[&,i](DeviceState gpu, GrainState& thisGrain){
					/* (async/sync) initialize grain's host/device environment (if necessary),
					 * called only once for lifetime of LoadBalancerX instance per device
					 */
					if(thisGrain.whichGpuComputedMeLastTime != gpu.gpuId)
					{
						thisGrain.cudaInputDevice[gpu.gpuId]=std::vector<float>(5); // simulating a cuda gpu buffer allocation
						thisGrain.cudaOutputDevice[gpu.gpuId]=std::vector<float>(5); // simulating a cuda gpu buffer allocation
						thisGrain.whichGpuComputedMeLastTime = gpu.gpuId;
					}
				},
				[&,i](DeviceState gpu, GrainState& thisGrain){
					/* (async) send input data to device (called for every run) */
					for(int j=0;j<pixelsPerGrain;j++)
						thisGrain.cudaInputDevice[gpu.gpuId][j]=input[i*pixelsPerGrain + j];
				},
				[&,i](DeviceState gpu, GrainState& thisGrain){
					/* (async) compute GPGPU task in device using input (called for every run) */

					// some simple color computation
					// (just simulating a cuda kernel)
					for(int j=0;j<pixelsPerGrain;j++)
						thisGrain.cudaOutputDevice[gpu.gpuId][j]=0.5f*thisGrain.cudaInputDevice[gpu.gpuId][j];
				},
				[&,i](DeviceState gpu, GrainState& thisGrain){
					/* (async) get results from device to host (called for every run) */

					for(int j=0;j<pixelsPerGrain;j++)
						output[i*pixelsPerGrain + j] = thisGrain.cudaOutputDevice[gpu.gpuId][j];
				},
				[&,i](DeviceState gpu, GrainState& thisGrain){
					/* (synchronized)synchronize this grain's work (called for every run) */

					// simulating cuda kernel synchronization
					// simulating different GPUs (bigger gpuId = low-end GPU)
					std::this_thread::sleep_for(std::chrono::milliseconds(2+gpu.gpuId));
				}
		));

	}
	lb.addDevice(LoadBalanceLib::ComputeDevice<DeviceState>({0})); // RTX3090
	lb.addDevice(LoadBalanceLib::ComputeDevice<DeviceState>({1})); // RTX3070
	lb.addDevice(LoadBalanceLib::ComputeDevice<DeviceState>({2})); // RTX3060 with overclock
	lb.addDevice(LoadBalanceLib::ComputeDevice<DeviceState>({3})); // RTX3060
	lb.addDevice(LoadBalanceLib::ComputeDevice<DeviceState>({4})); // for offloading to a server
	lb.addDevice(LoadBalanceLib::ComputeDevice<DeviceState>({5})); // maybe some CPU cores



	size_t nano;
	{

		for(int i=0;i<20;i++)
		{
			{
				LoadBalanceLib::Bench bench(&nano);
				lb.run();
			}
			std::cout<<nano<<"ns"<<std::endl;
			std::cout<<"performance ratios:"<<std::endl;
			auto performances = lb.getRelativePerformancesOfDevices();
			for(int i=0;i<performances.size();i++)
			{
				std::cout<<performances[i]<<"% ";
			}
			std::cout<<std::endl;
		}
	}





	for(int i=0;i<min(25,grains);i++)
	{
		std::cout<<output[i]<<std::endl;
	}

	return 0;
}
