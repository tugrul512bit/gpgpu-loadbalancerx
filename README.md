# gpgpu-loadbalancerx
Simple load-balancing library for balancing (gpugpu-type or other) workloads between gpus (or any devices) in a computer (or multiple computers if it is a cluster). 

On each run() call from LoadBalancerX instance, the work distribution becomes more fair (the faster GPU/CPU gets more work). API-overhead per run call is less than 150 microseconds(for FX8150 CPU + 7 devices) so the grains that are sent to devices should be taking enough time to benefit from run-time minimization optimization and number of grains should be high enough to let load-balancing trade enough grains between devices to minimize run-time. 
	
What can grain state be and what can grain do?
- Computation kernel for single 16x16 tile of an image, to be processed by 256 CUDA threads + its data transmissions over PCI-e
- Sending work to another computer and waiting for response by any means
- Anything that can be run in serial or parallel as long as it completes its own task within its scope
- Should have asynchronous methods in inputWork, computeWork and outputWork to have optimum performance
- must synchronize in outputWork function or syncWork function
	
What can device state have?
- Device settings to launch a kernel such as OpenCL context handle for a GPU-id
- CUDA GPU-id
- Object instance that holds I/O arrays for a GPU/FPGA/another CPU or even some network comm that offloads to a server
- Anything that needs some temporary state (to be used for grain computation)

How does it work?
- User adds devices with state objects or values
- User adds work grains to be repeated in each run() call
- Load balancer creates 1 dedicated CPU thread for each device
- Load balancer selects a grain and a device
- - If selected grain was not initialized in selected device, then runs the initWork function
- - runs inputWork function to copy data from host to device, user should use asynchronous functions inside for performance
- - runs computeWork function to compute, user should use asynchronous functions inside for performance
- - runs outputWork function to copy results from device to host, user should use asynchronous functions inside for performance
- - runs syncWork function to synchronize all previous async work with the host
- Load balancer synchronizes all dedicated device threads and returns to user with run-time minimization optimization for the next run() call
- After several repeats, it converges to a fair work distribution depending on performances of devices and run-time approaches to optimum level

![work](https://github.com/tugrul512bit/gpgpu-loadbalancerx/blob/main/canvas.png)
![how it works](https://github.com/tugrul512bit/gpgpu-loadbalancerx/blob/main/howitworks.png)

(image created in https://app.diagrams.net/)

```C++
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
					thisGrain.cudaInputDevice[gpu.gpuId]=std::vector<float>(pixelsPerGrain); // simulating a cuda gpu buffer allocation
					thisGrain.cudaOutputDevice[gpu.gpuId]=std::vector<float>(pixelsPerGrain); // simulating a cuda gpu buffer allocation
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
```

output:

```
140650356846593ns
performance ratios:
16.6667% 16.6667% 16.6667% 16.6667% 16.6667% 16.6667% 
1181025083ns
performance ratios:
30.7092% 20.8774% 15.848% 12.7355% 10.6629% 9.16699% 
655846495ns
performance ratios:
30.6198% 20.8904% 15.9645% 12.7305% 10.6415% 9.15345% 
660575262ns
performance ratios:
30.6731% 20.8994% 15.8065% 12.7349% 10.6392% 9.24704% 
655996932ns
performance ratios:
30.6568% 20.8807% 15.8166% 12.7384% 10.6613% 9.24612% 
655698083ns
performance ratios:
30.7377% 20.8877% 15.8029% 12.7014% 10.6361% 9.23423% 
659502176ns
performance ratios:
30.9245% 20.7333% 15.6754% 12.7932% 10.6956% 9.17793% 
657921971ns
performance ratios:
30.7195% 20.8781% 15.8102% 12.7176% 10.634% 9.24064% 
653907807ns
performance ratios:
31.0522% 20.7535% 15.7356% 12.6546% 10.6003% 9.20382% 
654945458ns
performance ratios:
31.3518% 20.7196% 15.7005% 12.6041% 10.5496% 9.07441% 
653337073ns
performance ratios:
31.1468% 20.7788% 15.74% 12.64% 10.5905% 9.10401% 
660147647ns
performance ratios:
30.7532% 20.8482% 15.8251% 12.716% 10.6346% 9.22301% 
653166648ns
performance ratios:
30.7406% 20.8485% 15.8166% 12.7168% 10.6442% 9.23332% 
654882018ns
performance ratios:
30.877% 20.8653% 15.7748% 12.7244% 10.6252% 9.13343% 
656351034ns
performance ratios:
30.6216% 20.8578% 16.0266% 12.74% 10.6182% 9.13572% 
659671523ns
performance ratios:
30.7534% 20.9119% 15.7982% 12.7349% 10.6378% 9.16384% 
654123455ns
performance ratios:
31.0321% 20.8127% 15.7659% 12.6766% 10.5949% 9.11777% 
657591513ns
performance ratios:
30.785% 20.8623% 15.8471% 12.7085% 10.6457% 9.15141% 
653089942ns
performance ratios:
31.1682% 20.7082% 15.8858% 12.6172% 10.5515% 9.06908% 
660573104ns
performance ratios:
30.7765% 20.8227% 15.8265% 12.7081% 10.7151% 9.15115% 
0
0.5
1
1.5
2
2.5
3
3.5
4
4.5
5
5.5
6
6.5
7
7.5
8
8.5
9
9.5
10
10.5
11
11.5
12

```
