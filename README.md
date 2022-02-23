# gpgpu-loadbalancerx
Simple load-balancing library for balancing (gpugpu-type or other) workloads between gpus (or any devices) in a computer (or multiple computers if it is a cluster). 

On each run() call from LoadBalancerX instance, the work distribution becomes more fair (the faster GPU/CPU gets more work). API-overhead per run call is less than 50 microseconds(for FX8150 CPU + 6 devices) and the grains that are sent to devices should be taking at least comparable time (to the API overhead) to benefit from run-time minimization optimization and number of grains should be high enough to let load-balancing trade enough grains between devices to minimize run-time. 
	
What can grain state be and what should a grain do?
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

How does it work? (wiki: https://github.com/tugrul512bit/gpgpu-loadbalancerx/wiki)
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
1181739684ns <------ bad start with 1.18 seconds
performance ratios:
16.6667% 16.6667% 16.6667% 16.6667% 16.6667% 16.6667% 

1074422792ns <------ faster
performance ratios:
19.4891% 17.5123% 16.5013% 15.8729% 15.4616% 15.1628% 

970820248ns  <------ faster
performance ratios:
22.2881% 18.3442% 16.3681% 15.0816% 14.2571% 13.661% 

861089248ns  <------ faster
performance ratios:
25.0974% 19.1938% 16.2083% 14.3014% 13.043% 12.156% 

756021596ns  <----- very close to optimum performance
performance ratios:
27.9079% 20.042% 16.0664% 13.4933% 11.8386% 10.652% 

659877380ns  <----- 0.65 seconds (2x performance of equal distribution)
performance ratios:
30.6945% 20.8847% 15.9153% 12.7279% 10.6311% 9.14648% 

659023379ns 
performance ratios:
30.69% 20.8868% 15.9086% 12.7347% 10.6316% 9.14839% 

655717420ns
performance ratios:
30.6905% 20.9072% 15.8724% 12.7362% 10.6319% 9.16177% 

655666552ns
performance ratios:
30.6857% 20.9003% 15.8632% 12.7275% 10.643% 9.18019% 

656228846ns
performance ratios:
30.6829% 20.8943% 15.8336% 12.7437% 10.6643% 9.18123% 

656478081ns
performance ratios:
30.6876% 20.9014% 15.8151% 12.7215% 10.6902% 9.18421% 

653987494ns
performance ratios:
30.6606% 20.898% 15.8112% 12.7402% 10.7102% 9.1799% 

654755954ns
performance ratios:
30.6738% 20.8798% 15.8082% 12.7467% 10.7251% 9.16641% 

655304544ns
performance ratios:
30.6821% 20.8701% 15.8057% 12.7455% 10.74% 9.15654% 

653814090ns
performance ratios:
30.7018% 20.8527% 15.7976% 12.7627% 10.7323% 9.15283% 

650336759ns
performance ratios:
30.7407% 20.8431% 15.7905% 12.7677% 10.7095% 9.14844% 

653457175ns
performance ratios:
30.7678% 20.8433% 15.7934% 12.7436% 10.7029% 9.14894% 

655730340ns
performance ratios:
30.7917% 20.8576% 15.7908% 12.736% 10.6823% 9.14162% 

655918707ns
performance ratios:
30.7905% 20.8661% 15.7948% 12.735% 10.6658% 9.14793% 

654000862ns
performance ratios:
30.7691% 20.8763% 15.8158% 12.7174% 10.6576% 9.16378% 

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
