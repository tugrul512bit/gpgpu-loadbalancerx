# gpgpu-loadbalancerx
Simple load-balancing library for balancing (gpugpu-type or other) workloads between gpus (or any devices) in a computer (or multiple computers if it is a cluster). 

On each run() call from LoadBalancerX instance, the work distribution becomes more fair (the faster GPU/CPU gets more work). API-overhead per run call is less than 40 microseconds(for FX8150 CPU + 7 devices) but grains that are sent to devices should be taking enough time to benefit from run-time minimization optimization. 
	
What can grain be?
- Computation kernel for single 16x16 tile of an image, processed by 256 CUDA threads + its data transmissions over PCI-e
- Sending work to another computer and waiting for response by any means
- Anything that can be run in serial or parallel as long as it completes its own task within its scope
	
What can device be?
- Device settings to launch a kernel such as OpenCL context handle for a GPU-id
- CUDA GPU-id
- Object instance that holds I/O arrays for a GPU/FPGA/another CPU
- Anything that needs some temporary state (to be used for grain computation)

How does it work?

![work](https://github.com/tugrul512bit/gpgpu-loadbalancerx/blob/main/canvas.png)
(image created in https://app.diagrams.net/)

```C++
std::vector<std::string> output(20);

// template parameter "State" (int below) can be any copyable class that contains any device-specific setting data such as GPU-identification for launching kernels
// for CUDA, it could be GPU-id, for an OpenCL wrapper it could be a context handle (created for each GPU) of a GPU
// for a cluster, it could be a networking function-object that sends data to other computers
LoadBalanceLib::LoadBalancerX<int> lb;


for(int i=0;i<20;i++)
{
	// load balancer selects necessary devices and feeds their "state" data to the selected work grain 
	// here it is just a simulated GPU-id value as integer
	lb.addWork(LoadBalanceLib::GrainOfWork<int>([&,i](int gpu){

		// simulating different GPUs (high-end = less sleep)
		std::this_thread::sleep_for(std::chrono::milliseconds(gpu*10+30));
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
```

output:

```
performance ratios:
27.3246% 20.5234% 16.4381% 13.6901% 11.7424% 10.2815% 
work:0 gpu:0
work:1 gpu:0
work:2 gpu:0
work:3 gpu:0
work:4 gpu:0
work:5 gpu:0
work:6 gpu:0
work:7 gpu:1
work:8 gpu:1
work:9 gpu:1
work:10 gpu:1
work:11 gpu:2
work:12 gpu:2
work:13 gpu:2
work:14 gpu:3
work:15 gpu:3
work:16 gpu:4
work:17 gpu:4
work:18 gpu:5
work:19 gpu:5

```
