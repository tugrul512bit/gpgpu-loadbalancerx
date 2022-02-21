# gpgpu-loadbalancerx
Simple load-balancing library for balancing (gpugpu or other) workloads between gpus (or any devices) in a computer.
```C++
std::vector<std::string> output(20);

LoadBalanceLib::LoadBalancerX<int> lb;


for(int i=0;i<20;i++)
{
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
