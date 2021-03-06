/*
 * LoadBalancerX.h
 *
 *  Created on: Feb 21, 2022
 *      Author: tugrul
 */

#ifndef LOADBALANCERX_H_
#define LOADBALANCERX_H_

#include<functional>
#include<thread>
#include<vector>
#include<chrono>
#include<memory>
#include<mutex>
#include<condition_variable>
#include<map>
#include<queue>
#include<iostream>

namespace LoadBalanceLib
{

	// writes elapsed time in nanoseconds to the variable pointed by targetPtr
	// elapsed = destruction time point - construction time point
	class Bench
	{
	public:
		Bench(size_t * targetPtr)
		{
			target=targetPtr;
			t1 =  std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::high_resolution_clock::now().time_since_epoch());
		}

		~Bench()
		{
			t2 =  std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::high_resolution_clock::now().time_since_epoch());
			*target= t2.count() - t1.count();
		}
	private:
		size_t * target;
		std::chrono::nanoseconds t1,t2;
	};

	/* single unit of work (i.e. input copy + kernel call + output copy + sync)
	 * State: device state that will be given by load balancer to each grain to select which device it is being run
	 * GrainState: to keep internal states of each grain if necessary
	 */
	template
	<typename State, typename GrainState>
	class GrainOfWork
	{
	public:
		GrainOfWork():	workInit([](State s, GrainState&){}),
						workInput([](State s, GrainState&){}),
						workCompute([](State s, GrainState&){}),
						workOutput([](State s, GrainState&){}),
						workSync([](State s, GrainState&){}),
						initialized(){ }

		/*
		 * workInitPrm: called only once per lifetime of LoadBalancerX instance, to initialize grain data / data inside device state (per device)
		 * 				can be synchronized algorithm
		 * workInputPrm: called on every run() method call of LoadBalancerX instance to load input data into device
		 * 				user should use asynchronous functions in this for optimal performance
		 * workComputePrm: called on every run() method call of LoadBalancerX instance to compute data in device
		 * 				user should use asynchronous functions in this for optimal performance
		 * workOutputPrm: called on every run() method call of LoadBalancerX instance to save output data from device into host environment
		 * 				user should use asynchronous functions in this for optimal performance
		 * workSyncPrm: called on every run() method call of LoadBalancerX instance to synchronize any and all asynchronous work inside
		 * 				workInputPrm, workComputePrm, workOutputPrm functions
		 * 				user must synchronize each grain's work either in this function or in any other work__Prm function
		 * 				this function is only given for extra readability and called last for every run() call for each grain
		 * grainStatePrm: internal state per grain to be used (if necessary)
		 */
		GrainOfWork(std::function<void(State, GrainState&)> workInitPrm,
					std::function<void(State, GrainState&)> workInputPrm,
					std::function<void(State, GrainState&)> workComputePrm,
					std::function<void(State, GrainState&)> workOutputPrm,
					std::function<void(State, GrainState&)> workSyncPrm
				): initialized()
		{
			workInit=workInitPrm;
			workInput=workInputPrm;
			workCompute=workComputePrm;
			workOutput=workOutputPrm;
			workSync=workSyncPrm;
		}

		// called only once for life time
		void init(State state, GrainState& gState){ if(workInit) workInit(state, gState);}
		void input(State state, GrainState& gState){ if(workInput) workInput(state, gState);}
		void compute(State state, GrainState& gState){ if(workCompute) workCompute(state, gState);}
		void output(State state, GrainState& gState){ if(workOutput) workOutput(state, gState);}
		void sync(State state, GrainState& gState){ if(workSync) workSync(state, gState);}
		bool isReady(int deviceIndex){ return initialized.find(deviceIndex) != initialized.end(); }
		void makeReady(int deviceIndex){ initialized[deviceIndex]=true; }

		GrainState& refGrainState (){ return grainState; }



	//private:
		// called once per lifetime of loadbalancerx per device
		std::function<void(State, GrainState&)> workInit;

		// called on every run method call of loadbalancerx
		// to copy input data to device
		std::function<void(State, GrainState&)> workInput;

		// called on every run method call of loadbalancerx
		// to compute data in device
		std::function<void(State, GrainState&)> workCompute;

		// called on every run method call of loadbalancerx
		// to copy output data from device to host
		std::function<void(State, GrainState&)> workOutput;

		// called on every run method call of loadbalancerx
		// to synchronize any and all asynchronized work given inside workInput, workCompute and workOutput
		// user must synchronize in this unless it is synchronized in other methods
		std::function<void(State, GrainState&)> workSync;

		std::map<int,bool> initialized;

		GrainState grainState;
		std::chrono::nanoseconds t1,t2;
	};

	template
	<typename State>
	class ComputeDevice
	{
	public:
		ComputeDevice():state(){  }
		ComputeDevice(State statePrm):state(statePrm){}
		State getState(){ return state; }
	private:
		State state;
	};

	template<typename GrainOfWork>
	class Load
	{
	public:
		int cmd; // 0:stop running, 1:compute
		size_t start;
		size_t grain;
		bool pipelined;
		GrainOfWork grainInfo;
	};

	class Response
	{
	public:
		int msg;
		size_t ns;
	};

	// thread-safe queue
	template<typename T, int sz>
	class ThreadsafeQueue
	{
	public:
		ThreadsafeQueue(){}
		void push(T t)
		{
			std::unique_lock<std::mutex> lc(m);
			q.push(t);
			c.notify_one();
		}

		size_t size()
		{
			std::unique_lock<std::mutex> lc(m);
			return q.size();
		}

		T pop()
		{
			std::unique_lock<std::mutex> lc(m);
			while(q.empty())
			{
				c.wait(lc);
			}
			T result = q.front();
			q.pop();
			return result;
		}
	private:
		std::queue<T> q;
		std::mutex m;
		std::condition_variable c;
	};

	template
	<typename State, typename GrainState>
	class FieldBlock
	{
	public:
		FieldBlock():initialized(false)
		{

		}
		std::vector<ComputeDevice<State>> devices;
		std::vector<GrainOfWork<State, GrainState>> totalWork;

		std::vector<double> performancesHistory;
		std::vector<size_t> nsDev;
		std::vector<size_t> grainDev;
		std::vector<size_t> startDev;
		std::vector<std::thread> thr;
		std::vector<double> performances;
		std::vector<std::shared_ptr<std::mutex>> mut;
		std::vector<bool> running;
		std::vector<bool> hasWork;
		std::vector<bool> workComplete;
		std::shared_ptr<std::mutex> mutGlobal;
		bool initialized;
		std::vector<std::shared_ptr<std::condition_variable>> cond;
		std::vector<std::shared_ptr<ThreadsafeQueue<Load<GrainOfWork<State,GrainState>>,    100>>> loadQueue;
		std::vector<std::shared_ptr<ThreadsafeQueue<Response,100>>> responseQueue;
	};


    template
    <typename DeviceState, typename GrainState>
    class GrainCache
    {
    public:
    	GrainOfWork<DeviceState, GrainState> getGrain(	size_t id,
    							std::function<void(DeviceState, GrainState&)> init,
    							std::function<void(DeviceState, GrainState&)> input,
    							std::function<void(DeviceState, GrainState&)> compute,
    							std::function<void(DeviceState, GrainState&)> output,
    							std::function<void(DeviceState, GrainState&)> sync
    							)
    	{
    		auto it = grains.find(id);
    		if(it!=grains.end())
    		{
    			it->second.workInit=init;
    			it->second.workInput=input;
    			it->second.workCompute=compute;
    			it->second.workOutput=output;
    			it->second.workSync=sync;
    			return it->second;
    		}
    		else
    		{
    			grains[id]=GrainOfWork<DeviceState, GrainState>(init,input,compute,output,sync);
    		}
    		return grains.at(id);
    	}
    private:
    	std::map<size_t,GrainOfWork<DeviceState, GrainState>> grains;
    };


	// GPGPU load balancing tool
	// distributes work between different graphics cards
	// in a way that minimizes total computation time
	template
	<typename State, typename GrainState>
	class LoadBalancerX
	{
	public:
		LoadBalancerX() // mutGlobal(std::make_shared<std::mutex>()),initialized(false)
		{
			fields=std::make_shared<FieldBlock<State, GrainState>>();
			fields->mutGlobal=std::make_shared<std::mutex>();
			runCount=0;
		}

		~LoadBalancerX()
		{

			for(size_t i=0; i<fields->thr.size(); i++)
			{
				fields->loadQueue[i]->push(Load<GrainOfWork<State,GrainState>>({0,0,0}));
			}

			for(size_t i=0; i<fields->thr.size(); i++)
			{
				fields->thr[i].join();
			}
		}

		void addWork(GrainOfWork<State, GrainState> work)
		{
			std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
			fields->totalWork.push_back(work);
		}
		void addDevice(ComputeDevice<State> devPrm)
		{
			size_t indexThr;
			{
				std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
				fields->initialized=false;
				fields->loadQueue.push_back(    std::make_shared<ThreadsafeQueue<Load<GrainOfWork<State,GrainState>>,    100>>());
				fields->responseQueue.push_back(std::make_shared<ThreadsafeQueue<Response,100>>());
				indexThr = fields->thr.size();

				fields->mut.push_back(std::make_shared<std::mutex>());
				fields->cond.push_back(std::make_shared<std::condition_variable>());
				{
					std::unique_lock<std::mutex> lg(*(fields->mut[indexThr]));
					fields->devices.push_back(devPrm);
					fields->running.push_back(true);
					fields->hasWork.push_back(false);
					fields->workComplete.push_back(true);

					fields->performances.push_back(1.0);
					fields->nsDev.push_back(1);
					fields->grainDev.push_back(1);
					fields->startDev.push_back(0);
				}

			}

			fields->thr.push_back(std::thread([&,indexThr](){

				State state;
				{
					std::unique_lock<std::mutex> lg(*(fields->mut[indexThr]));
					state = fields->devices[indexThr].getState();
				}
				bool isRunning = true;
				bool hasWrk = false;
				bool init=false;
				bool pipelined=false;
				size_t start = 0;
				size_t grain = 0;


				while(!init)
				{
					{
						std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
						init=fields->initialized;
					}
				}


				while(isRunning)
				{


					Load<GrainOfWork<State,GrainState>> load = fields->loadQueue[indexThr]->pop();
					if(load.cmd>0)
					{
						start = load.start;
						grain = load.grain;
						pipelined=load.pipelined;

						// single work sync request
						if(load.cmd==3)
						{

							GrainOfWork<State,GrainState> grainInfo = load.grainInfo;
							grainInfo.sync(state, grainInfo.refGrainState()); // user must synchronize in this unless it is synchronized in other methods
							grainInfo.t2=std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::high_resolution_clock::now().time_since_epoch());
							fields->responseQueue[indexThr]->push(Response({1,grainInfo.t2.count()-grainInfo.t1.count()}));
						}

						// single work request
						if(load.cmd==2)
						{

							size_t elapsedDevice = 0;
							{

								start = load.start;
								grain = load.grain; // 1
								GrainOfWork<State,GrainState> grainInfo = load.grainInfo;
								grainInfo.t1=std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::high_resolution_clock::now().time_since_epoch());
								if(!grainInfo.isReady(indexThr))
								{
									grainInfo.init(state, grainInfo.refGrainState()); // user should have asynchronous launch in this
									grainInfo.makeReady(indexThr);
								}
								grainInfo.input(state, grainInfo.refGrainState()); // user should have asynchronous launch in this
								grainInfo.compute(state, grainInfo.refGrainState()); // user should have asynchronous launch in this
								grainInfo.output(state, grainInfo.refGrainState()); // user should have asynchronous launch in this

								// creates a self-sync command at the end of queue (to let others run asynchronously)
								fields->loadQueue[indexThr]->push(Load<GrainOfWork<State,GrainState>>({3,0,0,false,grainInfo}));
							}



							continue;
						}

						hasWrk=true;
					}
					else if(load.cmd==0)
					{
						isRunning=false;
						hasWrk=false;
					}


					if(hasWrk && isRunning)
					{
						hasWrk=false;
						// compute grain
						size_t elapsedDevice;
						{
							Bench benchDevice(&elapsedDevice);
							if(grain>0)
							{
								const size_t first = start;
								const size_t last = first+grain;
								for(size_t j=first; j<last; j++)
								{
									if(!fields->totalWork[j].isReady(indexThr))
									{
										fields->totalWork[j].init(state, fields->totalWork[j].refGrainState()); // user should have asynchronous launch in this
										fields->totalWork[j].makeReady(indexThr);
									}
								}

								if(!pipelined || grain<3)
								{


									for(size_t j=first; j<last; j++)
									{
										fields->totalWork[j].input(state, fields->totalWork[j].refGrainState()); // user should have asynchronous launch in this
									}

									for(size_t j=first; j<last; j++)
									{
										fields->totalWork[j].compute(state, fields->totalWork[j].refGrainState()); // user should have asynchronous launch in this
									}

									for(size_t j=first; j<last; j++)
									{
										fields->totalWork[j].output(state, fields->totalWork[j].refGrainState()); // user should have asynchronous launch in this
									}


								}
								else
								{
									// 3-way concurrency by pipelining methods
									// input 1 input 2     input 3
									//         compute 1   compute 2   compute 3
									//                     output 1    output 2     output 3

									const size_t first = start+2;
									const size_t last = first+grain-2;
									fields->totalWork[start].input(state, fields->totalWork[start].refGrainState());
									fields->totalWork[start+1].input(state, fields->totalWork[start+1].refGrainState());
									fields->totalWork[start].compute(state, fields->totalWork[start].refGrainState());
									for(size_t j=first;j<last;j++)
									{
										fields->totalWork[j].input(state, fields->totalWork[j].refGrainState());
										fields->totalWork[j-1].compute(state, fields->totalWork[j-1].refGrainState());
										fields->totalWork[j-2].output(state, fields->totalWork[j-2].refGrainState());
									}
									fields->totalWork[last-1].compute(state, fields->totalWork[last-1].refGrainState());
									fields->totalWork[last-2].output(state, fields->totalWork[last-2].refGrainState());
									fields->totalWork[last-1].output(state, fields->totalWork[last-1].refGrainState());
								}

								for(size_t j=first; j<last; j++)
								{
									fields->totalWork[j].sync(state, fields->totalWork[j].refGrainState()); // user must synchronize in this unless it is synchronized in other methods
								}
							}
						}
						fields->responseQueue[indexThr]->push(Response({1,elapsedDevice}));
					}


				}

			}));
		}




		size_t runSingleAsync(GrainOfWork<State, GrainState> grain)
		{
			{
				std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
				fields->initialized=true;
			}

			const size_t totDev = fields->devices.size();

			unsigned int szMin = ((unsigned int)0)-1;
			int iMin = -1;

			bool space = false;
			while(!space)
			{
				for(size_t i=0; i<totDev; i++)
				{
					int sel=fields->loadQueue[i]->size();
					if(szMin>sel && sel<25)
					{
						szMin=sel;
						iMin=i;
						space=true;
					}
				}
			}


			fields->loadQueue[iMin]->push(Load<GrainOfWork<State,GrainState>>({2,0,0,false,grain}));

			return iMin;
		}

		// returns latency of grain's operation from being acquired by dedicated device thread to being sent to synchronization queue
		// most of this latency can be hidden behind other grains' operations
		size_t syncSingle(size_t id)
		{
			Response response = fields->responseQueue[id]->pop();
			if(response.msg==0)
			{
				std::cout<<"Error: compute failed in device-"<<id<<std::endl;
			}
			return response.ns;
		}

		/* returns elapsed time in nanoseconds (this is minimized by load-balancer)
		* pipelined: uses 3-way concurrency in launch pattern of input/compute/output/sync methods for supporting any CUDA/OpenCL-like efficient stream overlapping
		*/
		size_t run(bool pipelined = false)
		{

			{
				std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
				fields->initialized=true;
			}
			const size_t totWrk = fields->totalWork.size();
			const size_t totDev = fields->devices.size();

			const int numSmoothing = 5;
			const int curHistoryIndex = runCount % numSmoothing;
			double totPerf = 0;
			if(fields->performancesHistory.size()<totDev*numSmoothing)
			{
				fields->performancesHistory = std::vector<double>(totDev*numSmoothing);
				for(size_t i=0;i<totDev;i++)
				{
					for(int j=0;j<numSmoothing;j++)
					{
						fields->performancesHistory[j*totDev + i]=1.0/totDev;
					}
				}

			}

			// compute real performance
			totPerf=0.0f;
			for(size_t i=0;i<totDev;i++)
			{
				double perf = (fields->grainDev[i]+0.1)/(double)fields->nsDev[i];

				totPerf+=perf;
				fields->performances[i]=perf;
			}

			runCount++;

			size_t ct=0;
			for(size_t i=0;i<totDev;i++)
			{

				fields->performances[i]/=totPerf;

				// smoothing the performance measurement
				double smooth = 0.0;


				fields->performancesHistory[curHistoryIndex*totDev + i]=fields->performances[i];
				for(int j=0;j<numSmoothing;j++)
				{
					smooth += fields->performancesHistory[j*totDev + i];
				}
				smooth /= (double)numSmoothing;
				fields->performances[i]=smooth;
				fields->grainDev[i]=fields->performances[i]*totWrk;
				ct+=fields->grainDev[i];
			}



			// if all devices have 0 work or num work < num device or not enough work allocated
			size_t ctct=0;
			while(ct < totWrk)
			{

				fields->grainDev[ctct%totDev]++;
				ct++;ctct++;
			}

			ct=0;
			for(size_t i=0;i<totDev;i++)
			{

				fields->startDev[i]=ct;
				ct+=fields->grainDev[i];

			}


			size_t elapsedTotal;
			{
				Bench bench(&elapsedTotal);

				// parallel run for real work & time measurement
				for(size_t i=0; i<totDev; i++)
				{

					if(fields->grainDev[i]>0)
					{
						fields->loadQueue[i]->push(Load<GrainOfWork<State,GrainState>>({1,fields->startDev[i],fields->grainDev[i],pipelined}));

					}
				}

				for(size_t i=0; i<totDev; i++)
				{
					if(fields->grainDev[i]>0)
					{

						Response response = fields->responseQueue[i]->pop();
						if(response.msg==0)
						{
							std::cout<<"Error: compute failed in device-"<<i<<std::endl;
						}
						fields->nsDev[i]=response.ns;
					}
				}
			}
			return elapsedTotal;

		}

		// returns percentage of total system performance
		std::vector<double> getRelativePerformancesOfDevices()
		{
			std::vector<double> result;
			size_t sz=fields->performances.size();
			for(size_t i=0;i<sz;i++)
			{
				std::unique_lock<std::mutex> lg(*(fields->mut[i]));
				result.push_back(fields->performances[i]*100.0);
			}
			return result;
		}
	private:
		std::shared_ptr<FieldBlock<State, GrainState>> fields;
		int runCount;
	};

}

#endif /* LOADBALANCERX_H_ */
