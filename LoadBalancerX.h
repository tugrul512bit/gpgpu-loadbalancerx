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

	// single unit of work (i.e. a kernel call + data copy)
	template
	<typename State>
	class GrainOfWork
	{
	public:
		GrainOfWork(){ work=[](State state){ std::this_thread::sleep_for(std::chrono::seconds(1)); }; }
		GrainOfWork(std::function<void(State)> workPrm){ work=workPrm; }
		void run(State state){ if(work) work(state);}
	private:
		std::function<void(State)> work;
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

	// GPGPU load balancing tool
	// distributes work between different graphics cards
	// in a way that minimizes total computation time
	template
	<typename State>
	class LoadBalancerX
	{
	public:
		LoadBalancerX():mutGlobal(std::make_shared<std::mutex>()),initialized(false)
		{

		}

		~LoadBalancerX()
		{
			for(size_t i=0; i<thr.size(); i++)
			{
				{
					std::lock_guard<std::mutex> lg(*(mutGlobal));
					running[i]=false;
					initialized=true;
				}

				{

					std::lock_guard<std::mutex> lg(*(mut[i]));
					running[i]=false;
					initialized=true;
				}

			}

			for(size_t i=0; i<thr.size(); i++)
			{
				if(thr[i].joinable())
					thr[i].join();
			}
		}

		void addWork(GrainOfWork<State> work)
		{
			std::lock_guard<std::mutex> lg(*(mutGlobal));
			totalWork.push_back(work);
		}
		void addDevice(ComputeDevice<State> devPrm)
		{
			size_t indexThr;
			{
				std::lock_guard<std::mutex> lg(*(mutGlobal));
				initialized=false;

				indexThr = thr.size();

				mut.push_back(std::make_shared<std::mutex>());

				{
					std::lock_guard<std::mutex> lg(*(mut[indexThr]));
					devices.push_back(devPrm);
					running.push_back(true);
					hasWork.push_back(false);
					workComplete.push_back(true);

					performances.push_back(1.0);
					nsDev.push_back(1);
					grainDev.push_back(1);
					startDev.push_back(0);
				}

			}

			thr.push_back(std::thread([&,indexThr](){

				State state;
				{
					std::lock_guard<std::mutex> lg(*(mut[indexThr]));
					state = devices[indexThr].getState();
				}
				bool isRunning = true;
				bool hasWrk = false;
				bool init=false;
				size_t start = 0;
				size_t grain = 0;


				while(!init)
				{
					{
						std::lock_guard<std::mutex> lg(*(mutGlobal));
						init=initialized;
					}
				}

				bool globalSyncDone = false;
				while(isRunning)
				{

					{
						if(globalSyncDone)
						{
							std::lock_guard<std::mutex> lg(*(mut[indexThr]));
							isRunning = running[indexThr];
							hasWrk = hasWork[indexThr];
							start = startDev[indexThr];
							grain = grainDev[indexThr];
						}
						else
						{
							std::lock_guard<std::mutex> lg(*(mutGlobal));
							globalSyncDone=true;
							isRunning = running[indexThr];
							hasWrk = hasWork[indexThr];
							start = startDev[indexThr];
							grain = grainDev[indexThr];
						}

					}

					if(hasWrk)
					{
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
									totalWork[j].run(state);
								}
							}
						}

						{
							std::lock_guard<std::mutex> lg(*(mut[indexThr]));
							workComplete[indexThr]=true;
							nsDev[indexThr]=elapsedDevice;
						}

					}

				}
			}));
		}

		void run()
		{
			{
				std::lock_guard<std::mutex> lg(*(mutGlobal));
				initialized=true;
			}
			const size_t totWrk = totalWork.size();
			const size_t totDev = devices.size();


			// compute real performance
			double totPerf = 0;
			for(size_t i=0;i<totDev;i++)
			{
				double perf = (grainDev[i]+0.1)/(double)nsDev[i];
				totPerf+=perf;
				performances[i]=perf;
			}

			size_t ct=0;
			for(size_t i=0;i<totDev;i++)
			{
				std::lock_guard<std::mutex> lg(*(mut[i]));
				performances[i]/=totPerf;
				grainDev[i]=performances[i]*totWrk;
				ct+=grainDev[i];
			}

			// if all devices have 0 work or num work < num device
			size_t ctct=0;
			while(ct < totWrk)
			{
				std::lock_guard<std::mutex> lg(*(mut[ctct%totDev]));
				grainDev[ctct%totDev]++;
				ct++;ctct++;
			}

			ct=0;
			for(size_t i=0;i<totDev;i++)
			{
				std::lock_guard<std::mutex> lg(*(mut[i]));
				startDev[i]=ct;
				ct+=grainDev[i];

			}



			if(ns.size()>5)
				ns.erase(ns.begin());




			size_t elapsedTotal;
			{
				Bench bench(&elapsedTotal);

				// parallel run for real work & time measurement
				for(size_t i=0; i<totDev; i++)
				{
					// todo: optimize with dedicated threads
					std::lock_guard<std::mutex> lg(*(mut[i]));
					hasWork[i]=true;
					workComplete[i]=false;
				}

				for(size_t i=0; i<totDev; i++)
				{
					bool finish = false;
					while(!finish)
					{
						std::lock_guard<std::mutex> lg(*(mut[i]));
						finish = workComplete[i];
						std::this_thread::yield();
					}
				}
			}
			ns.push_back(elapsedTotal);

		}

		// returns percentage of total system performance
		std::vector<double> getRelativePerformancesOfDevices()
		{
			std::vector<double> result;
			size_t sz=performances.size();
			for(size_t i=0;i<sz;i++)
			{
				std::lock_guard<std::mutex> lg(*(mut[i]));
				result.push_back(performances[i]*100.0);
			}
			return result;
		}
	private:

		std::vector<ComputeDevice<State>> devices;
		std::vector<GrainOfWork<State>> totalWork;
		std::vector<size_t> ns;
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
	};

}

#endif /* LOADBALANCERX_H_ */
