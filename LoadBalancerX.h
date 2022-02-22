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

	template
	<typename State>
	class FieldBlock
	{
	public:
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
		std::vector<std::shared_ptr<std::condition_variable>> cond;
	};

	// GPGPU load balancing tool
	// distributes work between different graphics cards
	// in a way that minimizes total computation time
	template
	<typename State>
	class LoadBalancerX
	{
	public:
		LoadBalancerX() // mutGlobal(std::make_shared<std::mutex>()),initialized(false)
		{
			fields=std::make_shared<FieldBlock<State>>();
			fields->mutGlobal=std::make_shared<std::mutex>();
		}

		~LoadBalancerX()
		{

			for(size_t i=0; i<fields->thr.size(); i++)
			{

				{
					std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
					fields->running[i]=false;
					fields->initialized=true;
					fields->hasWork[i]=false;
				}

				{

					std::unique_lock<std::mutex> lg(*(fields->mut[i]));
					fields->running[i]=false;
					fields->initialized=true;
					fields->hasWork[i]=false;
				}

			}

			for(size_t i=0; i<fields->thr.size(); i++)
			{

				fields->cond[i]->notify_one();
				fields->thr[i].join();

			}
		}

		void addWork(GrainOfWork<State> work)
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
				size_t start = 0;
				size_t grain = 0;


				while(!init)
				{
					{
						std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
						init=fields->initialized;
					}
				}

				bool globalSyncDone = false;
				while(isRunning)
				{


					if(globalSyncDone)
					{
						std::unique_lock<std::mutex> lg(*(fields->mut[indexThr]));
						isRunning = fields->running[indexThr];
						hasWrk = fields->hasWork[indexThr];
						start = fields->startDev[indexThr];
						grain = fields->grainDev[indexThr];
					}
					else
					{
						std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
						globalSyncDone=true;
						isRunning = fields->running[indexThr];
						hasWrk = fields->hasWork[indexThr];
						start = fields->startDev[indexThr];
						grain = fields->grainDev[indexThr];
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
									fields->totalWork[j].run(state);
								}
							}
						}

						{
							std::unique_lock<std::mutex> lg(*(fields->mut[indexThr]));
							fields->workComplete[indexThr]=true;
							fields->hasWork[indexThr]=false;
							fields->nsDev[indexThr]=elapsedDevice;
							fields->cond[indexThr]->wait_for(lg,std::chrono::microseconds(1000));
						}

					}
					else
					{
						std::unique_lock<std::mutex> lg(*(fields->mut[indexThr]));
						fields->workComplete[indexThr]=true;
						fields->hasWork[indexThr]=false;
						fields->cond[indexThr]->wait_for(lg,std::chrono::microseconds(1000));
					}

				}

			}));
		}

		void run()
		{

			{
				std::unique_lock<std::mutex> lg(*(fields->mutGlobal));
				fields->initialized=true;
			}
			const size_t totWrk = fields->totalWork.size();
			const size_t totDev = fields->devices.size();


			// compute real performance
			double totPerf = 0;
			for(size_t i=0;i<totDev;i++)
			{
				double perf = (fields->grainDev[i]+0.1)/(double)fields->nsDev[i];
				totPerf+=perf;
				fields->performances[i]=perf;
			}

			size_t ct=0;
			for(size_t i=0;i<totDev;i++)
			{
				std::unique_lock<std::mutex> lg(*(fields->mut[i]));
				fields->performances[i]/=totPerf;
				fields->grainDev[i]=fields->performances[i]*totWrk;
				ct+=fields->grainDev[i];
			}

			// if all devices have 0 work or num work < num device
			size_t ctct=0;
			while(ct < totWrk)
			{
				std::unique_lock<std::mutex> lg(*(fields->mut[ctct%totDev]));
				fields->grainDev[ctct%totDev]++;
				ct++;ctct++;
			}

			ct=0;
			for(size_t i=0;i<totDev;i++)
			{
				std::unique_lock<std::mutex> lg(*(fields->mut[i]));
				fields->startDev[i]=ct;
				ct+=fields->grainDev[i];

			}



			if(fields->ns.size()>5)
				fields->ns.erase(fields->ns.begin());




			size_t elapsedTotal;
			{
				Bench bench(&elapsedTotal);

				// parallel run for real work & time measurement
				for(size_t i=0; i<totDev; i++)
				{

					if(fields->grainDev[i]>0)
					{
						std::unique_lock<std::mutex> lg(*(fields->mut[i]));
						fields->hasWork[i]=true;
						fields->workComplete[i]=false;
						fields->cond[i]->notify_one();
					}
				}

				for(size_t i=0; i<totDev; i++)
				{
					if(fields->grainDev[i]>0)
					{

						bool finish = false;
						while(!finish)
						{
							fields->cond[i]->notify_one();
							std::unique_lock<std::mutex> lg(*(fields->mut[i]));
							finish = fields->workComplete[i];
							std::this_thread::yield();
						}
					}
				}
			}
			fields->ns.push_back(elapsedTotal);

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
		std::shared_ptr<FieldBlock<State>> fields;
	};

}

#endif /* LOADBALANCERX_H_ */
