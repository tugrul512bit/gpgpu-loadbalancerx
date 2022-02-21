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
		void run(State state){ work(state);}
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
		LoadBalancerX()
		{

		}

		void addWork(GrainOfWork<State> work){ totalWork.push_back(work); }
		void addDevice(ComputeDevice<State> devPrm){ devices.push_back(devPrm); }
		void run()
		{
			const size_t totWrk = totalWork.size();
			const size_t totDev = devices.size();

			// initial guess for gpu performances
			while(performances.size()<totDev)
			{
				performances.push_back(1.0f/totDev);
				nsDev.push_back(1);
				grainDev.push_back(1);
				nDev.push_back(0);
			}


			// compute real performance
			double totPerf = 0;
			for(size_t i=0;i<totDev;i++)
			{
				double perf = grainDev[i]/(double)nsDev[i];
				totPerf+=perf;
				performances[i]=perf;
			}

			for(size_t i=0;i<totDev;i++)
			{
				performances[i]/=totPerf;
				nDev[i]=performances[i]*totWrk;

			}


			if(ns.size()>5)
				ns.erase(ns.begin());

			size_t elapsedTotal;
			size_t totalWorkCtr=0;
			size_t selectedDevice=0;

			for(auto & g:grainDev)
			{
				g=0;
			}
			for(auto & n:nsDev)
			{
				n=0;
			}

			{
				Bench bench(&elapsedTotal);
				for(auto w:totalWork)
				{
					size_t elapsedDevice;
					{
						Bench benchDevice(&elapsedDevice);
						w.run(devices[selectedDevice].getState());
					}
					nsDev[selectedDevice]+=elapsedDevice;
					grainDev[selectedDevice]++;
					totalWorkCtr++;
					if(totalWorkCtr == nDev[selectedDevice])
					{
						totalWorkCtr=0;
						selectedDevice++;
						if(selectedDevice==totDev)
							selectedDevice -= totDev;
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
		std::vector<size_t> nDev;
		std::vector<double> performances;
	};

}

#endif /* LOADBALANCERX_H_ */
