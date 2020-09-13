# Pipeline

Pipeline is a lib project that includes some basic components of a pipeline structure.


## What is pipeline
* A pipeline is consist of pipes, agents and threads.
* A pipeline can accelerate data processing by using multi threads of CPU.
* A pipeline is good at multi-step realtime signal processing system.

![avatar](Pipeline.png)
<center>Figure 1. An example of pipeline.</center>

## Components of pipeline
### Pipe
* A pipe is fixed size ring buffer, it supports multi-producers and multi-consumers working concurrently.   
* Producers push elements to the pipe and consumers pull elements from the pipe.  
* The producer is blocked when the pipe is full and the consumer is blocked when the pipe is empty.  

### Agent
* A agent owns multi threads that do the same job.
* It create its threads at the begining, then keeps all threads working concurrently when started.


### Thread
A thread is a worker in the pipeline. When a thread is started, it repeats the follow steps:
1. Wait for the input pipe to have enough elements to be pulled.
2. Wait for the output pipe to have enough space to pushed the produced item.
3. Process the input elements and generate ouput elements.
4. Tell the input pipe that a pack of elements are already read.
5. Tell the ouput pipe that a pack of elements are generated and pushed to the pipe.

The loop keeps running until any of step 1-3 return false.

![avatar](WorkloopOfPipeDrivenThread.png)
<center>Figure 2. Work loop of a typical pipe-driven thread.</center>

### PipeToken

* Their are two kinds of PipeToken: PipeReadToken and PipeWriteToken.
* When a section of elements are ready to be used ( written or read), the pipe pass a PipeToken to the user (producer or consumer)
* The PipeToken provides some informations including the pointer to the begin of the buffer sections, size of the section, whether the section is the start or end of a shot.
* The PipeToken satisfies RAII requiment, when destructed, it notify it's owner (the pipe) that it is done.

## How is a pipeline work
* All agents, threads and pipes are created before start
* All agents and threads works concurrently
* The fixed size pipes drives its consumers and producers
* If an agent runs too fast, it will need to wait for its previous or next agent
* While a thread (an agent) is waiting, the computing resource will be release for other threads
* If an agent runs too slow, it should be allocated more threads

## How to use
To use the pipeline framwork, one need to derive SyncAgentBase, SyncThreadBase and Pipe<T>. There are some examples of how to use pipe and how to use SyncAgent in the UTPipeline project.

## Copyright

