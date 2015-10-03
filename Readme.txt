This is the CPP version of the Online GASSOM, designed to encode multiple streams of inputs simultaniously. 

Due to the pains of CUDA not being OOP and havin gto hardcode some aspects in the kernels, the code only works for a handful of parameters. 

ie. AGENTSNUM=100, which means 100 simultanious streams, BASISDIM < 1024, due to limitation sin CUDA 2x, I can only launch that many threads in a block, and NUMBASIS <400, which means the number of subspaces. 

I hope to clean up the code and make it easily adaptable when i get some free time. But if anyone wants to use it, and have questions for the time, contact me on tnc<at>connect<dot>ust<dot>hk 

Tha Papers which describe the algorithm are also in the folder. 

   