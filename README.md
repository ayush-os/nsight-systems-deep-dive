# nsight-systems-deep-dive

nvcc -arch=sm_80 benchmark.cu -o bench_a100

### VECTOR ADD RESULTS WHEN N=10 and 1000 LOOPS 

#### only 52% of execution time is in kernels, rest is overhead
#### AKA CPU spends as much time launching as GPU spends executing
#### cudaMalloc consumes 97% of CUDA API runtime (wake hardware, allocate heap, establish primary context - so this should technically be excluded.) SEE BELOW FOR WHERE I do cudaFree(0) to give a more accurate cudaMalloc time

paperspace@psbqejul7uvv:~/nsight-systems-deep-dive$ sudo $(which nsys) profile   -t cuda,nvtx,osrt   --gpu-metrics-device=all   --cpuctxsw=none   --sample=cpu   --force-overwrite true   --stats=true   -o vector_add_profile_v3   ./bench_a100
GPU 0: General Metrics for NVIDIA GA100 (any frequency)
Launching kernel with 1 blocks...
Done.
Generating '/tmp/nsys-report-2ee6.qdstrm'
[1/8] [========================100%] vector_add_profile_v3.nsys-rep
[2/8] [========================100%] vector_add_profile_v3.sqlite
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style      Range    
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  ------------
     95.0        3,481,709          1  3,481,709.0  3,481,709.0  3,481,709  3,481,709          0.0  PushPop  Profile_Loop
      5.0          182,472          1    182,472.0    182,472.0    182,472    182,472          0.0  PushPop  Warmup      

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)     Min (ns)    Max (ns)     StdDev (ns)        Name     
 --------  ---------------  ---------  -------------  -------------  ---------  -----------  -------------  --------------
     67.3      335,987,431          2  167,993,715.5  167,993,715.5  2,971,823  333,015,608  233,376,198.5  sem_wait      
     20.0       99,806,189         17    5,870,952.3    1,427,621.0      6,930   33,322,960   11,011,658.0  poll          
     11.4       56,919,838        736       77,336.7        8,607.5          0   23,720,749      945,876.7  ioctl         
      0.5        2,704,139         51       53,022.3       10,500.0        605    1,872,013      260,137.2  mmap64        
      0.3        1,677,335         15      111,822.3       89,366.0     69,408      406,813       83,203.1  sem_timedwait 
      0.2        1,157,652         74       15,643.9       12,075.5          0      223,868       27,744.5  open64        
      0.0          246,404          4       61,601.0       56,108.0     45,943       88,245       18,768.8  pthread_create
      0.0          135,927         14        9,709.1        5,099.5          0       54,986       13,974.5  mmap          
      0.0          112,342         16        7,021.4        8,270.5          0       10,579        3,668.5  write         
      0.0           90,838         36        2,523.3        2,340.5          0       10,922        2,556.9  fopen         
      0.0           78,561          5       15,712.2        6,200.0      4,749       52,694       20,747.6  munmap        
      0.0           55,373         64          865.2          160.0          0       27,764        3,938.7  fgets         
      0.0           50,811         29        1,752.1          917.0          0       23,487        4,600.5  fclose        
      0.0           28,191         81          348.0          432.0          0        1,064          237.7  fcntl         
      0.0           20,417          2       10,208.5       10,208.5      8,929       11,488        1,809.5  putc          
      0.0           18,655          2        9,327.5        9,327.5      6,799       11,856        3,575.8  fread         
      0.0           16,495         20          824.8          483.0          0        2,929          887.3  read          
      0.0            9,731          6        1,621.8          690.5          0        4,499        2,058.9  open          
      0.0            9,165          1        9,165.0        9,165.0      9,165        9,165            0.0  pipe2         
      0.0            9,059          1        9,059.0        9,059.0      9,059        9,059            0.0  connect       
      0.0            8,135          2        4,067.5        4,067.5          0        8,135        5,752.3  socket        
      0.0            7,718          4        1,929.5          489.5        231        6,508        3,058.4  fwrite        
      0.0            4,062         13          312.5          345.0          0          918          237.0  dup           
      0.0            3,228         18          179.3          169.5          0          367           71.2  fflush        
      0.0                0          1            0.0            0.0          0            0            0.0  fopen64       
      0.0                0          1            0.0            0.0          0            0            0.0  bind          
      0.0                0          1            0.0            0.0          0            0            0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)  Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  --------  --------  -----------  ------------  ----------------------
     96.9      106,912,410          3  35,637,470.0   6,584.0     4,697  106,901,129  61,716,139.1  cudaMalloc            
      2.9        3,201,638      1,010       3,169.9   2,922.0         0      138,218       5,124.5  cudaLaunchKernel      
      0.2          179,454          3      59,818.0   9,850.0     2,925      166,679      92,609.1  cudaFree              
      0.0           48,161          3      16,053.7  18,247.0     6,098       23,816       9,060.3  cudaMemcpy            
      0.0            6,041          2       3,020.5   3,020.5         0        6,041       4,271.6  cudaDeviceSynchronize 
      0.0            2,512          1       2,512.0   2,512.0     2,512        2,512           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                          Name                         
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------------------------
    100.0        1,831,281      1,010   1,813.1   1,823.0     1,791     4,705         93.3  vectorAdd(const float *, const float *, float *, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------
     51.0            2,432      2   1,216.0   1,216.0     1,024     1,408        271.5  [CUDA memcpy HtoD]
     49.0            2,336      1   2,336.0   2,336.0     2,336     2,336          0.0  [CUDA memcpy DtoH]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
      0.000      2     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy DtoH]


#### RESULTS BUT NOW ADDED cudaFree(0) to do context creation way before the mallocs.

### cudaMalloc/Free/Memcpy are minimal. The real killer is cudaLaunchKernel which takes 3,145,161 and is called on every kernel launch
### this does the process of packaging args, validating dims, and adding command to gpu command queue, and it is what is resposnible for
### most of the overhead here.

paperspace@psbqejul7uvv:~/nsight-systems-deep-dive$ sudo $(which nsys) profile   -t cuda,nvtx,osrt   --gpu-metrics-device=all   --cpuctxsw=none   --sample=cpu   --force-overwrite true   --stats=true   -o vector_add_profile_v4   ./bench_a100
GPU 0: General Metrics for NVIDIA GA100 (any frequency)
Launching kernel with 1 blocks...
Done.
Generating '/tmp/nsys-report-a98b.qdstrm'
[1/8] [========================100%] vector_add_profile_v4.nsys-rep
[2/8] [========================100%] vector_add_profile_v4.sqlite
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style      Range    
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  ------------
     95.8        3,414,647          1  3,414,647.0  3,414,647.0  3,414,647  3,414,647          0.0  PushPop  Profile_Loop
      4.2          151,184          1    151,184.0    151,184.0    151,184    151,184          0.0  PushPop  Warmup      

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)     Min (ns)    Max (ns)     StdDev (ns)        Name     
 --------  ---------------  ---------  -------------  -------------  ---------  -----------  -------------  --------------
     68.1      350,446,170          2  175,223,085.0  175,223,085.0  2,700,963  347,745,207  243,983,124.7  sem_wait      
     19.5      100,167,455         17    5,892,203.2    1,550,771.0          0   38,094,297   11,179,534.0  poll          
     11.4       58,466,391        736       79,438.0        9,563.0          0   26,959,892    1,018,930.2  ioctl         
      0.5        2,552,486         51       50,048.7       11,052.0      8,845    1,668,793      231,656.1  mmap64        
      0.2        1,148,944         15       76,596.3       52,719.0     40,364      339,281       75,120.4  sem_timedwait 
      0.2        1,016,495         74       13,736.4       12,850.5      2,036       28,094        4,607.4  open64        
      0.1          301,452          4       75,363.0       77,304.0     49,869       96,975       20,551.5  pthread_create
      0.0          170,222         14       12,158.7        7,684.0      2,085       53,227       13,697.2  mmap          
      0.0          140,686         36        3,907.9        2,916.0      1,017       14,073        2,932.5  fopen         
      0.0          118,038         16        7,377.4        8,327.0      1,065        9,692        2,288.7  write         
      0.0           79,441         64        1,241.3          170.0        158       42,026        5,512.4  fgets         
      0.0           42,312          5        8,462.4       11,456.0          0       11,761        5,059.8  munmap        
      0.0           39,376         81          486.1          441.0        161        1,218          172.2  fcntl         
      0.0           36,976         29        1,275.0        1,184.0        736        2,061          335.4  fclose        
      0.0           26,911          6        4,485.2        4,019.5      1,421        9,539        2,931.7  open          
      0.0           25,183          2       12,591.5       12,591.5      7,727       17,456        6,879.4  putc          
      0.0           20,565          2       10,282.5       10,282.5      8,648       11,917        2,311.5  fread         
      0.0           18,050         20          902.5          597.0          0        3,395        1,029.9  read          
      0.0           17,938          2        8,969.0        8,969.0      5,622       12,316        4,733.4  socket        
      0.0           13,052          1       13,052.0       13,052.0     13,052       13,052            0.0  connect       
      0.0            8,803          4        2,200.8          463.0        226        7,651        3,637.1  fwrite        
      0.0            8,091          1        8,091.0        8,091.0      8,091        8,091            0.0  pipe2         
      0.0            4,871         13          374.7          346.0        269          818          140.2  dup           
      0.0            3,466          1        3,466.0        3,466.0      3,466        3,466            0.0  fopen64       
      0.0            3,016         18          167.6          157.5          0          399           71.4  fflush        
      0.0            1,846          1        1,846.0        1,846.0      1,846        1,846            0.0  bind          
      0.0            1,525          1        1,525.0        1,525.0      1,525        1,525            0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)  Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  --------  --------  -----------  ------------  ----------------------
     97.0      109,787,937          4  27,446,984.3  57,682.0     2,607  109,669,966  54,815,341.3  cudaFree              
      2.8        3,145,161      1,010       3,114.0   2,894.0         0      104,466       3,286.8  cudaLaunchKernel      
      0.2          177,866          3      59,288.7   6,880.0     4,461      166,525      92,877.3  cudaMalloc            
      0.0           46,996          3      15,665.3  18,386.0     6,421       22,189       8,228.5  cudaMemcpy            
      0.0           11,852          2       5,926.0   5,926.0     5,861        5,991          91.9  cudaDeviceSynchronize 
      0.0            1,980          1       1,980.0   1,980.0     1,980        1,980           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                          Name                         
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------------------------
    100.0        1,830,493      1,010   1,812.4   1,824.0     1,792     2,752         35.8  vectorAdd(const float *, const float *, float *, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------
     53.1            2,465      2   1,232.5   1,232.5     1,025     1,440        293.4  [CUDA memcpy HtoD]
     46.9            2,176      1   2,176.0   2,176.0     2,176     2,176          0.0  [CUDA memcpy DtoH]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
      0.000      2     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy DtoH]

Generated:
    /home/paperspace/nsight-systems-deep-dive/vector_add_profile_v4.nsys-rep
    /home/paperspace/nsight-systems-deep-dive/vector_add_profile_v4.sqlite
paperspace@psbqejul7uvv:~/nsight-systems-deep-dive$ 
