#include<cstdio>
#include<string>
#include<sstream>
#include<iostream>
#include<unistd.h>

void make_directory()
{
    if(!fork())
    {
        char *arg_list[] = { "mkdir", "/sdcard/PerfData", NULL };
        execvp("mkdir", arg_list);
    }
}

void do_monitor_process(int pid, int threadsBig, int threadsSmall, char* eventsToProfile, char* modelName)
{
    printf("Starting simpleperf...\n");
    char outputFileStr[100] = "", pidStr[20] = "";
    snprintf(outputFileStr, 100, "/sdcard/PerfData/perf_data_%s_%dB_%dS.csv", modelName, threadsBig, threadsSmall);
    snprintf(pidStr, 20, "%d", pid);
    char *arg_list[] = { "simpleperf", "stat", "--csv", "--per-thread", "-e", eventsToProfile, "-o", outputFileStr, "-p", pidStr, NULL };
    execvp("simpleperf", arg_list);
}

void do_pin_thread(int tid, int mask)
{
    printf("Pinning thread %d to CPU mask %x\n", tid, mask);
    if(!fork())
    {
        char maskStr[20] = "", tidStr[20] = "";
        snprintf(maskStr, 20, "%x", mask);
        snprintf(tidStr, 20, "%d", tid);
        char *arg_list[] = { "taskset", "-p", maskStr, tidStr, NULL };
        execvp("taskset", arg_list);
    }
    return;
}

int count_1_bits(int x)
{
    int num = 0;
    for(int i = 0; i < 32; i++)
    {
        num += x & 0x1;
        x = x >> 1;
    }
    return num;
}
//This function assumes there are 8 cores in total
bool generate_available_masks(int &bigMask, int &smallMask, int topo, int threadsBig, int threadsSmall)
{
    if(topo & (~0xff))
    {
        printf("This topology contains more than 8 cores, which is not supported!\n");
        return false;
    }
    int bigCoreAmt   = count_1_bits(topo);
    int smallCoreAmt = 8 - bigCoreAmt;
    printf("This CPU topology contains %d big cores and %d small cores.\n", bigCoreAmt, smallCoreAmt);

    if(threadsBig > bigCoreAmt || threadsBig < 0 || threadsSmall > smallCoreAmt || threadsSmall < 0 || threadsBig + threadsSmall == 0)
    {
        printf("Invalid thread specification!\n");
        return false;
    }

    //do the actual work
    bigMask = smallMask = 0;
    while(count_1_bits((bigMask)&topo) != threadsBig)
        bigMask++;
    while(count_1_bits((smallMask) & (~topo)) != threadsSmall)
        smallMask++;
    return true;
}

//      /data/local/tmp/perf_assist graph_mobilenet_50x_aarch64 cache-misses,L1-dcache-load-misses,raw-l2d-cache-lmiss-rd,raw-l2d-cache-inval,raw-l3d-cache-lmiss-rd,cpu-migrations,cpu-cycles f0 1 3
//      cat /sdcard/PerfData/
int main(int argc, char **argv)
{
    if(argc != 6)
    {
        printf("Wrong number of arguments! %d\n", argc);
        return 1;
    }

    unsigned int      x;
    std::stringstream ss;
    ss << std::hex << argv[3];
    ss >> x;

    int   topo         = static_cast<int>(x);
    int   threadsBig   = atoi(argv[4]);
    int   threadsSmall = atoi(argv[5]);
    int   bigMask = 0, smallMask = 0;
    char *modelToRun      = argv[1];
    char *eventsToProfile = argv[2];

    if(!generate_available_masks(bigMask, smallMask, topo, threadsBig, threadsSmall))
    {
        printf("Error generating masks!\n");
        return 1;
    }

    printf("Masks generated. Results below:\nTopology:%x\n# of big core threads:%d\n# of small core threads:%d\nBig core mask:%x\nSmall core mask:%x\n", topo, threadsBig, threadsSmall, bigMask, smallMask);


    
    int pid;
    if((pid = fork()) == 0)
    {
        //child process, used to run NN
        printf("Launching child process...\n");
        char pathStr[200] = "", threadStr[200] = "";
        snprintf(pathStr, 200, "/data/local/tmp/%s", modelToRun);
        snprintf(threadStr, 200, "--threads=%d", threadsBig + threadsSmall);
        char *arg_list[] = { pathStr , threadStr, NULL };
        execvp(pathStr, arg_list);
    }
    else
    {
        //parent process, used to pin child threads to their respective CPUs
        printf("Child PID == %d, fetching child thread IDs...\n", pid);
        char cmdStr[200] = "";
        snprintf(cmdStr, 200, "ps -eTf -p %d", pid);
        bool ok = false;
        int  neededRows   = threadsBig + threadsSmall + 1;
        int  TIDlist[10];
        while(!ok)
        {
            FILE *FileOpen;
            char  lines[200][200];
            int   curRows = 0;
            FileOpen = popen(cmdStr, "r");
            while(fgets(lines[curRows], sizeof lines[0], FileOpen))
            {
                curRows++;
            }
            if(curRows==neededRows)
            {
                std::stringstream ss;
                std::string str;
                for(int row = 1; row<neededRows; row++)
                {
                    ss << lines[row];
                    int col = 0;
                    while(std::getline(ss, str, '\040'))
                    {
                        if(str.length() != 0)
                        {
                            //valid token
                            col++;
                            if(col==3)
                            {
                                TIDlist[row - 1] = atoi(str.c_str());
                            }
                            printf(" || %s", str.c_str());
                        }
                    }
                    ss.clear();
                }
                ok = true;
            }
            pclose(FileOpen);
            //sleep(1);
        }
        printf("Finished fetching thread IDs:");
        for(int i=0; i<threadsBig+threadsSmall;i++)
        {
            printf(" %d", TIDlist[i]);        
        }
        printf("\n");
        
        //pin to small cores first
        int curMask = 1, pinned = 0;
        while(pinned < threadsSmall)
        {
            if(curMask & smallMask)
            {
                do_pin_thread(TIDlist[pinned++], curMask);
            }
            curMask = curMask << 1;
        }
        //and then big cores
        curMask = 1;
        while(pinned < threadsSmall + threadsBig)
        {
            if(curMask & bigMask)
            {
                do_pin_thread(TIDlist[pinned++], curMask);
            }
            curMask = curMask << 1;
        }

        //finally monitor the child thread using simpleperf
        make_directory();
        do_monitor_process(pid, threadsBig, threadsSmall, eventsToProfile, modelToRun);
    }
}