/*
 * Copyright (c) 2024 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>

// Include network class definitions
#include "graph_multinn.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to run multiple neural networks in parallel using the Compute Library's graph API */
class GraphMultiNNExample : public Example
{
public:
    GraphMultiNNExample() : cmd_parser(), common_opts(cmd_parser), common_params(), num_networks(nullptr), num_iterations(nullptr)
    {
        // Add num_networks option
        num_networks = cmd_parser.add_option<SimpleOption<int>>("num", 1);
        num_networks->set_help("Number of networks to run in parallel (0-4)");
        
        // Add num_iterations option
        num_iterations = cmd_parser.add_option<SimpleOption<int>>("iterations", 20);
        num_iterations->set_help("Number of iterations to run (default: 20)");
    }

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Get number of networks to run
        int num = num_networks->value();
        if(num < 0 || num > 4)
        {
            std::cerr << "Number of networks must be between 0 and 4" << std::endl;
            return false;
        }

        // 当num=0时，创建所有网络
        if(num == 0)
        {
            networks.push_back(std::make_unique<GraphMobilenetExample>());
            if(!networks[0]->do_setup(argc, argv))
                return false;
            
            networks.push_back(std::make_unique<GraphSqueezenetExample>());
            if(!networks[1]->do_setup(argc, argv))
                return false;
            
            networks.push_back(std::make_unique<GraphResNetV1_50Example>());
            if(!networks[2]->do_setup(argc, argv))
                return false;
            
            networks.push_back(std::make_unique<InceptionV3Example>());
            if(!networks[3]->do_setup(argc, argv))
                return false;
        }
        else
        {
            // 原有的代码逻辑
            if(num >= 1)
            {
                networks.push_back(std::make_unique<GraphMobilenetExample>());
                if(!networks[0]->do_setup(argc, argv))
                    return false;
            }
            if(num >= 2)
            {
                networks.push_back(std::make_unique<GraphSqueezenetExample>());
                if(!networks[1]->do_setup(argc, argv))
                    return false;
            }
            if(num >= 3)
            {
                networks.push_back(std::make_unique<GraphResNetV1_50Example>());
                if(!networks[2]->do_setup(argc, argv))
                    return false;
            }
            if(num >= 4)
            {
                networks.push_back(std::make_unique<InceptionV3Example>());
                if(!networks[3]->do_setup(argc, argv))
                    return false;
            }
        }

        return true;
    }

    void do_run() override
    {
        const int iterations = num_iterations->value();
        // 为每个网络创建运行时间累加器
        std::vector<long long> total_runtimes(networks.size(), 0);
        
        for(int i = 0; i < iterations; ++i)
        {
            if(num_networks->value() == 0)
            {
                // 顺序执行所有网络
                for(size_t net_idx = 0; net_idx < networks.size(); ++net_idx)
                {
                    auto net_start = std::chrono::high_resolution_clock::now();
                    networks[net_idx]->do_run();
                    auto net_end = std::chrono::high_resolution_clock::now();
                    auto net_duration = std::chrono::duration_cast<std::chrono::microseconds>(net_end - net_start).count();
                    total_runtimes[net_idx] += net_duration;
                }
            }
            else
            {
                // 并行执行代码
                std::vector<std::thread> threads;
                std::mutex runtime_mutex;  // 用于保护total_runtimes的访问

                for(size_t net_idx = 0; net_idx < networks.size(); ++net_idx)
                {
                    threads.emplace_back([this, &total_runtimes, &runtime_mutex, net_idx]() {
                        auto net_start = std::chrono::high_resolution_clock::now();
                        networks[net_idx]->do_run();
                        auto net_end = std::chrono::high_resolution_clock::now();
                        auto net_duration = std::chrono::duration_cast<std::chrono::microseconds>(net_end - net_start).count();
                        
                        // 线程安全地更新运行时间
                        std::lock_guard<std::mutex> lock(runtime_mutex);
                        total_runtimes[net_idx] += net_duration;
                    });
                }

                for(auto &thread : threads)
                {
                    thread.join();
                }
            }
        }

        // 输出每个网络的平均运行时间
        std::cout << "\nAverage runtime for each network:" << std::endl;
        for(size_t i = 0; i < networks.size(); ++i)
        {
            double avg_runtime = static_cast<double>(total_runtimes[i]) / iterations;
            printf("%s average runtime: %.2f microseconds\n", networks[i]->name().c_str(), avg_runtime);
        }
    }

    void do_teardown() override
    {
        // Teardown all networks
        for(auto &network : networks)
        {
            network->do_teardown();
        }
    }

private:
    CommandLineParser cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams common_params;
    SimpleOption<int> *num_networks;
    SimpleOption<int> *num_iterations;
    std::vector<std::unique_ptr<Example>> networks;
};

/** Main program for running multiple neural networks in parallel
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    GraphMultiNNExample example;
    bool setup_status = example.do_setup(argc, argv);
    
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - start).count();
    std::cout << "Setup time: " << setup_duration << " microseconds" << std::endl;
    
    if(!setup_status)
    {
        return 1;
    }

    example.do_run();
    
    auto run_end = std::chrono::high_resolution_clock::now();
    auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - setup_end).count();
    std::cout << "Run time: " << run_duration << " microseconds" << std::endl;
    
    example.do_teardown();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total time: " << total_duration << " microseconds" << std::endl;
    
    return 0;
} 