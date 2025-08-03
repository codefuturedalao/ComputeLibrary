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
#ifndef GRAPH_MULTINN_H
#define GRAPH_MULTINN_H

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include <time.h>

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

// Include network class definitions
/** Example demonstrating how to implement MobileNet's network using the Compute Library's graph API */
class GraphMobilenetExample : public Example
{
public:
    GraphMobilenetExample() : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetV1")
    {
        // Add model id option
        model_id_opt = cmd_parser.add_option<SimpleOption<int>>("model-id", 0);
        model_id_opt->set_help("Mobilenet model id (0: 1.0_224, else: 0.75_160");
    }
    GraphMobilenetExample(const GraphMobilenetExample &)            = delete;
    GraphMobilenetExample &operator=(const GraphMobilenetExample &) = delete;
    ~GraphMobilenetExample() override                               = default;
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if (common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get model parameters
        int model_id = model_id_opt->value();

        // Create input descriptor
        unsigned int spatial_size = (model_id == 0 || common_params.data_type == DataType::QASYMM8) ? 224 : 160;

        // Create input descriptor
        const TensorShape tensor_shape =
            permute_shape(TensorShape(spatial_size, spatial_size, 3U, common_params.batches), DataLayout::NCHW,
                          common_params.data_layout);
        TensorDescriptor input_descriptor =
            TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set graph hints
        graph << common_params.target << common_params.fast_math_hint;

        // Create core graph
        if (arm_compute::is_data_type_float(common_params.data_type))
        {
            create_graph_float(input_descriptor, model_id);
        }
        else
        {
            create_graph_qasymm(input_descriptor);
        }

        // Create common tail
        graph << ReshapeLayer(TensorShape(1001U)).set_name("Reshape") << SoftmaxLayer().set_name("Softmax")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        /*
        config.num_threads = common_params.threads;
        config.threads_affinity = common_params.threads_affinity;
        config.scheduler = (Scheduler::Type)common_params.scheduler;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;
        */
        config = consume_common_graph_config(common_params);

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }
    std::string name() override {  
        return "mobilenet";
    }
private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    SimpleOption<int> *model_id_opt{nullptr};
    CommonGraphParams  common_params;
    Stream             graph;

    void create_graph_float(TensorDescriptor &input_descriptor, int model_id)
    {
        float       depth_scale = (model_id == 0) ? 1.f : 0.75;
        std::string model_path =
            (model_id == 0) ? "/cnn_data/mobilenet_v1_1_224_model/" : "/cnn_data/mobilenet_v1_075_160_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if (!data_path.empty())
        {
            data_path += model_path;
        }

        graph << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
              << ConvolutionLayer(3U, 3U, 32U * depth_scale,
                                  get_weights_accessor(data_path, "Conv2d_0_weights.npy", DataLayout::NCHW),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
                     .set_name("Conv2d_0")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_variance.npy"),
                                         get_weights_accessor(data_path, "Conv2d_0_BatchNorm_gamma.npy"),
                                         get_weights_accessor(data_path, "Conv2d_0_BatchNorm_beta.npy"), 0.001f)
                     .set_name("Conv2d_0/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
                     .set_name("Conv2d_0/Relu6");
        graph << get_dwsc_node_float(data_path, "Conv2d_1", 64 * depth_scale, PadStrideInfo(1, 1, 1, 1),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_2", 128 * depth_scale,
                                     PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_3", 128 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_4", 256 * depth_scale,
                                     PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_5", 256 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_6", 512 * depth_scale,
                                     PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_7", 512 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_8", 512 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_9", 512 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_10", 512 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_11", 512 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_12", 1024 * depth_scale,
                                     PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_13", 1024 * depth_scale,
                                     PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL),
                                     PadStrideInfo(1, 1, 0, 0));
        graph
            << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, common_params.data_layout)).set_name("Logits/AvgPool_1a")
            << ConvolutionLayer(
                   1U, 1U, 1001U, get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy", DataLayout::NCHW),
                   get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"), PadStrideInfo(1, 1, 0, 0))
                   .set_name("Logits/Conv2d_1c_1x1");
    }

    void create_graph_qasymm(TensorDescriptor &input_descriptor)
    {
        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if (!data_path.empty())
        {
            data_path += "/cnn_data/mobilenet_qasymm8_model/";
        }

        // Quantization info taken from the AndroidNN QASYMM8 MobileNet example
        const QuantizationInfo in_quant_info = QuantizationInfo(0.0078125f, 128);

        const std::vector<QuantizationInfo> conv_weights_quant_info = {
            QuantizationInfo(0.02182667888700962f, 151), // conv0
            QuantizationInfo(0.004986600950360298f, 74)  // conv14
        };
        const std::vector<QuantizationInfo> conv_out_quant_info = {
            QuantizationInfo(0.023528477177023888f, 0), // conv0
            QuantizationInfo(0.16609922051429749f, 66)  // conv14
        };

        const std::vector<QuantizationInfo> depth_weights_quant_info = {
            QuantizationInfo(0.29219913482666016f, 110),  // dwsc1
            QuantizationInfo(0.40277284383773804f, 130),  // dwsc2
            QuantizationInfo(0.06053730100393295f, 160),  // dwsc3
            QuantizationInfo(0.01675807684659958f, 123),  // dwsc4
            QuantizationInfo(0.04105526953935623f, 129),  // dwsc5
            QuantizationInfo(0.013460792601108551f, 122), // dwsc6
            QuantizationInfo(0.036934755742549896f, 132), // dwsc7
            QuantizationInfo(0.042609862983226776f, 94),  // dwsc8
            QuantizationInfo(0.028358859941363335f, 127), // dwsc9
            QuantizationInfo(0.024329448118805885f, 134), // dwsc10
            QuantizationInfo(0.019366811960935593f, 106), // dwsc11
            QuantizationInfo(0.007835594937205315f, 126), // dwsc12
            QuantizationInfo(0.12616927921772003f, 211)   // dwsc13
        };

        const std::vector<QuantizationInfo> point_weights_quant_info = {
            QuantizationInfo(0.030420949682593346f, 121), // dwsc1
            QuantizationInfo(0.015148180536925793f, 104), // dwsc2
            QuantizationInfo(0.013755458407104015f, 94),  // dwsc3
            QuantizationInfo(0.007601846940815449f, 151), // dwsc4
            QuantizationInfo(0.006431614048779011f, 122), // dwsc5
            QuantizationInfo(0.00917122047394514f, 109),  // dwsc6
            QuantizationInfo(0.005300046876072884f, 140), // dwsc7
            QuantizationInfo(0.0049632852897048f, 127),   // dwsc8
            QuantizationInfo(0.007770895957946777f, 89),  // dwsc9
            QuantizationInfo(0.009658650495111942f, 99),  // dwsc10
            QuantizationInfo(0.005446993745863438f, 153), // dwsc11
            QuantizationInfo(0.00817922968417406f, 130),  // dwsc12
            QuantizationInfo(0.018048152327537537f, 95)   // dwsc13
        };

        graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_input_accessor(common_params, nullptr, false))
              << ConvolutionLayer(3U, 3U, 32U, get_weights_accessor(data_path, "Conv2d_0_weights.npy"),
                                  get_weights_accessor(data_path, "Conv2d_0_bias.npy"),
                                  PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), 1,
                                  conv_weights_quant_info.at(0), conv_out_quant_info.at(0))
                     .set_name("Conv2d_0")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f))
                     .set_name("Conv2d_0/Relu6");
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_1", 64U, PadStrideInfo(1U, 1U, 1U, 1U),
                                      PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(0),
                                      point_weights_quant_info.at(0));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_2", 128U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(1), point_weights_quant_info.at(1));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_3", 128U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(2), point_weights_quant_info.at(2));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_4", 256U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(3), point_weights_quant_info.at(3));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_5", 256U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(4), point_weights_quant_info.at(4));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_6", 512U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(5), point_weights_quant_info.at(5));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_7", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(6), point_weights_quant_info.at(6));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_8", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(7), point_weights_quant_info.at(7));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_9", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(8), point_weights_quant_info.at(8));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_10", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(9), point_weights_quant_info.at(9));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_11", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(10), point_weights_quant_info.at(10));
        graph << get_dwsc_node_qasymm(
            data_path, "Conv2d_12", 1024U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
            PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(11), point_weights_quant_info.at(11));
        graph
            << get_dwsc_node_qasymm(
                   data_path, "Conv2d_13", 1024U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR),
                   PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(12), point_weights_quant_info.at(12))
            << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, common_params.data_layout)).set_name("Logits/AvgPool_1a")
            << ConvolutionLayer(1U, 1U, 1001U, get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy"),
                                get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_bias.npy"),
                                PadStrideInfo(1U, 1U, 0U, 0U), 1, conv_weights_quant_info.at(1),
                                conv_out_quant_info.at(1))
                   .set_name("Logits/Conv2d_1c_1x1");
    }

    ConcatLayer get_dwsc_node_float(const std::string &data_path,
                                    std::string      &&param_path,
                                    unsigned int       conv_filt,
                                    PadStrideInfo      dwc_pad_stride_info,
                                    PadStrideInfo      conv_pad_stride_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(graph);
        sg << DepthwiseConvolutionLayer(
                  3U, 3U,
                  get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy", DataLayout::NCHW),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), dwc_pad_stride_info)
                  .set_name(total_path + "depthwise/depthwise")
           << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"), 0.001f)
                  .set_name(total_path + "depthwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
                  .set_name(total_path + "depthwise/Relu6")
           << ConvolutionLayer(1U, 1U, conv_filt,
                               get_weights_accessor(data_path, total_path + "pointwise_weights.npy", DataLayout::NCHW),
                               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), conv_pad_stride_info)
                  .set_name(total_path + "pointwise/Conv2D")
           << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_beta.npy"), 0.001f)
                  .set_name(total_path + "pointwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
                  .set_name(total_path + "pointwise/Relu6");

        return ConcatLayer(std::move(sg));
    }

    ConcatLayer get_dwsc_node_qasymm(const std::string &data_path,
                                     std::string      &&param_path,
                                     const unsigned int conv_filt,
                                     PadStrideInfo      dwc_pad_stride_info,
                                     PadStrideInfo      conv_pad_stride_info,
                                     QuantizationInfo   depth_weights_quant_info,
                                     QuantizationInfo   point_weights_quant_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(graph);

        sg << DepthwiseConvolutionLayer(3U, 3U, get_weights_accessor(data_path, total_path + "depthwise_weights.npy"),
                                        get_weights_accessor(data_path, total_path + "depthwise_bias.npy"),
                                        dwc_pad_stride_info, 1, std::move(depth_weights_quant_info))
                  .set_name(total_path + "depthwise/depthwise")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f))
                  .set_name(total_path + "depthwise/Relu6")
           << ConvolutionLayer(1U, 1U, conv_filt, get_weights_accessor(data_path, total_path + "pointwise_weights.npy"),
                               get_weights_accessor(data_path, total_path + "pointwise_bias.npy"), conv_pad_stride_info,
                               1, std::move(point_weights_quant_info))
                  .set_name(total_path + "pointwise/Conv2D")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f))
                  .set_name(total_path + "pointwise/Relu6");

        return ConcatLayer(std::move(sg));
    }
};

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API */
class GraphSqueezenetExample : public Example
{
public:
    GraphSqueezenetExample() : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "SqueezeNetV1")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if (common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3>     mean_rgb{{122.68f, 116.67f, 104.01f}};
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape =
            permute_shape(TensorShape(224U, 224U, 3U, common_params.batches), DataLayout::NCHW, operation_layout);
        TensorDescriptor input_descriptor =
            TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                     7U, 7U, 96U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
                     PadStrideInfo(2, 2, 0, 0))
                     .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("relu_conv1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                               PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                     .set_name("pool1")
              << ConvolutionLayer(
                     1U, 1U, 16U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire2/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire2/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire2", weights_layout, 64U, 64U).set_name("fire2/concat");
        graph << ConvolutionLayer(
                     1U, 1U, 16U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire3/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire3/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire3", weights_layout, 64U, 64U).set_name("fire3/concat");
        graph << ConvolutionLayer(
                     1U, 1U, 32U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire4/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire4/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire4", weights_layout, 128U, 128U).set_name("fire4/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                               PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                     .set_name("pool4")
              << ConvolutionLayer(
                     1U, 1U, 32U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire5/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire5/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire5", weights_layout, 128U, 128U).set_name("fire5/concat");
        graph << ConvolutionLayer(
                     1U, 1U, 48U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire6/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire6/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire6", weights_layout, 192U, 192U).set_name("fire6/concat");
        graph << ConvolutionLayer(
                     1U, 1U, 48U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire7/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire7/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire7", weights_layout, 192U, 192U).set_name("fire7/concat");
        graph << ConvolutionLayer(
                     1U, 1U, 64U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire8/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire8/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire8", weights_layout, 256U, 256U).set_name("fire8/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                               PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                     .set_name("pool8")
              << ConvolutionLayer(
                     1U, 1U, 64U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("fire9/squeeze1x1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("fire9/relu_squeeze1x1");
        graph << get_expand_fire_node(data_path, "fire9", weights_layout, 256U, 256U).set_name("fire9/concat");
        graph << ConvolutionLayer(
                     1U, 1U, 1000U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("conv10")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("relu_conv10")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("pool10")
              << FlattenLayer().set_name("flatten") << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;


       config = consume_common_graph_config(common_params);

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }
    std::string name() override {
        return "squeezenet";
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    ConcatLayer get_expand_fire_node(const std::string &data_path,
                                     std::string      &&param_path,
                                     DataLayout         weights_layout,
                                     unsigned int       expand1_filt,
                                     unsigned int       expand3_filt)
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, expand1_filt,
                                get_weights_accessor(data_path, total_path + "expand1x1_w.npy", weights_layout),
                                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                                PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/expand1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/relu_expand1x1");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(3U, 3U, expand3_filt,
                                get_weights_accessor(data_path, total_path + "expand3x3_w.npy", weights_layout),
                                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                                PadStrideInfo(1, 1, 1, 1))
                   .set_name(param_path + "/expand3x3")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/relu_expand3x3");

        return ConcatLayer(std::move(i_a), std::move(i_b));
    }
};

/** Example demonstrating how to implement ResNetV1_50 network using the Compute Library's graph API */
class GraphResNetV1_50Example : public Example
{
public:
    GraphResNetV1_50Example() : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "ResNetV1_50")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if (common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3>     mean_rgb{{122.68f, 116.67f, 104.01f}};
        std::unique_ptr<IPreprocessor> preprocessor =
            std::make_unique<CaffePreproccessor>(mean_rgb, false /* Do not convert to BGR */);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape =
            permute_shape(TensorShape(224U, 224U, 3U, common_params.batches), DataLayout::NCHW, operation_layout);
        TensorDescriptor input_descriptor =
            TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor),
                                                                 false /* Do not convert to BGR */))
              << ConvolutionLayer(
                     7U, 7U, 64U,
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 3, 3))
                     .set_name("conv1/convolution")
              << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_variance.npy"),
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_beta.npy"),
                     0.0000100099996416f)
                     .set_name("conv1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("conv1/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                               PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)))
                     .set_name("pool1/MaxPool");

        add_residual_block(data_path, "block1", weights_layout, 64, 3, 2);
        add_residual_block(data_path, "block2", weights_layout, 128, 4, 2);
        add_residual_block(data_path, "block3", weights_layout, 256, 6, 2);
        add_residual_block(data_path, "block4", weights_layout, 512, 3, 1);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("pool5")
              << ConvolutionLayer(
                     1U, 1U, 1000U,
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_weights.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_biases.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("logits/convolution")
              << FlattenLayer().set_name("predictions/Reshape") << SoftmaxLayer().set_name("predictions/Softmax")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        // config.num_threads        = common_params.threads;
        // config.use_tuner          = common_params.enable_tuner;
        // config.tuner_mode         = common_params.tuner_mode;
        // config.tuner_file         = common_params.tuner_file;
        // config.mlgo_file          = common_params.mlgo_file;
        // config.use_synthetic_type = arm_compute::is_data_type_quantized(common_params.data_type);
        // config.synthetic_type     = common_params.data_type;
        config = consume_common_graph_config(common_params);

        graph.finalize(common_params.target, config);

        return true;
    }

    void do_run() override
    {
        // Run graph
        graph.run();
    }
    std::string name() override {
        return "resnet50";
    }
private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    void add_residual_block(const std::string &data_path,
                            const std::string &name,
                            DataLayout         weights_layout,
                            unsigned int       base_depth,
                            unsigned int       num_units,
                            unsigned int       stride)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            std::stringstream unit_path_ss;
            unit_path_ss << "/cnn_data/resnet50_model/" << name << "_unit_" << (i + 1) << "_bottleneck_v1_";
            std::stringstream unit_name_ss;
            unit_name_ss << name << "/unit" << (i + 1) << "/bottleneck_v1/";

            std::string unit_path = unit_path_ss.str();
            std::string unit_name = unit_name_ss.str();

            unsigned int middle_stride = 1;

            if (i == (num_units - 1))
            {
                middle_stride = stride;
            }

            SubStream right(graph);
            right << ConvolutionLayer(1U, 1U, base_depth,
                                      get_weights_accessor(data_path, unit_path + "conv1_weights.npy", weights_layout),
                                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                      PadStrideInfo(1, 1, 0, 0))
                         .set_name(unit_name + "conv1/convolution")
                  << BatchNormalizationLayer(
                         get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_mean.npy"),
                         get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_variance.npy"),
                         get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_gamma.npy"),
                         get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_beta.npy"), 0.0000100099996416f)
                         .set_name(unit_name + "conv1/BatchNorm")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                         .set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(3U, 3U, base_depth,
                                      get_weights_accessor(data_path, unit_path + "conv2_weights.npy", weights_layout),
                                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                      PadStrideInfo(middle_stride, middle_stride, 1, 1))
                         .set_name(unit_name + "conv2/convolution")
                  << BatchNormalizationLayer(
                         get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_mean.npy"),
                         get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_variance.npy"),
                         get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_gamma.npy"),
                         get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_beta.npy"), 0.0000100099996416f)
                         .set_name(unit_name + "conv2/BatchNorm")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                         .set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(1U, 1U, base_depth * 4,
                                      get_weights_accessor(data_path, unit_path + "conv3_weights.npy", weights_layout),
                                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                      PadStrideInfo(1, 1, 0, 0))
                         .set_name(unit_name + "conv3/convolution")
                  << BatchNormalizationLayer(
                         get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_mean.npy"),
                         get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_variance.npy"),
                         get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_gamma.npy"),
                         get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_beta.npy"), 0.0000100099996416f)
                         .set_name(unit_name + "conv2/BatchNorm");

            if (i == 0)
            {
                SubStream left(graph);
                left << ConvolutionLayer(
                            1U, 1U, base_depth * 4,
                            get_weights_accessor(data_path, unit_path + "shortcut_weights.npy", weights_layout),
                            std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                            .set_name(unit_name + "shortcut/convolution")
                     << BatchNormalizationLayer(
                            get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_mean.npy"),
                            get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_variance.npy"),
                            get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_gamma.npy"),
                            get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_beta.npy"),
                            0.0000100099996416f)
                            .set_name(unit_name + "shortcut/BatchNorm");

                graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add)
                             .set_name(unit_name + "add");
            }
            else if (middle_stride > 1)
            {
                SubStream left(graph);
                left << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 1, common_params.data_layout,
                                                      PadStrideInfo(middle_stride, middle_stride, 0, 0), true))
                            .set_name(unit_name + "shortcut/MaxPool");

                graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add)
                             .set_name(unit_name + "add");
            }
            else
            {
                SubStream left(graph);
                graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add)
                             .set_name(unit_name + "add");
            }

            graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                         .set_name(unit_name + "Relu");
        }
    }
};

/** Example demonstrating how to implement InceptionV3's network using the Compute Library's graph API */
class InceptionV3Example : public Example
{
public:
    InceptionV3Example() : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "InceptionV3")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if (common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<TFPreproccessor>();

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape =
            permute_shape(TensorShape(299U, 299U, 3U, common_params.batches), DataLayout::NCHW, operation_layout);
        TensorDescriptor input_descriptor =
            TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph
            << common_params.target << common_params.fast_math_hint
            << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
            << ConvolutionLayer(3U, 3U, 32U,
                                get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_weights.npy",
                                                     weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(2, 2, 0, 0))
                   .set_name("Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   nullptr,
                   get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name("Conv2d_1a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Conv2d_1a_3x3/Relu")
            << ConvolutionLayer(3U, 3U, 32U,
                                get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_weights.npy",
                                                     weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
                   .set_name("Conv2d_2a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                   nullptr,
                   get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name("Conv2d_2a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Conv2d_2a_3x3/Relu")

            << ConvolutionLayer(3U, 3U, 64U,
                                get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_weights.npy",
                                                     weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 1, 1))
                   .set_name("Conv2d_2b_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                   nullptr,
                   get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name("Conv2d_2b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Conv2d_2b_3x3/Relu")

            << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                             PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                   .set_name("MaxPool_3a_3x3/MaxPool")

            << ConvolutionLayer(1U, 1U, 80U,
                                get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_weights.npy",
                                                     weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
                   .set_name("Conv2d_3b_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_BatchNorm_moving_variance.npy"),
                   nullptr,
                   get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name("Conv2d_3b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Conv2d_3b_1x1/Relu")

            << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_weights.npy",
                                                     weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
                   .set_name("Conv2d_4a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path,
                                        "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_BatchNorm_moving_variance.npy"),
                   nullptr,
                   get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name("Conv2d_4a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Conv2d_4a_3x3/Relu")

            << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                             PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                   .set_name("MaxPool_5a_3x3/MaxPool");

        graph << get_inception_node_A(data_path, "Mixed_5b", weights_layout, 64U, std::make_tuple(48U, 64U),
                                      std::make_tuple(64U, 96U, 96U), 32U)
                     .set_name("Mixed_5b/concat");
        graph << get_inception_node_A(data_path, "Mixed_5c", weights_layout, 64U, std::make_tuple(48U, 64U),
                                      std::make_tuple(64U, 96U, 96U), 64U, true)
                     .set_name("Mixed_5c/concat");
        graph << get_inception_node_A(data_path, "Mixed_5d", weights_layout, 64U, std::make_tuple(48U, 64U),
                                      std::make_tuple(64U, 96U, 96U), 64U)
                     .set_name("Mixed_5d/concat");

        graph << get_inception_node_B(data_path, "Mixed_6a", weights_layout, 384U, std::make_tuple(64U, 96U, 96U))
                     .set_name("Mixed_6a/concat");

        graph << get_inception_node_C(data_path, "Mixed_6b", weights_layout, 192U, std::make_tuple(128U, 128U, 192U),
                                      std::make_tuple(128U, 128U, 128U, 128U, 192U), 192U)
                     .set_name("Mixed_6b/concat");
        graph << get_inception_node_C(data_path, "Mixed_6c", weights_layout, 192U, std::make_tuple(160U, 160U, 192U),
                                      std::make_tuple(160U, 160U, 160U, 160U, 192U), 192U)
                     .set_name("Mixed_6c/concat");
        graph << get_inception_node_C(data_path, "Mixed_6d", weights_layout, 192U, std::make_tuple(160U, 160U, 192U),
                                      std::make_tuple(160U, 160U, 160U, 160U, 192U), 192U)
                     .set_name("Mixed_6d/concat");
        graph << get_inception_node_C(data_path, "Mixed_6e", weights_layout, 192U, std::make_tuple(192U, 192U, 192U),
                                      std::make_tuple(192U, 192U, 192U, 192U, 192U), 192U)
                     .set_name("Mixed_6e/concat");

        graph << get_inception_node_D(data_path, "Mixed_7a", weights_layout, std::make_tuple(192U, 320U),
                                      std::make_tuple(192U, 192U, 192U, 192U))
                     .set_name("Mixed_7a/concat");

        graph << get_inception_node_E(data_path, "Mixed_7b", weights_layout, 320U, std::make_tuple(384U, 384U, 384U),
                                      std::make_tuple(448U, 384U, 384U, 384U), 192U)
                     .set_name("Mixed_7b/concat");
        graph << get_inception_node_E(data_path, "Mixed_7c", weights_layout, 320U, std::make_tuple(384U, 384U, 384U),
                                      std::make_tuple(448U, 384U, 384U, 384U), 192U, true)
                     .set_name("Mixed_7c/concat");

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 8, operation_layout,
                                               PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL)))
                     .set_name("Logits/AvgPool_1a_8x8/AvgPool")
              << ConvolutionLayer(
                     1U, 1U, 1001U,
                     get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Logits_Conv2d_1c_1x1_weights.npy",
                                          weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Logits_Conv2d_1c_1x1_biases.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("Logits/Conv2d_1c_1x1/convolution")
              << ReshapeLayer(TensorShape(1001U)).set_name("Predictions/Reshape")
              << SoftmaxLayer().set_name("Predictions/Softmax") << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
       //  config.num_threads        = common_params.threads;
       //  config.use_tuner          = common_params.enable_tuner;
       //  config.tuner_mode         = common_params.tuner_mode;
       //  config.tuner_file         = common_params.tuner_file;
       //  config.mlgo_file          = common_params.mlgo_file;
       //  config.use_synthetic_type = arm_compute::is_data_type_quantized(common_params.data_type);
       //  config.synthetic_type     = common_params.data_type;

       config = consume_common_graph_config(common_params);
        graph.finalize(common_params.target, config);

        return true;
    }

    void do_run() override
    {
        graph.run();
    }
    std::string name() override {
        return "inception_v3";
    }
private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

private:
    ConcatLayer get_inception_node_A(const std::string                                   &data_path,
                                     std::string                                        &&param_path,
                                     DataLayout                                           weights_layout,
                                     unsigned int                                         a_filt,
                                     std::tuple<unsigned int, unsigned int>               b_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int> c_filters,
                                     unsigned int                                         d_filt,
                                     bool                                                 is_name_different = false)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";

        // This is due to a naming issue in the tf model
        std::string conv_id0 = "_0a_";
        std::string conv_id1 = "2d_0b_";
        if (is_name_different)
        {
            conv_id0 = "_0b_";
            conv_id1 = "_1_0c_";
        }

        SubStream i_a(graph);
        i_a << ConvolutionLayer(
                   1U, 1U, a_filt,
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, std::get<0>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 + "1x1_weights.npy",
                                        weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d" + conv_id0 + "1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 +
                                                                           "1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 +
                                                                           "1x1_BatchNorm_moving_variance.npy"),
                                       nullptr,
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 +
                                                                           "1x1_BatchNorm_beta.npy"),
                                       0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d" + conv_id0 + "1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d" + conv_id0 + "1x1/Relu")
            << ConvolutionLayer(
                   5U, 5U, std::get<1>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv" + conv_id1 + "5x5_weights.npy",
                                        weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 2, 2))
                   .set_name(param_path + "/Branch_1/Conv2d" + conv_id1 + "5x5/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path,
                                        total_path + "Branch_1_Conv" + conv_id1 + "5x5_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path,
                                        total_path + "Branch_1_Conv" + conv_id1 + "5x5_BatchNorm_moving_variance.npy"),
                   nullptr,
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv" + conv_id1 + "5x5_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d" + conv_id1 + "5x5/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d" + conv_id1 + "5x5/Relu");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                   1U, 1U, std::get<0>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<1>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                   .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<2>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                   .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/BatchNorm/batcnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/Relu");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, common_params.data_layout,
                                             PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
                   .set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(
                   1U, 1U, d_filt,
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    ConcatLayer get_inception_node_B(const std::string                                   &data_path,
                                     std::string                                        &&param_path,
                                     DataLayout                                           weights_layout,
                                     unsigned int                                         a_filt,
                                     std::tuple<unsigned int, unsigned int, unsigned int> b_filters)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                   3U, 3U, a_filt,
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name(param_path + "/Branch_0/Conv2d_1a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_0/Conv2d_1a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_1a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, std::get<0>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<1>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                   .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<2>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_1a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_1a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_1a_1x1/Relu");

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, common_params.data_layout,
                                             PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                   .set_name(param_path + "/Branch_2/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c));
    }

    ConcatLayer
    get_inception_node_C(const std::string                                   &data_path,
                         std::string                                        &&param_path,
                         DataLayout                                           weights_layout,
                         unsigned int                                         a_filt,
                         std::tuple<unsigned int, unsigned int, unsigned int> b_filters,
                         std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int> c_filters,
                         unsigned int                                                                     d_filt)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                   1U, 1U, a_filt,
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, std::get<0>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   7U, 1U, std::get<1>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(
                   1U, 7U, std::get<2>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
                   .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_0c_7x1/Relu");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                   1U, 1U, std::get<0>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   1U, 7U, std::get<1>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
                   .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/Relu")
            << ConvolutionLayer(
                   7U, 1U, std::get<2>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
                   .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/Relu")
            << ConvolutionLayer(
                   1U, 7U, std::get<3>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
                   .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/Relu")
            << ConvolutionLayer(
                   7U, 1U, std::get<4>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
                   .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/Relu");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, common_params.data_layout,
                                             PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
                   .set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(
                   1U, 1U, d_filt,
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    ConcatLayer get_inception_node_D(const std::string                                                 &data_path,
                                     std::string                                                      &&param_path,
                                     DataLayout                                                         weights_layout,
                                     std::tuple<unsigned int, unsigned int>                             a_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> b_filters)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                   1U, 1U, std::get<0>(a_filters),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<1>(a_filters),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name(param_path + "/Branch_0/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_0/Conv2d_1a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_1a_3x3/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, std::get<0>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   7U, 1U, std::get<1>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(
                   1U, 7U, std::get<2>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
                   .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<3>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_1a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_1a_3x3/Relu");

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, common_params.data_layout,
                                             PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
                   .set_name(param_path + "/Branch_2/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c));
    }

    ConcatLayer get_inception_node_E(const std::string                                                 &data_path,
                                     std::string                                                      &&param_path,
                                     DataLayout                                                         weights_layout,
                                     unsigned int                                                       a_filt,
                                     std::tuple<unsigned int, unsigned int, unsigned int>               b_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> c_filters,
                                     unsigned int                                                       d_filt,
                                     bool is_name_different = false)
    {
        // This is due to a naming issue in the tf model
        std::string conv_id = "_0b_";
        if (is_name_different)
        {
            conv_id = "_0c_";
        }

        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                   1U, 1U, a_filt,
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, std::get<0>(b_filters),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu");

        SubStream i_b1(i_b);
        i_b1 << ConvolutionLayer(
                    3U, 1U, std::get<1>(b_filters),
                    get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_weights.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 0))
                    .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/convolution")
             << BatchNormalizationLayer(
                    get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                    get_weights_accessor(data_path,
                                         total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                    nullptr, get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                    0.001f)
                    .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                    .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/Relu");

        SubStream i_b2(i_b);
        i_b2 << ConvolutionLayer(
                    1U, 3U, std::get<2>(b_filters),
                    get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id + "3x1_weights.npy",
                                         weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 1))
                    .set_name(param_path + "/Branch_1/Conv2d" + conv_id + "3x1/convolution")
             << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id +
                                                                            "3x1_BatchNorm_moving_mean.npy"),
                                        get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id +
                                                                            "3x1_BatchNorm_moving_variance.npy"),
                                        nullptr,
                                        get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id +
                                                                            "3x1_BatchNorm_beta.npy"),
                                        0.001f)
                    .set_name(param_path + "/Branch_1/Conv2d" + conv_id + "3x1/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                    .set_name(param_path + "/Branch_1/Conv2d" + conv_id + "3x1/Relu");

        // Merge b1 and b2
        i_b << ConcatLayer(std::move(i_b1), std::move(i_b2)).set_name(param_path + "/Branch_1/concat");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                   1U, 1U, std::get<0>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, std::get<1>(c_filters),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                   .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/Relu");

        SubStream i_c1(i_c);
        i_c1 << ConvolutionLayer(
                    3U, 1U, std::get<2>(c_filters),
                    get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_weights.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 0))
                    .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/convolution")
             << BatchNormalizationLayer(
                    get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_mean.npy"),
                    get_weights_accessor(data_path,
                                         total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_variance.npy"),
                    nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_beta.npy"),
                    0.001f)
                    .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                    .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/Relu");

        SubStream i_c2(i_c);
        i_c2 << ConvolutionLayer(
                    1U, 3U, std::get<3>(c_filters),
                    get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_weights.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 1))
                    .set_name(param_path + "/Branch_2/Conv2d_0d_3x1/convolution")
             << BatchNormalizationLayer(
                    get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_BatchNorm_moving_mean.npy"),
                    get_weights_accessor(data_path,
                                         total_path + "Branch_2_Conv2d_0d_3x1_BatchNorm_moving_variance.npy"),
                    nullptr, get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_BatchNorm_beta.npy"),
                    0.001f)
                    .set_name(param_path + "/Branch_2/Conv2d_0d_3x1/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                    .set_name(param_path + "/Branch_2/Conv2d_0d_3x1/Relu");

        // Merge i_c1 and i_c2
        i_c << ConcatLayer(std::move(i_c1), std::move(i_c2)).set_name(param_path + "/Branch_2/concat");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, common_params.data_layout,
                                             PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
                   .set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(
                   1U, 1U, d_filt,
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                   nullptr, get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                   0.001f)
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};


#endif /* GRAPH_MULTINN_H */ 