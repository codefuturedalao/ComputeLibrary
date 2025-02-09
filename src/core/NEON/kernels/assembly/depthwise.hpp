/*
 * Copyright (c) 2021-2024 Arm Limited.
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

#ifndef ACL_SRC_CORE_NEON_KERNELS_ASSEMBLY_DEPTHWISE_HPP
#define ACL_SRC_CORE_NEON_KERNELS_ASSEMBLY_DEPTHWISE_HPP

#pragma once

#include "arm_gemm.hpp"
#include "arm_gemm_local.hpp"
#include "depthwise_common.hpp"
#include "premultiply.hpp"

namespace arm_conv
{
namespace depthwise
{
struct DepthwiseConfig
{
    DepthwiseMethod method = DepthwiseMethod::DEFAULT;
    std::string     filter = "";

    DepthwiseConfig(DepthwiseMethod method) : method(method){};
    DepthwiseConfig(){};
};

struct DepthwiseArgs
{
    const CPUInfo *cpu_info;

    unsigned int kernel_rows, kernel_cols;
    unsigned int stride_rows, stride_cols;
    unsigned int dilation_rows, dilation_cols;

    unsigned int n_batches, input_rows, input_cols, input_channels;
    unsigned int output_rows, output_cols;
    unsigned int channel_multiplier;

    PaddingValues padding;

    arm_gemm::Activation activation;

    const DepthwiseConfig *config;

    bool fast_mode = false;

    DepthwiseArgs(const CPUInfo       *cpu_info,
                  unsigned int         kernel_rows,
                  unsigned int         kernel_cols,
                  unsigned int         stride_rows,
                  unsigned int         stride_cols,
                  unsigned int         dilation_rows,
                  unsigned int         dilation_cols,
                  unsigned int         n_batches,
                  unsigned int         input_rows,
                  unsigned int         input_cols,
                  unsigned int         input_channels,
                  unsigned int         output_rows,
                  unsigned int         output_cols,
                  unsigned int         channel_multiplier,
                  PaddingValues        padding,
                  arm_gemm::Activation activation,

                  const DepthwiseConfig *config)
        : cpu_info(cpu_info),
          kernel_rows(kernel_rows),
          kernel_cols(kernel_cols),
          stride_rows(stride_rows),
          stride_cols(stride_cols),
          dilation_rows(dilation_rows),
          dilation_cols(dilation_cols),
          n_batches(n_batches),
          input_rows(input_rows),
          input_cols(input_cols),
          input_channels(input_channels),
          output_rows(output_rows),
          output_cols(output_cols),
          channel_multiplier(channel_multiplier),
          padding(padding),
          activation(activation),
          config(config)
    {
    }

    DepthwiseArgs(const CPUInfo         *cpu_info,
                  unsigned int           kernel_rows,
                  unsigned int           kernel_cols,
                  unsigned int           stride_rows,
                  unsigned int           stride_cols,
                  unsigned int           n_batches,
                  unsigned int           input_rows,
                  unsigned int           input_cols,
                  unsigned int           input_channels,
                  unsigned int           output_rows,
                  unsigned int           output_cols,
                  unsigned int           channel_multiplier,
                  PaddingValues          padding,
                  arm_gemm::Activation   activation,
                  const DepthwiseConfig *config)
        : DepthwiseArgs(cpu_info,
                        kernel_rows,
                        kernel_cols,
                        stride_rows,
                        stride_cols,
                        1,
                        1,
                        n_batches,
                        input_rows,
                        input_cols,
                        input_channels,
                        output_rows,
                        output_cols,
                        channel_multiplier,
                        padding,
                        activation,
                        config)
    {
    }
};

template <typename TInput>
struct Tile
{
    TInput *array;

    unsigned int tile_rows     = 0;
    unsigned int tile_cols     = 0;
    unsigned int tile_channels = 0;

    Tile(TInput *array, unsigned int tile_rows, unsigned int tile_cols, unsigned int tile_channels)
        : array(array), tile_rows(tile_rows), tile_cols(tile_cols), tile_channels(tile_channels)
    {
    }

    Tile() : Tile(nullptr, 0, 0, 0)
    {
    }

    void load_from(const TInput      *input,
                   const unsigned int ld_row,
                   const unsigned int ld_col,
                   const unsigned int n_rows,
                   const unsigned int n_cols,
                   const int          input_i,
                   const int          input_j,
                   const unsigned int channel_multiplier) const
    {
        const auto pad_top  = input_i < 0 ? -input_i : 0;
        const auto pad_left = input_j < 0 ? -input_j : 0;

        const auto padded_rows = std::min(n_rows - input_i, tile_rows) - pad_top;
        const auto padded_cols = std::min(n_cols - input_j, tile_cols) - pad_left;

        if (padded_rows < tile_rows || padded_cols < tile_cols)
        {
            memset(array, 0, tile_rows * tile_cols * tile_channels * sizeof(TInput));
        }

        do_premultiply<TInput>((TInput *)input + std::max(input_i, 0) * ld_row + std::max(input_j, 0) * ld_col, ld_row,
                               ld_col, array + pad_top * tile_cols * tile_channels + pad_left * tile_channels,
                               tile_cols * tile_channels, tile_channels, padded_rows, padded_cols,
                               tile_channels / channel_multiplier, channel_multiplier);
    }
};

template <typename TInput, typename TWeight, typename TOutput>
class DepthwiseCommon : public IDepthwiseCommon
{
protected:
    const DepthwiseArgs m_args; // Copy of arguments
    std::string         m_name{};

public:
    DepthwiseCommon(const DepthwiseArgs &args) : m_args(args){};
    DepthwiseCommon(DepthwiseCommon &)            = delete;
    DepthwiseCommon &operator=(DepthwiseCommon &) = delete;

    std::string name() const override
    {
        return m_name;
    }

    void set_name(std::string name)
    {
        // Only allow the name to be set once
        if (m_name.empty())
        {
            m_name = name;
        }
    }
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override {
        ARM_COMPUTE_UNUSED(thread_count);
        ARM_COMPUTE_UNUSED(platform);
        return arm_compute::ICPPKernel::default_mws;
    }

    void execute(const void *const  input,
                 const void *const  parameters,
                 void *const        output,
                 void *const        working_space,
                 const ThreadInfo & info) const override final
    {
        const size_t ld_input_col    = m_args.input_channels;
        const size_t ld_input_row    = ld_input_col * m_args.input_cols;
        const size_t ld_input_batch  = ld_input_row * m_args.input_rows;
        const size_t ld_output_col   = m_args.input_channels * m_args.channel_multiplier;
        const size_t ld_output_row   = ld_output_col * m_args.output_cols;
        const size_t ld_output_batch = ld_output_row * m_args.output_rows;

        execute(input, ld_input_col, ld_input_row, ld_input_batch, parameters, output, ld_output_col, ld_output_row,
                ld_output_batch, working_space, info);
    }

    void execute(const void *const  input,
                 size_t             ld_input_col,
                 size_t             ld_input_row,
                 size_t             ld_input_batch,
                 const void *const  parameters,
                 void *const        output,
                 size_t             ld_output_col,
                 size_t             ld_output_row,
                 size_t             ld_output_batch,
                 void *const        working_space,
                 const ThreadInfo & info) const override final
    {
        execute(m_args.n_batches, m_args.input_rows, m_args.input_cols, m_args.input_channels, m_args.padding, input,
                ld_input_col, ld_input_row, ld_input_batch, parameters, m_args.output_rows, m_args.output_cols, output,
                ld_output_col, ld_output_row, ld_output_batch, working_space, info);
    }

    void execute(unsigned int         batches,
                 unsigned int         input_height,
                 unsigned int         input_width,
                 unsigned int         channels,
                 const PaddingValues &padding,
                 const void          *input,
                 size_t               ld_input_col,
                 size_t               ld_input_row,
                 size_t               ld_input_batch,
                 const void          *parameters,
                 unsigned int         output_height,
                 unsigned int         output_width,
                 void                *output,
                 size_t               ld_output_col,
                 size_t               ld_output_row,
                 size_t               ld_output_batch,
                 void                *working_space,
                 const ThreadInfo & info) const override final
    {
        // Construct a new set of arguments to reflect that we might have been
        // passed different input/output tensors. Dilation is handled at this
        // level; so we set the dilation in the arguments to zero.
        DepthwiseArgs args(this->m_args);
        args.n_batches      = batches;
        args.input_rows     = input_height;
        args.input_cols     = input_width;
        args.input_channels = channels;
        args.output_rows    = output_height;
        args.output_cols    = output_width;
        args.padding        = padding;
        args.dilation_rows = args.dilation_cols = 1;

        // std::printf("111 ld_input_col, ld_input_row, ld_output_col, ld_output_row: %zu, %zu, %zu, %zu\n", ld_input_col, ld_input_row, ld_output_col, ld_output_row);
        auto ld_input_col_d  = ld_input_col * m_args.dilation_cols;
        auto ld_input_row_d  = ld_input_row * m_args.dilation_rows;
        auto ld_output_col_d = ld_output_col * m_args.dilation_cols;
        auto ld_output_row_d = ld_output_row * m_args.dilation_rows;

        for (size_t drow = 0; drow < m_args.dilation_rows; drow++)
        {
            size_t start_i;
            std::tie(args.output_rows, args.input_rows, start_i, args.padding.top, args.padding.bottom) =
                get_reduced_view_for_dilation(output_height, input_height, drow, m_args.dilation_rows,
                                              m_args.kernel_rows, m_args.stride_rows, padding.top);

            auto input_row  = static_cast<const TInput *>(input) + start_i * ld_input_row;
            auto output_row = static_cast<TOutput *>(output) + drow * ld_output_row;

            if (args.output_rows)
            {
                for (size_t dcol = 0; dcol < m_args.dilation_cols; dcol++)
                {
                    size_t start_j;
                    std::tie(args.output_cols, args.input_cols, start_j, args.padding.left, args.padding.right) =
                        get_reduced_view_for_dilation(output_width, input_width, dcol, m_args.dilation_cols,
                                                      m_args.kernel_cols, m_args.stride_cols, padding.left);

                    const TInput *input_col  = input_row + start_j * ld_input_col;
                    TOutput      *output_col = output_row + dcol * ld_output_col;

                    if (args.output_cols)
                    {
                        //TODO: Add ftrace to look why the threads cannot parallelize well 
                        this->execute_internal(args, input_col, ld_input_col_d, ld_input_row_d, ld_input_batch,
                                               parameters, output_col, ld_output_col_d, ld_output_row_d,
                                               ld_output_batch, working_space, info);
                    }
                }
            }
        }
    }

protected:
    virtual void execute_internal(const DepthwiseArgs &instance_args,
                                  const void          *input,
                                  size_t               ld_input_col,
                                  size_t               ld_input_row,
                                  size_t               ld_input_batch,
                                  const void          *parameters,
                                  void                *output,
                                  size_t               ld_output_col,
                                  size_t               ld_output_row,
                                  size_t               ld_output_batch,
                                  void                *working_space,
                                  const ThreadInfo & info) const = 0;

    virtual bool uses_premultiply() const
    {
        return true;
    }
};

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput>
using UniqueDepthwiseCommon = std::unique_ptr<DepthwiseCommon<TInput, TWeight, TOutput>>;

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
KernelDescription get_depthwise_method(const DepthwiseArgs &, const OutputStage & = {});

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
UniqueDepthwiseCommon<TInput, TWeight, TOutput> depthwise(const DepthwiseArgs &, const OutputStage & = {});

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
std::vector<KernelDescription> get_compatible_kernels(const DepthwiseArgs &, const OutputStage & = {});

} // namespace depthwise
} // namespace arm_conv

#endif // ACL_SRC_CORE_NEON_KERNELS_ASSEMBLY_DEPTHWISE_HPP
