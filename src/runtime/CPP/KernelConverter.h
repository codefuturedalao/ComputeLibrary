#ifndef ARM_COMPUTE_KERNELCONVERTER_H
#define ARM_COMPUTE_KERNELCONVERTER_H

#include <vector>
#include <utility>
#include <cstddef>
#include <iostream>

namespace arm_compute
{

/** Kernel conversion utilities for transforming workload between different GEMM kernels */
class KernelConverter
{
public:
    // Make LCM functions publicly accessible for scheduling
    static unsigned int gcd(unsigned int a, unsigned int b) {
        while (b != 0) {
            unsigned int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    
    static unsigned int lcm(unsigned int a, unsigned int b) {
        return (a * b) / gcd(a, b);
    }

    /** Structure to hold kernel conversion configuration */
    struct ConversionConfig
    {
        unsigned int primary_out_height;      /**< Primary kernel output height */
        unsigned int primary_out_width;       /**< Primary kernel output width */
        unsigned int alternate_out_height;    /**< Alternate kernel output height */
        unsigned int alternate_out_width;     /**< Alternate kernel output width */
        unsigned int output_matrix_height;    /**< Total output matrix height */
        unsigned int output_matrix_width;     /**< Total output matrix width */
        
        // Computed values
        unsigned int primary_block_rows;      /**< Number of row blocks for primary kernel */
        unsigned int primary_block_cols;      /**< Number of col blocks for primary kernel */
        unsigned int alternate_block_rows;    /**< Number of row blocks for alternate kernel */
        unsigned int alternate_block_cols;    /**< Number of col blocks for alternate kernel */
        
        // Conversion ratio
        unsigned int merge_block_height;      /**< Height of merged block in alternate units */
        unsigned int merge_block_width;       /**< Width of merged block in alternate units */
        
        ConversionConfig(unsigned int p_height, unsigned int p_width,
                        unsigned int a_height, unsigned int a_width,
                        unsigned int m_height, unsigned int m_width)
            : primary_out_height(p_height), primary_out_width(p_width),
              alternate_out_height(a_height), alternate_out_width(a_width),
              output_matrix_height(m_height), output_matrix_width(m_width)
        {
            // Calculate workload block counts
            primary_block_rows = (output_matrix_height + primary_out_height - 1) / primary_out_height;
            primary_block_cols = (output_matrix_width + primary_out_width - 1) / primary_out_width;
            alternate_block_rows = (output_matrix_height + alternate_out_height - 1) / alternate_out_height;
            alternate_block_cols = (output_matrix_width + alternate_out_width - 1) / alternate_out_width;
            
            // Calculate merge block size using LCM to ensure area compatibility
            // LCM ensures both kernels can cover the same area perfectly
            merge_block_height = lcm(primary_out_height, alternate_out_height) / alternate_out_height;
            merge_block_width = lcm(primary_out_width, alternate_out_width) / alternate_out_width;
            
            std::cout << "KernelConverter Config:" << std::endl;
            std::cout << "  Primary: " << primary_out_height << "x" << primary_out_width 
                     << " -> blocks: " << primary_block_rows << "x" << primary_block_cols << std::endl;
            std::cout << "  Alternate: " << alternate_out_height << "x" << alternate_out_width 
                     << " -> blocks: " << alternate_block_rows << "x" << alternate_block_cols << std::endl;
            std::cout << "  Merge block: " << merge_block_height << "x" << merge_block_width << std::endl;
        }
    };
    
    /** Convert primary kernel workload index to alternate kernel workload indices
     *
     * @param[in]  config         Conversion configuration
     * @param[in]  primary_idx    Primary kernel workload index
     * @param[out] alternate_indices  Vector of alternate kernel workload indices
     *
     * @return True if conversion successful
     */
    static bool convert_primary_to_alternate(const ConversionConfig &config, 
                                            size_t primary_idx,
                                            std::vector<size_t> &alternate_indices)
    {
        alternate_indices.clear();
        
        // Convert 1D index to 2D coordinates for primary kernel
        size_t primary_row = primary_idx / config.primary_block_cols;
        size_t primary_col = primary_idx % config.primary_block_cols;
        
        if (primary_row >= config.primary_block_rows) {
            return false;
        }
        
        // Calculate the range of alternate kernel blocks that correspond to this primary block
        size_t alt_start_row = primary_row * config.merge_block_height;
        size_t alt_start_col = primary_col * config.merge_block_width;
        size_t alt_end_row = std::min(alt_start_row + config.merge_block_height, 
                                     static_cast<size_t>(config.alternate_block_rows));
        size_t alt_end_col = std::min(alt_start_col + config.merge_block_width,
                                     static_cast<size_t>(config.alternate_block_cols));
        
        // Generate alternate indices (row-major order for hybrid kernels)
        for (size_t row = alt_start_row; row < alt_end_row; ++row) {
            for (size_t col = alt_start_col; col < alt_end_col; ++col) {
                size_t alt_idx = row * config.alternate_block_cols + col;
                alternate_indices.push_back(alt_idx);
            }
        }
        
        return !alternate_indices.empty();
    }
    
    /** Convert alternate kernel workload indices to primary kernel workload index
     *
     * @param[in]  config         Conversion configuration
     * @param[in]  alternate_indices  Vector of alternate kernel workload indices
     * @param[out] primary_idx    Primary kernel workload index
     *
     * @return True if conversion successful
     */
    static bool convert_alternate_to_primary(const ConversionConfig &config,
                                            const std::vector<size_t> &alternate_indices,
                                            size_t &primary_idx)
    {
        if (alternate_indices.empty()) {
            return false;
        }
        
        // Take the first alternate index to determine the primary block
        size_t alt_idx = alternate_indices[0];
        size_t alt_row = alt_idx / config.alternate_block_cols;
        size_t alt_col = alt_idx % config.alternate_block_cols;
        
        // Convert to primary coordinates
        size_t primary_row = alt_row / config.merge_block_height;
        size_t primary_col = alt_col / config.merge_block_width;
        
        if (primary_row >= config.primary_block_rows || primary_col >= config.primary_block_cols) {
            return false;
        }
        
        primary_idx = primary_row * config.primary_block_cols + primary_col;
        return true;
    }
    
    /** Check if a set of alternate workload indices can be merged
     *
     * @param[in] config           Conversion configuration
     * @param[in] alternate_indices Vector of alternate kernel workload indices
     *
     * @return True if indices form a complete mergeable block
     */
    static bool can_merge_alternate_indices(const ConversionConfig &config,
                                           const std::vector<size_t> &alternate_indices)
    {
        if (alternate_indices.empty()) {
            return false;
        }
        
        // Check if indices form a rectangular block
        std::vector<std::pair<size_t, size_t>> coords;
        for (size_t idx : alternate_indices) {
            size_t row = idx / config.alternate_block_cols;
            size_t col = idx % config.alternate_block_cols;
            coords.emplace_back(row, col);
        }
        
        // Sort by row, then by column
        std::sort(coords.begin(), coords.end());
        
        // Check if it forms a complete rectangular block
        if (coords.size() != static_cast<size_t>(config.merge_block_height * config.merge_block_width)) {
            // May be a partial block at boundaries
            size_t expected_size = std::min(static_cast<size_t>(config.merge_block_height), 
                                          static_cast<size_t>(config.alternate_block_rows) - coords[0].first) *
                                  std::min(static_cast<size_t>(config.merge_block_width),
                                          static_cast<size_t>(config.alternate_block_cols) - coords[0].second);
            if (coords.size() != expected_size) {
                return false;
            }
        }
        
        // Verify contiguous rectangular pattern
        size_t start_row = coords[0].first;
        size_t start_col = coords[0].second;
        size_t rows = 0, cols = 0;
        
        // Count dimensions
        for (auto &coord : coords) {
            if (coord.first == start_row && coord.second >= start_col) {
                cols = std::max(cols, coord.second - start_col + 1);
            }
            if (coord.second == start_col && coord.first >= start_row) {
                rows = std::max(rows, coord.first - start_row + 1);
            }
        }
        
        // Verify all positions are present
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                auto it = std::find(coords.begin(), coords.end(),
                                  std::make_pair(start_row + r, start_col + c));
                if (it == coords.end()) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

} // namespace arm_compute

#endif // ARM_COMPUTE_KERNELCONVERTER_H 