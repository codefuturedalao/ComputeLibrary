/*
 * Copyright (c) 2018-2020, 2022-2024 Arm Limited.
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

#include "arm_gemm.hpp"

#include "kernel_weight_format.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <cstring>

namespace arm_gemm {

/* Structure describing an implementation.  For each supported combination
 * of types, a static list of these structures is built up to describe the
 * implementations available.
 */
template<typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
struct GemmImplementation {
    const GemmMethod                                                               method;
    const char *                                                                   name;
    const KernelWeightFormat                                                       kernel_weight_format = KernelWeightFormat::NON_FIXED;
    std::function<bool(const GemmArgs &, const OutputStage &)>                     is_supported = {};
    std::function<uint64_t(const GemmArgs &, const OutputStage &)>                 cycle_estimate = {};
    std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &, const OutputStage &)>  instantiate = {};

    bool do_is_supported(const GemmArgs &args, const OutputStage &os) const {
        // Check supplied is_supported() function first.
        if (is_supported != nullptr && !is_supported(args, os)) {
            return false;
        }

        // Check weight format is appropriate.
        if (args._fixed_format == false) {
            // Can't return a fixed format kernel if we weren't asked for one.
            return (kernel_weight_format == KernelWeightFormat::NON_FIXED);
        } else {
            // Fixed format kernel requested: if this is a non-fixed format kernel we can't use it.
            if (kernel_weight_format == KernelWeightFormat::NON_FIXED) {
                return false;
            }

            // If there's no config, or the config says ANY then this one is OK.
            if (!args._cfg || args._cfg->weight_format == WeightFormat::ANY) {
                return true;
            }

            // If we get here it means there is a config and it specifies a format.  Check it matches this kernel.
            // NOTE: this will execute SVE instructions if it's an SVE kernel, so it's important that is_supported()
            // was called above first.
            return (args._cfg->weight_format == get_weight_format(kernel_weight_format, sizeof(Tlop)));
        }
    }

    uint64_t do_cycle_estimate(const GemmArgs &args, const OutputStage &os) const {
        if (cycle_estimate != nullptr) {
            return cycle_estimate(args, os);
        } else {
            return 0;
        }
    }

    GemmCommon<Tlop, Trop, Tret> *do_instantiate(const GemmArgs &args, const OutputStage &os) const {
        return instantiate(args, os);
    }

    static GemmImplementation with_estimate(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &, const OutputStage &)> is_supported, std::function<uint64_t(const GemmArgs &, const OutputStage &)> cycle_estimate,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &, const OutputStage &)> instantiate) {
        GemmImplementation impl(m,n);

        impl.is_supported=is_supported;
        impl.cycle_estimate=cycle_estimate;
        impl.instantiate=instantiate;

        return impl;
    }

    GemmImplementation(const GemmImplementation &) = default;
    GemmImplementation & operator= (const GemmImplementation &) = default;

    GemmImplementation(GemmMethod m, const char * n) : method(m), name(n) {}

    GemmImplementation(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &, const OutputStage &)> is_supported, std::function<bool(const GemmArgs &, const OutputStage &)> is_recommended,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &, const OutputStage &)> instantiate) :
                       method(m), name(n), is_supported(is_supported),
                       cycle_estimate( [is_recommended](const GemmArgs &args, const OutputStage &os) { return (is_recommended == nullptr) ? 0 : (is_recommended(args, os) ? 0 : UINT64_MAX); } ),
                       instantiate(instantiate) {   }

    GemmImplementation(GemmMethod m, const char *n, KernelWeightFormat kwf,
                       std::function<bool(const GemmArgs &, const OutputStage &)> is_supported, std::function<bool(const GemmArgs &, const OutputStage &)> is_recommended,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &, const OutputStage &)> instantiate) :
                       method(m), name(n), kernel_weight_format(kwf), is_supported(is_supported),
                       cycle_estimate( [is_recommended](const GemmArgs &args, const OutputStage &os) { return (is_recommended == nullptr) ? 0 : (is_recommended(args, os) ? 0 : UINT64_MAX); } ),
                       instantiate(instantiate) {   }
};

/* Slightly different version of above for straightforward GEMMs with no
 * output stage, so the std::functions there don't have to deal with the
 * unnecessary second argument.  */
template<typename Tlop, typename Trop, typename Tret>
struct GemmImplementation<Tlop, Trop, Tret, Nothing> {
    const GemmMethod                                          method;
    const char *                                              name;
    const KernelWeightFormat                                  kernel_weight_format = KernelWeightFormat::NON_FIXED;
    std::function<bool(const GemmArgs &)>                     is_supported = {};
    std::function<uint64_t(const GemmArgs &)>                 cycle_estimate = {};
    std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)>  instantiate = {};

    bool do_is_supported(const GemmArgs &args, const Nothing &) const {
        // Check supplied is_supported() function first.
        if (is_supported != nullptr && !is_supported(args)) {
            return false;
        }

        // Check weight format is appropriate.
        if (args._fixed_format == false) {
            // Can't return a fixed format kernel if we weren't asked for one.
            return (kernel_weight_format == KernelWeightFormat::NON_FIXED);
        } else {
            // Fixed format kernel requested: if this is a non-fixed format kernel we can't use it.
            if (kernel_weight_format == KernelWeightFormat::NON_FIXED) {
                return false;
            }

            // If there's no config, or the config says ANY then this one is OK.
            if (!args._cfg || args._cfg->weight_format == WeightFormat::ANY) {
                return true;
            }

            // If we get here it means there is a config and it specifies a format.  Check it matches this kernel.
            // NOTE: this will execute SVE instructions if it's an SVE kernel, so it's important that is_supported()
            // was called above first.
            return (args._cfg->weight_format == get_weight_format(kernel_weight_format, sizeof(Tlop)));
        }
    }

    uint64_t do_cycle_estimate(const GemmArgs &args, const Nothing &) const {
        if (cycle_estimate != nullptr) {
            return cycle_estimate(args);
        } else {
            return 0;
        }
    }

    GemmCommon<Tlop, Trop, Tret> *do_instantiate(const GemmArgs &args, const Nothing &) const {
        return instantiate(args);
    }

    static GemmImplementation with_estimate(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &)> is_supported, std::function<uint64_t(const GemmArgs &)> cycle_estimate,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)> instantiate) {
        GemmImplementation impl(m,n);

        impl.is_supported=is_supported;
        impl.cycle_estimate=cycle_estimate;
        impl.instantiate=instantiate;

        return impl;
    }

    static GemmImplementation with_estimate(GemmMethod m, const char *n, KernelWeightFormat f,
                       std::function<bool(const GemmArgs &)> is_supported, std::function<uint64_t(const GemmArgs &)> cycle_estimate,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)> instantiate) {
        GemmImplementation impl(m,n,f);

        impl.is_supported=is_supported;
        impl.cycle_estimate=cycle_estimate;
        impl.instantiate=instantiate;

        return impl;
    }

    GemmImplementation(const GemmImplementation &) = default;
    GemmImplementation & operator= (const GemmImplementation &) = default;

    GemmImplementation(GemmMethod m, const char *n, KernelWeightFormat f=KernelWeightFormat::NON_FIXED) : method(m), name(n), kernel_weight_format(f) {}

    GemmImplementation(GemmMethod m, const char *n,
                       std::function<bool(const GemmArgs &)> is_supported, std::function<bool(const GemmArgs &)> is_recommended,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)> instantiate) :
                       method(m), name(n), is_supported(is_supported),
                       cycle_estimate( [is_recommended](const GemmArgs &args) -> uint64_t { return (is_recommended == nullptr) ? 0 : (is_recommended(args) ? 0 : UINT64_MAX); } ),
                       instantiate(instantiate) {   }

    GemmImplementation(GemmMethod m, const char *n, KernelWeightFormat kwf,
                       std::function<bool(const GemmArgs &)> is_supported, std::function<bool(const GemmArgs &)> is_recommended,
                       std::function<GemmCommon<Tlop, Trop, Tret> *(const GemmArgs &)> instantiate) :
                       method(m), name(n), kernel_weight_format(kwf), is_supported(is_supported),
                       cycle_estimate( [is_recommended](const GemmArgs &args) -> uint64_t { return (is_recommended == nullptr) ? 0 : (is_recommended(args) ? 0 : UINT64_MAX); } ),
                       instantiate(instantiate) {   }
};

/* Provides the list of implementation descriptors which is processed by the
 * other functions.
 *
 * A specialised version is provided for each supported combination of types.
 * The end of the list is indicated by a sentinel descriptor with
 * method==GemmMethod::DEFAULT.  */
template<typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
const GemmImplementation<Tlop, Trop, Tret, OutputStage> *gemm_implementation_list();

/*
 * Select a GEMM implementation for the given arguments.
 *
 * The logic here returns the method on the list which supports the
 * requested problem parameters, matches the provided filters (method and/or
 * name string match) and offers the lowest cycle estimate.  A cycle
 * estimate of '0' is treated as a special value, causing the corresponding
 * method to be selected immediately.
 *
 * If no method supports the requested parameters and passes the filters,
 * this function returns false and doesn't touch the provided pointer
 * reference.
 */
template<typename Tlop, typename Trop, typename Tret, class OutputStage>
bool find_implementation(const GemmArgs &args, const OutputStage &os, const GemmImplementation<Tlop, Trop, Tret, OutputStage> * &impl) {
    auto gemms = gemm_implementation_list<Tlop, Trop, Tret, OutputStage>();
    const GemmConfig *cfg = args._cfg;

    const GemmImplementation<Tlop, Trop, Tret, OutputStage> *saved_impl = nullptr;
    uint64_t best_estimate = 0;

    // std::cout << "find_implementation : -------------" << typeid(Trop).name() << ", " << typeid(Tret).name() << ", " << args._Msize << ", " << args._Nsize << ", " << args._Ksize << ", " << args._nmulti << ", " << args._nbatches << std::endl;
    for (const GemmImplementation<Tlop, Trop, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Skip if this implementation doesn't support these args. */
        if (!i->do_is_supported(args, os)) {
            // std::cout << "Unsupported: " << i->name << std::endl;
            continue;
        }

        /* Skip if a specific method is requested and this is a different one. */
        if (cfg && cfg->method != GemmMethod::DEFAULT && i->method != cfg->method) {
            //std::cout << "Best : " << i->name << " method unmatch" << std::endl;
            continue;
        }

        /* Skip if a filter is to be applied and it doesn't match. */
        if (cfg && cfg->filter != "" && !strstr(i->name, cfg->filter.c_str())) {
            // std::cout << "filter unmatch: " << i->name << std::endl;
            continue;
        }

        /* Test the cycle estimate */
        uint64_t estimate = i->do_cycle_estimate(args, os);
        // std::cout << "Impl Name : " << i->name << " , " << estimate << std::endl;

        /* Short circuit - if the estimate is zero, return this one immediately. */
        if (estimate==0) {
            impl=i;
            return true;
        }

        if (strcmp(i->name, "a64_hybrid_fp32_mla_6x16") == 0) {
        //if (strcmp(i->name, "a64_sgemm_8x12") == 0) {
            // std::cout << "best ipick : " << i->name << std::endl;
            impl=i;
            return true;
        }

        /* Otherwise, remember this is our best so far if we don't yet have
         * a valid candidate, or we beat the estimate.  */
        if ((saved_impl == nullptr) || (estimate < best_estimate)) {
            saved_impl = i;
            best_estimate = estimate;
        }
    }

    /* Return whichever method gave the best estimate. */
    if (saved_impl != nullptr) {
        // std::cout << "Best Pick : " << saved_impl->name << std::endl;
        impl = saved_impl;
        return true;
    }

    return false;
}

template<typename Tlop, typename Trop, typename Tret, class OutputStage>
std::vector<KernelDescription> get_compatible_kernels(const GemmArgs &args, const OutputStage &os) {
    std::vector<KernelDescription> res;

    /* Find out what the default implementation in so we can set the flag accordingly later. */
    const GemmImplementation<Tlop, Trop, Tret, OutputStage> *default_impl;
    find_implementation(args, os, default_impl);

    auto gemms = gemm_implementation_list<Tlop, Trop, Tret, OutputStage>();

    for (const GemmImplementation<Tlop, Trop, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Check that this implementation supports the presented problem. */

        if (!i->do_is_supported(args, os)) {
            continue;
        }

        res.push_back(KernelDescription(i->method, i->name, i==default_impl, i->do_cycle_estimate(args, os)));
    }

    return res;
}

template<typename Tlop, typename Trop, typename Tret, class OutputStage>
bool has_opt_gemm(WeightFormat &wf, const GemmArgs &args, const OutputStage &os) {
    const GemmImplementation<Tlop, Trop, Tret, OutputStage> *impl;
    const bool success =  find_implementation<Tlop, Trop, Tret, OutputStage>(args, os, impl);
    if (success)
      wf = UniqueGemmCommon<Tlop, Trop, Tret>(impl->do_instantiate(args, os))->get_config().weight_format;
    return success;
}

template<typename Tlop, typename Trop, typename Tret, class OutputStage>
UniqueGemmCommon<Tlop, Trop, Tret> gemm(const GemmArgs &args, const OutputStage &os) {
    const GemmImplementation<Tlop, Trop, Tret, OutputStage> *impl;

    if (find_implementation<Tlop, Trop, Tret, OutputStage>(args, os, impl)) {
        //std::cout << "Best: " << impl->name << std::endl;
        return UniqueGemmCommon<Tlop, Trop, Tret>(impl->do_instantiate(args, os));
    }

    return UniqueGemmCommon<Tlop, Trop, Tret>(nullptr);
}

template<typename Tlop, typename Trop, typename Tret, class OutputStage>
KernelDescription get_gemm_method(const GemmArgs &args, const OutputStage &os) {
    const GemmImplementation<Tlop, Trop, Tret, OutputStage> *impl;

    //std::cout << "Best : get_gemm_method" << std::endl;
    if (find_implementation<Tlop, Trop, Tret>(args, os, impl)) {
        return KernelDescription(impl->method, impl->name);
    }

    /* This shouldn't happen - there should always be at least one valid implementation. */
    return KernelDescription();
}


} // namespace arm_gemm
