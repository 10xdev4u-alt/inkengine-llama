#include "debug.h"

#include "common.h"
#include "log.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

struct common_debug_cb_user_data::impl {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;
    bool                    abort_on_nan{false};
};

common_debug_cb_user_data::common_debug_cb_user_data() : pimpl(std::make_unique<impl>()) {}
common_debug_cb_user_data::~common_debug_cb_user_data() = default;

common_debug_cb_user_data::common_debug_cb_user_data(common_params & params, const std::vector<std::string> & filter_patterns, bool abort_on_nan)
    : pimpl(std::make_unique<impl>())
{
    for (const auto & pattern : filter_patterns) {
        try {
            std::string anchored_pattern = "^" + pattern;
            pimpl->tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
        } catch (const std::regex_error & e) {
            throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
        }
    }
    pimpl->abort_on_nan = abort_on_nan;

    params.cb_eval           = common_debug_cb_eval;
    params.cb_eval_user_data = this;
}

static std::string common_ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

#define INDENT "    "

static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }

    if (abort_on_nan) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\n");
            exit(0);
        }
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (common_debug_cb_user_data *) user_data;
    auto * pimpl = cb_data->pimpl.get();

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true;  // Always retrieve data
    }

    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, common_ggml_ne_string(src1).c_str());
    }

    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
            common_ggml_ne_string(t).c_str());
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        pimpl->data.resize(n_bytes);
        ggml_backend_tensor_get(t, pimpl->data.data(), 0, n_bytes);
    }

    uint8_t * data = is_host ? (uint8_t *) t->data : pimpl->data.data();

    if (!ggml_is_quantized(t->type) && matches_filter) {
        common_debug_print_tensor(data, t->type, t->ne, t->nb, 3, pimpl->abort_on_nan);
    }

    static const std::string dump_dir = []() {
        const char * value = std::getenv("LLAMA_DUMP_TENSORS_DIR");
        return value != nullptr && value[0] != '\0' ? std::string(value) : std::string();
    }();
    static const bool dump_ptr = []() {
        const char * value = std::getenv("LLAMA_DUMP_TENSOR_PTR");
        if (value == nullptr) {
            return false;
        }
        const std::string text(value);
        return text == "1" || text == "on" || text == "true";
    }();
    static std::unordered_map<std::string, int> dump_occurrences;

    // Dump only when an explicit filter is set, so we never dump the whole graph.
    if (!dump_dir.empty() && matches_filter && !pimpl->tensor_filters.empty()) {
        std::string basename = t->name;
        for (char & c : basename) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
            if (!ok) {
                c = '_';
            }
        }
        const int occurrence = dump_occurrences[basename]++;
        const std::string filename = basename + "__" + std::to_string(occurrence) + ".bin";

        const std::filesystem::path dir(dump_dir);
        try {
            std::filesystem::create_directories(dir);
        } catch (const std::exception & e) {
            LOG_ERR("%s: cannot create dump dir: %s\n", __func__, e.what());
            return true;
        }

        const size_t n_bytes = ggml_nbytes(t);
        {
            std::ofstream out(dir / filename, std::ios::binary | std::ios::trunc);
            out.write((const char *) data, (std::streamsize) n_bytes);
            if (!out) {
                LOG_ERR("%s: failed to write %s\n", __func__, filename.c_str());
                return true;
            }
        }
        {
            std::ofstream out(dir / "manifest.tsv", std::ios::app);
            out << t->name << '\t' << ggml_type_name(t->type);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                out << '\t' << t->ne[i];
            }
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                out << '\t' << t->nb[i];
            }
            out << '\t' << n_bytes << '\t' << filename << '\n';
            if (!out) {
                LOG_ERR("%s: failed to append manifest in %s\n", __func__, dump_dir.c_str());
            }
        }
        if (dump_ptr) {
            char ptr[32] = { 0 };
            snprintf(ptr, sizeof(ptr), "%p", (const void *) t);
            std::ofstream out(dir / "manifest_ptr.tsv", std::ios::app);
            out << filename << '\t' << ptr << '\n';
            if (!out) {
                LOG_ERR("%s: failed to append pointer manifest in %s\n", __func__, dump_dir.c_str());
            }
        }
    }

    return true;
}
