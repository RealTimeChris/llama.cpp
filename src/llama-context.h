#pragma once

#include <atomic>
#include <chrono>
#include <mutex>

#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "llama-adapter.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama.h"

namespace test {

// from
// https://stackoverflow.com/questions/16337610/how-to-know-if-a-type-is-a-specialization-of-stdvector
template <typename, template <typename...> typename> constexpr bool is_specialization_v = false;

template <template <typename...> typename value_type, typename... arg_types>
constexpr bool is_specialization_v<value_type<arg_types...>, value_type> = true;

template <typename value_type> concept time_type = is_specialization_v<value_type, std::chrono::duration>;

template <time_type value_type = std::chrono::nanoseconds> class stop_watch {
  public:
    using hr_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                        std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    static constexpr bool lock_free{ std::atomic<value_type>::is_always_lock_free };
    using time_type = std::conditional_t<lock_free, value_type, uint64_t>;

    stop_watch(uint64_t newTime) noexcept { total_time_units.store(time_type{ newTime }, std::memory_order_release); }

    stop_watch & operator=(stop_watch && other) noexcept {
        if (this != &other) {
            total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
            start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
        }
        return *this;
    }

    stop_watch(stop_watch && other) noexcept { *this = std::move(other); }

    stop_watch & operator=(const stop_watch & other) noexcept {
        if (this != &other) {
            total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
            start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
        }
        return *this;
    }

    stop_watch(const stop_watch & other) noexcept { *this = other; }

    bool has_time_elapsed() noexcept {
        return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >=
                total_time_units.load(std::memory_order_acquire));
    }

    void add_time() noexcept {
        //std::unique_lock lock{ mutex };
        values.emplace_back(total_time_elapsed());
        //lock.release();
        reset();
    }

    uint64_t get_count() noexcept { return values.size(); }

    uint64_t get_average(time_type newTimeValue = time_type{}) noexcept {
        std::unique_lock lock{ mutex };
        uint64_t         total_time{};
        for (auto & value : values) {
            total_time += get_value_as_uint(value);
        }
        return total_time / ((values.size() > 0) ? values.size() : 1);
    }

    void reset(time_type newTimeValue = time_type{}) noexcept {
        if (newTimeValue != time_type{}) {
            total_time_units.store(newTimeValue, std::memory_order_release);
        }
        start_time_units.store(get_current_time(), std::memory_order_release);
    }

    uint64_t get_total_wait_time() const noexcept {
        return get_value_as_uint(total_time_units.load(std::memory_order_acquire));
    }

    time_type total_time_elapsed() noexcept {
        return get_current_time() - start_time_units.load(std::memory_order_acquire);
    }

    uint64_t total_time_elapsed_uint64() noexcept {
        return get_value_as_uint(get_current_time()) -
               get_value_as_uint(start_time_units.load(std::memory_order_acquire));
    }

  protected:
    std::atomic<time_type> total_time_units{};
    std::atomic<time_type> start_time_units{};
    std::vector<time_type> values{};
    std::mutex             mutex{};

    time_type get_current_time() {
        if constexpr (lock_free) {
            return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch());
        } else {
            return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch()).count();
        }
    }

    uint64_t get_value_as_uint(time_type time) {
        if constexpr (lock_free) {
            return time.count();
        } else {
            return time;
        }
    }
};
}  // namespace test

inline test::stop_watch stop_watch_val{ 0 };

#include <map>
#include <vector>

struct llama_model;
struct llama_kv_cache;

class llama_io_read_i;
class llama_io_write_i;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(const llama_model & model, llama_context_params params);

    ~llama_context();

    void synchronize();

    const llama_model &   get_model() const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    ggml_context * get_ctx_compute() const;

    uint32_t n_ctx() const;
    uint32_t n_ctx_per_seq() const;
    uint32_t n_batch() const;
    uint32_t n_ubatch() const;
    uint32_t n_seq_max() const;

    uint32_t n_threads() const;
    uint32_t n_threads_batch() const;

    llama_kv_cache *       get_kv_self();
    const llama_kv_cache * get_kv_self() const;

    void kv_self_update();

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    void attach_threadpool(ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings(bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapter_lora(llama_adapter_lora * adapter, float scale);

    bool rm_adapter_lora(llama_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end);

    int encode(llama_batch & inp_batch);
    int decode(llama_batch & inp_batch);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id);
    size_t state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size);

    bool state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity,
                         size_t * n_token_count_out);

    bool state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count);

    size_t state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out,
                               size_t n_token_capacity, size_t * n_token_count_out);

    size_t state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens,
                               size_t n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void                    perf_reset();

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    void opt_epoch(ggml_opt_dataset_t dataset, ggml_opt_result_t result_train, ggml_opt_result_t result_eval,
                   int64_t idata_split, ggml_opt_epoch_callback callback_train, ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(ggml_opt_dataset_t dataset, ggml_opt_result_t result, const std::vector<llama_token> & tokens,
                        const std::vector<llama_token> & labels_sparse, llama_batch & batch,
                        ggml_opt_epoch_callback callback, bool train, int64_t idata_in_loop, int64_t ndata_in_loop,
                        int64_t t_loop_start);

  private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    int32_t output_reserve(int32_t n_outputs);

    //
    // graph
    //

  public:
    int32_t graph_max_nodes() const;

    // zero-out inputs and create the ctx_compute for the compute graph
    ggml_cgraph * graph_init();

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

  private:
    llm_graph_result_ptr graph_build(ggml_context * ctx, ggml_cgraph * gf, const llama_ubatch & ubatch,
                                     llm_graph_type gtype);

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data(llama_io_read_i & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id);
    size_t state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;

    llama_cross cross;  // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0;  // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    int32_t n_outputs     = 0;        // number of actually-used outputs in the current ubatch or last logical batch
    int32_t n_outputs_max = 0;        // capacity (of tokens positions) for the output buffers

    std::vector<int32_t> output_ids;  // map batch token positions to ids of the logits and embd buffers

    ggml_backend_sched_ptr sched;

    ggml_backend_t                backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    ggml_context_ptr ctx_compute;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0;  // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0;  // number of eval calls
};
