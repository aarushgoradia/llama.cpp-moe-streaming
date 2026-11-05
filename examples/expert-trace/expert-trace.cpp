// record the experts the MoE router selects (ffn_moe_topk tensor), one CSV row per token and layer:
//   tag,phase,pos,layer,e0,...,e{k-1}
// env: EXPERT_TRACE_OUT (output file, appended), EXPERT_TRACE_TAG (row label)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct trace_state {
    FILE *            out   = nullptr;
    std::string       tag;
    const char *      phase = "prefill";
    llama_pos         pos0  = 0;
    std::map<int, int> seen;          // layer -> tokens traced so far in this call (a call can span several ubatches)
    std::vector<uint8_t> buf;
    size_t            n_rows = 0;
};

static const char * TOPK_PREFIX = "ffn_moe_topk-";

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;

    const bool is_topk = strncmp(t->name, TOPK_PREFIX, strlen(TOPK_PREFIX)) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_I32);

    const int layer = atoi(t->name + strlen(TOPK_PREFIX));
    const int64_t k        = t->ne[0]; // experts used per token
    const int64_t n_tokens = t->ne[1];

    // top-k is a strided view of the argsort result
    const size_t n_bytes = ggml_nbytes(t);
    st->buf.resize(n_bytes);
    ggml_backend_tensor_get(t, st->buf.data(), 0, n_bytes);

    int & done = st->seen[layer];
    for (int64_t j = 0; j < n_tokens; ++j) {
        fprintf(st->out, "%s,%s,%d,%d", st->tag.c_str(), st->phase, (int) (st->pos0 + done + j), layer);
        for (int64_t i = 0; i < k; ++i) {
            const int32_t e = *(const int32_t *) (st->buf.data() + j*t->nb[1] + i*t->nb[0]);
            fprintf(st->out, ",%d", e);
        }
        fputc('\n', st->out);
        st->n_rows++;
    }
    done += (int) n_tokens;

    return true;
}

static bool process(llama_context * ctx, trace_state & st, common_batch & batch, const char * phase, llama_pos pos0) {
    st.phase = phase;
    st.pos0  = pos0;
    st.seen.clear();
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * out_path = getenv("EXPERT_TRACE_OUT");
    const char * tag      = getenv("EXPERT_TRACE_TAG");

    trace_state st;
    st.tag = tag ? tag : "run";
    st.out = fopen(out_path ? out_path : "expert-trace.csv", "a");
    if (!st.out) {
        LOG_ERR("%s: cannot open output file\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt (use -p or -f)\n", __func__);
        return 1;
    }
    LOG_INF("%s: prompt tokens = %zu, n_predict = %d\n", __func__, tokens.size(), params.n_predict);

    common_sampler * smpl = common_sampler_init(model, params.sampling);

    for (int32_t i = 0; i < llama_model_meta_count(model); ++i) {
        char key[256];
        char val[64];
        llama_model_meta_key_by_index(model, i, key, sizeof(key));
        const std::string k = key;
        const char * name = string_ends_with(k, ".expert_count")      ? "n_expert" :
                            string_ends_with(k, ".expert_used_count") ? "n_expert_used" : nullptr;
        if (name) {
            llama_model_meta_val_str_by_index(model, i, val, sizeof(val));
            LOG_INF("%s: %s = %s\n", __func__, name, val);
        }
    }

    // all prompt tokens are outputs: otherwise the last layer skips all but the final token and the trace has a gap
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.add(tokens[i], (llama_pos) i, 0, true);
    }

    const int64_t t_start = ggml_time_us();
    if (!process(ctx, st, batch, "prefill", 0)) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }

    llama_pos pos = (llama_pos) tokens.size();
    int n_gen = 0;
    while (params.n_predict < 0 || n_gen < params.n_predict) {
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        LOG("%s", common_token_to_piece(ctx, id).c_str());

        batch.clear();
        batch.add(id, pos, 0, true);
        if (!process(ctx, st, batch, "decode", pos)) {
            LOG_ERR("%s: decode failed at pos %d\n", __func__, pos);
            return 1;
        }
        pos++;
        n_gen++;
    }
    const double secs = (ggml_time_us() - t_start) / 1e6;

    LOG("\n");
    LOG_INF("%s: tag=%s generated=%d rows=%zu time=%.1fs\n", __func__, st.tag.c_str(), n_gen, st.n_rows, secs);

    fclose(st.out);
    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
