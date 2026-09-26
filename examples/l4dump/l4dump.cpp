// l4dump: dump post-RoPE K (all prompt positions) and decode-step Q for selected attention layers,
// for the L4 page-selection recall study. Output: <out>/K_<il>.f16 ([n_tok][n_kv_head][d]) and
// <out>/Q_<il>.f32 ([n_steps][n_head][d]), plus meta.txt.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <algorithm>

struct dump_state {
    std::set<int> layers;
    std::string   out_dir;
    bool          decode_phase = false;
    std::vector<uint8_t> buf;
};

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return -1;
    return atoi(name + n + 1);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const int lk = layer_of(t->name, "Kcur");
    const int lq = layer_of(t->name, "Qcur");
    const bool want = (lk >= 0 && st->layers.count(lk) && !st->decode_phase) || (lq >= 0 && st->layers.count(lq) && st->decode_phase);
    if (ask) return want;
    if (!want) return true;
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const size_t nb = ggml_nbytes(t);
    st->buf.resize(nb);
    ggml_backend_tensor_get(t, st->buf.data(), 0, nb);
    const float * f = (const float *) st->buf.data();
    const int64_t n = ggml_nelements(t);
    if (lk >= 0) {
        std::ofstream o(st->out_dir + "/K_" + std::to_string(lk) + ".f16", std::ios::app | std::ios::binary);
        std::vector<ggml_fp16_t> h(n);
        for (int64_t i = 0; i < n; ++i) h[i] = ggml_fp32_to_fp16(f[i]);
        o.write((const char *) h.data(), n * sizeof(ggml_fp16_t));
    } else {
        std::ofstream o(st->out_dir + "/Q_" + std::to_string(lq) + ".f32", std::ios::app | std::ios::binary);
        o.write((const char *) f, n * sizeof(float));
    }
    return true;
}

int main(int argc, char ** argv) {
    // usage: l4dump -m model.gguf -f prompt.txt -c N --l4-layers 3,31,63 --l4-out dir --l4-steps 8
    std::string layers_arg = "3,31,63", out_dir = "l4dump_out";
    int n_steps = 8;
    std::vector<char *> rest;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--l4-layers") && i + 1 < argc) { layers_arg = argv[++i]; continue; }
        if (!strcmp(argv[i], "--l4-out")    && i + 1 < argc) { out_dir    = argv[++i]; continue; }
        if (!strcmp(argv[i], "--l4-steps")  && i + 1 < argc) { n_steps    = atoi(argv[++i]); continue; }
        rest.push_back(argv[i]);
    }
    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;

    dump_state st;
    st.out_dir = out_dir;
    for (size_t p = 0; p < layers_arg.size();) {
        size_t q = layers_arg.find(',', p); if (q == std::string::npos) q = layers_arg.size();
        st.layers.insert(atoi(layers_arg.substr(p, q - p).c_str())); p = q + 1;
    }
    std::string mk = "mkdir -p " + out_dir + " && rm -f " + out_dir + "/K_* " + out_dir + "/Q_*";
    if (system(mk.c_str()) != 0) return 1;

    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true, true);
    const int n_ctx = llama_n_ctx(ctx);
    if ((int) toks.size() + n_steps + 1 > n_ctx) toks.resize(n_ctx - n_steps - 1);
    fprintf(stderr, "prompt tokens: %zu, layers: %s\n", toks.size(), layers_arg.c_str());

    const int n_batch = llama_n_batch(ctx);
    for (size_t i = 0; i < toks.size(); i += n_batch) {
        const int n = std::min<int>(n_batch, (int) toks.size() - (int) i);
        llama_batch b = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(ctx, b)) { fprintf(stderr, "prefill decode failed at %zu\n", i); return 1; }
        if ((i / n_batch) % 32 == 0) fprintf(stderr, "prefill %zu/%zu\n", i + n, toks.size());
    }
    st.decode_phase = true;
    llama_token tok = 0;
    std::string gen;
    for (int s = 0; s < n_steps; ++s) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int nv = llama_vocab_n_tokens(vocab);
        tok = (llama_token) (std::max_element(logits, logits + nv) - logits);
        gen += common_token_to_piece(ctx, tok);
        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    std::ofstream m(out_dir + "/meta.txt");
    m << "n_prompt " << toks.size() << "\nn_steps " << n_steps << "\nlayers " << layers_arg
      << "\nn_head " << llama_model_n_head(model) << "\nn_head_kv " << llama_model_n_head_kv(model)
      << "\nhead_dim 256\ngenerated " << gen << "\n";
    fprintf(stderr, "done. generated: %s\n", gen.c_str());
    llama_backend_free();
    return 0;
}
