#include "models.h"
#include <cinttypes>

namespace {
struct llm_graph_input_tokens_only : public llm_graph_input_i {
    void set_input(const llama_ubatch * ubatch) override {
        if (ubatch->token) {
            const int64_t n_tokens = ubatch->n_tokens;
            ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens * ggml_element_size(tokens));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    }

    ggml_tensor * tokens = nullptr;
};
}

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    // Read the target model's hidden size from assistant metadata
    uint32_t backbone_hidden_size = 0;
    if (ml.get_key(llm_kv(LLM_KV_ASSISTANT_BACKBONE_HIDDEN_SIZE), backbone_hidden_size, false)) {
        hparams.n_embd_out_impl = backbone_hidden_size;
        LLAMA_LOG_INFO("%s: using assistant backbone_hidden_size=%u as n_embd_out\n", 
                __func__, backbone_hidden_size);
    }

    hparams.n_embd_inp_impl = hparams.n_embd_out();

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.f_attention_scale = 1.0f;

    LLAMA_LOG_INFO("%s: BEFORE base loader values:\n", __func__);
    LLAMA_LOG_INFO("%s:   n_layer() = %u, n_layer_all = %u, n_layer_nextn = %u\n", 
            __func__, hparams.n_layer(), hparams.n_layer_all, hparams.n_layer_nextn);
    LLAMA_LOG_INFO("%s:   n_head_arr[0] = %u, n_ff_arr[0] = %u\n", 
            __func__, hparams.n_head_arr[0], hparams.n_ff_arr[0]);
    LLAMA_LOG_INFO("%s:   n_embd_head_k_full = %u, n_embd_head_v_full = %u\n", 
            __func__, hparams.n_embd_head_k_full, hparams.n_embd_head_v_full);
    LLAMA_LOG_INFO("%s:   swa_layers[0] = %u\n", __func__, hparams.swa_layers[0]);

    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer());

    LLAMA_LOG_INFO("%s: AFTER SWA pattern read: swa_layers[0] = %u\n", 
            __func__, hparams.swa_layers[0]);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);
    GGML_UNUSED(n_kv_shared_layers);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn == 0) {
        hparams.n_layer_nextn = hparams.n_layer_all;
    }
    // For gemma4-assistant, all layers are nextn layers
    GGML_ASSERT(hparams.n_layer_nextn == hparams.n_layer_all &&
            "gemma4-assistant expects n_layer_nextn to equal n_layer_all");

    LLAMA_LOG_INFO("%s: AFTER n_layer_nextn set: n_layer() = %u\n", 
            __func__, hparams.n_layer());

    // n_layer() = n_layer_all - n_layer_nextn = 0 (no regular layers)

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT,    hparams.n_head_arr,    hparams.n_layer_all);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer_all);
    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,     hparams.n_ff_arr,      hparams.n_layer_all);

    LLAMA_LOG_INFO("%s: AFTER array re-read:\n", __func__);
    LLAMA_LOG_INFO("%s:   n_head_arr[0] = %u, n_head_kv_arr[0] = %u, n_ff_arr[0] = %u\n", 
            __func__, hparams.n_head_arr[0], hparams.n_head_kv_arr[0], hparams.n_ff_arr[0]);

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,           hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,     hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  hparams.f_norm_rms_eps);
    
    LLAMA_LOG_INFO("%s: BEFORE head_k/v re-read: n_embd_head_k_full = %u\n", 
            __func__, hparams.n_embd_head_k_full);
    
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH,         hparams.n_embd_head_k_full, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH,       hparams.n_embd_head_v_full, false);
    
    LLAMA_LOG_INFO("%s: AFTER head_k/v re-read: n_embd_head_k_full = %u, n_embd_head_v_full = %u\n", 
            __func__, hparams.n_embd_head_k_full, hparams.n_embd_head_v_full);
    
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,     hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,   hparams.n_embd_head_v_swa);

    LLAMA_LOG_INFO("%s: FINAL values:\n", __func__);
    LLAMA_LOG_INFO("%s:   n_embd_head_k_swa = %u, n_embd_head_v_swa = %u\n", 
            __func__, hparams.n_embd_head_k_swa, hparams.n_embd_head_v_swa);
    LLAMA_LOG_INFO("%s:   n_swa = %u\n", __func__, hparams.n_swa);
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 assistant requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 assistant requires n_embd_head_k_swa == n_embd_head_v_swa");
    }
    if (hparams.n_embd_out() == n_embd) {
        throw std::runtime_error("Gemma 4 assistant requires embedding_length_out to carry the target hidden size");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);

    output_norm  = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    mtp_post_proj = create_tensor(tn(LLM_TENSOR_ASSISTANT_POST_PROJ, "weight"), { n_embd, hparams.n_embd_out() }, 0);
    int rope_freqs_flag = 0;
    const int64_t n_embd_backbone = hparams.n_embd_out();

    for (int i = 0; i < (int) hparams.n_layer_nextn; ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff        = hparams.n_ff(i);

        if (i < 4 || i == (int) hparams.n_layer_nextn - 1) {
            LLAMA_LOG_INFO("%s: layer %d: n_head=%" PRId64 ", n_embd_head=%" PRId64 ", n_ff=%" PRId64 ", is_swa=%d\n",
                    __func__, i, n_head, n_embd_head, n_ff, hparams.is_swa(i));
        }

        if (i == 0) {
            mtp_pre_proj = create_tensor(tn(LLM_TENSOR_ASSISTANT_PRE_PROJ, "weight"), { 2 * n_embd_backbone, n_embd }, 0);
        }

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), { n_embd, n_embd_head * n_head }, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), { n_embd_head * n_head, n_embd }, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), { n_embd_head }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, nullptr, i), { 1u }, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head / 2 }, TENSOR_NOT_REQUIRED | rope_freqs_flag);
            if (layer.rope_freqs != nullptr) {
                rope_freqs_flag = TENSOR_DUPLICATED;
            }
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), { n_embd }, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params) :
llm_graph_context(params) {
    const int64_t n_embd_backbone = hparams.n_embd_out();
    const bool has_target_ctx = cparams.ctx_other != nullptr;

    if (!has_target_ctx) {
        auto inp = std::make_unique<llm_graph_input_tokens_only>();
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens_stub", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;
        ggml_tensor * cur = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
        res->add_input(std::move(inp));

        if (hparams.n_embd < n_embd_backbone) {
            cur = ggml_pad(ctx0, cur, n_embd_backbone - hparams.n_embd, 0, 0, 0);
        }
        cur = ggml_scale(ctx0, cur, sqrtf((float) n_embd_backbone));
        cb(cur, "inp_embd_stub", -1);

        ggml_tensor * xh = ggml_concat(ctx0, cur, cur, 0);
        cb(xh, "inp_xh_stub", -1);

        cur = ggml_mul_mat(ctx0, model.mtp_pre_proj, xh);
        cb(cur, "pre_proj_stub", -1);

        for (int il = 0; il < (int) hparams.n_layer_nextn; ++il) {
            ggml_tensor * residual = cur;

            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm_stub", il);

            cur = build_lora_mm(model.layers[il].wq, cur);
            cb(cur, "attn_q_proj_stub", il);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_out_stub", il);

            cur = ggml_add(ctx0, cur, residual);
            cb(cur, "attn_residual_stub", il);

            residual = cur;
            cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm_stub", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out_stub", il);

            cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm_stub", il);

            cur = ggml_add(ctx0, cur, residual);
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled_stub", il);
        }

        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "result_norm_stub", -1);

        ggml_tensor * logits = build_lora_mm(model.output, cur);
        cb(logits, "result_output_stub", -1);
        res->t_logits = logits;

        ggml_tensor * h_next = ggml_mul_mat(ctx0, model.mtp_post_proj, cur);
        cb(h_next, "h_nextn_stub", -1);
        res->t_h_nextn = h_next;

        ggml_build_forward_expand(gf, logits);
        ggml_build_forward_expand(gf, h_next);
        return;
    }

    ggml_tensor * inp_tokens;
    ggml_tensor * inp_h;
    {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_backbone);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        inp_tokens = inp->tokens;
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, ubatch.n_tokens);
        cb(inp->embd, "inp_h", -1);
        ggml_set_input(inp->embd);
        inp_h = inp->embd;
        res->t_inp_embd = inp->embd;

        res->add_input(std::move(inp));
    }
    const auto * model_other = llama_get_model(cparams.ctx_other);
    ggml_tensor * x = ggml_get_rows(ctx0, model_other->tok_embd, inp_tokens);
    x = ggml_scale(ctx0, x, sqrtf((float) n_embd_backbone));
    cb(x, "inp_embd_target", -1);

    ggml_tensor * xh = ggml_concat(ctx0, x, inp_h, 0);
    cb(xh, "inp_xh", -1);

    ggml_tensor * cur = ggml_mul_mat(ctx0, model.mtp_pre_proj, xh);
    cb(cur, "pre_proj", -1);

    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inpL = cur;

    for (int il = 0; il < (int) hparams.n_layer_nextn; ++il) {
        const bool is_swa = hparams.is_swa(il);
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head = hparams.n_head(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        ggml_tensor * cur_norm = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur_norm, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur_norm);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);

        if (il == (int) hparams.n_layer_nextn - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);
        cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        cb(cur, "out_scaled", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * logits = build_lora_mm(model.output, cur);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_tensor * h_next = ggml_mul_mat(ctx0, model.mtp_post_proj, cur);
    cb(h_next, "h_nextn", -1);
    res->t_h_nextn = h_next;

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, h_next);
}
