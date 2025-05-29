/* main
    common_init - only logging
    common_params --> this strucutre is the core structure which holds all model parameters
        example to some parameters -
            int32_t n_predict             =    -1; // new tokens to predict
            int32_t n_ctx                 =  4096; // context size
            int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
            int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
            int32_t n_keep                =     0; // number of tokens to keep from initial prompt
            int32_t n_chunks              =    -1; // max number of chunks to process (-1 = unlimited)
            int32_t n_parallel            =     1; // number of parallel sequences to decode
            int32_t n_sequences           =     1; // number of sequences to decode
            int32_t grp_attn_n            =     1; // group-attention factor
            int32_t grp_attn_w            =   512; // group-attention width
            int32_t n_print               =    -1; // print token count every n tokens (-1 = disabled) 

            std::string model                = ""; // model path                                                    // NOLINT
            std::string model_alias          = ""; // model alias                                                   // NOLINT
            std::string model_url            = ""; // model url to download                                         // NOLINT
            std::string hf_token             = ""; // HF token                                                      // NOLINT
            std::string hf_repo              = ""; // HF repo                                                       // NOLINT
            std::string hf_file              = ""; // HF file                                                       // NOLINT
            std::string prompt               = "";                                                                  // NOLINT
            std::string prompt_file          = ""; // store the external prompt file name                           // NOLINT
            std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state             // NOLINT
            std::string input_prefix         = ""; // string to prefix user inputs with                             // NOLINT
            std::string input_suffix         = ""; // string to suffix user inputs with                             // NOLINT
            std::string lookup_cache_static  = ""; // path of static ngram cache file for lookup decoding           // NOLINT
            std::string lookup_cache_dynamic = ""; // path of dynamic ngram cache file for lookup decoding          // NOLINT
            std::string logits_file          = ""; // file for saving *all* logits    

    llama_backend_init();
        ggml_init();
            ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16); -- a 256kb buffer
            return ctx;  --> this is like scratch space for further computation and not to store weights

    llama_numa_init(params.numa);  --> llama_numa_init prepares the threads and resources for execution 
    ***********
    ******below code loads model from hf, url or from file*****
    common_init_result llama_init = common_init_from_params(params);
    model = llama_init.model.get();
    ctx = llama_init.context.get();
    
    CONTINUE from line 160 of main.cpp
    
    */