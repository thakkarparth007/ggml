#include <iostream>
#include <sstream>
#include "utils.h"
#include "crow_all.h"
#include "common.h"

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

int main(int argc, char** argv) {

    gpt_params params;
    params.model = "models/gpt-j-6B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);


    crow::SimpleApp app;

    gpt_vocab vocab;
    gptj_model model;

    int64_t t_load_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!gptj_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }


    CROW_ROUTE(app, "/")([](){
        return "Hello world";
    });

    CROW_ROUTE(app, "/copilot_internal/v2/token")([](){
        //return "Hello world";

        crow::json::wvalue response = {{"token","1"}, {"expires_at", static_cast<std::uint64_t>(2600000000)}, {"refresh_in",900}};

        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "application/json");
        res.body = response.dump();
        return res;
    });


    CROW_ROUTE(app, "/v1/engines/codegen/completions").methods(crow::HTTPMethod::POST)
    ([&model, &vocab, &params](const crow::request& req) {
        crow::json::rvalue data = crow::json::load(req.body);

        if(!data.has("prompt") && !data.has("input_ids")){
            crow::response res;
            res.code = 400;
            res.set_header("Content-Type", "application/json");
            res.body = "{\"message\":\"you must specify a prompt or input_ids\"}";
            return res;
        }

        // tokenize the prompt
        std::vector<gpt_vocab::id> embd_inp;
        if (data.has("prompt")) {
            std::string prompt = data["prompt"].s();
            embd_inp = ::codegen_tokenize(vocab, prompt);
        } else {
            crow::json::rvalue input_ids = data["input_ids"];
            for (auto id : input_ids.lo()) {
                embd_inp.push_back(id.i());
            }
        }

        printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        printf("\n");

        // std::string suffix = data["suffix"].s();
        int maxTokens = 200;
        if(data.has("max_tokens")){
            maxTokens = data["max_tokens"].i();
        }
        std::string modelName = "";
        std::string suffix = "";
        float temperature = 0.6;

        data["model"].s();

        if(data.has("suffix")){
            suffix = data["suffix"].s();
        }


        if(data.has("temperature")){
            temperature = data["temperature"].d();
        }


        std::vector<float> logits;

        std::vector<gpt_vocab::id> embd;

    size_t mem_per_token = 0;
    gptj_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);
    int n_past = 0;
    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;
    std::mt19937 rng(params.seed);
    std::stringstream ss;

    int32_t n_predict = std::min(maxTokens, model.hparams.n_ctx - (int) embd_inp.size());

    for (int i = embd.size(); i < embd_inp.size() + n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gptj_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                crow::response res;
                res.code = 500;
                res.body = "{\"message\":\"failed to predict\"}";
                res.add_header("Content-Type","application/json");

                return res;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);

            // display text
            for (auto id : embd) {
                ss << vocab.id_to_token[id].c_str();
                //printf("%s", vocab.id_to_token[id].c_str());
            }
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        } 


        //fflush(stdout);

        // end of text token
        if (embd.back() == 50256) {
            break;
        }
    }

        boost::uuids::uuid uuid = boost::uuids::random_generator()();


        // Generate a mock response based on the input parameters
        crow::json::wvalue choice = {
            {"text", ss.str()},
            {"index",0},
            {"finish_reason", "length"},
            {"logprobs", nullptr}
        };
        crow::json::wvalue::list choices = {choice};


        crow::json::wvalue usage = {
            {"completion_tokens", n_past},
            {"prompt_tokens", static_cast<std::uint64_t>(embd_inp.size())},
            {"total_tokens", static_cast<std::uint64_t>(n_past + embd_inp.size())}
        };

        crow::json::wvalue response = {
            {"id", boost::lexical_cast<std::string>(uuid)},
            {"model", "codegen"},
            {"object","text_completion"},
            {"created", static_cast<std::int64_t>(std::time(nullptr))},
            {"choices", choices },
            {"usage", usage}
        };

        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "application/json");

        res.body = response.dump(); //ss.str();
        return res;
    });

    app.port(18080).run();

}