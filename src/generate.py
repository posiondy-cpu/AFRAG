import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,3,4,5'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6,7'

from pydoc import text
import numpy as np
import logging
from regex import T
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re
from pydantic import BaseModel
from openai import OpenAI
from text2vec import SentenceModel, cos_sim
import random
import string
class SubQuestion(BaseModel):
    questions: list[str]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id=self.tokenizer.eos_token_id
    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_length,
                return_dict_in_generate = True,
                output_scores = True,
                # temperature = 0.0,
                do_sample=False,
                pad_token_id = self.tokenizer.eos_token_id
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs

        else:
            outputs = self.model.generate(
                input_ids = input_ids,
                max_new_tokens = max_length,
                attention_mask = attention_mask,
                # temperature = 0.0,
                do_sample=False
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            # 避免这种情况
            # text = """Blind Shaft was released on September 20, 2002.
            # Question 2: Who directed Blind Shaft?
            # Answer:xxx.
            # Question 3：xxx?
            # Answer：xxx."""

            doc = nlp(text)
            sentences = list(doc.sents)

            # 提取第一个 "Question" 之前的句子
            result_sentences = []
            for sentence in sentences:
                if sentence.text.strip().startswith("Question"):
                    break
                result_sentences.append(sentence.text.strip())

            # 拼接保留的句子为最终结果
            result_text = " ".join(result_sentences)
            return result_text.strip(), None, None

    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_length,
            return_dict_in_generate = True,
            output_scores = True,
            # temperature = 0.0,
            do_sample=False,
            pad_token_id = self.tokenizer.eos_token_id
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])
        def convert_to_string(input_data):
            if isinstance(input_data, bytes):
                return input_data.decode('utf-8')
            else:
                return input_data

        tokens = [convert_to_string(t) for t in tokens]
        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(str(self.space_token)) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>' or tokens[i-1] == '<|endoftext|>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1


        # regular tokens
        seqlist = []
        # attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(str(self.space_token), "")
            # value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            # attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.to(torch.float).cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
        else:
            seqentropies = None
        # 删除text中Question之后的内容，tokens也要对应删除

        # 找到"Question"在text中的位置
        question_pos = -1
        for i, token in enumerate(tokens):
            if "Question" in token:
                question_pos = i
                break
        # print("Question pos:", question_pos)
        if question_pos != -1:
            # 保留"Question"之前的内容
            text = text[:question_pos]

            try:
                token_index = question_pos
                tokens = tokens[:token_index]
                seqlist = seqlist[:token_index]
                # attns = attns[:token_index]
                if seqlogprobs:
                    seqlogprobs = seqlogprobs[:token_index]
                if seqentropies:
                    seqentropies = seqentropies[:token_index]
            except ValueError:
                pass
        return text, seqlist, None, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated,
            "token_count": self.token - other_counter.token,
            "sentence_count": self.sentence - other_counter.sentence
        }


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        self.enbedding = SentenceModel('shibing624/text2vec-base-multilingual')

        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name,
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path,
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError

        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
            )
            return docs[0]
        else:
            raise NotImplementedError

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""

    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            docs = self.retrieve(question, topk=self.retrieve_topk)
            # 对 topk 个 passage 生成 prompt
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1

        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt,
                self.generate_max_length,
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)

        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)

        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)]
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0

                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None



    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        tl, tr = 0, len(tokens)
        curr_st = len(tokens) - len(curr_tokens)
        attns = []
        for r in range_:
            att = torch.zeros(atten_tmp.shape[0], input_length)
            for i in range(r[0], r[1] + 1):
                att += atten_tmp[:, i]
            att /= (r[1] - r[0] + 1)
            att = torch.mean(att, dim=0)
            att = att[tl:tr]
            if att.shape[0] > 1:
                att = att / sum(att[1:]).item()
            attns.append(att)

        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(tr - tl)
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if match(tok):
                real_pairs.append((att, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # print('####', prompt)
            # prompt += case + " " + text
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_max_length,
                # self.attention_solver,
                use_entropy = self.method == "dragin",
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)

            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all

                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)

                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)

                elif self.query_formulation == "real_words":
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext,
                        curr_tokens = curr_tokens,
                        curr_hit = curr_hit,
                    )
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                logger.info(f"Retrieve question: {retrieve_question}")
                logger.info(f"Prompt: {prompt}")

                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

                # print("### retrieve_question ###")
                # print(retrieve_question)
                # context = "### Context: ###\n"
                # for i, doc in enumerate(docs):
                #     context += f"[] {doc}\n"
                # print(context)
                # print(text)

            # 判断 token 的个数要少于 generate_max_length
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        # print("#" * 20)
        return text




class ARAM(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

        self.client = client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-WUE0159f560e702e1bde0f9ea2c3dd935f1b089833f956V1"
        )

        self.prompt = """
        Question: When did the director of film Hypocrite (Film) die?
        Decomposition: 1. Who is the director of the film Hypocrite? 2. When did this director die?
        Question: Are both Kurram Garhi and Trojkrsti located in the same country?
        Decomposition: 1. Where is Kurram Garhi located? 2. Where is Trojkrsti located? 3. Are these locations in the same country?
        Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
        Decomposition: 1. Who directed Coolie No. 1 (1995)? 2. Who directed The Sensational Trial? 3. What is the nationality of each director? 4. Do these directors share the same nationality?
        Question: Who is Boraqchin (Wife Of Ögedei)'s father-in-law?
        Decomposition: 1. Who was Ögedei's father?
        Question: Who was born first out of Martin Hodge and Ivania Martinich?
        Decomposition: 1. What is the birthdate of Martin Hodge? 2. What is the birthdate of Ivania Martinich? 3. Which of these individuals was born first?
        Question: When did the director of film Laughter In Hell die?
        Decomposition: 1. Who directed the film Laughter In Hell? 2. When did this director die?
        Decompose the questions as necessary, following the examples above.
        Question:
        """

    def modifier(self, question, modify_prob=0.2):
        # 随机增、删、改单词中的字母，操作概率通过参数控制，且保证至少进行一次修改
        def random_modify(word, modify_prob):
            modified = False
            word_list = list(word)

            # 创建一个副本以避免在遍历过程中修改列表
            original_length = len(word_list)

            for i in range(original_length):
                if random.random() < modify_prob:
                    operation = random.choice(['add', 'delete', 'replace'])
                    if operation == 'add':
                        # 在当前位置添加一个随机字母
                        new_char = random.choice(string.ascii_letters)
                        word_list.insert(i, new_char)
                        modified = True
                    elif operation == 'delete' and len(word_list) > 1 and i < len(word_list) - 1:
                        # 删除当前位置的字母
                        word_list.pop(i)
                        modified = True
                    elif operation == 'replace' and i < len(word_list) - 1:
                        # 替换当前位置的字母
                        new_char = random.choice(string.ascii_letters)
                        word_list[i] = new_char
                        modified = True

            # 如果没有进行任何修改，强制进行一次替换操作
            if not modified and len(word_list) > 0:
                index = random.choice(range(len(word_list)))
                new_char = random.choice(string.ascii_letters)
                word_list[index] = new_char

            return ''.join(word_list)
        doc = nlp(question)

        entities = [ent.text for ent in doc.ents]


        # 修改实体中的单词
        new_entities = [random_modify(entity, modify_prob) for entity in entities]

        # 替换原始问题中的实体
        new_question = question
        for old, new in zip(entities, new_entities):
            new_question = new_question.replace(old, new, 1)

        return new_question

    def inference(self, question, demo, case):
        # 问题分解
        print(f'Original question is:{question}')
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": question},
            ],
            response_format=SubQuestion,
            temperature=0
        )

        event = completion.choices[0].message.parsed

        ############################################################
        text = ""
        # threshold = 0.65 # 相似度阈值
        # modify_prob = 0.2 # 单词修改概率
        # alpha = 0.3 # 平均相似度和修改问题相似度的权重
        # theta = 0.82    # token概率阈值

        for i, e in enumerate(event.questions):
            cur_question = f"Question: {e}\n"
            # print(cur_question)
            prompt = text + cur_question + "Answer: "
            original_answer, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                max_length=self.generate_max_length,
                use_entropy = False,
                use_logprob = True
            )

            if self.use_counter == True:
                self.counter.add_generate(original_answer, self.generator.tokenizer)

            propn = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
            
            
            hullucination = False
            for i, token in enumerate(tokens):
                # print(token, exp(logprobs[i]))
                # 删除token中Question后的内容
                if "Question" in token:
                    tokens = tokens[:i]
                    logprobs =logprobs[:i]
                    break
            original_answer = ' '.join(tokens)
            original_answer = re.sub(r'<[^>]+>', '', original_answer)
            doc = nlp(original_answer)
            for ent in doc.ents:
                if ent.label_ in propn:
                    #遍历tokens，找到匹配的实体
                    l = ent[-1].i - ent[0].i + 1
                    # print("l: ", l)
                    start_token_index=-1
                    for i, token in enumerate(tokens):
                        #去除token前后的标点空格
                        token = re.sub(r'|^[^a-zA-Z]+|[^a-zA-Z]+$|^[\s\W]+|[\s\W]+$', '', token).strip()
                        if token in ent.text:
                            j = i
                            # print("j: ", j)
                            # print("token: ", token)

                            while j < len(tokens) and re.sub(r'|^[^a-zA-Z]+|[^a-zA-Z]+$|^[\s\W]+|[\s\W]+$', '', tokens[j]).strip() in ent.text:
                                j += 1
                            # print("j: ", j)
                            # print("j - i: ", j - i)
                            if (j - i == l and l == 1) or (j - i >= 2):
                                start_token_index = i
                                break
                    if start_token_index == -1:
                        # print("Can't find the entity.")
                        continue

                    for i in range(start_token_index, start_token_index + l):
                        # print(i, tokens[i], exp(logprobs[i]))
                        if i < len(tokens) and exp(logprobs[i]) < self.theta and (i > 0 or len(tokens) == 1):
                            hullucination = True
                            break
                    if hullucination:
                        break
            if hullucination:
                print("Hullucination")


            o_question = e
            k = 2 # 修改问题的次数
            modified_questions = [self.modifier(o_question, modify_prob=self.modify_prob) for _ in range(k)]
            modified_answers = []

            for mq in modified_questions:
                prompt = text + f"Question: {mq}\n" + "Answer: "
                modified_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                modified_answers.append(modified_answer)

            original_embedding = self.enbedding.encode(original_answer)
            modified_embeddings = [self.enbedding.encode(ma) for ma in modified_answers]
            similarities = [cos_sim(me, original_embedding) for me in modified_embeddings]
            avg_sim = sum(similarities) / len(similarities)

            # 计算修改问题的回答之间的相似度
            m_similarities = [cos_sim(modified_embeddings[i], modified_embeddings[j])
                              for i in range(k) for j in range(i + 1, k)]
            avg_m_similarity = sum(m_similarities) / len(m_similarities)

            combined_similarity = self.alpha * avg_sim + (1 - self.alpha) * avg_m_similarity[0][0]
            # 高于阈值说明需要检索更多信息


            # if (combined_similarity > self.hallucination_threshold and modified_questions[0] != modified_questions[1]) or hullucination:
            if hullucination:    
                # print("Need more information.")
                # 删去概率低于阈值的tokend的内容，如果第一次出现低于阈值的token索引大于4则删除后面的内容
                filtered_tokens = []
                first_low_prob_index = None

                for i, token in enumerate(tokens):
                    prob = exp(logprobs[i])
                    if prob >= self.theta or i == 0 or i == 1:
                        filtered_tokens.append(token)
                    else:
                        if first_low_prob_index is None:
                            first_low_prob_index = i
                # print("first_low_prob_index: ", first_low_prob_index)
                # 如果第一次低于阈值的 token 索引大于 4，则删除后面的内容
                if first_low_prob_index is not None and first_low_prob_index > 4:
                    filtered_tokens = tokens[:first_low_prob_index]


                tokens = filtered_tokens
                use_question = False
                keywords = []
                if len(tokens) > 4 and len(tokens) <= 20: # 适当且足够信息
                    cleaned_keywords = ' '.join(tokens)
                    doc = nlp(cleaned_keywords)
                    keywords = [ent.text for ent in doc.ents if ent.label_ in propn]
                    if not keywords: # 无意义
                        use_question = True
                if keywords == [] or use_question:
                    print("*****Use question.*****")
                    # 只需要之前问题的答案和当前问题中的实体作为上下文检索,
                    pattern = r"Answer: (.*?)(?=\nQuestion\d+:|$)"
                    matches = re.findall(pattern, text, re.DOTALL)
                    retrieve_question = " ".join([match.strip() for match in matches]) + " " + e

                    # 替换连字符
                    retrieve_question = re.sub(r'[-()]', ' ', retrieve_question)

                    print("retrieve_question: ", retrieve_question)
                    doc = nlp(retrieve_question)
                    keywords = []
                    for token in doc:
                        if token.ent_type_ or token.pos_ in ["NOUN", "PROPN", "VERB"]:
                            keywords.append(token.text)
                    # keywords = " ".join(keywords)
                    # 拼接剩余的tokens
                    retrieve_question += ' '.join(tokens)

                    # 移除标记内容如 <0x0A> 等，但保留 </s>
                    retrieve_question = re.sub(r'<(?!/s>).+?>', '', retrieve_question)
                    keywords = [re.sub(r'</s', '', keyword) for keyword in keywords]
                    cleaned_keywords = " ".join(keywords)

                print(f"keywords: {cleaned_keywords}")
                _docs_ids, docs = self.retriever.retrieve([cleaned_keywords], topk=2)
                prompt = "Context:"
                prompt += '\n'.join([doc for doc in docs[0]])
                prompt += "\n Answer the following question: \n"
                prompt += cur_question
                original_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                # print(prompt)
                if self.use_counter == True:
                    self.counter.add_generate(original_answer, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                # print("Generated Answer: ", original_answer)
            text += cur_question
            text += "Answer: " + original_answer + "\n"
            print(text)

        prompt = text
        prompt += "\n{}Based on the above questions and answers, provide a brief answer without any explanations or additional information. Start with \"So the answer is \". ".format(question)
        final_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
        # 删除数字和点
        if final_answer.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            final_answer = final_answer.split(".", 1)[1].strip()

        # print(final_answer)
        return final_answer.strip()






class ARAM_SC(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

        self.client = client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-xmM274b35dcca1780cbf4a2842fd822a95ef1304b1aJ1PwT"
        )

        self.prompt = """
        Question: When did the director of film Hypocrite (Film) die?
        Decomposition: 1. Who is the director of the film Hypocrite? 2. When did this director die?
        Question: Are both Kurram Garhi and Trojkrsti located in the same country?
        Decomposition: 1. Where is Kurram Garhi located? 2. Where is Trojkrsti located? 3. Are these locations in the same country?
        Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
        Decomposition: 1. Who directed Coolie No. 1 (1995)? 2. Who directed The Sensational Trial? 3. What is the nationality of each director? 4. Do these directors share the same nationality?
        Question: Who is Boraqchin (Wife Of Ögedei)'s father-in-law?
        Decomposition: 1. Who was Ögedei's father?
        Question: Who was born first out of Martin Hodge and Ivania Martinich?
        Decomposition: 1. What is the birthdate of Martin Hodge? 2. What is the birthdate of Ivania Martinich? 3. Which of these individuals was born first?
        Question: When did the director of film Laughter In Hell die?
        Decomposition: 1. Who directed the film Laughter In Hell? 2. When did this director die?
        Decompose the questions as necessary, following the examples above.
        Question:
        """

    def modifier(self, question, modify_prob=0.2):
        # 随机增、删、改单词中的字母，操作概率通过参数控制，且保证至少进行一次修改
        def random_modify(word, modify_prob):
            modified = False
            word_list = list(word)

            # 创建一个副本以避免在遍历过程中修改列表
            original_length = len(word_list)

            for i in range(original_length):
                if random.random() < modify_prob:
                    operation = random.choice(['add', 'delete', 'replace'])
                    if operation == 'add':
                        # 在当前位置添加一个随机字母
                        new_char = random.choice(string.ascii_letters)
                        word_list.insert(i, new_char)
                        modified = True
                    elif operation == 'delete' and len(word_list) > 1 and i < len(word_list) - 1:
                        # 删除当前位置的字母
                        word_list.pop(i)
                        modified = True
                    elif operation == 'replace' and i < len(word_list) - 1:
                        # 替换当前位置的字母
                        new_char = random.choice(string.ascii_letters)
                        word_list[i] = new_char
                        modified = True

            # 如果没有进行任何修改，强制进行一次替换操作
            if not modified and len(word_list) > 0:
                index = random.choice(range(len(word_list)))
                new_char = random.choice(string.ascii_letters)
                word_list[index] = new_char

            return ''.join(word_list)
        doc = nlp(question)

        entities = [ent.text for ent in doc.ents]


        # 修改实体中的单词
        new_entities = [random_modify(entity, modify_prob) for entity in entities]

        # 替换原始问题中的实体
        new_question = question
        for old, new in zip(entities, new_entities):
            new_question = new_question.replace(old, new, 1)

        return new_question

    def inference(self, question, demo, case):
        # 问题分解
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": question},
            ],
            response_format=SubQuestion,
            temperature=0
        )

        event = completion.choices[0].message.parsed

        ############################################################
        text = ""
        # threshold = 0.65 # 相似度阈值
        # modify_prob = 0.2 # 单词修改概率
        # alpha = 0.3 # 平均相似度和修改问题相似度的权重
        # theta = 0.82    # token概率阈值
        # print("self check")
        for i, e in enumerate(event.questions):
            cur_question = f"Question: {e}\n"
            # print(cur_question)

            modify_quesiton = """
If the given question is already specific, return it as is. Otherwise, modify it to make it more precise. Example:
Question: Who was John V, Prince Of Anhalt-Zerbst's father?
Answer: Ernest I, Prince of Anhalt-Dessau.</s>
Original Question: "When did this father die?"
Modified Question: "When did Ernest I, Prince of Anhalt-Dessau die?"
Now, process the following question accordingly:
Context:{}
Original Question:{}
Modified Question:
""".format(text, e)
            # 修改问题
            if i != 0:
                e = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": modify_quesiton},
                    ],
                    temperature=0
                ).choices[0].message.content
            cur_question = f"Question: {e}\n"
            # print(cur_question)
            prompt = text + cur_question + "Provide a brief answer without any explanations or additional information.Answer: "
            # original_answer, _, _ = self.generator.generate(prompt, max_length=64)
            original_answer, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                max_length=self.generate_max_length,
                # self.attention_solver,
                use_entropy = False,
                use_logprob = True
            )

            if self.use_counter == True:
                self.counter.add_generate(original_answer, self.generator.tokenizer)

            propn = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
            hullucination = False
            for i, token in enumerate(tokens):
                # print(i, token, exp(logprobs[i]))
                # 删除token中Question后的内容
                if "Question" in token:
                    tokens = tokens[:i]
                    logprobs = logprobs[:i]
                    break
            original_answer = ' '.join(tokens)
            # 删除<xx>标签
            original_answer = re.sub(r'<[^>]+>', '', original_answer)
            # print("tokens: ", tokens)
            # if "</s>" in tokens[-1]:
            #     tokens[-1] = tokens[-1].replace("</s>", "")
            #     original_answer = ' '.join(tokens)
            # print("original_answer: ", original_answer)
            doc = nlp(original_answer)
            for ent in doc.ents:
                # print(ent.text, ent.label_)
                if ent.label_ in propn:
                    #遍历tokens，找到匹配的实体
                    l = ent[-1].i - ent[0].i + 1
                    # print("l: ", l)
                    start_token_index=-1
                    for i, token in enumerate(tokens):
                        # 删除所有<>标记，如</s>
                        token = re.sub(r'<[^>]+>', '', token)
                        # 去除token首尾所有非字母数字字符
                        token = re.sub(r'^\W+|\W+$', '', token).strip()
                        # print("token: ", token)
                        if token in ent.text:
                            j = i
                            # print("j: ", j)
                            # print("token: ", token)

                            while j < len(tokens) and re.sub(r'^\W+|\W+$', '', re.sub(r'<[^>]+>', '', tokens[j])).strip() in ent.text:
                                j += 1
                            # print("j: ", j)
                            # print("j - i: ", j - i)
                            if (j - i == l and l == 1) or (j - i >= 2):
                                start_token_index = i
                                break
                    if start_token_index == -1:
                        # print("Can't find the entity.")
                        continue
                    # print("find start_token_index: ", start_token_index)
                    # print("l: ", l)
                    for i in range(start_token_index, start_token_index + l):
                        if i > len(tokens) - 1:
                            break
                        if i < len(tokens) and exp(logprobs[i]) < self.theta and (i > 0 or len(tokens) == 1):
                            hullucination = True
                            break
                        # print(i, tokens[i], exp(logprobs[i]))
                    if hullucination:
                        break

            t = ""
            if hullucination:
                print("Hullucination")
            else:
                print("self checking")
                o_question = e
                n_question1 = self.modifier(o_question,modify_prob=self.modify_prob)
                n_question2 = self.modifier(o_question,modify_prob=self.modify_prob)

                prompt = text + f"Question: {n_question1}\n" + "Provide a brief answer without any explanations or additional information.Answer: "
                modified_answer1, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                prompt = text + f"Question: {n_question2}\n" + "Provide a brief answer without any explanations or additional information.Answer: "
                modified_answer2 , _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)

                # 让模型自己判断输出中的分歧
                prompt = f"""
Based on the following questions and answers, determine if the model demonstrates relevant knowledge to answer the questions. In addition, ensure that all entities mentioned in the questions are real. If any answer indicates that an entity is "not a real [entity]" or similar, treat the response as NO. Respond with [YES or NO] only.
Original Question: {o_question}
Original Answer: {original_answer}
Random Modified Question 1: {n_question1}
Answer to Random Modified Question 1: {modified_answer1}
Random Modified Question 2: {n_question2}
Answer to Random Modified Question 2: {modified_answer2}
Does the model demonstrate relevant knowledge to answer these questions? Respond with [YES or NO]:
Output:
"""


                t = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are an expert in evaluating whether a model has sufficient knowledge of a given topic."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0
                ).choices[0].message.content

            flag  = True
            if "YES" in t.upper():
                flag = False
                print("YES")
            elif "NO" in t.upper():
                print("NO")

            if (flag) or hullucination:
                print("Need more information.")
                # 删去概率低于阈值的tokend的内容，如果第一次出现低于阈值的token索引大于4则删除后面的内容
                filtered_tokens = []
                first_low_prob_index = None

                for i, token in enumerate(tokens):
                    prob = exp(logprobs[i])
                    if prob >= self.theta or i == 0 or i == 1:
                        filtered_tokens.append(token)
                    else:
                        if first_low_prob_index is None:
                            first_low_prob_index = i
                # print("first_low_prob_index: ", first_low_prob_index)
                # 如果第一次低于阈值的 token 索引大于 4，则删除后面的内容
                if first_low_prob_index is not None and first_low_prob_index > 4:
                    filtered_tokens = tokens[:first_low_prob_index]


                tokens = filtered_tokens
                use_question = False
                keywords = []
                if len(tokens) > 4 and len(tokens) <= 20: # 适当且足够信息
                    cleaned_keywords = ' '.join(tokens)
                    doc = nlp(cleaned_keywords)
                    keywords = [ent.text for ent in doc.ents if ent.label_ in propn]
                    if not keywords: # 无意义
                        use_question = True
                    else:
                        print("*****Use tokens.*****")
                if keywords == [] or use_question:
                    print("*****Use question.*****")
                    # 只需要之前问题的答案和当前问题中的实体作为上下文检索,
                    pattern = r"Answer: (.*?)(?=\nQuestion\d+:|$)"
                    matches = re.findall(pattern, text, re.DOTALL)
                    retrieve_question = " ".join([match.strip() for match in matches]) + " " + e

                    # 替换连字符
                    retrieve_question = re.sub(r'[-()]', ' ', retrieve_question)

                    # print("retrieve_question: ", retrieve_question)
                    doc = nlp(retrieve_question)
                    keywords = []
                    for token in doc:
                        if token.ent_type_ or token.pos_ in ["NOUN", "PROPN", "VERB"]:
                            keywords.append(token.text)
                    # keywords = " ".join(keywords)
                    # 拼接剩余的tokens
                    retrieve_question += ' '.join(tokens)

                    # 移除标记内容如 <0x0A> 等，但保留 </s>
                    retrieve_question = re.sub(r'<(?!/s>).+?>', '', retrieve_question)
                    keywords = [re.sub(r'</s', '', keyword) for keyword in keywords]
                    cleaned_keywords = " ".join(keywords)

                _docs_ids, docs = self.retriever.retrieve([cleaned_keywords], topk=2)
                # print("docs: ", docs)
                prompt = "Context: "
                prompt += '\n'.join([doc for doc in docs[0]])
                prompt += "\nAnswer the following question: "
                prompt += text
                prompt += cur_question + "Answer: "
                original_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                # print("Context: ", prompt)
                if self.use_counter == True:
                    self.counter.add_generate(original_answer, self.generator.tokenizer)
                    self.counter.hallucinated += 1

            text += cur_question
            text += "Answer: " + original_answer + "\n"

        prompt = text
        prompt += "\nProvide a brief answer without any explanations or additional information.\nSo {} \nAnswer: ".format(question)
        final_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
        # 删除数字和点
        if final_answer.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            final_answer = final_answer.split(".", 1)[1].strip()
        return final_answer.strip()




class ARAM_SC_1(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

        # model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
        # # self.critic = AutoTokenizer.from_pretrained(model_name_or_path, proxies={'http': '10.21.181.155:7890','https': '10.21.181.155:7890'})
        # # logger.info(f"Loading model from {model_name_or_path}")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, proxies={'http': '10.21.181.155:7890','https': '10.21.181.155:7890'})
        # self.model_config = AutoConfig.from_pretrained(model_name_or_path,proxies={'http': '10.21.181.155:7890','https': '10.21.181.155:7890'},
        #             trust_remote_code = "falcon" in model_name_or_path)
        # self.critic = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", proxies={'http': '10.21.181.155:7890','https': '10.21.181.155:7890'},
        #             trust_remote_code = "falcon" in model_name_or_path)
        # if self.model_config.model_type == "llama":
        #     self.space_token = "▁"
        # else:
        #     self.space_token = self.tokenizer.tokenize(' ')[0]

        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        self.client = client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-xmM274b35dcca1780cbf4a2842fd822a95ef1304b1aJ1PwT"
        )

        # self.client = client = OpenAI(
        #     # base_url="https://api.gptsapi.net/v1",
        #     api_key="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"
        # )

        self.prompt = """
        Question: When did the director of film Hypocrite (Film) die?
        Decomposition: 1. Who is the director of the film Hypocrite? 2. When did this director die?
        Question: Are both Kurram Garhi and Trojkrsti located in the same country?
        Decomposition: 1. Where is Kurram Garhi located? 2. Where is Trojkrsti located? 3. Are these locations in the same country?
        Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
        Decomposition: 1. Who directed Coolie No. 1 (1995)? 2. Who directed The Sensational Trial? 3. What is the nationality of each director? 4. Do these directors share the same nationality?
        Question: Who is Boraqchin (Wife Of Ögedei)'s father-in-law?
        Decomposition: 1. Who was Ögedei's father?
        Question: Who was born first out of Martin Hodge and Ivania Martinich?
        Decomposition: 1. What is the birthdate of Martin Hodge? 2. What is the birthdate of Ivania Martinich? 3. Which of these individuals was born first?
        Question: When did the director of film Laughter In Hell die?
        Decomposition: 1. Who directed the film Laughter In Hell? 2. When did this director die?
        Decompose the questions as necessary, following the examples above.
        Question:
        """

    def modifier(self, question, modify_prob=0.2):
        # 随机增、删、改单词中的字母，操作概率通过参数控制，且保证至少进行一次修改
        def random_modify(word, modify_prob):
            modified = False
            word_list = list(word)

            # 创建一个副本以避免在遍历过程中修改列表
            original_length = len(word_list)

            for i in range(original_length):
                if random.random() < modify_prob:
                    operation = random.choice(['add', 'delete', 'replace'])
                    if operation == 'add':
                        # 在当前位置添加一个随机字母
                        new_char = random.choice(string.ascii_letters)
                        word_list.insert(i, new_char)
                        modified = True
                    elif operation == 'delete' and len(word_list) > 1 and i < len(word_list) - 1:
                        # 删除当前位置的字母
                        word_list.pop(i)
                        modified = True
                    elif operation == 'replace' and i < len(word_list) - 1:
                        # 替换当前位置的字母
                        new_char = random.choice(string.ascii_letters)
                        word_list[i] = new_char
                        modified = True

            # 如果没有进行任何修改，强制进行一次替换操作
            if not modified and len(word_list) > 0:
                index = random.choice(range(len(word_list)))
                new_char = random.choice(string.ascii_letters)
                word_list[index] = new_char

            return ''.join(word_list)
        doc = nlp(question)

        entities = [ent.text for ent in doc.ents]


        # 修改实体中的单词
        new_entities = [random_modify(entity, modify_prob) for entity in entities]

        # 替换原始问题中的实体
        new_question = question
        for old, new in zip(entities, new_entities):
            new_question = new_question.replace(old, new, 1)

        return new_question

    def inference(self, question, demo, case):


        ############################################################
        text = ""
        # threshold = 0.65 # 相似度阈值
        # modify_prob = 0.2 # 单词修改概率
        # alpha = 0.3 # 平均相似度和修改问题相似度的权重
        # theta = 0.82    # token概率阈值
        print("self check")
        for i, e in enumerate(demo):
            cur_question = f"Question: {e}\n"
            # print(cur_question)

            modify_quesiton = """
If the given question is already specific, return it as is. Otherwise, modify it to make it more precise. Example:
Question: Who was John V, Prince Of Anhalt-Zerbst's father?
Answer: Ernest I, Prince of Anhalt-Dessau.</s>
Original Question: "When did this father die?"
Modified Question: "When did Ernest I, Prince of Anhalt-Dessau die?"
Now, process the following question accordingly:
Context:{}
Original Question:{}
Modified Question:
""".format(text, e)
            # 修改问题
            if i != 0:
                e = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": modify_quesiton},
                    ],
                    temperature=0
                ).choices[0].message.content
            cur_question = f"Question: {e}\n"
            # print(cur_question)
            prompt = text + cur_question + "Provide a brief answer without any explanations or additional information.Answer: "
            # original_answer, _, _ = self.generator.generate(prompt, max_length=64)
            original_answer, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                max_length=self.generate_max_length,
                # self.attention_solver,
                use_entropy = False,
                use_logprob = True
            )

            if self.use_counter == True:
                self.counter.add_generate(original_answer, self.generator.tokenizer)

            propn = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
            hullucination = False
            for i, token in enumerate(tokens):
                # print(i, token, exp(logprobs[i]))
                # 删除token中Question后的内容
                if "Question" in token:
                    tokens = tokens[:i]
                    logprobs = logprobs[:i]
                    break
            original_answer = ' '.join(tokens)
            # 删除<xx>标签
            original_answer = re.sub(r'<[^>]+>', '', original_answer)
            # print("tokens: ", tokens)
            # if "</s>" in tokens[-1]:
            #     tokens[-1] = tokens[-1].replace("</s>", "")
            #     original_answer = ' '.join(tokens)
            # print("original_answer: ", original_answer)
            doc = nlp(original_answer)
            for ent in doc.ents:
                # print(ent.text, ent.label_)
                if ent.label_ in propn:
                    #遍历tokens，找到匹配的实体
                    l = ent[-1].i - ent[0].i + 1
                    # print("l: ", l)
                    start_token_index=-1
                    for i, token in enumerate(tokens):
                        # 删除所有<>标记，如</s>
                        token = re.sub(r'<[^>]+>', '', token)
                        # 去除token首尾所有非字母数字字符
                        token = re.sub(r'^\W+|\W+$', '', token).strip()
                        # print("token: ", token)
                        if token in ent.text:
                            j = i
                            # print("j: ", j)
                            # print("token: ", token)

                            while j < len(tokens) and re.sub(r'^\W+|\W+$', '', re.sub(r'<[^>]+>', '', tokens[j])).strip() in ent.text:
                                j += 1
                            # print("j: ", j)
                            # print("j - i: ", j - i)
                            if (j - i == l and l == 1) or (j - i >= 2):
                                start_token_index = i
                                break
                    if start_token_index == -1:
                        # print("Can't find the entity.")
                        continue
                    # print("find start_token_index: ", start_token_index)
                    # print("l: ", l)
                    for i in range(start_token_index, start_token_index + l):
                        if i > len(tokens) - 1:
                            break
                        if i < len(tokens) and exp(logprobs[i]) < self.theta and (i > 0 or len(tokens) == 1):
                            hullucination = True
                            break
                        # print(i, tokens[i], exp(logprobs[i]))
                    if hullucination:
                        break

            t = ""
            if hullucination:
                print("Hullucination")
            else:
                print("self checking")
                o_question = e
                n_question1 = self.modifier(o_question,modify_prob=self.modify_prob)
                n_question2 = self.modifier(o_question,modify_prob=self.modify_prob)

                prompt = text + f"Question: {n_question1}\n" + "Provide a brief answer without any explanations or additional information.Answer: "
                modified_answer1, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                prompt = text + f"Question: {n_question2}\n" + "Provide a brief answer without any explanations or additional information.Answer: "
                modified_answer2 , _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)

                # 让模型自己判断输出中的分歧
                prompt = f"""
Based on the following questions and answers, determine if the model demonstrates relevant knowledge to answer the questions. In addition, ensure that all entities mentioned in the questions are real. If any answer indicates that an entity is "not a real [entity]" or similar, treat the response as NO. Respond with [YES or NO] only.
Original Question: {o_question}
Original Answer: {original_answer}
Random Modified Question 1: {n_question1}
Answer to Random Modified Question 1: {modified_answer1}
Random Modified Question 2: {n_question2}
Answer to Random Modified Question 2: {modified_answer2}
Does the model demonstrate relevant knowledge to answer these questions? Respond with [YES or NO]:
Output:
"""

                # new_text, t, attns, lp, entropies = self.generator.generate_attn(
                #     prompt,
                #     10,
                #     # self.attention_solver,
                #     use_entropy = False,
                #     use_logprob = True
                # )
                # print(new_text)
                # # yes token prob > \theta or no
                t = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are an expert in evaluating whether a model has sufficient knowledge of a given topic."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0
                ).choices[0].message.content
                # print(prompt)
                # print(t)
            flag    = True
            if "YES" in t.upper():
                flag = False
                print("YES")
            elif "NO" in t.upper():
                print("NO")

            if (flag) or hullucination:
                print("Need more information.")
                # 删去概率低于阈值的tokend的内容，如果第一次出现低于阈值的token索引大于4则删除后面的内容
                filtered_tokens = []
                first_low_prob_index = None

                for i, token in enumerate(tokens):
                    prob = exp(logprobs[i])
                    if prob >= self.theta or i == 0 or i == 1:
                        filtered_tokens.append(token)
                    else:
                        if first_low_prob_index is None:
                            first_low_prob_index = i
                # print("first_low_prob_index: ", first_low_prob_index)
                # 如果第一次低于阈值的 token 索引大于 4，则删除后面的内容
                if first_low_prob_index is not None and first_low_prob_index > 4:
                    filtered_tokens = tokens[:first_low_prob_index]


                tokens = filtered_tokens
                # print("tokens: ", tokens)
                # if ',' in cleaned_keywords:
                #     extracted_content = cleaned_keywords.split(',', 1)[1].strip()
                # else:
                #     extracted_content = cleaned_keywords
                use_question = False
                keywords = []
                if len(tokens) > 4 and len(tokens) <= 20: # 适当且足够信息
                    cleaned_keywords = ' '.join(tokens)
                    doc = nlp(cleaned_keywords)
                    keywords = [ent.text for ent in doc.ents if ent.label_ in propn]
                    if not keywords: # 无意义
                        use_question = True
                        # keywords = [ent.text for ent in doc.ents if ent.label_ in ["NOUN", "PROPN", "VERB"]]
                        # if len(keywords) >= 20:
                        #     use_question = True
                    else:
                        print("*****Use tokens.*****")
                if keywords == [] or use_question:
                    print("*****Use question.*****")
                    # 只需要之前问题的答案和当前问题中的实体作为上下文检索,
                    pattern = r"Answer: (.*?)(?=\nQuestion\d+:|$)"
                    matches = re.findall(pattern, text, re.DOTALL)
                    retrieve_question = " ".join([match.strip() for match in matches]) + " " + e

                    # 替换连字符
                    retrieve_question = re.sub(r'[-()]', ' ', retrieve_question)

                    # print("retrieve_question: ", retrieve_question)
                    doc = nlp(retrieve_question)
                    keywords = []
                    for token in doc:
                        if token.ent_type_ or token.pos_ in ["NOUN", "PROPN", "VERB"]:
                            keywords.append(token.text)
                    # keywords = " ".join(keywords)
                    # 拼接剩余的tokens
                    retrieve_question += ' '.join(tokens)

                    # 移除标记内容如 <0x0A> 等，但保留 </s>
                    retrieve_question = re.sub(r'<(?!/s>).+?>', '', retrieve_question)
                    keywords = [re.sub(r'</s', '', keyword) for keyword in keywords]
                    cleaned_keywords = " ".join(keywords)

                # # keywords为空直接使用问题进行检索
                # if not keywords:
                #     cleaned_keywords = e
                # else:
                #     cleaned_keywords = " ".join(keywords)

                # print(f"keywords: {cleaned_keywords}")
                _docs_ids, docs = self.retriever.retrieve([cleaned_keywords], topk=2)
                # print("docs: ", docs)
                prompt = "Context: "
                prompt += '\n'.join([doc for doc in docs[0]])
                prompt += "\nAnswer the following question: "
                prompt += text
                prompt += cur_question + "Answer: "
                original_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                # print("Context: ", prompt)
                if self.use_counter == True:
                    self.counter.add_generate(original_answer, self.generator.tokenizer)
                    self.counter.hallucinated += 1
            # print("Generated Answer: ", original_answer)

            text += cur_question
            text += "Answer: " + original_answer + "\n"
        # print("#" * 20)
        # print(text)
        # logger.info(f"ARAM: {text}")
        # print("#" * 20)
        prompt = text
        prompt += "\nProvide a brief answer without any explanations or additional information.\nSo {} \nAnswer: ".format(question)
        final_answer, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
        # 删除数字和点
        if final_answer.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            final_answer = final_answer.split(".", 1)[1].strip()
        # # 保留第一句话
        # final_answer = final_answer.split(".")[0] + "."
        # print("prompt: ", prompt)
        # print(final_answer)
        return final_answer.strip()
