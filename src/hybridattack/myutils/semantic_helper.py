import numpy as np


class SemanticHelper:

    def __init__(self, embeddings, tokenizer):
        self.word2idx = None
        self.idx2word = None
        raw = embeddings.weight.data.cpu().numpy()
        self.embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.tokenizer = tokenizer
        self.build_vocab()

    def build_vocab(self):
        self.idx2word = {}
        self.word2idx = {}

        print("Building vocab...")
        vocab = self.tokenizer.get_vocab()
        for token, idx in vocab.items():
            self.idx2word[idx] = token
            self.word2idx[token] = idx

    def is_number(self, input):
        try:
            t = float(input)
            return True
        except:
            return False

    def get_cos_sim(self, in_words_idx):
        embeddings = self.embeddings[in_words_idx]
        all_embeddings = self.embeddings
        cos_sim = np.matmul(embeddings, all_embeddings.T)
        return cos_sim

    def recover_whole_word(self, sub_word, ori_doc_token_ids_list, word_re):
        ori_doc_token_ids_list = [t.item() for t in ori_doc_token_ids_list]

        word_index = ori_doc_token_ids_list.index(word_re.word2idx[sub_word])
        right_word_index = word_index + 1
        while (right_word_index < len(ori_doc_token_ids_list)) and (
            word_re.idx2word[ori_doc_token_ids_list[right_word_index]].startswith("##")
        ):
            word_index = word_index + 1
            right_word_index = word_index + 1
        subword_list = []
        subword_token_id_list = []
        now_sub_word = sub_word
        now_word_index = word_index
        subword_list.append(now_sub_word)
        subword_token_id_list.append(word_re.word2idx[now_sub_word])
        while (now_sub_word.startswith("##")) and (now_word_index > 0):
            now_word_index = now_word_index - 1
            now_sub_word = word_re.idx2word[ori_doc_token_ids_list[now_word_index]]

            subword_list.append(now_sub_word)
            subword_token_id_list.append(word_re.word2idx[now_sub_word])

        whole_word = subword_list[-1]

        assert len(subword_list) > 1

        for i in range(len(subword_list) - 2, -1, -1):
            current_sub_word = subword_list[i][2:]
            whole_word += current_sub_word

        first_word = subword_list[-1]
        tail_word = whole_word[len(first_word) :]
        if self.is_number(first_word) and (tail_word in self.word2idx):
            whole_word = tail_word
            subword_token_id_list.pop(0)
        self.subword_neighbor_dict[whole_word] = subword_token_id_list
        return whole_word

    def pick_most_similar_words_batch(
        self, src_words, ori_doc_token_ids_list, word_re, ret_count=10, threshold=0.0
    ):
        in_words_idx = []
        in_words = []
        out_words = []

        self.subword_neighbor_dict = {}
        for src_word in src_words:
            if src_word.startswith("##"):
                src_word = self.recover_whole_word(
                    src_word, ori_doc_token_ids_list, word_re
                )

            if src_word in self.word2idx:
                in_words_idx.append(self.word2idx[src_word])
                in_words.append(src_word)
            else:
                out_words.append(src_word)

        cos_sim = self.get_cos_sim(in_words_idx)

        # Exclude the word itself from similarities
        for idx, in_word_idx in enumerate(in_words_idx):
            cos_sim[idx, in_word_idx] = -np.inf
        sim_order = np.argsort(-cos_sim)[:, 1 : 1 + ret_count]

        sim_values = np.take_along_axis(cos_sim, sim_order, axis=1)

        sim_words = {}

        print("sim_order: ",sim_order)
        print("sim_values: ",sim_values)
        for idx, in_word in enumerate(in_words):
            sim_word_ids = sim_order[idx]
            sim_word_scores = sim_values[idx]
            mask = sim_word_scores >= threshold
            sim_word_ids = sim_word_ids[mask]
            sim_word_tokens = [self.idx2word[word_id] for word_id in sim_word_ids]
            sim_words[in_word] = sim_word_tokens
            # score can be accessed thru sim_word_scores[mask]

        return sim_words, sim_values, self.subword_neighbor_dict

    # seems no changes are needed
    def pick_nearest_word_with_embedding(
        self,
        importance_tokens,
        importance_token_id_list,
        ori_doc_token_ids_list,
        ori_embedding,
        att_embedding,
        word_re,
        attacker_word_number,
    ):
        new_doc_token_ids_list = ori_doc_token_ids_list.copy()
        assert len(importance_tokens) == len(importance_token_id_list)
        m = attacker_word_number
        ti_huan_num = 0
        for i in range(len(importance_tokens)):
            word = importance_tokens[i]
            word_id = importance_token_id_list[i]
            for j in range(len(ori_doc_token_ids_list)):
                ori_word_id = ori_doc_token_ids_list[j]
                if ori_word_id == word_id:
                    word_vecor = att_embedding[word_id]
                    max_sim, max_id = word_re.get_max_sim_word(
                        word_vecor, ori_embedding
                    )
                    replace_word_id = max_id.cpu()[0].item()
                    new_doc_token_ids_list[j] = replace_word_id
                    ti_huan_num += 1
                    if ti_huan_num == m:
                        break
            if ti_huan_num == m:
                break
        return new_doc_token_ids_list

    def pick_most_similar_words_batch_with_itself(
        self, src_words, ret_count=10, threshold=0.0
    ):

        in_words_idx = []
        in_words = []
        out_words = []
        for src_word in src_words:
            if src_word in self.word2idx:
                in_words_idx.append(self.word2idx[src_word])
                in_words.append(src_word)
            else:
                out_words.append(src_word)

        cos_sim = self.get_cos_sim(in_words_idx)

        sim_order = np.argsort(-cos_sim)[:, :ret_count]
        sim_values = np.take_along_axis(cos_sim, sim_order, axis=1)
        sim_words = {}

        for idx, in_word_idx in enumerate(in_words_idx):
            sim_word_ids = sim_order[idx]
            sim_word_scores = sim_values[idx]
            mask = sim_word_scores >= threshold
            sim_word_ids = sim_word_ids[mask]
            sim_word_tokens = [self.idx2word[word_id] for word_id in sim_word_ids]
            sim_words[in_word_idx] = sim_word_tokens
            # append the word itself to its similarity list
            if in_word_idx not in sim_word_tokens:
                sim_words[in_word_idx].append(in_word_idx)

            # sim_value = self.cos_sim_matrix[in_word_idx][sim_order[idx]]
            # mask = sim_value >= threshold
            # sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            # sim_word = [self.idx2word[id] for id in sim_word]
            # sim_word.append(in_words[idx])
            # sim_words[in_words[idx]] = sim_word
            # sim_values.append(sim_value)
        return sim_words, sim_values
