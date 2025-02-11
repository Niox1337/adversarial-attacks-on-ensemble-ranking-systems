import torch


class PGD:
    def __init__(self, model, attack_word_idx:list[torch.Tensor], device):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.attack_word_idx = torch.stack(attack_word_idx, dim=0)
        self.device = device
        self.momentum = 0

    def attack(
        self,
        epsilon=1.0,
        alpha=0.3,
        emb_name="electra.embeddings.word_embeddings.weight",
        is_first_attack=False,
    ):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    #attack_word_idx = [int(idx.item()) for idx in self.attack_word_idx]
                    #param = param.to(device)
                    #param.grad = param.grad.to(device)
                    r_matrix = torch.zeros(param.grad.shape).to(param.device)

                    # param.grad shape: torch.Size([30522, 768])
                    # norm shape: torch.Size([])
                    # self.attack_word_idx: [tensor(7099), tensor(3893), tensor(102)]

                    r_matrix[self.attack_word_idx, :] = (param.grad / norm)[self.attack_word_idx, :]

                    # r_matrix[self.attack_word_idx] = (param.grad / norm)[
                    #     self.attack_word_idx
                    # ]
                    r_at = -alpha * r_matrix
                    param.data.add_(r_at)

    def attack_with_momentum(
        self,
        epsilon=1.0,
        alpha=0.3,
        emb_name="electra.embeddings.word_embeddings.weight",
        is_first_attack=False,
        mu=1,
    ):
        # change the emb_name to the embedding parameter name of your model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_matrix = torch.zeros(param.grad.shape).to(self.device)
                    r_matrix[self.attack_word_idx] = (param.grad / norm)[
                        self.attack_word_idx
                    ]

                    self.momentum = self.momentum + r_matrix
                    r_at = -alpha * self.momentum
                    param.data.add_(r_at)

    def restore(self, emb_name="electra.embeddings.word_embeddings.weight"):
        # emb_name xxx
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
