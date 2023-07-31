MODEL_NAME = "databricks/dolly-v2-12b"



# class CNNModule(pl.LightningModule):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.layer1 = nn.Conv2d(1, 64, kernel_size=1)
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc2= nn.Sequential(
#             nn.Linear(2304, num_classes))
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc2(out)
#         return out

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)

    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def configure_optimizers(self):
    #     return Adam(self.parameters(), lr=0.02)

class DollyV2Model(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.predictions = []
        self.references = []

    def forward(self, batch):
        outputs = self.model(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            print(self.trainer.model)
        return torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, eps=self.eps)

def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)


def main(args):
    torch.set_float32_matmul_precision('medium')

    print("Loading dataset...")
    dataset = load_dataset("tiny_shakespeare", cache_dir="/scratch/gilbreth/dchawra/cache/")
    train_loader = DataLoader(dataset, batch_size=128)



    print("Loading model...")

    trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy="ddp",
                         plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)])
    model = TextGenerationModule("gpt2", learning_rate=1e-4)

    logger = tb.TensorBoardLogger("lightning_logs", name="mnisttest")

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # TRAIN
    main(args)