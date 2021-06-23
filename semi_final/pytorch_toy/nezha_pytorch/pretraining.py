# coding:utf-8
import os
import warnings
from helper.configuration import NeZhaConfig
from helper.modeling import NeZhaForMaskedLM
import directory
from transformers import BertTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments
from run import RunConfig
from helper.seed import seed

run_config = RunConfig()

warnings.filterwarnings('ignore')


def pretrain():
    config = NeZhaConfig.from_pretrained(directory.BERT_CONFIG_PATH)

    model = NeZhaForMaskedLM(config=config)

    tokenizer = BertTokenizer.from_pretrained(directory.VOCAB_PATH)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=directory.CORPUS_PATH,
        block_size=run_config.seq_len
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=directory.PRETRAIN_DIR,
        overwrite_output_dir=True,
        num_train_epochs=run_config.num_pretrain_epochs,
        per_device_train_batch_size=run_config.batch_size,
        save_steps=10000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(directory.PRETRAIN_DIR)


if __name__ == '__main__':
    if not os.path.exists(directory.PRETRAIN_DIR):
        os.makedirs(directory.PRETRAIN_DIR)

    seed()

    pretrain()
