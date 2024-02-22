import sys
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup,AdamW
from transformers import GPTNeoXForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer, seed_everything
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import transformers
import torch
import copy
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
from data_process import DataCollatorForSupervisedDataset, smart_tokenizer_and_embedding_resize
from data_process import MRtoTextDataset as SupervisedDataset
import pandas as pd

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)


import os
import glob

class StopOnTokens(StoppingCriteria):
	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		stop_ids = [29]

		for stop_id in stop_ids:
			if input_ids[0][-1] == stop_id:
				return True
		return False

stop = StopOnTokens()

def find_latest_checkpoint(output_dir):
    checkpoint_files = glob.glob(os.path.join(output_dir, "*.ckpt"))
    if not checkpoint_files:
        raise ValueError("No checkpoint files found in the output directory.")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="togethercomputer/RedPajama-INCITE-Instruct-3B-v1")

@dataclass
class DataArguments:
	# train/val data path
    train_data_path: str = field(default='/PATH/TO/TRAINING_FILE', metadata={"help": "Path to the training data."})
    val_data_path: str = field(default='/PATH/TO/EVALUATION_FILE', metadata={"help": "Path to the training data."})
    test_data_path: str = field(default='/PATH/TO/TESTING_FILE', metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: str = field(default='/PATH/TO/MAIN_DIRECTORY')
    num_train_epochs:int = 15
    per_device_train_batch_size:int = field(default=1)
    per_device_eval_batch_size:int = field(default=1)
    gradient_accumulation_steps:int = field(default=8)
    evaluation_strategy:str = field(default='epoch')
    save_strategy:str = field(default='steps')
    save_steps:int = field(default=2000)
    save_total_limit:int = field(default=1)
    learning_rate:float = field(default=2e-5)
    warmup_ratio:float = field(default=0.03)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.val_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    val_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)
    return train_dataloader, val_dataloader

def make_eval_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.val_data_path, test=True)
    print(eval_dataset[0])
    val_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size)
    return val_dataloader

class LLAMA_Wiki(LightningModule):
    def __init__(self,
        tokenizer = None,
        num_new_tokens = 0,
        use_pretrained = True,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 100,
        weight_decay: float = 0.0,
        train_batch_size: int = 64,
        eval_batch_size: int = 64,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer', 'num_new_tokens'])
        self.tokenizer = tokenizer
        self.model_path = 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1'

        self.model = GPTNeoXForCausalLM.from_pretrained(self.model_path)

        lora_r = 4
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05

        lora_target_modules = [
            "query_key_value", "xxx"
        ]

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, self.lora_config)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        outputs = self.model(**batch)
        # Logging to TensorBoard (if installed) by default
        loss = outputs['loss']
        self.log("train_loss", loss, batch_size=self.hparams.train_batch_size, logger=True, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        outputs = self.model(**batch)
        # Logging to TensorBoard (if installed) by default
        loss = outputs['loss']
        self.log("val_loss", loss, batch_size=self.hparams.eval_batch_size, logger=True, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def forward(self, instruction):
        sample_inputs = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
        input_length = len(sample_inputs[0])
        sample_outputs = self.model.generate(input_ids = sample_inputs, max_new_tokens=75,do_sample=True,temperature=0.7,top_p=1,top_k=0, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, forced_eos_token_id=tokenizer.eos_token_id)
        output_texts = []

        for i, beam_output in enumerate(sample_outputs):
            output_text = tokenizer.decode(beam_output[input_length:].tolist(), skip_special_tokens=True).lstrip('\n').strip('\n')
            output_text = output_text.split('\n')[0]
            print("{}: {}".format(i, output_text))
            output_texts.append(output_text)

        return output_texts


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer = AdamW(model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_utterance = batch['source'][0]
        pred_out = self(input_utterance)
        gt_out = batch['target'][0]
        return [input_utterance, pred_out, gt_out]



if __name__ == "__main__":
    seed_everything(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", add_eos_token=True)
    tokenizer.add_special_tokens({'eos_token':'<|endoftext|>'})
    num_new_tokens = 0
    tokenizer.pad_token_id = 0



    # uncomment 'training = False' when perform testing
    training = True
    #training = False
    if training:
        train_dataloader, val_dataloader = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
        batch_size = 256
        num_devices = 5
        aval_devices = [0, 1, 2, 3, 4]
        gradient_accumulation_steps = batch_size // (training_args.per_device_train_batch_size*num_devices)
        # init the autoencoder
        llama = LLAMA_Wiki(tokenizer, num_new_tokens, train_batch_size=training_args.per_device_train_batch_size, eval_batch_size=training_args.per_device_eval_batch_size)
        deep_speed = DeepSpeedStrategy(
                            stage=3,
                            offload_optimizer=True,
                            offload_parameters=True,
                        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=training_args.output_dir,
            filename='{epoch}-{val_loss:.2f}',
            monitor="val_loss"
        )
        trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=training_args.num_train_epochs,
                        accumulate_grad_batches=gradient_accumulation_steps,
                        accelerator="gpu", devices=aval_devices,
                        strategy = 'ddp',
                        callbacks=[checkpoint_callback])

        trainer.fit(llama, train_dataloader, val_dataloader)


    else:
        val_dataloader = make_eval_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
        device = torch.device('cuda:0')
        output_path = "/data2/winson/redpj"

        # we can change this to the our choice to make it consistent
        latest_checkpoint_path = find_latest_checkpoint(training_args.output_dir)
        #latest_checkpoint_path ="/PATH/TO/MAIN_DIRECTORY/BEST_CHECKPOINT_FOR_INFERENCE"
        #print('latest checkpoint path is:', latest_checkpoint_path)
        llama = LLAMA_Wiki.load_from_checkpoint(latest_checkpoint_path, strict=False, tokenizer=tokenizer, num_new_tokens=num_new_tokens, use_pretrained=True).to(device)
        llama.eval()
        tester = Trainer(default_root_dir=training_args.output_dir, accelerator="gpu", devices=[2])
        predictions = tester.predict(llama, val_dataloader)

        df = pd.DataFrame(predictions)
        df.to_csv('./output_new2.csv', index=False, header=False)
        print('end')
