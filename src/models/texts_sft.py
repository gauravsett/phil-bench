import pandas as pd
import trlx
from huggingface import HuggingFaceModel
from trlx.data.default_configs import TRLConfig, TrainConfig, ModelConfig, TokenizerConfig, OptimizerConfig, SchedulerConfig, SFTConfig


class SFTModel(HuggingFaceModel):
    
    def __init__(self, config):
        super().__init__(config)
    
    def train(self, texts, config, save=None):
        trainer.train(texts)
        trainer = trlx.train(
            samples=texts,
            eval_prompts=[""]*5,
            config=config,
            metric_fn=lambda samples, **kwargs: {"lengths": [float(len(sample)) for sample in samples]},
        )
        if save:
            trainer.save_pretrained(save)

def main():
    texts = pd.read_feather("../data/semantic_scholar_papers.feather")["text"].tolist()
    config = {
        "name": "pythia_410_texts_sft",
        "model": "EleutherAI/pythia-410m-deduped",
    }
    sft_model = SFTModel(config)
    train_config = TRLConfig(
        train=TrainConfig(
            seq_length=512,
            epochs=10,
            total_steps=len(texts),
            batch_size=16,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateSFTTrainer",
        ),
        model=ModelConfig(model_path=config["model"], num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path=config["model"], truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="linear", kwargs=dict(start_factor=1)),
        method=SFTConfig(
            name="sftconfig",
            gen_kwargs=dict(max_new_tokens=256, top_k=0, top_p=1.0, do_sample=True),
        ),
    )
    sft_model.train(texts, train_config, save=f"./weights/{config['name']}")


if __name__ == "__main__":
    main()
