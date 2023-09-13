import pandas as pd
import trlx
from huggingface import HuggingFaceModel
from trlx.data.default_configs import TRLConfig, TrainConfig, ModelConfig, TokenizerConfig, OptimizerConfig, SchedulerConfig, SFTConfig
from trlx.models.modeling_ilql import ILQLConfig


class ILQLModel(HuggingFaceModel):
    
    def __init__(self, config):
        super().__init__(config)
    
    def train(self, texts, rewards, config, save=None):
        trainer.train(texts)
        trainer = trlx.train(
            samples=texts,
            rewards=rewards,
            eval_prompts=[""]*5,
            config=config,
            metric_fn=lambda samples, **kwargs: {"lengths": [float(len(sample)) for sample in samples]},
        )
        if save:
            trainer.save_pretrained(save)

def main():
    data = pd.read_feather("../data/semantic_scholar_papers.feather")
    texts = data["text"].tolist()
    rewards = data["citations"].tolist()
    config = {
        "name": "pythia_410_citations_ilql",
        "model": "EleutherAI/pythia-410m-deduped",
    }
    sft_model = ILQLModel(config)
    train_config = TRLConfig(
        train=TrainConfig(
            seq_length=512,
            epochs=20,
            total_steps=len(texts) * 20,
            batch_size=16,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateILQLTrainer",
        ),
        model=ModelConfig(model_path=config["model"], num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(
            tokenizer_path=config["model"], truncation_side="right"
        ),
        optimizer=OptimizerConfig(
            name="adam", kwargs=dict(lr=6e-4)
        ),
        scheduler=SchedulerConfig(
            name="linear", kwargs=dict(start_factor=1)
        ),
        method=ILQLConfig(
            name="ilqlconfig",
            tau=0.7,
            gamma=0.99,
            cql_scale=0.1,
            awac_scale=1,
            alpha=0.001,
            beta=0,
            steps_for_target_q_sync=5,
            two_qs=True,
            gen_kwargs=dict(
                max_new_tokens=256, top_k=20, beta=1, temperature=1.0
            ),
        ),
    )
    sft_model.train(
        texts, rewards, train_config, save=f"./weights/{config['name']}"
    )