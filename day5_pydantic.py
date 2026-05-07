#Pydantic is like dataclasses but with automatic validation built in.
#It's the standard for data validation in production ML systems — FastAPI uses it, 
#most ML serving frameworks use it.

from pydantic import BaseModel, field_validator
from typing import Optional

# Basic pydantic model
class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str="default"
    device: str = "cpu"

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v:float) -> float:
        if v<=0:
            raise ValueError("learning rate must be positive")
        return v
    
    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: int) -> int:
        if v <=0:
            raise ValueError("batch size must be positive")
        return v

# Nested model
class ExperimentConfig(BaseModel):
    name: str
    training: TrainingConfig
    tags: list[str] = []
    notes: Optional[str] = None

# Valid config
config = TrainingConfig(
    learning_rate =0.001,
    batch_size=32,
    epochs=10,
    model_name="GPT-2"
)
print(config)
print(f"learning rate: {config.learning_rate}")

# Auto type conversion
config2 = TrainingConfig(
    learning_rate="0.001", # type: ignore[arg-type]
    batch_size="32",    # type: ignore[arg-type]
    epochs=10
)
print(f"\nauto convertedd: {config2.learning_rate} type={type(config2.learning_rate)}")

# Nested config
experiment = ExperimentConfig(
    name="run_001",
    training=config,
    tags=["baseline", "gpt2"],
    notes="first experiment"
)
print(f"\nexperiment: {experiment.name}")
print(f"training config: {experiment.training.model_name}")

# Invalid config
try:
    bad = TrainingConfig(
        learning_rate=-0.001,
        batch_size=32,
        epochs=10
    )
except Exception as e:
    print(f"\ncaught: {e}")
