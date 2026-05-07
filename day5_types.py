from typing import TypeVar, Protocol
from dataclasses import dataclass, field

# TypeVar - flexible but consistent types of data, can any type like int str ..
T = TypeVar("T")

def first_item(items: list[T]) -> T: 
    return items[0]

def last_item(items: list[T]) -> T:
    return items[-1]

# protocol - structural subtyping see if fit or predict
class trainable(Protocol): 
    def fit(self, data: list[float]) ->None: ...
    def predict(self, x: float) -> float: ...

class LinearModel: 
    def fit(slef, data: list[float]) -> None:
        print(f"training  on {len(data)} samples" )     # len(data) is a function counts how many variables in (data)

    def predict(self, x: float) -> float:
        return x * 2.0

def train(model: trainable, data: list[float]) -> None:
    model.fit(data)
    result: float = model.predict(3.0)
    print(f"prediction: {result}")

# Dataclasses
@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str = "default"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.learning_rate <=0:
            raise ValueError("learning rate must be positive")
        if self.batch_size <=0:
            raise ValueError("batch size must be positive")
        
# Using them
print(first_item([1,2,3]))
print(first_item(["GPT", "BERT", "Llama"]))
print(last_item([10, 20, 30]))

model = LinearModel()
train(model, [1.0, 2.0, 3.0, 4.0, 5.0])

config = TrainingConfig(
    learning_rate = 0.001,
    batch_size=32,
    epochs=10,
    model_name="GPT-2"
)
print(config)

try: 
    bad_config = TrainingConfig(
        learning_rate=-0.001,
        batch_size=32,
        epochs=10
    )
except ValueError as e:
    print(f"caught error: {e}")