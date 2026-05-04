#blue print for ai model
class Model:
    def __init__ (self, name: str, layers: int, hidden_size: int ) ->None:
       self.name = name
       self.layers = layers
       self.hidden_size = hidden_size

    def describe(self) -> str:
        return f"{self.name} -layers: {self.layers}, hidden_size:{self.hidden_size}"
    
    def is_large(self) -> bool:
            return self.layers >24
    

#After blueprint, we can create instances of the model
#Inheritance
class TransformerModel(Model):
    def __init__ (self, name: str, layers: int, hidden_size: int, heads: int) -> None:
          super().__init__(name, layers, hidden_size)
          self.heads = heads
          
    def describe(self) ->str:
        base: str = super().describe()
        return f"{base}, heads:{self.heads})"
    
#Creating objects
gpt: Model = Model("Gpt-2", 12, 768)
bert: Model = Model("Bert-large", 24, 1024)
gpt4: TransformerModel = TransformerModel("Gpt-4", 96, 12288, 96)

print(gpt.describe())
print(bert.describe())
print(gpt4.describe())
print()

print(gpt.is_large())
print(bert.is_large())
print(gpt4.is_large())
        