#Lists
scores: list[int]= [95, 87, 92, 78, 95, 100]

print(scores[0])
print(scores[-1])
print(scores[1:4])
print(len(scores))
print(max(scores))
print(min(scores))
print(sum(scores))

scores.append(88)
scores.sort()
print(scores)

#Dictionaries
model: dict[str,str] = {
    "name": "GPT",
    "type": "transformer",
    "creator": "OpenAI",
}

print(model["name"])
print(model.keys())
print(model.values())

model["year"] ="2023"
print(model)