#for loops
numbers:list[int] = [1, 2, 3, 4, 5]

for num in numbers:
    print(num)

#loop with range
print()

for i in range(5):
    print(i)

print()

#Loop with index
for i, num in enumerate(numbers):
    print(f"index{i} = {num}")

print()

#while loops
count: int =0
while count <5:
    print(count)
    count +=1

print()


#Loop over dictionary 
model: dict[str, str] = {
    "name": "GPT",
    "type": "transformer"
}

for key, value in model.items():
    print(f"{key}: {value}")