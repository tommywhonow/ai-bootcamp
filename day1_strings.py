sentence: str = "machine learning is the future of technology"

print(sentence.upper())
print(sentence.title())
print(sentence.capitalize())

print(sentence.find("learning"))
print(sentence.count("e"))
print("future" in sentence)

words: list[str] = sentence.split(" ")
print(words)
print(" | ".join(words))

messy: str = "   hello world   "
print(messy.strip())

print(sentence.replace("learning", "intelligence"))