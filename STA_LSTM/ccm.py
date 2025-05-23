# Writing "Hello Sir!" to a new file greeting.txt
with open("greeting.txt", "w") as file:
    file.write("Hello Sir!")
# Appending new lines to the file greeting.txt
with open("greeting.txt", "a") as file:
    file.write("\nHow are you?")
    file.write("\nHope you are fine")
with open("greeting.txt", "r") as file:
    contents = file.read()

print(contents)

