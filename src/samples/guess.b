'functions'
func floor(x): x // 1
func ceil(x): if x - x // 1 == 0 then x else x // 1 + 1
func round(x): if x - x // 1 >= 0.5 then ceil(x) else floor(x)
func randint(x, y): round(x + rand() * (y - x))

'main'
number = randint(1, 10)
print("number between 1 and 10 was chosen")
while true
    guess = inputNum()
    if guess == number then break
    if guess > number then print("number is smaller")
    if guess < number then print("number is bigger")
end
print("you guessed the number correctly")