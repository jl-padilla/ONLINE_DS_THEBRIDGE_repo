import random

# generamos un numero 

randow_number = random.randint(0, 10)

intentos = 3

while intentos > 0:
    eleccion = input("Adivina en que numero pienso del 1 al 10!: ")
    if eleccion == randow_number:
        print("You win!!")
        break
    else:
        print(f"You lose!! {eleccion} es incorrecto")
        intentos -= 1

