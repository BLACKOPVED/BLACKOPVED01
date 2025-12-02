# Personal Introduction Program
# This program asks for user information and displays a friendly welcome message

def main():
    # Get user input
    name = input("What is your name? ")
    age = input("How old are you? ")
    hobby = input("What is your hobby? ")
    
    # Display friendly welcome message
    print("\n" + "="*50)
    print("Welcome to the Personal Introduction Program!")
    print("="*50)
    print(f"\nHello {name}! 👋")
    print(f"It's great to meet you!")
    print(f"\nHere's what we learned about you:")
    print(f"  • Name: {name}")
    print(f"  • Age: {age}")
    print(f"  • Hobby: {hobby}")
    print(f"\nWe hope you have a wonderful day pursuing your passion for {hobby}!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
