# Simple Student Grade Calculator
# This program converts marks to grades and displays encouraging messages

def get_grade_and_message(marks):
    """
    Convert marks to grade and return grade with encouraging message.
    Marks should be between 0 and 100.
    """
    if marks < 0 or marks > 100:
        return "Invalid", "Please enter marks between 0 and 100."
    
    if marks >= 90:
        return "A", "🌟 Outstanding! You're a superstar! Keep up the amazing work!"
    elif marks >= 80:
        return "B", "✨ Excellent work! You're doing great! Keep striving for excellence!"
    elif marks >= 70:
        return "C", "👍 Good job! You're on the right track. A little more effort will take you further!"
    elif marks >= 60:
        return "D", "📚 You passed! But there's room for improvement. Keep practicing!"
    elif marks >= 50:
        return "E", "⚠️  You barely passed. Put in more effort next time to improve your grades!"
    else:
        return "F", "❌ You didn't pass this time. Don't give up! Study harder and try again!"

def main():
    print("\n" + "="*60)
    print("WELCOME TO THE STUDENT GRADE CALCULATOR")
    print("="*60 + "\n")
    
    try:
        # Get student name
        student_name = input("Enter student name: ")
        
        # Get marks from user
        marks = float(input("Enter marks (0-100): "))
        
        # Calculate grade and get message
        grade, message = get_grade_and_message(marks)
        
        # Display results
        print("\n" + "-"*60)
        print(f"Student Name: {student_name}")
        print(f"Marks Obtained: {marks}/100")
        print(f"Grade: {grade}")
        print(f"\nMessage: {message}")
        print("-"*60 + "\n")
        
    except ValueError:
        print("❌ Error: Please enter a valid number for marks!")

if __name__ == "__main__":
    main()