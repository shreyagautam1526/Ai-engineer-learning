from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

json_schema = {
    "title": "Student",
    "description": "A student object",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The name of the student"
        },
        "age": {
            "type": "integer",
            "description": "The age of the student"
        },
        "email": {
            "type": "string",
            "description": "The email address of the student"
        },
        "grade": {
            "type": "number",
            "description": "The grade of the student",
            "minimum": 0.0,
            "maximum": 100.0
        },
        "is_enrolled": {
            "type": "boolean",
            "description": "Indicates whether the student is currently enrolled"
        },
        "courses": {
            "type": "array",
            "description": "A list of courses the student is enrolled in",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["name", "age", "email", "grade", "is_enrolled", "courses"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("Tell me about a fictional student named Alice who is 20 years old, has an email address alice@example.com, a grade of 85.5, is enrolled, and is taking Math and Science courses.")

print("RESPONSE TYPE:", type(result))
print("RESPONSE:", result)
print("NAME:", result["name"])
print("AGE:", result["age"])
print("EMAIL:", result["email"])
print("GRADE:", result["grade"])
print("IS_ENROLLED:", result["is_enrolled"])
print("COURSES:", result["courses"])
