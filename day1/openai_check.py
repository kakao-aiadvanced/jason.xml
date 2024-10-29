from openai import OpenAI
client = OpenAI()


content = """
Translate the following word to Korean as in examples:

Examples:
1. book: 책
2. cheese: 치즈
3. water: 물
4. bread: 빵
5. apple: 사과

Task:
"dog":
"""


content = """
Convert the following natural language requests into SQL queries:
1. "Show all employees with a salary greater than $50,000.": SELECT * FROM employees WHERE salary > 50000;
2. "List all products that are out of stock.": SELECT * FROM products WHERE stock = 0;
3. "Find the names of students who scored above 90 in math.": SELECT name FROM students WHERE math_score > 90;
4. "Retrieve the details of orders placed in the last 30 days.": SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);
5. "Get the count of customers from each city.": SELECT city, COUNT(*) FROM customers GROUP BY city;

Request: "Find the average salary of employees in the marketing department."
SQL Query:
"""

content = """
아래 질문을 2가지 버전의 프롬프트로 만들어줘

질문 : 
"피카츄의 모습에 대해 설명해줘"
"""

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"{content}",
        }
    ]
)

print(completion.choices[0].message.content)