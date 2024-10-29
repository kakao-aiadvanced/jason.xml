import ollama

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

response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': f'{content}',
    },
])
print(response['message']['content'])