from flask import Flask, render_template, request
from transformers import pipeline
from difflib import SequenceMatcher
import pandas as pd
from PyPDF2 import PdfReader
import openai

app = Flask(__name__)

# Set your Azure OpenAI credentials
openai.api_type = "azure"
openai.api_base = "https://react-ticketing-openai.openai.azure.com/"
openai.api_version = "2024-08-01-preview"
openai.api_key = "2AeDKaG1FvNa4oECqkYx0EZIvNKaXSeCW3TwmGOpbGEgi6CdW654JQQJ99ALACYeBjFXJ3w3AAABACOGIoUJ"

# Read CSV file to get solutions
def read_csv_solutions(csv_file):
    try:
        data = pd.read_csv(csv_file)
        data = data.fillna('')  # Fill any missing values with empty strings
        solutions = []
        for _, row in data.iterrows():
            solutions.append({
                'Issue': row['Description'],  # Assuming 'Description' is the issue column
                'Resolution Steps': row['Resolution Steps']
            })
        return solutions
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

# Read PDF file to get solutions
def read_pdf_solutions(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Parse the text to find solutions
        solutions = []
        for line in text.split('\n'):
            if ':' in line:
                issue, solution = line.split(':', 1)
                solutions.append({'Issue': issue.strip(), 'Resolution Steps': solution.strip()})
        return solutions
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return []

# GPT-4o-mini model for solution generation using Azure OpenAI
def get_solution_from_azure_gpt(issue):
    try:
        # Call the OpenAI GPT-4o-mini model hosted on Azure
        response = openai.ChatCompletion.create(
            engine="gpt-4o-mini",  # Your engine name for Azure GPT-4o-mini
            messages=[
                {"role": "system", "content": "You are an expert assistant helping resolve issues."},
                {"role": "user", "content": f"The issue is: {issue}"}
            ],
            max_tokens=150,
            temperature=0.7
        )

        resolution = response['choices'][0]['message']['content'].strip()
        return {"Resolution Steps": resolution}
    except Exception as e:
        print(f"Error fetching solution: {e}")
        return None

def find_best_match(user_issue, solutions):
    best_match = None
    highest_similarity = 0
    for solution in solutions:
        similarity = SequenceMatcher(None, user_issue.lower(), solution['Issue'].lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = solution
    return best_match, highest_similarity

def resolve_issue(user_issue, csv_file, pdf_file=None):
    results = {'CSV': None, 'PDF': None, 'GPT-4o-mini': None}
    accuracy_scores = {'CSV': 0, 'PDF': 0, 'GPT-4o-mini': 0}

    # Read solutions from CSV
    csv_solutions = read_csv_solutions(csv_file)
    csv_match, csv_similarity = find_best_match(user_issue, csv_solutions)
    if csv_match:
        results['CSV'] = csv_match['Resolution Steps']
        accuracy_scores['CSV'] = csv_similarity * 100

    # Read solutions from PDF (optional)
    if pdf_file:
        pdf_solutions = read_pdf_solutions(pdf_file)
        pdf_match, pdf_similarity = find_best_match(user_issue, pdf_solutions)
        if pdf_match:
            results['PDF'] = pdf_match['Resolution Steps']
            accuracy_scores['PDF'] = pdf_similarity * 100

    # Get the solution from the Azure GPT-4o-mini model ok?
    gpt_solution = get_solution_from_azure_gpt(user_issue)
    if gpt_solution:
        results['GPT-4o-mini'] = gpt_solution['Resolution Steps']
        accuracy_scores['GPT-4o-mini'] = 100

    return results, accuracy_scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    csv_file = "Database_Error_codes_Tickets.csv"
    pdf_file = "e12152-User Guide.pdf"
    user_issue = request.form['issue']

    # Resolve the issue
    results, accuracy_scores = resolve_issue(user_issue, csv_file, pdf_file)

    return render_template(
        'result.html',
        user_issue=user_issue,
        results=results,
        accuracy_scores=accuracy_scores
    )

if __name__ == '__main__':
    app.run(debug=True)
