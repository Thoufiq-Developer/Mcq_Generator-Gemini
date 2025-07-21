import PyPDF2
import json

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception:
            raise Exception("Error reading PDF file.")
    elif file.name.endswith(".txt"):
        try:
            return file.read().decode("utf-8")
        except Exception:
            raise Exception("Error reading text file.")
    else:
        raise Exception("Unsupported file format.")

def get_table_data(response):
    quiz_output = response.get("quiz", "")
    clean_quiz = quiz_output.strip().replace("```json", "").replace("```", "")

    try:
        parsed_quiz = json.loads(clean_quiz)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_quiz, dict):
        return None

    table = []
    for mcq_id, mcq_data in parsed_quiz.items():
        mcq = mcq_data.get("mcq", "")
        options = '|'.join(f"{opt}:{val}" for opt, val in mcq_data.get("options", {}).items())
        correct = mcq_data.get("correct", "")
        table.append({"MCQ": mcq, "Choice": options, "Correct": correct})

    return table
