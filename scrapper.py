import requests
import json
import time

API_URL = "https://www.examprepper.co/_next/data/2f-2l-vFyCUM74XGjywMD/exam/12/{}.json?id=12&page={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.examprepper.co/exam/12/1"
} # Avoid getting blocked
PAGE_RANGE = range(1, 60)  # Pages 1 to 59

def fetch_questions(page_num):
    """Fetch questions from API and return parsed data."""
    try:
        response = requests.get(API_URL.format(page_num, page_num), headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        questions = []
        for item in data.get("pageProps", {}).get("questions", []):
            question_text = item.get("question_text", "").replace("\n", "\n> ").replace("`", "'")
            # Sort options by key (A, B, C, D)
            sorted_options = dict(sorted(item.get("choices", {}).items()))
            correct_answer = item.get("answer", "Unknown")
            questions.append({
                "question": question_text, 
                "options": sorted_options,
                "correct_answer": correct_answer
            })
        return questions
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page_num}: {e}")
        return []

def scrape_all():
    """Scrapes all pages and saves results to a JSON file."""
    all_questions = []

    for page in PAGE_RANGE:
        print(f"Scraping page {page}...")
        all_questions.extend(fetch_questions(page))
        time.sleep(1)  # Respectful crawling, avoid overload

    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=4)

    print(f"Scraping completed! {len(all_questions)} questions saved.")

def convert_to_md():
    """Converts the JSON file to a Markdown file."""
    print("Markdown conversion in progress...")
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open("questions.md", "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            f.write(f"> [!question] Question {i}\n")
            f.write(f"> {question['question']}\n>\n")
            for option, explanation in question["options"].items():
                f.write(f"> **{option}.** {explanation}\n")
            f.write("\n> [!info]- Answer & Explanation\n")
            correct_answer = question["correct_answer"]
            f.write(f"> **Answer:** {correct_answer}\n> \n> \n> **Explanation:**\n> \n")
            f.write("\n---\n\n")

    print("Markdown conversion completed!")

if __name__ == "__main__":
    scrape_all()
    convert_to_md()