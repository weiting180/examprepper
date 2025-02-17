import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.examprepper.co/exam/12/{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # Avoid getting blocked
PAGE_RANGE = range(1, 60)  # Pages 1 to 59

def scrape_page(page_num):
    """Fetches and parses a single exam page, extracting multiple-choice questions."""
    url = BASE_URL.format(page_num)
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Failed to fetch page {page_num}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    questions = []

    # Modify this selector to match the real structure of the questions
    for question_div in soup.find_all("div", class_="question-container"):
        question_text = question_div.find("div", class_="question-text").get_text(strip=True)

        options = []
        for option_div in question_div.find_all("div", class_="option"):
            options.append(option_div.get_text(strip=True))

        questions.append({"question": question_text, "options": options})

    return questions

def scrape_all():
    """Scrapes all pages and saves results to a JSON file."""
    all_questions = []

    for page in PAGE_RANGE:
        print(f"Scraping page {page}...")
        all_questions.extend(scrape_page(page))
        time.sleep(1)  # Respectful crawling, avoid overload

    with open("questions.json", "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=4)

    print(f"Scraping completed! {len(all_questions)} questions saved.")

if __name__ == "__main__":
    scrape_all()