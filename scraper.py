from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import re
import time
import os

documents_dir = "./documents/"
if not os.path.exists(documents_dir):
    os.makedirs(documents_dir)

def parse_date_from_text(text):
    # Mapping of French month names to month numbers
    months = {
        "janvier": "01", "février": "02", "mars": "03", "avril": "04",
        "mai": "05", "juin": "06", "juillet": "07", "août": "08",
        "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"
    }

    # Define the regular expression pattern for dates
    date_pattern = r"(\d{1,2}) (janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre) (\d{4}) à (\d{1,2})h(\d{2})"

    # Find all dates in the text
    dates = re.findall(date_pattern, text)

    # Parse found dates into datetime objects
    date_objects = []
    for day, month, year, hour, minute in dates:
        month_number = months[month]
        date_str = f"{year}-{month_number}-{day} {hour}:{minute}"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        date_objects.append(date_obj)

    # Find the most recent date
    if not date_objects:
        return None  # No dates found
    most_recent_date = max(date_objects)

    # Format the most recent date
    formatted_date = most_recent_date.strftime("%Y-%m-%d")

    # Append the suffix
    result = f"{formatted_date}_le_monde"

    return result

# Test the function with various dates
phrase_1 = "Publié le 18 mars 2024 à 03h09, modifié le 20 avril 2024 à 11h30"
phrase_2 = "Publié le 21 février 2023 à 16h26"

# Setup Chrome WebDriver
# service = Service(ChromeDriverManager().install())
options = Options()
options.page_load_strategy = 'eager'
driver = webdriver.Chrome(options=options)

# URL to open
url = 'https://www.lemonde.fr/coree-du-sud/'

# Open the URL
driver.get(url)

# Example wait for a consent button and click it if exists
time.sleep(2)  # Adjust this delay based on your observation of how long it typically takes for the consent button to become clickable
try:
    first_button = driver.find_element(By.CLASS_NAME, 'gdpr-lmd-button')
    first_button.click()
    print("avoid pop_up ...")
except Exception as e:
    print("No consent button found or error clicking it:", e)

# Wait for a moment after handling the consent
time.sleep(5)

# Find all elements that match the class name 'thread'
threads = driver.find_elements(By.CLASS_NAME, 'thread')
print("get threads elements ...")

# Iterate over the found 'thread' divs, click each, and print the date from the article page.
for index, thread in enumerate(threads):
    if index == 12:
        break
    # Threads may need to be found again due to potential changes in the DOM
    current_thread = driver.find_elements(By.CLASS_NAME, 'thread')[index]
    try:
        date_element = current_thread.find_element(By.CLASS_NAME, 'meta__date')
        parsed_date = parse_date_from_text(date_element.text)
    except Exception as e:
        print(f"Could not find the date element on the article page: {e}")

    current_thread.click()
    print("click on article ", index)

    time.sleep(5)
    try:
        paragraphs = driver.find_elements(By.CLASS_NAME, 'article__paragraph')
        file_path = os.path.join(documents_dir, parsed_date + ".txt")

        with open(file_path, 'w', encoding='utf-8') as file:
            for paragraph in paragraphs:
                file.write(paragraph.text + "\n\n")  # Writing each paragraph followed by a newline for readability
        print(f"Content saved to {file_path}")
    except Exception as e:
        print(f"Could not find the paragraphs or save the file for the article at index {index}: {e}")


    # Go back to the list of threads after printing the date
    driver.back()

    # Wait again before proceeding to the next thread
    time.sleep(8)

# Close the browser when done
driver.quit()
