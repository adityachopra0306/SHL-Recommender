from selenium import webdriver
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd

TEST_TYPE_MAP = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

print(list(TEST_TYPE_MAP.values()))

HOME_PREPACKAGED_ROW_SELECTOR = "tr[data-course-id]"
HOME_PREPACKAGED_LINK_SELECTOR = "td.custom__table-heading__title a"
HOME_PREPACKAGED_ADAPTIVE_CIRCLE_SELECTOR = "span.catalogue__circle.-yes"

HOME_INDIVIDUAL_ROW_SELECTOR = "tr[data-entity-id]"
HOME_INDIVIDUAL_LINK_SELECTOR = "td.custom__table-heading__title a"
HOME_INDIVIDUAL_ADAPTIVE_CIRCLE_SELECTOR = "span.catalogue__circle.-yes"

packaged_urls = ["https://www.shl.com/products/product-catalog/?start=12&type=2"] + ['https://www.shl.com/products/product-catalog/?start=' + str(i) + '&type=2&type=2' for i in range(24, 133, 12)]
individual_urls = ["https://www.shl.com/products/product-catalog/?start=12&type=1"] + ['https://www.shl.com/products/product-catalog/?start=' + str(i) + '&type=1&type=1' for i in range(24, 373, 12)]

print(packaged_urls[7])
print(individual_urls[19])

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Scraper for one page
def scrape_assessment_page(url):
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(2)
    
    entry = {
        'url': url,             # URL
        'title' : '',
        'description': '',
        'remote_support': '',
        'duration': '',
        'test_types': []
    }
    
    try:
        # TITLE
        entry['title'] = driver.find_element(By.CSS_SELECTOR, 'body > main > div.product-catalogue.module > div > div.row.content__container.typ > h1').text
    except:
        pass

    try:
        # DESCRIPTION
        entry['description'] = driver.find_element(By.CSS_SELECTOR, 'body > main > div.product-catalogue.module > div > div:nth-child(2) > div.col-12.col-md-8 > div > div:nth-child(1) > p').text
    except:
        pass

    try:
        # DURATION
        entry['duration'] = driver.find_element(By.CSS_SELECTOR, 'body > main > div.product-catalogue.module > div > div:nth-child(2) > div.col-12.col-md-8 > div > div:nth-child(4) > p').text
    except:
        pass
    
    # TYPES
    elements = driver.find_elements(By.CSS_SELECTOR, "span.product-catalogue__key")
    entry['test_types'] = [
    TEST_TYPE_MAP[el.text.strip()]
    for el in elements
    if el.text.strip() in TEST_TYPE_MAP
]

    try:
        driver.find_element(By.CSS_SELECTOR, 'body > main > div.product-catalogue.module > div > div:nth-child(2) > div.col-12.col-md-8 > div > div:nth-child(4) > div > p:nth-child(2) > span').text
        entry['remote_support'] = 'yes'
    except:
        entry['remote_support'] = 'no'

    driver.quit()
    return entry


def homepage(pageurl, rowselector, linkselector, adaptiveselector):
    main_driver = webdriver.Chrome()
    main_driver.get(pageurl)
    time.sleep(3)

    rows = main_driver.find_elements(By.CSS_SELECTOR, rowselector)
    entries = []

    for row in rows:
        try:
            link = row.find_element(By.CSS_SELECTOR, linkselector)
            url = link.get_attribute("href")

            try:
                row.find_elements(By.TAG_NAME, "td")[2].find_element(By.CSS_SELECTOR, adaptiveselector)
                adaptive_support = "yes"
            except:
                adaptive_support = "no"

            entries.append({"url": url, "adaptive_support": adaptive_support})
        except Exception as e:
            print(f"Error processing row: {e}")

    main_driver.quit()

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_entry = {
            executor.submit(scrape_assessment_page, entry["url"]): entry for entry in entries
        }

        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                data = future.result()
                data["adaptive_support"] = entry["adaptive_support"]
                results.append(data)
            except Exception as e:
                print(f"Error scraping {entry['url']}: {e}")

    print(pageurl, len(results))
    return results

def main():
    prepackaged, individual = [], []

    prepackaged.extend(homepage(pageurl="https://www.shl.com/products/product-catalog/", rowselector=HOME_PREPACKAGED_ROW_SELECTOR, linkselector=HOME_PREPACKAGED_LINK_SELECTOR, adaptiveselector = HOME_PREPACKAGED_ADAPTIVE_CIRCLE_SELECTOR))
    print(len(prepackaged))
    
    for i in prepackaged:
        print(i)
        print("\n\n")

    individual.extend(homepage(pageurl="https://www.shl.com/products/product-catalog/", rowselector=HOME_INDIVIDUAL_ROW_SELECTOR, linkselector=HOME_INDIVIDUAL_LINK_SELECTOR, adaptiveselector = HOME_INDIVIDUAL_ADAPTIVE_CIRCLE_SELECTOR))
    print(len(prepackaged))
    
    for i in individual:
        print(i)
        print("\n\n")

    for indurl in individual_urls:
        individual.extend(homepage(pageurl=indurl, rowselector=HOME_INDIVIDUAL_ROW_SELECTOR, linkselector=HOME_INDIVIDUAL_LINK_SELECTOR, adaptiveselector = HOME_INDIVIDUAL_ADAPTIVE_CIRCLE_SELECTOR))

    for packurl in packaged_urls:
        prepackaged.extend(homepage(pageurl=packurl, rowselector=HOME_PREPACKAGED_ROW_SELECTOR, linkselector=HOME_PREPACKAGED_LINK_SELECTOR, adaptiveselector = HOME_PREPACKAGED_ADAPTIVE_CIRCLE_SELECTOR))   

    df_prepackaged = pd.DataFrame(prepackaged)
    df_prepackaged.to_csv("prepackaged_assessments.csv", index=False)

    df_individual = pd.DataFrame(individual)
    df_individual.to_csv("individual_assessments.csv", index=False)

    print("Saved", len(prepackaged), len(individual))

if __name__ == "__main__":
    main()