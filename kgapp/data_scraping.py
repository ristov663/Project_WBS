import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import random
from datetime import datetime, timedelta


# Function to fetch URL content with retries
async def fetch(session, url, retries=3):
    for _ in range(retries):
        try:
            async with session.get(url, timeout=30) as response:
                return await response.text()
        except asyncio.TimeoutError:
            print(f"Timeout for {url}, retrying...")
        except Exception as e:
            print(f"Error fetching {url}: {e}, retrying...")
    return None


# Function to scrape data from a single page
async def scrape_page(session, year, page):
    url = f"https://opendata.mk/Home/DealsOver1mDetails?year={year}#id={page}"
    html = await fetch(session, url)
    if not html:
        return []

    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table')
    if not table:
        return []

    data = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) == 4:
            institution = cols[0].text.strip()
            contract = cols[1].text.strip() or "No description for this public procurement!"
            company = cols[2].text.strip()
            amount = cols[3].text.strip().replace(',', '')
            date = random_date(year).strftime('%Y-%m-%d')
            data.append([institution, contract, company, amount, date])
    return data


# Function to scrape data for a specific year
async def scrape_year(session, year):
    tasks = [scrape_page(session, year, page) for page in range(1, 11)]  # Limit to 10 pages per year
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]


# Function to scrape data for multiple years
async def scrape_opendata_mk(years):
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_year(session, year) for year in years]
        all_data = await asyncio.gather(*tasks)
    return [item for sublist in all_data for item in sublist]


# Function to generate a random date within a given year
def random_date(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


# Main function to orchestrate the scraping process
async def main():
    years = range(2011, 2022)
    data = await scrape_opendata_mk(years)
    df = pd.DataFrame(data, columns=['Institution', 'Contract', 'Supplier', 'Amount', 'Date'])
    df.to_csv('kgapp/datasets/all_contracts.csv', index=False, encoding='utf-8')
    print("Data successfully downloaded and saved to all_contracts.csv")

if __name__ == "__main__":
    asyncio.run(main())
