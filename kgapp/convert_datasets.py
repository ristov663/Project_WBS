import csv
import urllib.parse
from datetime import datetime


# Function to convert dates to ISO 8601 format
def convert_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%d.%m.%Y')
        return date_obj.date().isoformat()
    except ValueError:
        return date_str  # If conversion fails, return the original value


# Function to encode URIs
def encode_uri(uri_str):
    return urllib.parse.quote(uri_str, safe=":/?&=+#")


# Path to your CSV file
csv_file = 'kgapp/datasets/datasets.csv'
output_file = 'kgapp/datasets/datasets_converted.csv'

# Reading the original CSV file and writing to the new one
with open(csv_file, newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames  # Reading column headers
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write headers to the new file
    writer.writeheader()

    # Process each row, converting dates and URIs
    for row in reader:
        # Convert all dates in the 'Date' column
        row['Date'] = convert_date(row['Date'])

        # Encode URIs
        row['Contract'] = encode_uri(row['Contract'])
        row['Supplier'] = encode_uri(row['Supplier'])
        row['Institution'] = encode_uri(row['Institution'])

        # Write the row to the new CSV
        writer.writerow(row)

print(f"New data has been saved to {output_file}")
