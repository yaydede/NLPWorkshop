# Generate Sample Sales Data CSV
import csv
import random
from datetime import datetime, timedelta

# Function to generate random dates
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds()))
    )

# Generate sample data
data = []
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch']
regions = ['North', 'South', 'East', 'West', 'Central']

for _ in range(10000):  # Generate 10,000 rows
    date = random_date(start_date, end_date)
    product = random.choice(products)
    quantity = random.randint(1, 10)
    price = random.uniform(100, 1000)
    region = random.choice(regions)
    
    data.append([date.strftime('%Y-%m-%d'), product, quantity, price, region])

# Write to CSV
with open('sales_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Product', 'Quantity', 'Price', 'Region'])  # Header
    writer.writerows(data)

print("CSV file 'sales_data.csv' has been generated.")