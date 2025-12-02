import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Sample sales dataset
data = {
    'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19',
             '2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23', '2024-01-24'],
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Monitor',
                'Mouse', 'Laptop', 'Keyboard', 'Monitor', 'Laptop'],
    'Quantity': [2, 5, 3, 1, 2, 8, 3, 4, 1, 2],
    'Price': [1200, 25, 75, 1200, 300, 25, 1200, 75, 300, 1200]
}

# Create DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df['Total_Sale'] = df['Quantity'] * df['Price']

print("=" * 60)
print("SALES ANALYSIS REPORT")
print("=" * 60)
print(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Display the dataset
print("Raw Sales Data:")
print("-" * 60)
print(df.to_string(index=False))
print()

# 1. Total Sales
total_sales = df['Total_Sale'].sum()
print("=" * 60)
print("1. TOTAL SALES")
print("=" * 60)
print(f"Total Revenue: ${total_sales:,.2f}")
print(f"Number of Transactions: {len(df)}")
print(f"Average Transaction Value: ${df['Total_Sale'].mean():,.2f}")
print()

# 2. Best-Selling Product (by quantity)
product_quantity = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
print("=" * 60)
print("2. BEST-SELLING PRODUCT (By Quantity)")
print("=" * 60)
print("\nProducts Sold (Quantity):")
for product, qty in product_quantity.items():
    print(f"  {product:15s}: {qty:3d} units")
print(f"\nBest-Selling Product: {product_quantity.index[0]} ({product_quantity.iloc[0]} units)")
print()

# 3. Best-Selling Product (by revenue)
product_revenue = df.groupby('Product')['Total_Sale'].sum().sort_values(ascending=False)
print("=" * 60)
print("3. BEST-SELLING PRODUCT (By Revenue)")
print("=" * 60)
print("\nRevenue by Product:")
for product, revenue in product_revenue.items():
    print(f"  {product:15s}: ${revenue:,.2f}")
print(f"\nTop Revenue Product: {product_revenue.index[0]} (${product_revenue.iloc[0]:,.2f})")
print()

# 4. Additional Insights
print("=" * 60)
print("4. ADDITIONAL INSIGHTS")
print("=" * 60)
avg_price = df.groupby('Product')['Price'].mean()
print("\nAverage Price per Product:")
for product, price in avg_price.items():
    print(f"  {product:15s}: ${price:,.2f}")
print()

print(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Total Unique Products: {df['Product'].nunique()}")
print()

# 5. Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Total Sales Revenue: ${total_sales:,.2f}")
print(f"✓ Best Seller (Quantity): {product_quantity.index[0]}")
print(f"✓ Best Seller (Revenue): {product_revenue.index[0]}")
print(f"✓ Most Expensive Product: {avg_price.idxmax()} (${avg_price.max():,.2f})")
print("=" * 60)

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart for quantity sold
axes[0].bar(product_quantity.index, product_quantity.values, color='skyblue', edgecolor='navy')
axes[0].set_title('Units Sold by Product', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Product')
axes[0].set_ylabel('Quantity Sold')
axes[0].grid(axis='y', alpha=0.3)

# Bar chart for revenue
axes[1].bar(product_revenue.index, product_revenue.values, color='lightcoral', edgecolor='darkred')
axes[1].set_title('Revenue by Product', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Product')
axes[1].set_ylabel('Revenue ($)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nAnalysis complete! Charts displayed above.")