import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

# ============================================================================
# DATA GENERATION - Realistic Customer Sales Data
# ============================================================================
np.random.seed(42)

# Generate dates for the last 12 months
end_date = datetime(2024, 11, 30)
start_date = end_date - timedelta(days=365)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Customer segments
customer_segments = ['Premium', 'Regular', 'Occasional']
product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
regions = ['North', 'South', 'East', 'West', 'Central']

# Generate customer base
customers = []
for i in range(150):
    segment = np.random.choice(customer_segments, p=[0.15, 0.45, 0.40])
    customers.append({
        'Customer_ID': f'CUST{i+1:04d}',
        'Customer_Name': f'Customer {i+1}',
        'Segment': segment,
        'Region': np.random.choice(regions),
        'Join_Date': start_date + timedelta(days=np.random.randint(0, 300))
    })

df_customers = pd.DataFrame(customers)

# Generate transactions
transactions = []
transaction_id = 1

for _, customer in df_customers.iterrows():
    # Number of purchases based on segment
    if customer['Segment'] == 'Premium':
        num_purchases = np.random.randint(15, 40)
        avg_amount = 250
    elif customer['Segment'] == 'Regular':
        num_purchases = np.random.randint(5, 20)
        avg_amount = 120
    else:  # Occasional
        num_purchases = np.random.randint(1, 8)
        avg_amount = 80
    
    for _ in range(num_purchases):
        purchase_date = customer['Join_Date'] + timedelta(
            days=np.random.randint(0, (end_date - customer['Join_Date']).days + 1)
        )
        
        category = np.random.choice(product_categories)
        base_price = avg_amount + np.random.normal(0, avg_amount * 0.3)
        quantity = np.random.randint(1, 5)
        unit_price = max(10, base_price / quantity)
        
        transactions.append({
            'Transaction_ID': f'TXN{transaction_id:06d}',
            'Customer_ID': customer['Customer_ID'],
            'Date': purchase_date,
            'Category': category,
            'Quantity': quantity,
            'Unit_Price': round(unit_price, 2),
            'Total_Amount': round(unit_price * quantity, 2),
            'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'], 
                                              p=[0.45, 0.25, 0.15, 0.15])
        })
        transaction_id += 1

df_transactions = pd.DataFrame(transactions)

# Merge customer and transaction data
df = pd.merge(df_transactions, df_customers, on='Customer_ID')

# Additional calculated fields
df['Month'] = df['Date'].dt.to_period('M')
df['Month_Name'] = df['Date'].dt.strftime('%b %Y')
df['Quarter'] = df['Date'].dt.to_period('Q')
df['Day_of_Week'] = df['Date'].dt.day_name()

# ============================================================================
# ANALYSIS BEGINS
# ============================================================================
print("="*90)
print(" "*25 + "CUSTOMER SALES ANALYSIS DASHBOARD")
print("="*90)
print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Analysis Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print("="*90)

# ============================================================================
# SECTION 1: EXECUTIVE SUMMARY
# ============================================================================
print("\n" + "="*90)
print("SECTION 1: EXECUTIVE SUMMARY")
print("="*90)

total_revenue = df['Total_Amount'].sum()
total_transactions = len(df)
total_customers = df['Customer_ID'].nunique()
avg_transaction_value = df['Total_Amount'].mean()
avg_customer_lifetime_value = df.groupby('Customer_ID')['Total_Amount'].sum().mean()

print(f"\n💰 TOTAL REVENUE:              ${total_revenue:,.2f}")
print(f"📊 TOTAL TRANSACTIONS:         {total_transactions:,}")
print(f"👥 TOTAL CUSTOMERS:            {total_customers:,}")
print(f"💵 AVG TRANSACTION VALUE:      ${avg_transaction_value:,.2f}")
print(f"⭐ AVG CUSTOMER LIFETIME VALUE: ${avg_customer_lifetime_value:,.2f}")
print(f"📈 AVG TRANSACTIONS/CUSTOMER:  {total_transactions/total_customers:.1f}")

# ============================================================================
# SECTION 2: TOP CUSTOMERS ANALYSIS
# ============================================================================
print("\n\n" + "="*90)
print("SECTION 2: TOP CUSTOMERS ANALYSIS")
print("="*90)

# Calculate customer metrics
customer_metrics = df.groupby('Customer_ID').agg({
    'Total_Amount': ['sum', 'count', 'mean'],
    'Date': ['min', 'max'],
    'Segment': 'first',
    'Region': 'first',
    'Customer_Name': 'first'
}).reset_index()

customer_metrics.columns = ['Customer_ID', 'Total_Spent', 'Purchase_Count', 
                            'Avg_Purchase', 'First_Purchase', 'Last_Purchase',
                            'Segment', 'Region', 'Customer_Name']

customer_metrics['Days_Since_Last_Purchase'] = (end_date - customer_metrics['Last_Purchase']).dt.days
customer_metrics = customer_metrics.sort_values('Total_Spent', ascending=False)

print("\n🏆 TOP 10 CUSTOMERS BY REVENUE:")
print("-"*90)
print(f"{'Rank':<6}{'Customer ID':<15}{'Name':<20}{'Revenue':<15}{'Orders':<10}{'Segment':<12}{'Region'}")
print("-"*90)

for idx, row in customer_metrics.head(10).iterrows():
    print(f"{customer_metrics.index.get_loc(idx)+1:<6}{row['Customer_ID']:<15}{row['Customer_Name']:<20}"
          f"${row['Total_Spent']:>10,.2f}   {row['Purchase_Count']:>5}    "
          f"{row['Segment']:<12}{row['Region']}")

# Customer segmentation analysis
print("\n\n📊 CUSTOMER SEGMENT BREAKDOWN:")
print("-"*90)
segment_analysis = df.groupby('Segment').agg({
    'Customer_ID': 'nunique',
    'Total_Amount': ['sum', 'mean'],
    'Transaction_ID': 'count'
}).round(2)

segment_analysis.columns = ['Customers', 'Total_Revenue', 'Avg_Transaction', 'Total_Transactions']
segment_analysis['Revenue_Share'] = (segment_analysis['Total_Revenue'] / total_revenue * 100).round(1)
segment_analysis['Avg_Orders_per_Customer'] = (segment_analysis['Total_Transactions'] / segment_analysis['Customers']).round(1)

print(segment_analysis.to_string())

# ============================================================================
# SECTION 3: PURCHASING PATTERNS
# ============================================================================
print("\n\n" + "="*90)
print("SECTION 3: PURCHASING PATTERNS ANALYSIS")
print("="*90)

# Monthly trends
print("\n📅 MONTHLY SALES TREND:")
print("-"*90)
monthly_sales = df.groupby('Month_Name').agg({
    'Total_Amount': 'sum',
    'Transaction_ID': 'count'
}).round(2)
monthly_sales.columns = ['Revenue', 'Transactions']
print(monthly_sales.tail(6).to_string())

# Category performance
print("\n\n🏷️  PRODUCT CATEGORY PERFORMANCE:")
print("-"*90)
category_performance = df.groupby('Category').agg({
    'Total_Amount': ['sum', 'mean'],
    'Transaction_ID': 'count',
    'Quantity': 'sum'
}).round(2)
category_performance.columns = ['Total_Revenue', 'Avg_Transaction', 'Num_Transactions', 'Units_Sold']
category_performance = category_performance.sort_values('Total_Revenue', ascending=False)
category_performance['Revenue_Share_%'] = (category_performance['Total_Revenue'] / total_revenue * 100).round(1)
print(category_performance.to_string())

# Day of week analysis
print("\n\n📆 SALES BY DAY OF WEEK:")
print("-"*90)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_analysis = df.groupby('Day_of_Week')['Total_Amount'].agg(['sum', 'mean', 'count']).round(2)
day_analysis = day_analysis.reindex(day_order)
day_analysis.columns = ['Total_Revenue', 'Avg_Transaction', 'Num_Transactions']
print(day_analysis.to_string())

# Regional performance
print("\n\n🗺️  REGIONAL PERFORMANCE:")
print("-"*90)
regional_performance = df.groupby('Region').agg({
    'Total_Amount': 'sum',
    'Customer_ID': 'nunique',
    'Transaction_ID': 'count'
}).round(2)
regional_performance.columns = ['Total_Revenue', 'Unique_Customers', 'Total_Transactions']
regional_performance['Avg_Revenue_per_Customer'] = (regional_performance['Total_Revenue'] / 
                                                     regional_performance['Unique_Customers']).round(2)
regional_performance = regional_performance.sort_values('Total_Revenue', ascending=False)
print(regional_performance.to_string())

# ============================================================================
# SECTION 4: CUSTOMER BEHAVIOR INSIGHTS
# ============================================================================
print("\n\n" + "="*90)
print("SECTION 4: CUSTOMER BEHAVIOR INSIGHTS")
print("="*90)

# RFM Analysis (Recency, Frequency, Monetary)
print("\n📊 RFM ANALYSIS (Top Customers by Combined Score):")
print("-"*90)

# Calculate RFM scores
customer_metrics['Recency_Score'] = pd.qcut(customer_metrics['Days_Since_Last_Purchase'], 
                                             q=5, labels=[5,4,3,2,1], duplicates='drop')
customer_metrics['Frequency_Score'] = pd.qcut(customer_metrics['Purchase_Count'].rank(method='first'), 
                                               q=5, labels=[1,2,3,4,5], duplicates='drop')
customer_metrics['Monetary_Score'] = pd.qcut(customer_metrics['Total_Spent'].rank(method='first'), 
                                              q=5, labels=[1,2,3,4,5], duplicates='drop')

customer_metrics['RFM_Score'] = (customer_metrics['Recency_Score'].astype(int) + 
                                 customer_metrics['Frequency_Score'].astype(int) + 
                                 customer_metrics['Monetary_Score'].astype(int))

customer_metrics['Customer_Type'] = customer_metrics['RFM_Score'].apply(
    lambda x: 'Champions' if x >= 13 else 
              'Loyal' if x >= 10 else 
              'Potential' if x >= 7 else 
              'At Risk' if x >= 5 else 'Lost'
)

rfm_summary = customer_metrics.groupby('Customer_Type').agg({
    'Customer_ID': 'count',
    'Total_Spent': 'mean',
    'Purchase_Count': 'mean'
}).round(2)
rfm_summary.columns = ['Count', 'Avg_Revenue', 'Avg_Orders']
print(rfm_summary.to_string())

# Payment method analysis
print("\n\n💳 PAYMENT METHOD PREFERENCES:")
print("-"*90)
payment_analysis = df.groupby('Payment_Method').agg({
    'Total_Amount': ['sum', 'count', 'mean']
}).round(2)
payment_analysis.columns = ['Total_Revenue', 'Num_Transactions', 'Avg_Transaction']
payment_analysis['Usage_%'] = (payment_analysis['Num_Transactions'] / total_transactions * 100).round(1)
print(payment_analysis.to_string())

# ============================================================================
# SECTION 5: KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n\n" + "="*90)
print("SECTION 5: KEY INSIGHTS & RECOMMENDATIONS")
print("="*90)

print("\n🔍 KEY FINDINGS:")
print("-"*90)

# Finding 1: Revenue concentration
top_10_revenue = customer_metrics.head(10)['Total_Spent'].sum()
top_10_percentage = (top_10_revenue / total_revenue * 100)
print(f"\n1. REVENUE CONCENTRATION:")
print(f"   • Top 10 customers generate ${top_10_revenue:,.2f} ({top_10_percentage:.1f}% of total revenue)")
print(f"   • Top customer segment: {segment_analysis['Total_Revenue'].idxmax()}")

# Finding 2: Best performing category
best_category = category_performance.index[0]
best_category_revenue = category_performance.loc[best_category, 'Total_Revenue']
print(f"\n2. PRODUCT PERFORMANCE:")
print(f"   • Best category: {best_category} (${best_category_revenue:,.2f})")
print(f"   • Most transactions: {category_performance['Num_Transactions'].idxmax()}")

# Finding 3: Regional insights
best_region = regional_performance.index[0]
print(f"\n3. GEOGRAPHIC INSIGHTS:")
print(f"   • Top region: {best_region} (${regional_performance.loc[best_region, 'Total_Revenue']:,.2f})")
print(f"   • Most customers: {regional_performance['Unique_Customers'].idxmax()} region")

# Finding 4: Customer retention
at_risk_customers = len(customer_metrics[customer_metrics['Days_Since_Last_Purchase'] > 90])
print(f"\n4. CUSTOMER RETENTION:")
print(f"   • Active customers (purchased in last 30 days): {len(customer_metrics[customer_metrics['Days_Since_Last_Purchase'] <= 30])}")
print(f"   • At-risk customers (no purchase in 90+ days): {at_risk_customers}")

print("\n\n💡 STRATEGIC RECOMMENDATIONS:")
print("-"*90)
print("1. 🎯 VIP PROGRAM: Launch exclusive program for Premium segment (highest CLV)")
print("2. 📧 RE-ENGAGEMENT: Email campaign for 'At Risk' customers with personalized offers")
print(f"3. 🏆 CATEGORY FOCUS: Expand '{best_category}' inventory and marketing")
print(f"4. 🗺️  REGIONAL EXPANSION: Replicate {best_region} region strategies in other areas")
print("5. 💳 PAYMENT INCENTIVES: Promote digital wallet usage with cashback offers")
print("6. 📅 WEEKEND PROMOTIONS: Boost sales on slower weekdays with targeted campaigns")
print("7. 🔄 UPSELL STRATEGY: Increase average order value through bundling")
print("8. 📊 LOYALTY PROGRAM: Implement points system to increase purchase frequency")

# ============================================================================
# SECTION 6: COMPREHENSIVE DASHBOARD VISUALIZATIONS
# ============================================================================
print("\n\n" + "="*90)
print("GENERATING COMPREHENSIVE SALES DASHBOARD...")
print("="*90)

fig = plt.figure(figsize=(20, 14))
fig.suptitle('CUSTOMER SALES PERFORMANCE DASHBOARD', fontsize=20, fontweight='bold', y=0.995)

# Chart 1: Monthly Revenue Trend with Moving Average
ax1 = plt.subplot(3, 4, 1)
monthly_data = df.groupby(df['Date'].dt.to_period('M'))['Total_Amount'].sum()
monthly_data.index = monthly_data.index.to_timestamp()
ax1.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2.5, 
         markersize=8, color='#2E86AB', label='Monthly Revenue')
ax1.fill_between(monthly_data.index, monthly_data.values, alpha=0.3, color='#2E86AB')

# Moving average
ma = monthly_data.rolling(window=3).mean()
ax1.plot(ma.index, ma.values, '--', linewidth=2, color='#E63946', label='3-Month MA')

ax1.set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Month', fontsize=10, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3)
plt.xticks(rotation=45)

# Chart 2: Top 10 Customers Bar Chart
ax2 = plt.subplot(3, 4, 2)
top_10 = customer_metrics.head(10)
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
bars = ax2.barh(range(10), top_10['Total_Spent'].values, color=colors_gradient, edgecolor='black')
ax2.set_yticks(range(10))
ax2.set_yticklabels(top_10['Customer_ID'].values, fontsize=9)
ax2.set_xlabel('Total Revenue ($)', fontsize=10, fontweight='bold')
ax2.set_title('Top 10 Customers by Revenue', fontsize=12, fontweight='bold', pad=10)
ax2.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, top_10['Total_Spent'].values)):
    ax2.text(val, i, f' ${val:,.0f}', va='center', fontsize=8, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Chart 3: Customer Segment Distribution (Pie Chart)
ax3 = plt.subplot(3, 4, 3)
segment_counts = df.groupby('Segment')['Customer_ID'].nunique()
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = ax3.pie(segment_counts.values, labels=segment_counts.index, 
                                     autopct='%1.1f%%', colors=colors_pie,
                                     startangle=90, textprops={'fontweight': 'bold', 'fontsize': 10})
ax3.set_title('Customer Segment Distribution', fontsize=12, fontweight='bold', pad=10)

# Chart 4: Category Performance
ax4 = plt.subplot(3, 4, 4)
cat_revenue = df.groupby('Category')['Total_Amount'].sum().sort_values(ascending=True)
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(cat_revenue)))
bars = ax4.barh(cat_revenue.index, cat_revenue.values, color=colors_cat, edgecolor='black')
ax4.set_xlabel('Revenue ($)', fontsize=10, fontweight='bold')
ax4.set_title('Revenue by Product Category', fontsize=12, fontweight='bold', pad=10)
for bar, val in zip(bars, cat_revenue.values):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f' ${val/1000:.1f}K', 
             va='center', fontsize=9, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Chart 5: Regional Performance
ax5 = plt.subplot(3, 4, 5)
region_revenue = df.groupby('Region')['Total_Amount'].sum().sort_values(ascending=False)
bars = ax5.bar(region_revenue.index, region_revenue.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
               edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
ax5.set_title('Revenue by Region', fontsize=12, fontweight='bold', pad=10)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'${height/1000:.0f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Chart 6: Sales by Day of Week
ax6 = plt.subplot(3, 4, 6)
day_revenue = df.groupby('Day_of_Week')['Total_Amount'].sum().reindex(day_order)
colors_days = ['#FF6B6B' if x in ['Saturday', 'Sunday'] else '#4ECDC4' for x in day_order]
bars = ax6.bar(range(7), day_revenue.values, color=colors_days, edgecolor='black', linewidth=1.5)
ax6.set_xticks(range(7))
ax6.set_xticklabels([d[:3] for d in day_order], fontsize=9)
ax6.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
ax6.set_title('Sales by Day of Week', fontsize=12, fontweight='bold', pad=10)
ax6.grid(axis='y', alpha=0.3)

# Chart 7: Customer Purchase Frequency Distribution
ax7 = plt.subplot(3, 4, 7)
purchase_freq = customer_metrics['Purchase_Count']
ax7.hist(purchase_freq, bins=20, color='#45B7D1', edgecolor='black', alpha=0.7)
ax7.axvline(purchase_freq.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {purchase_freq.mean():.1f}')
ax7.axvline(purchase_freq.median(), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {purchase_freq.median():.1f}')
ax7.set_xlabel('Number of Purchases', fontsize=10, fontweight='bold')
ax7.set_ylabel('Number of Customers', fontsize=10, fontweight='bold')
ax7.set_title('Customer Purchase Frequency', fontsize=12, fontweight='bold', pad=10)
ax7.legend(fontsize=8)
ax7.grid(axis='y', alpha=0.3)

# Chart 8: Average Transaction Value by Segment
ax8 = plt.subplot(3, 4, 8)
segment_avg = df.groupby('Segment')['Total_Amount'].mean().sort_values(ascending=False)
bars = ax8.bar(segment_avg.index, segment_avg.values, 
               color=['#FFD700', '#C0C0C0', '#CD7F32'], edgecolor='black', linewidth=1.5)
ax8.set_ylabel('Avg Transaction ($)', fontsize=10, fontweight='bold')
ax8.set_title('Avg Transaction by Segment', fontsize=12, fontweight='bold', pad=10)
for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# Chart 9: Customer Lifetime Value Distribution
ax9 = plt.subplot(3, 4, 9)
clv_data = customer_metrics['Total_Spent']
ax9.boxplot([clv_data[customer_metrics['Segment']=='Premium'],
             clv_data[customer_metrics['Segment']=='Regular'],
             clv_data[customer_metrics['Segment']=='Occasional']],
            labels=['Premium', 'Regular', 'Occasional'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='navy'),
            medianprops=dict(color='red', linewidth=2))
ax9.set_ylabel('Customer Lifetime Value ($)', fontsize=10, fontweight='bold')
ax9.set_title('CLV by Customer Segment', fontsize=12, fontweight='bold', pad=10)
ax9.grid(axis='y', alpha=0.3)

# Chart 10: Payment Method Distribution
ax10 = plt.subplot(3, 4, 10)
payment_dist = df.groupby('Payment_Method')['Total_Amount'].sum().sort_values(ascending=False)
colors_payment = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
wedges, texts, autotexts = ax10.pie(payment_dist.values, labels=payment_dist.index,
                                      autopct='%1.1f%%', colors=colors_payment,
                                      startangle=45, textprops={'fontweight': 'bold', 'fontsize': 9})
ax10.set_title('Payment Method Revenue Share', fontsize=12, fontweight='bold', pad=10)

# Chart 11: RFM Customer Type Distribution
ax11 = plt.subplot(3, 4, 11)
rfm_dist = customer_metrics['Customer_Type'].value_counts()
colors_rfm = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#95A5A6']
bars = ax11.bar(rfm_dist.index, rfm_dist.values, color=colors_rfm[:len(rfm_dist)], 
                edgecolor='black', linewidth=1.5)
ax11.set_ylabel('Number of Customers', fontsize=10, fontweight='bold')
ax11.set_title('RFM Customer Classification', fontsize=12, fontweight='bold', pad=10)
ax11.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax11.text(bar.get_x() + bar.get_width()/2., height,
              f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax11.grid(axis='y', alpha=0.3)

# Chart 12: Quarterly Revenue Comparison
ax12 = plt.subplot(3, 4, 12)
quarterly_revenue = df.groupby(df['Date'].dt.to_period('Q'))['Total_Amount'].sum()
quarters = [str(q) for q in quarterly_revenue.index]
bars = ax12.bar(range(len(quarters)), quarterly_revenue.values, 
                color=plt.cm.plasma(np.linspace(0.2, 0.8, len(quarters))),
                edgecolor='black', linewidth=1.5)
ax12.set_xticks(range(len(quarters)))
ax12.set_xticklabels(quarters, rotation=45, fontsize=9)
ax12.set_ylabel('Revenue ($)', fontsize=10, fontweight='bold')
ax12.set_title('Quarterly Revenue Trend', fontsize=12, fontweight='bold', pad=10)
for bar in bars:
    height = bar.get_height()
    ax12.text(bar.get_x() + bar.get_width()/2., height,
              f'${height/1000:.0f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax12.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('customer_sales_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Comprehensive Customer Sales Analysis Complete!")
print("📊 Dashboard with 12 visualizations generated successfully")
print("📁 Saved as 'customer_sales_dashboard.png'")
print("="*90)