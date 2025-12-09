import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, levene, ttest_ind, chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

print("="*100)
print(" "*30 + "STATISTICAL HYPOTHESIS TESTING & ANALYSIS")
print("="*100)
print("\nThis program performs comprehensive statistical analysis including:")
print("• Hypothesis Testing (t-tests, ANOVA, Chi-square)")
print("• Correlation Analysis (Pearson, Spearman)")
print("• Marketing Effectiveness Analysis")
print("• Revenue Prediction Modeling")
print("="*100)

# ============================================================================
# GENERATE REALISTIC SALES & MARKETING DATA
# ============================================================================
np.random.seed(42)

# Generate 12 months of data
months = pd.date_range('2024-01-01', periods=12, freq='M')
regions = ['North', 'South', 'East', 'West']
channels = ['Online', 'Retail', 'Wholesale']
campaigns = ['Social Media', 'TV', 'Email', 'SEO', 'Print']

data = []
for month in months:
    for region in regions:
        # Marketing spend with some correlation to revenue
        social_spend = np.random.uniform(5000, 15000)
        tv_spend = np.random.uniform(10000, 30000)
        email_spend = np.random.uniform(2000, 8000)
        seo_spend = np.random.uniform(3000, 10000)
        print_spend = np.random.uniform(5000, 12000)
        
        total_marketing = social_spend + tv_spend + email_spend + seo_spend + print_spend
        
        # Revenue influenced by marketing spend + random factors
        base_revenue = 50000 + np.random.normal(0, 10000)
        marketing_effect = (social_spend * 1.5 + tv_spend * 1.2 + 
                          email_spend * 2.0 + seo_spend * 1.8 + print_spend * 0.8)
        
        # Regional multipliers
        regional_multipliers = {'North': 1.3, 'South': 1.1, 'East': 1.0, 'West': 0.9}
        
        revenue = (base_revenue + marketing_effect) * regional_multipliers[region] + np.random.normal(0, 15000)
        revenue = max(revenue, 30000)  # Ensure positive revenue
        
        # Generate correlated metrics
        customer_count = int(revenue / np.random.uniform(150, 250))
        conversion_rate = np.random.uniform(0.02, 0.08)
        avg_order_value = revenue / customer_count if customer_count > 0 else 0
        
        data.append({
            'Month': month,
            'Region': region,
            'Revenue': round(revenue, 2),
            'Marketing_Spend': round(total_marketing, 2),
            'Social_Media_Spend': round(social_spend, 2),
            'TV_Spend': round(tv_spend, 2),
            'Email_Spend': round(email_spend, 2),
            'SEO_Spend': round(seo_spend, 2),
            'Print_Spend': round(print_spend, 2),
            'Customer_Count': customer_count,
            'Conversion_Rate': round(conversion_rate, 4),
            'Avg_Order_Value': round(avg_order_value, 2),
            'Channel': np.random.choice(channels, p=[0.5, 0.3, 0.2])
        })

df = pd.DataFrame(data)

# Calculate ROI
df['ROI'] = ((df['Revenue'] - df['Marketing_Spend']) / df['Marketing_Spend'] * 100).round(2)
df['Revenue_per_Customer'] = (df['Revenue'] / df['Customer_Count']).round(2)

print(f"\n✓ Generated dataset with {len(df)} observations")
print(f"✓ Time period: {df['Month'].min().strftime('%B %Y')} to {df['Month'].max().strftime('%B %Y')}")
print(f"✓ Regions analyzed: {', '.join(df['Region'].unique())}")

# Display sample data
print("\n" + "="*100)
print("SAMPLE DATA (First 5 Rows)")
print("="*100)
print(df.head().to_string(index=False))

# ============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n\n" + "="*100)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("="*100)

print("\n📊 KEY METRICS SUMMARY:")
print("-"*100)
print(f"Total Revenue:           ${df['Revenue'].sum():,.2f}")
print(f"Total Marketing Spend:   ${df['Marketing_Spend'].sum():,.2f}")
print(f"Average Monthly Revenue: ${df['Revenue'].mean():,.2f} ± ${df['Revenue'].std():,.2f}")
print(f"Average Marketing Spend: ${df['Marketing_Spend'].mean():,.2f} ± ${df['Marketing_Spend'].std():,.2f}")
print(f"Average ROI:             {df['ROI'].mean():.2f}%")
print(f"Total Customers:         {df['Customer_Count'].sum():,}")

print("\n📈 REVENUE STATISTICS BY REGION:")
print("-"*100)
revenue_by_region = df.groupby('Region')['Revenue'].agg(['mean', 'std', 'min', 'max', 'count'])
revenue_by_region.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'Count']
print(revenue_by_region.round(2).to_string())

# ============================================================================
# SECTION 2: HYPOTHESIS TESTING
# ============================================================================
print("\n\n" + "="*100)
print("SECTION 2: HYPOTHESIS TESTING")
print("="*100)

# Test 1: Independent Samples t-test
# H0: Mean revenue is equal between North and South regions
# H1: Mean revenue is different between North and South regions
print("\n" + "─"*100)
print("TEST 1: INDEPENDENT SAMPLES T-TEST")
print("─"*100)
print("Research Question: Is there a significant difference in revenue between North and South regions?")
print("\nHypotheses:")
print("  H₀ (Null): μ_North = μ_South (No difference in mean revenue)")
print("  H₁ (Alternative): μ_North ≠ μ_South (Significant difference exists)")

north_revenue = df[df['Region'] == 'North']['Revenue']
south_revenue = df[df['Region'] == 'South']['Revenue']

# Check normality assumption
shapiro_north = shapiro(north_revenue)
shapiro_south = shapiro(south_revenue)

print(f"\nAssumption Check (Normality):")
print(f"  North region: Shapiro-Wilk p-value = {shapiro_north.pvalue:.4f}")
print(f"  South region: Shapiro-Wilk p-value = {shapiro_south.pvalue:.4f}")
print(f"  {'✓ Data appears normally distributed (p > 0.05)' if min(shapiro_north.pvalue, shapiro_south.pvalue) > 0.05 else '⚠ Data may not be normally distributed'}")

# Levene's test for equal variances
levene_stat, levene_p = levene(north_revenue, south_revenue)
print(f"\nAssumption Check (Equal Variances):")
print(f"  Levene's test p-value = {levene_p:.4f}")
print(f"  {'✓ Variances are equal (p > 0.05)' if levene_p > 0.05 else '⚠ Variances are unequal'}")

# Perform t-test
t_stat, t_pvalue = ttest_ind(north_revenue, south_revenue)

print(f"\nTest Results:")
print(f"  Sample Sizes: North = {len(north_revenue)}, South = {len(south_revenue)}")
print(f"  Mean Revenue (North): ${north_revenue.mean():,.2f}")
print(f"  Mean Revenue (South): ${south_revenue.mean():,.2f}")
print(f"  Difference in Means: ${north_revenue.mean() - south_revenue.mean():,.2f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pvalue:.4f}")

alpha = 0.05
print(f"\nDecision (α = {alpha}):")
if t_pvalue < alpha:
    print(f"  ✓ REJECT NULL HYPOTHESIS (p < {alpha})")
    print(f"  Conclusion: There IS a statistically significant difference in revenue between North and South regions.")
    effect_size = abs(north_revenue.mean() - south_revenue.mean()) / np.sqrt((north_revenue.std()**2 + south_revenue.std()**2) / 2)
    print(f"  Cohen's d (Effect Size): {effect_size:.4f}")
else:
    print(f"  ✗ FAIL TO REJECT NULL HYPOTHESIS (p ≥ {alpha})")
    print(f"  Conclusion: There is NO statistically significant difference in revenue between regions.")

# Test 2: One-Way ANOVA
# H0: Mean revenue is equal across all regions
# H1: At least one region has different mean revenue
print("\n\n" + "─"*100)
print("TEST 2: ONE-WAY ANOVA (Analysis of Variance)")
print("─"*100)
print("Research Question: Is there a significant difference in revenue across ALL four regions?")
print("\nHypotheses:")
print("  H₀ (Null): μ_North = μ_South = μ_East = μ_West (All means are equal)")
print("  H₁ (Alternative): At least one region has a different mean revenue")

# Prepare data for ANOVA
groups = [df[df['Region'] == region]['Revenue'].values for region in regions]
f_stat, anova_pvalue = f_oneway(*groups)

print(f"\nDescriptive Statistics by Region:")
for region in regions:
    region_data = df[df['Region'] == region]['Revenue']
    print(f"  {region:6s}: Mean = ${region_data.mean():,.2f}, SD = ${region_data.std():,.2f}, n = {len(region_data)}")

print(f"\nTest Results:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {anova_pvalue:.4f}")

print(f"\nDecision (α = {alpha}):")
if anova_pvalue < alpha:
    print(f"  ✓ REJECT NULL HYPOTHESIS (p < {alpha})")
    print(f"  Conclusion: There IS a statistically significant difference in revenue across regions.")
    print(f"  Recommendation: Conduct post-hoc tests (e.g., Tukey HSD) to identify which specific regions differ.")
else:
    print(f"  ✗ FAIL TO REJECT NULL HYPOTHESIS (p ≥ {alpha})")
    print(f"  Conclusion: There is NO statistically significant difference in revenue across regions.")

# Test 3: Chi-Square Test of Independence
# H0: Channel type is independent of Region
# H1: Channel type is associated with Region
print("\n\n" + "─"*100)
print("TEST 3: CHI-SQUARE TEST OF INDEPENDENCE")
print("─"*100)
print("Research Question: Is there an association between sales channel and region?")
print("\nHypotheses:")
print("  H₀ (Null): Channel and Region are independent (no association)")
print("  H₁ (Alternative): Channel and Region are associated (dependent)")

# Create contingency table
contingency_table = pd.crosstab(df['Region'], df['Channel'])
print(f"\nContingency Table (Observed Frequencies):")
print(contingency_table.to_string())

chi2, chi_pvalue, dof, expected = chi2_contingency(contingency_table)

print(f"\nTest Results:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  Degrees of freedom: {dof}")
print(f"  p-value: {chi_pvalue:.4f}")

print(f"\nDecision (α = {alpha}):")
if chi_pvalue < alpha:
    print(f"  ✓ REJECT NULL HYPOTHESIS (p < {alpha})")
    print(f"  Conclusion: There IS a statistically significant association between channel and region.")
else:
    print(f"  ✗ FAIL TO REJECT NULL HYPOTHESIS (p ≥ {alpha})")
    print(f"  Conclusion: There is NO statistically significant association between channel and region.")

# ============================================================================
# SECTION 3: CORRELATION ANALYSIS
# ============================================================================
print("\n\n" + "="*100)
print("SECTION 3: CORRELATION ANALYSIS")
print("="*100)

# Pearson Correlation: Marketing Spend vs Revenue
print("\n" + "─"*100)
print("ANALYSIS 1: MARKETING SPEND vs REVENUE CORRELATION")
print("─"*100)

pearson_corr, pearson_p = pearsonr(df['Marketing_Spend'], df['Revenue'])
spearman_corr, spearman_p = spearmanr(df['Marketing_Spend'], df['Revenue'])

print(f"\nPearson Correlation Coefficient (Linear Relationship):")
print(f"  r = {pearson_corr:.4f}")
print(f"  p-value = {pearson_p:.4f}")

# Interpret correlation strength
if abs(pearson_corr) >= 0.7:
    strength = "STRONG"
elif abs(pearson_corr) >= 0.5:
    strength = "MODERATE"
elif abs(pearson_corr) >= 0.3:
    strength = "WEAK"
else:
    strength = "VERY WEAK/NONE"

direction = "positive" if pearson_corr > 0 else "negative"

print(f"  Interpretation: {strength} {direction} correlation")
print(f"  R² (Coefficient of Determination): {pearson_corr**2:.4f} ({pearson_corr**2*100:.1f}% of variance explained)")

print(f"\nSpearman Rank Correlation (Monotonic Relationship):")
print(f"  ρ (rho) = {spearman_corr:.4f}")
print(f"  p-value = {spearman_p:.4f}")

print(f"\nStatistical Significance:")
if pearson_p < 0.05:
    print(f"  ✓ Correlation is statistically significant (p < 0.05)")
    print(f"  Conclusion: Marketing spend and revenue are significantly correlated.")
else:
    print(f"  ✗ Correlation is NOT statistically significant (p ≥ 0.05)")

# Individual Marketing Channel Correlations
print("\n\n" + "─"*100)
print("ANALYSIS 2: INDIVIDUAL MARKETING CHANNEL EFFECTIVENESS")
print("─"*100)

channels_to_test = ['Social_Media_Spend', 'TV_Spend', 'Email_Spend', 'SEO_Spend', 'Print_Spend']
channel_correlations = []

print(f"\nCorrelation of each marketing channel with Revenue:")
print(f"{'Channel':<20} {'Correlation':<15} {'p-value':<15} {'Significance':<20} {'ROI Impact'}")
print("-"*100)

for channel in channels_to_test:
    corr, p_val = pearsonr(df[channel], df['Revenue'])
    sig = "✓ Significant" if p_val < 0.05 else "✗ Not Significant"
    
    # Calculate average ROI for this channel
    avg_revenue_per_dollar = df['Revenue'].sum() / df[channel].sum()
    
    channel_correlations.append({
        'Channel': channel.replace('_Spend', ''),
        'Correlation': corr,
        'p_value': p_val,
        'Significant': p_val < 0.05
    })
    
    print(f"{channel.replace('_Spend', ''):<20} {corr:>8.4f}      {p_val:>10.4f}    {sig:<20} ${avg_revenue_per_dollar:.2f}")

# Rank channels by effectiveness
channel_corr_df = pd.DataFrame(channel_correlations).sort_values('Correlation', ascending=False)
print(f"\n📊 MARKETING CHANNEL RANKING (by correlation strength):")
for idx, row in channel_corr_df.iterrows():
    print(f"  {idx+1}. {row['Channel']}: r = {row['Correlation']:.4f} {'✓' if row['Significant'] else '✗'}")

# Correlation Matrix
print("\n\n" + "─"*100)
print("ANALYSIS 3: COMPREHENSIVE CORRELATION MATRIX")
print("─"*100)

correlation_vars = ['Revenue', 'Marketing_Spend', 'Customer_Count', 'Conversion_Rate', 
                   'Avg_Order_Value', 'ROI']
corr_matrix = df[correlation_vars].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3).to_string())

print("\n🔍 KEY INSIGHTS FROM CORRELATION MATRIX:")
print("-"*100)

# Find strongest correlations
correlations_list = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        correlations_list.append({
            'Variable 1': corr_matrix.columns[i],
            'Variable 2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

correlations_df = pd.DataFrame(correlations_list).sort_values('Correlation', key=abs, ascending=False)

print("\nTop 5 Strongest Correlations:")
for idx, row in correlations_df.head(5).iterrows():
    print(f"  • {row['Variable 1']} ↔ {row['Variable 2']}: r = {row['Correlation']:.4f}")

# ============================================================================
# SECTION 4: REGRESSION ANALYSIS & PREDICTION
# ============================================================================
print("\n\n" + "="*100)
print("SECTION 4: SIMPLE LINEAR REGRESSION - REVENUE PREDICTION")
print("="*100)

from scipy.stats import linregress

# Simple Linear Regression: Marketing Spend → Revenue
X = df['Marketing_Spend'].values
y = df['Revenue'].values

slope, intercept, r_value, p_value, std_err = linregress(X, y)

print(f"\nRegression Equation: Revenue = {intercept:.2f} + {slope:.4f} × Marketing_Spend")
print(f"\nModel Parameters:")
print(f"  Intercept (β₀): ${intercept:,.2f}")
print(f"  Slope (β₁): {slope:.4f}")
print(f"  Interpretation: For every $1 increase in marketing spend, revenue increases by ${slope:.2f}")
print(f"\nModel Performance:")
print(f"  R-squared (R²): {r_value**2:.4f} ({r_value**2*100:.1f}% of variance explained)")
print(f"  Standard Error: ${std_err:,.2f}")
print(f"  p-value: {p_value:.4e}")

# Make predictions
predicted_revenue = intercept + slope * X
residuals = y - predicted_revenue
rmse = np.sqrt(np.mean(residuals**2))

print(f"  RMSE (Root Mean Square Error): ${rmse:,.2f}")

# Example predictions
print(f"\n📈 EXAMPLE PREDICTIONS:")
print("-"*100)
test_spends = [30000, 50000, 70000, 100000]
for spend in test_spends:
    pred_revenue = intercept + slope * spend
    print(f"  Marketing Spend: ${spend:,} → Predicted Revenue: ${pred_revenue:,.2f}")

# ============================================================================
# SECTION 5: KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n\n" + "="*100)
print("SECTION 5: STATISTICAL INSIGHTS & RECOMMENDATIONS")
print("="*100)

print("\n🔍 KEY FINDINGS:")
print("-"*100)

# Finding 1: Regional differences
best_region = df.groupby('Region')['Revenue'].mean().idxmax()
worst_region = df.groupby('Region')['Revenue'].mean().idxmin()
print(f"\n1. REGIONAL PERFORMANCE:")
print(f"   • Best performing region: {best_region}")
print(f"   • Lowest performing region: {worst_region}")
print(f"   • Statistical significance: {'Confirmed by ANOVA' if anova_pvalue < 0.05 else 'Not significant'}")

# Finding 2: Marketing effectiveness
best_channel = channel_corr_df.iloc[0]
print(f"\n2. MARKETING EFFECTIVENESS:")
print(f"   • Most effective channel: {best_channel['Channel']} (r = {best_channel['Correlation']:.3f})")
print(f"   • Overall marketing-revenue correlation: {pearson_corr:.3f} ({strength})")
print(f"   • Average ROI: {df['ROI'].mean():.1f}%")

# Finding 3: Predictive power
print(f"\n3. PREDICTIVE MODELING:")
print(f"   • Marketing spend explains {r_value**2*100:.1f}% of revenue variation")
print(f"   • Expected return: ${slope:.2f} per dollar spent")
print(f"   • Model reliability: {'High' if r_value**2 > 0.5 else 'Moderate' if r_value**2 > 0.3 else 'Low'}")

print("\n\n💡 STRATEGIC RECOMMENDATIONS:")
print("-"*100)
print(f"1. 🎯 CHANNEL OPTIMIZATION: Increase budget allocation to {channel_corr_df.iloc[0]['Channel']}")
print(f"2. 📍 REGIONAL STRATEGY: Replicate {best_region} region tactics in underperforming areas")
print(f"3. 💰 BUDGET ALLOCATION: Optimal spend appears to be in ${df['Marketing_Spend'].quantile(0.75):,.0f} - ${df['Marketing_Spend'].max():,.0f} range")
print(f"4. 📊 CONTINUOUS TESTING: Conduct A/B tests to validate channel effectiveness")
print(f"5. 🔄 PORTFOLIO BALANCING: Maintain diverse channel mix while prioritizing top performers")

# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================
print("\n\n" + "="*100)
print("GENERATING STATISTICAL VISUALIZATIONS...")
print("="*100)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('STATISTICAL ANALYSIS DASHBOARD', fontsize=20, fontweight='bold', y=0.995)

# Plot 1: Scatter plot with regression line
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(df['Marketing_Spend'], df['Revenue'], alpha=0.6, s=100, c=df['ROI'], 
           cmap='RdYlGn', edgecolors='black')
ax1.plot(X, predicted_revenue, 'r--', linewidth=3, label=f'y = {intercept:.0f} + {slope:.2f}x')
ax1.set_xlabel('Marketing Spend ($)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=11, fontweight='bold')
ax1.set_title(f'Marketing Spend vs Revenue\n(R² = {r_value**2:.3f}, p < 0.001)', 
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
cbar = plt.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label('ROI (%)', fontsize=10)

# Plot 2: Box plot - Revenue by Region
ax2 = plt.subplot(2, 3, 2)
region_data = [df[df['Region'] == r]['Revenue'].values for r in regions]
bp = ax2.boxplot(region_data, labels=regions, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
ax2.set_ylabel('Revenue ($)', fontsize=11, fontweight='bold')
ax2.set_title(f'Revenue Distribution by Region\n(ANOVA p-value = {anova_pvalue:.4f})', 
             fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Correlation heatmap
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax3)
ax3.set_title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold', pad=10)

# Plot 4: Marketing channel effectiveness
ax4 = plt.subplot(2, 3, 4)
channels_sorted = channel_corr_df.sort_values('Correlation')
colors = ['green' if c > 0.5 else 'orange' if c > 0.3 else 'red' 
         for c in channels_sorted['Correlation']]
ax4.barh(channels_sorted['Channel'], channels_sorted['Correlation'], color=colors, 
        edgecolor='black', alpha=0.7)
ax4.set_xlabel('Correlation with Revenue', fontsize=11, fontweight='bold')
ax4.set_title('Marketing Channel Effectiveness', fontsize=12, fontweight='bold')
ax4.axvline(0, color='black', linewidth=0.8)
ax4.grid(axis='x', alpha=0.3)

# Plot 5: Residual plot
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(predicted_revenue, residuals, alpha=0.6, s=80, edgecolors='black')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Revenue ($)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
ax5.set_title(f'Residual Plot\n(RMSE = ${rmse:,.0f})', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# Plot 6: ROI distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(df['ROI'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax6.axvline(df['ROI'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["ROI"].mean():.1f}%')
ax6.axvline(df['ROI'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df["ROI"].median():.1f}%')
ax6.set_xlabel('ROI (%)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Return on Investment Distribution', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Statistical Analysis Complete!")
print("📊 Comprehensive hypothesis testing, correlation analysis, and visualizations generated")
print("="*100)