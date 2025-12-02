import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create comprehensive student performance dataset
np.random.seed(42)

students = []
for i in range(100):
    # Generate correlated scores (students good in one subject tend to be good in others)
    base_ability = np.random.normal(70, 15)
    
    student = {
        'Student_ID': f'STU{i+1:03d}',
        'Gender': np.random.choice(['Male', 'Female'], p=[0.52, 0.48]),
        'Study_Hours': np.random.randint(1, 25),
        'Attendance': np.random.randint(60, 100),
        'Previous_Grade': np.random.choice(['A', 'B', 'C', 'D'], p=[0.15, 0.35, 0.35, 0.15]),
        'Extracurricular': np.random.choice(['Yes', 'No'], p=[0.4, 0.6]),
        'Parental_Support': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
        'Math_Score': int(np.clip(base_ability + np.random.normal(0, 8), 0, 100)),
        'Science_Score': int(np.clip(base_ability + np.random.normal(0, 8), 0, 100)),
        'English_Score': int(np.clip(base_ability + np.random.normal(0, 10), 0, 100)),
        'History_Score': int(np.clip(base_ability + np.random.normal(0, 9), 0, 100))
    }
    students.append(student)

df = pd.DataFrame(students)

# Calculate additional metrics
df['Total_Score'] = df['Math_Score'] + df['Science_Score'] + df['English_Score'] + df['History_Score']
df['Average_Score'] = df['Total_Score'] / 4
df['Grade'] = pd.cut(df['Average_Score'], 
                      bins=[0, 50, 60, 70, 80, 100],
                      labels=['F', 'D', 'C', 'B', 'A'])

print("="*80)
print("COMPREHENSIVE STUDENT PERFORMANCE ANALYSIS")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Students Analyzed: {len(df)}")
print("="*80)
print()

# ============================================================================
# SECTION 1: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("="*80)

print("\n1.1 Overall Performance Summary:")
print("-"*80)
print(f"Average Math Score:    {df['Math_Score'].mean():.2f} ± {df['Math_Score'].std():.2f}")
print(f"Average Science Score: {df['Science_Score'].mean():.2f} ± {df['Science_Score'].std():.2f}")
print(f"Average English Score: {df['English_Score'].mean():.2f} ± {df['English_Score'].std():.2f}")
print(f"Average History Score: {df['History_Score'].mean():.2f} ± {df['History_Score'].std():.2f}")
print(f"\nOverall Average:       {df['Average_Score'].mean():.2f}")
print(f"Median Score:          {df['Average_Score'].median():.2f}")
print(f"Standard Deviation:    {df['Average_Score'].std():.2f}")

print("\n1.2 Grade Distribution:")
print("-"*80)
grade_dist = df['Grade'].value_counts().sort_index(ascending=False)
for grade, count in grade_dist.items():
    percentage = (count / len(df)) * 100
    print(f"Grade {grade}: {count:2d} students ({percentage:.1f}%)")

print("\n1.3 Study Habits:")
print("-"*80)
print(f"Average Study Hours:   {df['Study_Hours'].mean():.2f} hours/week")
print(f"Average Attendance:    {df['Attendance'].mean():.2f}%")
print(f"Students with High Parental Support: {len(df[df['Parental_Support']=='High'])} ({len(df[df['Parental_Support']=='High'])/len(df)*100:.1f}%)")
print(f"Students in Extracurriculars: {len(df[df['Extracurricular']=='Yes'])} ({len(df[df['Extracurricular']=='Yes'])/len(df)*100:.1f}%)")

# ============================================================================
# SECTION 2: COMPARATIVE ANALYSIS
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 2: COMPARATIVE ANALYSIS")
print("="*80)

print("\n2.1 Performance by Gender:")
print("-"*80)
gender_stats = df.groupby('Gender')['Average_Score'].agg(['mean', 'median', 'std', 'count'])
print(gender_stats.to_string())

print("\n2.2 Performance by Parental Support Level:")
print("-"*80)
support_stats = df.groupby('Parental_Support')['Average_Score'].agg(['mean', 'median', 'count'])
support_stats = support_stats.reindex(['High', 'Medium', 'Low'])
print(support_stats.to_string())

print("\n2.3 Impact of Extracurricular Activities:")
print("-"*80)
extra_stats = df.groupby('Extracurricular')['Average_Score'].agg(['mean', 'median', 'std'])
print(extra_stats.to_string())

# ============================================================================
# SECTION 3: CORRELATION ANALYSIS
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 3: CORRELATION ANALYSIS")
print("="*80)

print("\n3.1 Study Hours vs Performance:")
print("-"*80)
correlation_study = df['Study_Hours'].corr(df['Average_Score'])
print(f"Correlation coefficient: {correlation_study:.3f}")
if correlation_study > 0.5:
    print("Finding: STRONG positive correlation - More study hours significantly improve performance")
elif correlation_study > 0.3:
    print("Finding: MODERATE positive correlation - Study hours have noticeable impact")
else:
    print("Finding: WEAK correlation - Study hours show limited direct impact")

print("\n3.2 Attendance vs Performance:")
print("-"*80)
correlation_attendance = df['Attendance'].corr(df['Average_Score'])
print(f"Correlation coefficient: {correlation_attendance:.3f}")
if correlation_attendance > 0.5:
    print("Finding: STRONG positive correlation - Attendance is crucial for performance")
elif correlation_attendance > 0.3:
    print("Finding: MODERATE positive correlation - Regular attendance helps performance")
else:
    print("Finding: WEAK correlation - Attendance shows limited direct impact")

# ============================================================================
# SECTION 4: KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n\n" + "="*80)
print("SECTION 4: KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n📊 KEY FINDINGS:")
print("-"*80)

# Finding 1: Top performers
top_10_percent = df.nlargest(10, 'Average_Score')
print(f"\n1. Top 10% Performance Characteristics:")
print(f"   - Average study hours: {top_10_percent['Study_Hours'].mean():.1f} hrs/week")
print(f"   - Average attendance: {top_10_percent['Attendance'].mean():.1f}%")
print(f"   - With extracurriculars: {(top_10_percent['Extracurricular']=='Yes').sum()}/{len(top_10_percent)}")

# Finding 2: Subject performance
subject_means = df[['Math_Score', 'Science_Score', 'English_Score', 'History_Score']].mean()
strongest_subject = subject_means.idxmax().replace('_Score', '')
weakest_subject = subject_means.idxmin().replace('_Score', '')
print(f"\n2. Subject Performance:")
print(f"   - Strongest subject: {strongest_subject} (avg: {subject_means.max():.1f})")
print(f"   - Weakest subject: {weakest_subject} (avg: {subject_means.min():.1f})")
print(f"   - Performance gap: {subject_means.max() - subject_means.min():.1f} points")

# Finding 3: At-risk students
at_risk = df[df['Average_Score'] < 60]
print(f"\n3. At-Risk Students (scoring below 60%):")
print(f"   - Count: {len(at_risk)} students ({len(at_risk)/len(df)*100:.1f}%)")
if len(at_risk) > 0:
    print(f"   - Average study hours: {at_risk['Study_Hours'].mean():.1f} hrs/week")
    print(f"   - Average attendance: {at_risk['Attendance'].mean():.1f}%")

print("\n\n💡 RECOMMENDATIONS:")
print("-"*80)
print("1. STUDY INTERVENTION: Students studying <10 hours/week need additional support")
print("2. ATTENDANCE MONITORING: Students with <75% attendance require intervention")
print(f"3. SUBJECT FOCUS: Implement remedial programs for {weakest_subject}")
print("4. PARENTAL ENGAGEMENT: Increase parental involvement programs")
print("5. EXTRACURRICULAR BALANCE: Encourage balanced participation in activities")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

fig = plt.figure(figsize=(16, 12))

# Chart 1: Bar Chart - Average Scores by Subject
ax1 = plt.subplot(2, 3, 1)
subjects = ['Math', 'Science', 'English', 'History']
scores = [df['Math_Score'].mean(), df['Science_Score'].mean(), 
          df['English_Score'].mean(), df['History_Score'].mean()]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax1.bar(subjects, scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax1.set_title('Average Performance by Subject', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{score:.1f}', ha='center', fontweight='bold')

# Chart 2: Pie Chart - Grade Distribution
ax2 = plt.subplot(2, 3, 2)
grade_colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#95A5A6']
wedges, texts, autotexts = ax2.pie(grade_dist.values, labels=grade_dist.index, 
                                     autopct='%1.1f%%', colors=grade_colors,
                                     startangle=90, textprops={'fontweight': 'bold'})
ax2.set_title('Grade Distribution', fontsize=13, fontweight='bold', pad=15)

# Chart 3: Scatter Plot - Study Hours vs Average Score
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(df['Study_Hours'], df['Average_Score'], 
                      c=df['Average_Score'], cmap='RdYlGn', 
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Study Hours per Week', fontsize=11, fontweight='bold')
ax3.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax3.set_title('Study Hours vs Performance', fontsize=13, fontweight='bold', pad=15)
ax3.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Score')
z = np.polyfit(df['Study_Hours'], df['Average_Score'], 1)
p = np.poly1d(z)
ax3.plot(df['Study_Hours'].sort_values(), p(df['Study_Hours'].sort_values()), 
         "r--", linewidth=2, label=f'Trend (r={correlation_study:.2f})')
ax3.legend()

# Chart 4: Box Plot - Performance by Parental Support
ax4 = plt.subplot(2, 3, 4)
support_order = ['High', 'Medium', 'Low']
box_data = [df[df['Parental_Support']==level]['Average_Score'].values for level in support_order]
bp = ax4.boxplot(box_data, labels=support_order, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', color='navy'),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(color='navy'),
                 capprops=dict(color='navy'))
ax4.set_xlabel('Parental Support Level', fontsize=11, fontweight='bold')
ax4.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax4.set_title('Performance by Parental Support', fontsize=13, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)

# Chart 5: Histogram - Score Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(df['Average_Score'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
ax5.axvline(df['Average_Score'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["Average_Score"].mean():.1f}')
ax5.axvline(df['Average_Score'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {df["Average_Score"].median():.1f}')
ax5.set_xlabel('Average Score', fontsize=11, fontweight='bold')
ax5.set_ylabel('Number of Students', fontsize=11, fontweight='bold')
ax5.set_title('Distribution of Average Scores', fontsize=13, fontweight='bold', pad=15)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Chart 6: Grouped Bar Chart - Gender Performance by Subject
ax6 = plt.subplot(2, 3, 6)
gender_subject_data = df.groupby('Gender')[['Math_Score', 'Science_Score', 
                                             'English_Score', 'History_Score']].mean()
x = np.arange(len(subjects))
width = 0.35
bars1 = ax6.bar(x - width/2, gender_subject_data.iloc[0], width, 
                label=gender_subject_data.index[0], color='#FF9999', edgecolor='black')
bars2 = ax6.bar(x + width/2, gender_subject_data.iloc[1], width, 
                label=gender_subject_data.index[1], color='#66B2FF', edgecolor='black')
ax6.set_xlabel('Subject', fontsize=11, fontweight='bold')
ax6.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax6.set_title('Gender Comparison by Subject', fontsize=13, fontweight='bold', pad=15)
ax6.set_xticks(x)
ax6.set_xticklabels(subjects)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('student_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Analysis complete! Comprehensive report and visualizations generated.")
print("📁 Charts saved as 'student_performance_analysis.png'")
print("="*80)