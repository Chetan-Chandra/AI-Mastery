# **AI Mastery Program — Lesson 2**

## **Mathematics for Machine Learning: Statistics and Probability**

Lesson 2

Beginner-friendly

3–4 hours

Today we'll learn the mathematics behind machine learning using a practical customer dataset. You don't need to be strong at mathematics already. I'll explain what each concept means, why ML needs it, and how to implement it in Python.

**Today's outcome:** You'll be able to describe a dataset statistically, understand how spread and probability work, identify relationships between variables, and detect outliers.

## **1. Why does machine learning need statistics?**

Imagine your company has customer data:

| Customer | Monthly charges | Support tickets | Churned |
| -------- | --------------- | --------------- | ------- |
| A        | ₹800            | 1               | No      |
| B        | ₹2,500          | 4               | Yes     |
| C        | ₹1,200          | 0               | No      |
| D        | ₹4,000          | 6               | Yes     |
| E        | ₹1,500          | 2               | No      |

Before training a model, we need to understand the data.

Questions statistics helps us answer:

- What is the average monthly charge?
- What is a typical number of support tickets?
- Are charges spread closely together or widely?
- Are unusually high charges present?
- Do customers with more support tickets tend to churn more often?

These answers help us understand the data and prepare it for ML.

## **2. Population vs sample**

Suppose a telecom company has 1 million customers.

- **Population:** All 1 million customers.
- **Sample:** A smaller group selected from those customers, perhaps 10,000.

Why use a sample? Analysing every record may be expensive or unnecessary. A representative sample can help us estimate patterns in the full population.

**Think of it like a data pipeline**

**Population**

# **1,000,000**

All customer records

Sampling

**Sample**

# **10,000**

Records used for analysis

Important: A sample should represent the population. If we randomly select only customers from one city, the sample may not represent the entire customer base.

## **3. Mean, median and mode**

These are three ways to describe the centre of a dataset.

Let's use this dataset:

`[10, 20, 20, 30, 40]`

### **A. Mean — the average**

Add all values and divide by the number of values.

\\[ Mean = \frac{10+20+20+30+40}{5}=24 \\]

Python:

```
import numpy as npvalues = [10, 20, 20, 30, 40]print("Mean:", np.mean(values))
```

Result: `24.0`

### **B. Median — the middle value**

First, arrange the numbers in ascending order. The middle value is the median.

`[10, 20, 20, 30, 40]`

Median = `20`

For an even number of values, average the two middle values.

```
print("Median:", np.median(values))
```

### **C. Mode — the most frequently occurring value**

In our example, `20` appears twice, more than any other value.

```
from scipy import statsprint("Mode:", stats.mode(values, keepdims=False).mode)
```

Result: `20`

### **When should you use each?**

| Measure | Useful when                       |
| ------- | --------------------------------- |
| Mean    | You want the overall average      |
| Median  | Your data contains extreme values |
| Mode    | You want the most common value    |

**Real-world example:** Salaries are ₹20K, ₹25K, ₹30K, ₹35K and ₹5 lakh.

The mean gets pulled upward by the ₹5 lakh salary. The median better represents the typical salary in this small group.

## **4. Range and percentiles**

### **A. Range — how far apart are the smallest and largest values?**

\\[ Range = Maximum - Minimum \\]

For `[10, 20, 20, 30, 40]`:

\\[ Range = 40-10=30 \\]

```
print("Minimum:", np.min(values))print("Maximum:", np.max(values))print("Range:", np.max(values) - np.min(values))
```

A larger range means the values cover a wider interval. But range only looks at the two extreme values, so it can be heavily affected by outliers.

### **B. Percentiles — where does a value stand relative to the rest?**

A percentile tells us the value below which a given percentage of observations falls.

For example:

- 25th percentile: 25% of observations are at or below this value.
- 50th percentile: the median.
- 75th percentile: 75% of observations are at or below this value.
- 90th percentile: 90% of observations are at or below this value.

```
print("25th percentile:", np.percentile(values, 25))print("50th percentile:", np.percentile(values, 50))print("75th percentile:", np.percentile(values, 75))print("90th percentile:", np.percentile(values, 90))
```

In Pandas, you can calculate these together:

```
import pandas as pdseries = pd.Series(values)print(series.describe())
```

You'll see count, mean, standard deviation, minimum, quartiles and maximum.

**Data engineering connection:** Percentiles are useful when analysing pipeline execution times. For example, the 95th percentile latency tells you the time within which 95% of runs completed. This is often more informative than the average.

## **5. Variance — how spread out are the values?**

Consider two teams' daily processing times, in minutes:

- Team A: `[9, 10, 10, 11, 10]`
- Team B: `[2, 6, 10, 14, 18]`

Both have a mean of 10 minutes. But Team B's values are much more spread out.

Variance measures how far values are spread around their mean.

The basic idea:

1. Calculate the mean.
2. Subtract the mean from each value.
3. Square each difference.
4. Calculate the average of those squared differences.

For a population:

\\[ \sigma^2 = \frac{\sum (x_i-\mu)^2}{N} \\]

Don't worry about memorising the formula yet.

- \\(x_i\\): each observation
- \\(\mu\\): population mean
- \\(N\\): number of observations
- \\(\sigma^2\\): population variance

Python:

```
team_a = [9, 10, 10, 11, 10]team_b = [2, 6, 10, 14, 18]print("Team A variance:", np.var(team_a))print("Team B variance:", np.var(team_b))
```

Result:

- Team A: `0.4`
- Team B: `32.0`

Team B has a much higher variance.

**Important distinction:** `np.var(data)` calculates population variance by default. When your data is a sample and you want to estimate population variance, use `np.var(data, ddof=1)`. This applies the sample correction.

## **6. Standard deviation — spread in the original unit**

Variance is useful mathematically, but it is expressed in squared units. Standard deviation makes the spread easier to interpret.

\\[ Standard\ Deviation = \sqrt{Variance} \\]

For our teams:

```
print("Team A std:", np.std(team_a))print("Team B std:", np.std(team_b))
```

- Team A standard deviation: approximately `0.63 minutes`
- Team B standard deviation: approximately `5.66 minutes`

**How to interpret it:** A larger standard deviation means the observations are more spread out around the mean.

For machine learning, standard deviation is important because features may have very different scales. For example, customer age may range from 18 to 80, while annual income may range from 100,000 to 2,000,000.

Some ML algorithms are sensitive to these differences. Standardisation transforms a feature using:

\\[ z = \frac{x-\mu}{\sigma} \\]

This is the same basic idea behind the `StandardScaler` you used in Lesson 1.

## **7. Probability fundamentals**

Probability measures how likely an event is.

Its value ranges from 0 to 1:

| Probability | Meaning    |
| ----------- | ---------- |
| 0           | Impossible |
| 0.25        | 25% chance |
| 0.50        | 50% chance |
| 0.75        | 75% chance |
| 1           | Certain    |

For example, if 100 customers contain 30 customers who churned:

\\[ P(Churn)=\frac{30}{100}=0.30 \\]

The observed churn rate is 30%.

### **Probability in machine learning**

Remember this from Lesson 1?

```
probabilities = model.predict_proba(X_test)
```

A model might return:

```
Customer A: [0.85, 0.15]
Customer B: [0.20, 0.80]
Customer C: [0.55, 0.45]
```

For a binary classification model, the two values represent its estimated probabilities for classes 0 and 1.

If class 1 means churn:

- Customer A: 15% estimated churn probability.
- Customer B: 80% estimated churn probability.
- Customer C: 45% estimated churn probability.

With the default 0.5 threshold, Customer B is classified as churn, while A and C are classified as not churn.

Remember: a predicted probability is an estimate, not a guarantee that the customer will churn.

## **8. Normal distribution — the bell-shaped curve**

A normal distribution is a common pattern in statistics.

Many observations cluster around the mean, with fewer observations appearing far from it.

[13: Distribution – Applied Statistics for Quantitative Research: A Practical Guide with Jamovi](https://images.openai.com/static-rsc-4/C-uB7gPwRrwi9xSMgSunz_GIUZSkiClYSVivxgSm6rtcA7cM3WbrTX7PulqnqXWFycQcPojvSUGTkTKlKyKY0oEHW5o12YCjm1L44yEi3vWBhfsG-NUsg1oVxwo2ECvov0Ls7PCLnV4NFUDzxvay4deyxWhxUP0uFy9q5XYMHxU?purpose=inline)

[odp.library.tamu.edu](https://odp.library.tamu.edu/appliedstatswithjamovi/chapter/13-distribution/?utm_source=chatgpt.com)

For an ideal normal distribution:

- Approximately 68% of values fall within 1 standard deviation of the mean.
- Approximately 95% fall within 2 standard deviations.
- Approximately 99.7% fall within 3 standard deviations.

Example:

Suppose a data pipeline's average runtime is 30 minutes and its standard deviation is 5 minutes.

- Within 1 standard deviation: approximately 25–35 minutes.
- Within 2 standard deviations: approximately 20–40 minutes.
- Within 3 standard deviations: approximately 15–45 minutes.

This is useful for understanding unusual observations, but be careful: real-world data is not always normally distributed. Customer spending, transaction amounts and pipeline runtimes can be skewed.

## **9. Correlation and covariance**

These concepts help us understand relationships between two variables.

Imagine we want to understand whether the number of support tickets is related to customer churn.

### **A. Correlation**

Correlation measures the strength and direction of a linear relationship between two variables.

Its value ranges from -1 to +1.

| Correlation | Meaning                              |
| ----------- | ------------------------------------ |
| +1          | Perfect positive linear relationship |
| +0.8        | Strong positive relationship         |
| 0           | No linear relationship               |
| -0.8        | Strong negative relationship         |
| -1          | Perfect negative linear relationship |

Examples:

- More support tickets and higher churn tendency: potentially positive correlation.
- More customer tenure and lower churn tendency: potentially negative correlation.

**Important:** Correlation does not prove causation. Two variables can move together without one causing the other.

### **B. Covariance**

Covariance indicates whether two variables tend to move in the same or opposite directions.

- Positive covariance: they tend to move together.
- Negative covariance: when one increases, the other tends to decrease.
- Near-zero covariance: little linear co-movement.

Unlike correlation, covariance is not restricted to -1 and +1. Its magnitude depends on the units and scale of the variables.

### **Python implementation**

Let's create a small example:

```
data = pd.DataFrame({    "support_tickets": [0, 1, 2, 3, 4, 5],    "churn_score": [5, 10, 15, 25, 35, 45],    "tenure": [60, 55, 48, 35, 25, 12]})print("Correlation:")print(data.corr())print("\nCovariance:")print(data.cov())
```

To visualise the relationships:

```
import matplotlib.pyplot as pltplt.scatter(data["support_tickets"], data["churn_score"])plt.xlabel("Support Tickets")plt.ylabel("Churn Score")plt.title("Support Tickets vs Churn Score")plt.show()
```

A useful Pandas shortcut:

```
print(df.corr(numeric_only=True))
```

This calculates correlations between the numeric columns in your Lesson 1 dataset.

## **10. Outlier detection using Python**

An outlier is a value that is unusually far from the rest of the observations.

Imagine these transaction amounts:

`[500, 600, 650, 700, 750, 800, 850, 900, 950, 50000]`

Most transactions are below ₹1,000, but one transaction is ₹50,000.

That value is a potential outlier.

Outliers can be:

- Genuine unusual business activity.
- Data-entry mistakes.
- Duplicate or incorrectly processed records.
- Fraudulent or suspicious transactions.

Don't automatically delete an outlier. First investigate why it exists.

### **Method 1: IQR (Interquartile Range)**

IQR is the difference between the 75th percentile and the 25th percentile.

\\[ IQR = Q3-Q1 \\]

The common IQR rule defines potential outliers as:

\\[ Lower = Q1-1.5(IQR) \\]

\\[ Upper = Q3+1.5(IQR) \\]

Values outside these boundaries are flagged as potential outliers.

Let's implement it:

```
transactions = pd.Series([    500, 600, 650, 700, 750,    800, 850, 900, 950, 50000])Q1 = transactions.quantile(0.25)Q3 = transactions.quantile(0.75)IQR = Q3 - Q1lower_bound = Q1 - 1.5 * IQRupper_bound = Q3 + 1.5 * IQRoutliers = transactions[    (transactions < lower_bound) |    (transactions > upper_bound)]print("Q1:", Q1)print("Q3:", Q3)print("IQR:", IQR)print("Lower bound:", lower_bound)print("Upper bound:", upper_bound)print("Potential outliers:")print(outliers)
```

You should see ₹50,000 flagged as a potential outlier.

### **Method 2: Z-score**

A Z-score tells us how many standard deviations an observation is away from the mean.

\\[ Z = \frac{x-\mu}{\sigma} \\]

Example:

- Z = 0: exactly at the mean.
- Z = 1: one standard deviation above the mean.
- Z = -2: two standard deviations below the mean.

A common rule flags values with an absolute Z-score greater than 3.

```
from scipy.stats import zscorez_scores = zscore(transactions)outliers_z = transactions[abs(z_scores) > 3]print(outliers_z)
```

**When to use which?**

| IQR                                       | Z-score                                                               |
| ----------------------------------------- | --------------------------------------------------------------------- |
| Doesn't require normally distributed data | Works best as a screening method when data is reasonably well-behaved |
| Uses quartiles                            | Uses mean and standard deviation                                      |
| Often useful for skewed data              | Can be influenced by extreme values                                   |

Neither method automatically tells you whether an observation is wrong or dangerous. It only helps identify values worth investigating.

## **11. Practical implementation — analyse your Lesson 1 dataset**

Now let's apply today's concepts to the customer churn dataset you already created.

Open your existing notebook: `01_customer_churn_prediction.ipynb`.

Create a new section called `Lesson 2 - Statistical Analysis`.

### **Step 1: Descriptive statistics**

```
print(df.describe())
```

This gives you:

- Count
- Mean
- Standard deviation
- Minimum
- 25th percentile
- Median (50th percentile)
- 75th percentile
- Maximum

### **Step 2: Calculate mean, median and mode**

```
for column in df.columns:    print(f"\nColumn: {column}")    print("Mean:", df[column].mean())    print("Median:", df[column].median())    print("Mode:", df[column].mode().iloc[0])
```

### **Step 3: Check the standard deviation**

```
print(df.std(numeric_only=True))
```

Which feature has the highest standard deviation? Remember that the features have different units and scales, so don't compare their importance using standard deviation alone.

### **Step 4: Check correlations**

```
correlation = df.corr(numeric_only=True)print(correlation)
```

Visualise them:

```
import matplotlib.pyplot as pltplt.figure(figsize=(8, 5))plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)plt.colorbar()plt.xticks(    range(len(correlation.columns)),    correlation.columns,    rotation=45)plt.yticks(    range(len(correlation.columns)),    correlation.columns)plt.title("Customer Churn Correlation Matrix")plt.tight_layout()plt.show()
```

Look specifically at the correlation of `churn` with:

- `tenure`
- `monthly_charges`
- `support_tickets`
- `late_payments`

Because churn is encoded as 0 and 1, these correlations can help describe how features are linearly associated with the churn label. They do not prove that any feature causes churn.

### **Step 5: Detect outliers in monthly charges**

```
Q1 = df["monthly_charges"].quantile(0.25)Q3 = df["monthly_charges"].quantile(0.75)IQR = Q3 - Q1lower = Q1 - 1.5 * IQRupper = Q3 + 1.5 * IQRoutliers = df[    (df["monthly_charges"] < lower) |    (df["monthly_charges"] > upper)]print("Potential outlier count:", len(outliers))print(outliers.head())
```

### **Step 6: Visualise monthly charges**

```
plt.boxplot(df["monthly_charges"])plt.title("Monthly Charges - Outlier Detection")plt.ylabel("Monthly Charges")plt.show()
```

[Understanding Data with Five-Number Summary Statistics | by Sarah Disviana | Medium](https://images.openai.com/static-rsc-4/JqCP0Lfoq6c9zZD7g9Psv5jQfVrZBpNQ4x_0iU_BK_r2NWLEBWAnG33VUd6bGjr7XqpOx3082W6-AKSMYoE6-7iStYK1CYymZc3_GpBFgXSykFoGhY9a_gm-FJ-xD2fWlLfAVnGy3nGh3B3bg2eMHtN3iCQjdN9hssKF66IOZdk?purpose=inline)

[medium.com](https://medium.com/%40disvianas/understanding-data-with-five-number-summary-statistics-ea4f5c93d738?utm_source=chatgpt.com)

In a box plot:

- The line inside the box is the median.
- The box covers the middle 50% of observations.
- The whiskers show the non-outlier range under the box plot's rule.
- Individual points beyond the whiskers are potential outliers.

## **12. What you should remember from Lesson 2**

| Concept             | Meaning                                   | Typical application               |
| ------------------- | ----------------------------------------- | --------------------------------- |
| Mean                | Average                                   | Average transaction amount        |
| Median              | Middle value                              | Typical salary or customer spend  |
| Mode                | Most frequent value                       | Most common product category      |
| Percentile          | Relative position                         | 95th percentile pipeline runtime  |
| Variance            | Squared spread                            | Measuring variability             |
| Standard deviation  | Spread in original units                  | Understanding feature variability |
| Probability         | Likelihood of an event                    | Churn prediction                  |
| Normal distribution | Bell-shaped distribution                  | Statistical analysis              |
| Correlation         | Strength/direction of linear relationship | Feature analysis                  |
| Covariance          | Direction of co-movement                  | Statistical relationships         |
| IQR                 | Middle 50% spread                         | Outlier detection                 |
| Z-score             | Distance from mean in standard deviations | Outlier screening                 |

## **13. Knowledge check — mini quiz**

Try answering without looking back at the lesson.

**Lesson 2 quiz**

6/6 answered

**1. A dataset contains salaries ₹20K, ₹25K, ₹30K, ₹35K and ₹5 lakh. Which measure best represents a typical salary?**

Mean

Median

Maximum

Variance

**Correct**

The median is less affected by the unusually high ₹5 lakh salary.

**2. What does a high standard deviation indicate?**

All values are identical

Values are widely spread around the mean

The mean is zero

There are no outliers

**Correct**

A high standard deviation indicates greater spread.

**3. What is the range of [10, 20, 30, 40]?**

10

20

30

40

**Correct**

Range = maximum − minimum = 40 − 10 = 30.

**4. A correlation of -0.9 indicates:**

Strong positive linear relationship

Strong negative linear relationship

No relationship

Proof of causation

**Correct**

-0.9 indicates a strong negative linear relationship, not causation.

**5. Which method uses Q1, Q3 and IQR to flag potential outliers?**

Z-score

IQR method

Mean

Correlation

**Correct**

The IQR method uses the first and third quartiles.

**6. If a model estimates a customer's churn probability as 0.80, what does that mean?**

The customer definitely will churn

The estimated probability of churn is 80%

The model has 80% accuracy

The customer has churned already

**Correct**

It is an estimated probability, not a guarantee or model accuracy.

# **6 / 6**

Try again

## **14. Practical assignment**

Complete these tasks in your notebook:

1. Calculate the mean, median and mode of `monthly_charges`.
2. Calculate the 25th, 50th and 90th percentiles of `monthly_charges`.
3. Identify which feature has the highest standard deviation.
4. Find the correlation between `support_tickets` and `churn`.
5. Find potential outliers in `monthly_charges` using IQR.
6. Explain in your own words why correlation does not prove causation.

**Three interview questions to practise:**

- What is the difference between variance and standard deviation?
- What is the difference between correlation and covariance?
- How would you identify outliers in a dataset using Python?

## **Your progress**

**AI Mastery Program**

In progress

Lesson 1 — AI and ML fundamentals

Lesson 2 — Statistics and probability

Current

Lesson 3 — Linear algebra for ML

Upcoming

Next: vectors, matrices, dimensions, dot products and how ML models use them.

**Your next step:** Complete the practical assignment and quiz. Share your quiz score and the Python outputs (especially the correlation matrix and outlier count). I'll review your understanding, correct mistakes, and then move you to Lesson 3.