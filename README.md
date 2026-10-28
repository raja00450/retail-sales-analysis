# Retail Sales Analysis

**Data source:** Sample retail dataset (generated)

## Data Cleaning
- Rows: 5025 -> 5000 after cleaning
- Duplicates removed: 25
- Missing values filled (median for numbers, mode for text): 100
- Outliers capped using IQR rule: {'Sales': 153, 'Profit': 470}

## Key Numbers
- Total Sales: 3,212,720
- Total Profit: 173,895 (margin 5.4%)

## Insights
1. **Technology** is the most profitable category.
2. **Furnishings** has the lowest profit among sub-categories.
3. **Central** is the top region by sales.
4. Sales peak in month **12** (seasonal demand).
5. Discount and Profit correlation is **-0.56**: higher discounts reduce profit.
6. Most important profit driver in the model: **Discount**.

## Prediction Model (predicting Profit)
| Model | R2 | MAE |
|---|---|---|
| Linear Regression | 0.639 | 40.22 |
| Random Forest | 0.854 | 19.00 |

Best model: **Random Forest**.

## Conclusion
- Limit heavy discounts (above 30%), because they push margins negative.
- Focus marketing and stock on the most profitable category and peak months.
- Review or reprice the weakest sub-category.

## Charts
![](outputs/1_monthly_sales_trend.png)
![](outputs/2_category_sales_profit.png)
![](outputs/3_subcategory_profit.png)
![](outputs/4_region_sales.png)
![](outputs/5_discount_vs_margin.png)
![](outputs/6_correlation_heatmap.png)
![](outputs/7_feature_importance.png)
![](outputs/8_actual_vs_predicted.png)

## How to run
```
pip install pandas matplotlib seaborn scikit-learn
python retail_analysis.py
```
