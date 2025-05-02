import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from contextlib import redirect_stdout

# 1. Load dataset
df = pd.read_csv('possum.csv')

# 2. Select predictors and target, then drop rows with ANY missing values
predictors = [
    'hdlngth', 'skullw', 'totlngth', 'taill', 'footlgth',
    'earconch', 'eye', 'chest', 'belly', 'sex', 'site', 'Pop'
]
df_model = df[['age'] + predictors].dropna()

# 3. Fit linear regression with categorical encodings
formula = (
    "age ~ hdlngth + skullw + totlngth + taill + footlgth + earconch + eye "
    "+ chest + belly + C(sex) + C(site) + C(Pop)"
)
model = smf.ols(formula, data=df_model).fit()

# 4. Metrics
r2 = model.rsquared
rmse = np.sqrt(mean_squared_error(df_model['age'], model.fittedvalues))

# 5. Create results file
with open('basic_linear_results.txt', 'w') as f:
    with redirect_stdout(f):
        print("POSSUM AGE PREDICTION - LINEAR REGRESSION")
        print("=" * 70)
        print(f"\nModel Performance (n={len(df_model)})")
        print("-" * 30)
        print(f"R-squared: {round(r2, 3)}")
        print(f"Root Mean Squared Error (years): {round(rmse, 3)}")
        
        print("\nREGRESSION COEFFICIENTS")
        print("-" * 25)
        coef_table = model.summary2().tables[1].reset_index().rename(columns={'index': 'Predictor'})
        print(coef_table[['Predictor', 'Coef.', 'Std.Err.', 'P>|t|']].to_string())
        
        print("\nFULL MODEL SUMMARY")
        print("-" * 25)
        print(model.summary())

# 6. Create visualization plots
plt.figure(figsize=(15, 5))

# Predicted vs Actual plot
plt.subplot(1, 2, 1)
plt.scatter(df_model['age'], model.fittedvalues, alpha=0.5)
plt.plot([df_model['age'].min(), df_model['age'].max()],
         [df_model['age'].min(), df_model['age'].max()], 
         linestyle='--', color='black', label='Perfect Prediction')
plt.xlabel("Actual Age (years)")
plt.ylabel("Predicted Age (years)")
plt.title("Predicted vs. Actual Possum Age")
plt.legend()

# Coefficient plot
plt.subplot(1, 2, 2)
coefs = model.params[1:]  # Skip intercept
plt.bar(range(len(coefs)), coefs)
plt.xticks(range(len(coefs)), coefs.index, rotation=45, ha='right')
plt.title("Coefficients")
plt.tight_layout()

# Save plot
plt.savefig('basic_linear_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete! Results have been saved to 'basic_linear_results.txt' and plots have been saved as 'basic_linear_analysis.png'")

