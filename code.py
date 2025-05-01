import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
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
mse = mean_squared_error(df_model['age'], model.fittedvalues)
rmse = np.sqrt(mse)  # Calculate RMSE manually

# 5. Create results file
with open('results.txt', 'w') as f:
    with redirect_stdout(f):
        print("POSSUM AGE PREDICTION - LINEAR REGRESSION ANALYSIS")
        print("=" * 50)
        
        print("\nMODEL PERFORMANCE METRICS")
        print("-" * 25)
        print(f"R-squared: {round(r2, 3)}")
        print(f"Root Mean Squared Error (years): {round(rmse, 3)}")
        
        print("\nREGRESSION COEFFICIENTS")
        print("-" * 25)
        coef_table = model.summary2().tables[1].reset_index().rename(columns={'index': 'Predictor'})
        print(coef_table[['Predictor', 'Coef.', 'Std.Err.', 'P>|t|']].to_string())
        
        print("\nFULL MODEL SUMMARY")
        print("-" * 25)
        print(model.summary())
        
        print("\nNote: The predicted vs. actual plot has been saved as 'predicted_vs_actual.png'")

# 6. Create predicted vs. actual plot
plt.figure(figsize=(10, 6))
plt.scatter(df_model['age'], model.fittedvalues, alpha=0.5)
plt.xlabel("Actual Age (years)")
plt.ylabel("Predicted Age (years)")
plt.title("Predicted vs. Actual Possum Age")
plt.plot([df_model['age'].min(), df_model['age'].max()],
         [df_model['age'].min(), df_model['age'].max()], 
         linestyle='--', color='red', label='Perfect Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete! Results have been saved to 'results.txt' and the plot has been saved as 'predicted_vs_actual.png'")
