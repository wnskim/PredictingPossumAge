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
    'earconch', 'eye', 'chest', 'belly', 'site', 'Pop'
]
df_model = df[['age'] + predictors + ['sex']].dropna()

# 3. Split data by sex
df_female = df_model[df_model['sex'] == 'f']
df_male = df_model[df_model['sex'] == 'm']

# 4. Create formula without sex term
formula = (
    "age ~ hdlngth + skullw + totlngth + taill + footlgth + earconch + eye "
    "+ chest + belly + C(site) + C(Pop)"
)

# 5. Fit models for each sex
model_female = smf.ols(formula, data=df_female).fit()
model_male = smf.ols(formula, data=df_male).fit()

# 6. Calculate metrics for each model
metrics = {
    'Female': {
        'r2': model_female.rsquared,
        'rmse': np.sqrt(mean_squared_error(df_female['age'], model_female.fittedvalues)),
        'n': len(df_female)
    },
    'Male': {
        'r2': model_male.rsquared,
        'rmse': np.sqrt(mean_squared_error(df_male['age'], model_male.fittedvalues)),
        'n': len(df_male)
    }
}

# 7. Create results file
with open('sex_specific_results.txt', 'w') as f:
    with redirect_stdout(f):
        print("POSSUM AGE PREDICTION - SEX-SPECIFIC LINEAR REGRESSION ANALYSIS")
        print("=" * 60)
        
        for sex in ['Female', 'Male']:
            print(f"\n{sex.upper()} MODEL (n={metrics[sex]['n']})")
            print("-" * 30)
            print(f"R-squared: {round(metrics[sex]['r2'], 3)}")
            print(f"Root Mean Squared Error (years): {round(metrics[sex]['rmse'], 3)}")
            
            print("\nREGRESSION COEFFICIENTS")
            print("-" * 25)
            model = model_female if sex == 'Female' else model_male
            coef_table = model.summary2().tables[1].reset_index().rename(columns={'index': 'Predictor'})
            print(coef_table[['Predictor', 'Coef.', 'Std.Err.', 'P>|t|']].to_string())
            
            print("\nFULL MODEL SUMMARY")
            print("-" * 25)
            print(model.summary())
            print("\n" + "=" * 60)

# 8. Create visualization plots
plt.figure(figsize=(15, 5))

# Predicted vs Actual by Sex subplot
plt.subplot(1, 2, 1)
plt.scatter(df_female['age'], model_female.fittedvalues, alpha=0.5, label='Female', color='red')
plt.scatter(df_male['age'], model_male.fittedvalues, alpha=0.5, label='Male', color='blue')
plt.plot([df_model['age'].min(), df_model['age'].max()],
         [df_model['age'].min(), df_model['age'].max()], 
         linestyle='--', color='black', label='Perfect Prediction')
plt.xlabel("Actual Age (years)")
plt.ylabel("Predicted Age (years)")
plt.title("Predicted vs. Actual Possum Age by Sex")
plt.legend()

# Coefficient comparison subplot
plt.subplot(1, 2, 2)
width = 0.35
x = np.arange(len(model_female.params[1:]))  # Skip intercept
plt.bar(x - width/2, model_female.params[1:], width, label='Female', color='red', alpha=0.6)
plt.bar(x + width/2, model_male.params[1:], width, label='Male', color='blue', alpha=0.6)
plt.xticks(x, model_female.params.index[1:], rotation=45, ha='right')
plt.title("Coefficient Comparison Between Sexes")
plt.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('sex_specific_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete! Results have been saved to 'sex_specific_results.txt' and plots have been saved as 'sex_specific_analysis.png'")