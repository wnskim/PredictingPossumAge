# Possum Age Prediction

This project predicts the age of possums using linear regression with interaction terms, based on morphological and categorical data.

## Project Structure

- **possum.csv**  
  The dataset containing possum measurements and metadata.  
  Columns include:  
  - `case`, `site`, `Pop`, `sex`, `age`, `hdlngth`, `skullw`, `totlngth`, `taill`, `footlgth`, `earconch`, `eye`, `chest`, `belly`

- **interaction_analysis.py**  
  Main analysis script.  
  - Loads the dataset, selects predictors, and drops rows with missing values.
  - Fits a linear regression model with interaction terms between key measurements and categorical variables.
  - Outputs model performance metrics (R², RMSE), regression coefficients, and a full model summary to `interaction_results.txt`.
  - Generates and saves two plots to `interaction_analysis.png`:
    - Predicted vs. Actual Age
    - Regression Coefficients (including interaction terms)

- **interaction_results.txt**  
  Text file with model performance, coefficients, and a detailed summary.

- **interaction_analysis.png**  
  Visualization of model predictions and coefficients.

- **requirements.txt**  
  Lists all Python dependencies required to run the analysis.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure `possum.csv` is in the project directory.
2. Run the analysis script:

   ```bash
   python interaction_analysis.py
   ```

3. Results will be saved to:
   - `interaction_results.txt` (text summary)
   - `interaction_analysis.png` (plots)

## Output Interpretation

- **R² and RMSE**: Indicate model fit and prediction error.
- **Regression Coefficients**: Show the effect of each variable and interaction on age.
- **Plots**: Visualize prediction accuracy and the importance of each predictor.

## Notes

- The model uses interaction terms to capture combined effects of key measurements.
- If you wish to modify predictors or interactions, edit the `formula` variable in `interaction_analysis.py`.
- The dataset must not contain missing values for the selected predictors.
