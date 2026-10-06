# Final submission for BT2024016

Name: Tanmay Dixit. The improved models have already been trained. Use the new
predictions and report together.

## Required items

1. `BT2024016_report.pdf`
2. `BT2024016_pred_var1.csv`
3. `BT2024016_pred_var2.csv`
4. Your GitHub repository URL

The PDF has three pages. Each prediction CSV has 1,000 rows under header `y`.
Do not add an index, sort rows, round values or combine the files. Holdout
prediction files are evidence, not the required prediction submissions.

## Next steps

1. Extract the final-submission ZIP. The PDF and prediction CSVs are at its
   root. Review your name and roll number in the PDF.
2. Create a GitHub repository, such as `ml-assignment-1-bt2024016`, or update
   your existing one. Upload the contents of the enclosed
   `polynomial_regression_assignment` folder, preserving its structure.
   Include source files, notebook, configuration, requirements, experiments,
   evidence, tests and README. The `outputs` folder also supplies trained JSON
   models for inference. Upload extracted files, not only the ZIP. Ensure
   your instructor can access the repository.
3. Submit the PDF, both CSVs and the repository URL through your course portal.
   Follow its deadline and archive requirements. The supplied assignment PDF
   did not specify a portal or deadline.

Colab does not need to be rerun. For reproduction, open the updated notebook,
run from the top and upload all four original CSVs. Your name and roll are
filled in. Setup preserves Colab's scientific packages and installs ReportLab
only if needed.

New holdout MSE: 0.278873 for var1 and 0.274358 for var2. These scores use the
same labelled-data holdout as before. Hidden-test scores remain unknown.
Read the report so you can explain degree, regularization, cross-validation,
final refitting and the fixed averages used.
