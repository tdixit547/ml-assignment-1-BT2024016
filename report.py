"""Create the assignment report from measured results."""
from pathlib import Path
from xml.sax.saxutils import escape


def write_report(metadata, out_dir):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    out_dir = Path(out_dir)
    roll = metadata['roll']
    path = out_dir / f'{roll}_report.pdf'
    font_dir = Path(reportlab.__file__).parent / 'fonts'
    for name, filename in [('ReportSans','Vera.ttf'),('ReportSans-Bold','VeraBd.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
    pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold',
                                 italic='ReportSans',boldItalic='ReportSans-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('TitleMain',fontName='ReportSans-Bold',fontSize=22,leading=27,
                              textColor=colors.HexColor('#8B1E2D'),alignment=TA_LEFT,spaceAfter=12))
    styles.add(ParagraphStyle('Section',fontName='ReportSans-Bold',fontSize=11.5,leading=15,
                              textColor=colors.HexColor('#8B1E2D'),spaceBefore=11,spaceAfter=6))
    styles.add(ParagraphStyle('TextMain',fontName='ReportSans',fontSize=9.6,leading=13.3,spaceAfter=7))
    story=[]
    def text(value,style='TextMain'):
        story.append(Paragraph(value,styles[style]))

    text('Polynomial Regression','TitleMain')
    text(f"ML Assignment 1 | {escape(metadata['student_name'])} | {escape(roll)}")
    text('Data and objective','Section')
    text('Separate polynomial regression models predict the Net Power Score (var1) and Thermal Anomaly '
         'Score (var2). Each supplied training file has 1,000 labelled rows. Each test file has 1,000 '
         'rows without target labels. All six inputs are used for var1 and all three for var2. '
         'The four files contain no missing values. Original test-row order is preserved.')
    text('Polynomial models','Section')
    text('Each term has a total degree equal to the sum of its exponents. For example, x1^2 x2 '
         'has total degree 3. The search includes powers and interactions, with degree capped at 10 '
         'for var1 and 20 for var2. Monomial and Legendre bases are compared. Both represent polynomials. '
         'A fixed average of polynomial fits is also a polynomial, with degree no greater than the '
         'largest component degree. No tree, kernel with non-polynomial terms, or neural network is used.')
    text('Model selection','Section')
    text('A seed-42 split gives 800 development rows and 200 holdout rows per '
         'problem. Broad searches on the development rows compare ordinary least squares, Ridge, '
         'Lasso, relaxed Lasso, and automatic relevance determination (ARD) regression. ARD learns '
         'a separate shrinkage level for each coefficient. Expanded columns are standardized using '
         'only the fitting fold. Sparse term selection and any coefficient refit also stay inside that fold.')
    text('The initial expanded Ridge search covers all-feature degrees 1-6 for var1 and 1-12 for var2, '
         'plus reduced-feature controls. Further sparse searches test var1 degrees 4, 5, 6, 8, 10 and '
         'var2 degrees 6, 8, 10, 12, 16, 20, followed by a finer penalty grid near the leading degrees. '
         'Ridge basis and degree-penalty searches extend to degree 8 and 20, respectively. The complete '
         'grids and attempted numerical fits are recorded with the code.')
    text('Nine individual finalists and two fixed equal-weight averages are compared per problem. '
         'Each is evaluated with three five-fold partitions of the same 800 development rows '
         '(seeds 42, 137, 2026). The lowest mean MSE over these 15 folds determines the final choice. '
         'These are model-selection scores, not an unbiased final test estimate. Repeated folds overlap.')
    text('Validation and final fitting','Section')
    text('The chosen settings are fixed before the final holdout evaluation. The 200-row holdout '
         'was examined during preliminary experiments and excluded from subsequent tuning, so it '
         'is a reused validation set. After recording MSE and R-squared, the fixed models are '
         'refitted on all 1,000 labelled rows to create the submission predictions.')
    text('MSE is the mean squared prediction error; lower is better. R-squared is 1 - SSE/SST; '
         'closer to 1 is better. Hidden-test scores cannot be computed because test targets were not supplied.')

    explanations={
        1:('Selected degree: 5. The model averages two degree-5 monomial fits with equal weights: '
           'Lasso (alpha = 0.008) and ARD (maximum 500 iterations, tolerance 0.0001). '
           'Both use all six features and start from 461 polynomial terms, excluding the intercept. '
           'Coefficient shrinkage limits the influence of unsupported powers and interactions.'),
        2:('Selected degree: 12. The model equally averages a degree-12 Legendre Ridge fit '
           '(alpha = 0.01) and a degree-9 monomial relaxed-Lasso fit (alpha = 0.001). '
           'For Ridge, a standardized term of total degree k is divided by k^2 before fitting, '
           'so high-degree standardized coefficients receive a stronger penalty. The relaxed-Lasso '
           'fit averages its Lasso coefficients with a Ridge refit (alpha = 0.01) on selected terms. '
           'The two expansions contain 454 and 219 terms, excluding intercepts.'),
    }
    for result in metadata['results']:
        p=result['problem']
        story.append(PageBreak())
        title='Steam turbine optimization' if p==1 else 'Thermal reservoir mapping'
        text(f'var{p}: {title}','TitleMain')
        text('Selected model','Section')
        text(explanations[p])
        text('Measured results','Section')
        text(f"Repeated development CV MSE: {result['repeated_cv_mse']:.6f}. "
             f"Individual fold MSE standard deviation: {result['cv_fold_sd']:.6f}. "
             'This dispersion is descriptive; it is not a confidence interval.')
        measured=result['holdout']
        text(f"Selected-model holdout MSE: {measured['mse']:.6f}. "
             f"Holdout R-squared: {measured['r2']:.6f}. "
             'The chart compares representative candidates on the same 15 development folds. '
             'The full comparison covers 11 finalists per problem.')
        story.append(Spacer(1,5))
        story.append(Image(str(out_dir/f'var{p}_diagnostics.png'),width=504,height=323))
        text('Selection rationale and final predictions','Section')
        if p==1:
            text('Degree-5 Lasso had lower repeated CV MSE than the tested degree-6 Lasso and dense '
                 'degree-5 Ridge finalists. Averaging the degree-5 Lasso and ARD fits gave the lowest '
                 'mean CV MSE among the finalists. Its advantage over Lasso alone is small and '
                 'does not establish a statistically significant difference. Test inputs are more often '
                 'near the feature boundaries than training inputs, so random-split performance may '
                 'not fully represent hidden-test performance.')
        else:
            text('Degree-9 relaxed Lasso and degree-12 Ridge were competitive individual candidates. '
                 'Their fixed average had lower mean repeated CV MSE than either component and '
                 'was the best of the tested finalists. The resulting polynomial has maximum '
                 'total degree 12. Hidden-test performance remains unknown.')
        text(f"Final training uses all {result['final_fit_rows']:,} labelled rows. "
             f"{escape(result['prediction_file'])} contains {result['test_rows']:,} finite predictions "
             'under the single header y. The saved JSON model supports inference without retraining.')

    def footer(canvas,document):
        canvas.saveState(); canvas.setFont('ReportSans',8); canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(42,24,f'ML Assignment 1 | {roll}')
        canvas.drawRightString(A4[0]-42,24,str(document.page)); canvas.restoreState()
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=38,
                          title=f'Polynomial Regression - {roll}',author=metadata['student_name'])
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return path
