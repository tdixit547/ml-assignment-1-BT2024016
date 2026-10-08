"""Create the assignment report from measured results."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import pandas as pd
import numpy as np

CV = pd.DataFrame(json.loads('[{"problem":1,"candidate":"lasso5_005","degree":5,"cv_mse":0.3069783882,"fold_mse_sd":0.0286804056,"mean_active_terms":162.8666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.005, \\"relax\\": 0.0}","fold_1_mse":0.2919284709,"fold_2_mse":0.2574978761,"fold_3_mse":0.3171859095,"fold_4_mse":0.3574285246,"fold_5_mse":0.3068051436,"fold_6_mse":0.305175708,"fold_7_mse":0.2838473418,"fold_8_mse":0.3155298125,"fold_9_mse":0.3406678251,"fold_10_mse":0.2981605674,"fold_11_mse":0.2809439382,"fold_12_mse":0.2618000561,"fold_13_mse":0.3236790769,"fold_14_mse":0.3396375832,"fold_15_mse":0.3243879885},{"problem":1,"candidate":"lasso5_008","degree":5,"cv_mse":0.3015181389,"fold_mse_sd":0.0264967651,"mean_active_terms":132.3333333333,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.008, \\"relax\\": 0.0}","fold_1_mse":0.2835496456,"fold_2_mse":0.2639936303,"fold_3_mse":0.3206975151,"fold_4_mse":0.3535447455,"fold_5_mse":0.2938955917,"fold_6_mse":0.2998501599,"fold_7_mse":0.2763363458,"fold_8_mse":0.3067682025,"fold_9_mse":0.3306763196,"fold_10_mse":0.2922639366,"fold_11_mse":0.2784228001,"fold_12_mse":0.2604758077,"fold_13_mse":0.3250179365,"fold_14_mse":0.3207267842,"fold_15_mse":0.316552663},{"problem":1,"candidate":"lasso5_010","degree":5,"cv_mse":0.303916564,"fold_mse_sd":0.0262792254,"mean_active_terms":119.8666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.01, \\"relax\\": 0.0}","fold_1_mse":0.2805556254,"fold_2_mse":0.2723285389,"fold_3_mse":0.3254093119,"fold_4_mse":0.3570146662,"fold_5_mse":0.2904853889,"fold_6_mse":0.3012101434,"fold_7_mse":0.2779121708,"fold_8_mse":0.308821793,"fold_9_mse":0.32797692,"fold_10_mse":0.2969560179,"fold_11_mse":0.2829649696,"fold_12_mse":0.2639262245,"fold_13_mse":0.3286822578,"fold_14_mse":0.320803949,"fold_15_mse":0.3237004825},{"problem":1,"candidate":"relaxed5_015","degree":5,"cv_mse":0.3020244903,"fold_mse_sd":0.0240945511,"mean_active_terms":99.5333333333,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.015, \\"relax\\": 0.5}","fold_1_mse":0.2852130178,"fold_2_mse":0.2716009686,"fold_3_mse":0.3167938874,"fold_4_mse":0.3515578121,"fold_5_mse":0.2967817515,"fold_6_mse":0.3014751199,"fold_7_mse":0.2683020832,"fold_8_mse":0.3101026035,"fold_9_mse":0.3232699583,"fold_10_mse":0.2941709129,"fold_11_mse":0.2826136751,"fold_12_mse":0.2685671447,"fold_13_mse":0.3206043881,"fold_14_mse":0.324199624,"fold_15_mse":0.3151144071},{"problem":1,"candidate":"relaxed5_020","degree":5,"cv_mse":0.3072887509,"fold_mse_sd":0.0251544955,"mean_active_terms":89.6666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.02, \\"relax\\": 0.5}","fold_1_mse":0.2853194418,"fold_2_mse":0.2749473838,"fold_3_mse":0.3339333162,"fold_4_mse":0.3598428891,"fold_5_mse":0.2914380972,"fold_6_mse":0.3072839082,"fold_7_mse":0.288873635,"fold_8_mse":0.3126863139,"fold_9_mse":0.3128692676,"fold_10_mse":0.2988636898,"fold_11_mse":0.285929385,"fold_12_mse":0.272294394,"fold_13_mse":0.3316983567,"fold_14_mse":0.3193263334,"fold_15_mse":0.3340248519},{"problem":1,"candidate":"refit5_030","degree":5,"cv_mse":0.320594226,"fold_mse_sd":0.035688642,"mean_active_terms":75.8666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.03, \\"relax\\": 1.0}","fold_1_mse":0.2934249938,"fold_2_mse":0.2772572476,"fold_3_mse":0.3210364444,"fold_4_mse":0.3718785031,"fold_5_mse":0.3112280157,"fold_6_mse":0.3243809709,"fold_7_mse":0.2601417912,"fold_8_mse":0.3193293014,"fold_9_mse":0.3561084771,"fold_10_mse":0.3120297404,"fold_11_mse":0.2803631558,"fold_12_mse":0.3003767313,"fold_13_mse":0.3403718908,"fold_14_mse":0.3773476286,"fold_15_mse":0.3636384979},{"problem":1,"candidate":"lasso6_010","degree":6,"cv_mse":0.3245725968,"fold_mse_sd":0.0292792487,"mean_active_terms":152.0,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 6, \\"alpha\\": 0.01, \\"relax\\": 0.0}","fold_1_mse":0.296768533,"fold_2_mse":0.309357634,"fold_3_mse":0.342505176,"fold_4_mse":0.3562136518,"fold_5_mse":0.303838618,"fold_6_mse":0.3227841927,"fold_7_mse":0.2987390434,"fold_8_mse":0.3254085213,"fold_9_mse":0.3646394071,"fold_10_mse":0.3123079968,"fold_11_mse":0.310764008,"fold_12_mse":0.2706754024,"fold_13_mse":0.3278334878,"fold_14_mse":0.3446699037,"fold_15_mse":0.3820833767},{"problem":1,"candidate":"ard5","degree":5,"cv_mse":0.3154109855,"fold_mse_sd":0.0299541447,"mean_active_terms":109.6,"components":1,"spec":"{\\"method\\": \\"ard\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5}","fold_1_mse":0.3150344126,"fold_2_mse":0.284177117,"fold_3_mse":0.3244278578,"fold_4_mse":0.3729315223,"fold_5_mse":0.339666811,"fold_6_mse":0.3220688396,"fold_7_mse":0.2873840405,"fold_8_mse":0.2952856402,"fold_9_mse":0.3241410702,"fold_10_mse":0.2918800772,"fold_11_mse":0.2768501962,"fold_12_mse":0.2834028517,"fold_13_mse":0.3394432904,"fold_14_mse":0.366400071,"fold_15_mse":0.3080709846},{"problem":1,"candidate":"ridge5_legendre","degree":5,"cv_mse":0.4575586938,"fold_mse_sd":0.0546523873,"mean_active_terms":461.0,"components":1,"spec":"{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 5, \\"alpha\\": 0.1, \\"penalty\\": 2.0}","fold_1_mse":0.4887354618,"fold_2_mse":0.399244041,"fold_3_mse":0.3984327318,"fold_4_mse":0.5285261207,"fold_5_mse":0.4687362931,"fold_6_mse":0.4398902347,"fold_7_mse":0.4180094949,"fold_8_mse":0.4370696439,"fold_9_mse":0.5335329455,"fold_10_mse":0.5243029983,"fold_11_mse":0.4429901207,"fold_12_mse":0.3676579165,"fold_13_mse":0.4744672211,"fold_14_mse":0.4103628306,"fold_15_mse":0.5314223521},{"problem":1,"candidate":"average_lasso_ard","degree":5,"cv_mse":0.2993067544,"fold_mse_sd":0.0270330634,"mean_active_terms":null,"components":2,"spec":"{\\"components\\": [{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.008, \\"relax\\": 0.0}, {\\"method\\": \\"ard\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5}], \\"weights\\": [1.0, 1.0]}","fold_1_mse":0.2930911912,"fold_2_mse":0.2580649326,"fold_3_mse":0.315521459,"fold_4_mse":0.35299085,"fold_5_mse":0.3076912325,"fold_6_mse":0.3042943883,"fold_7_mse":0.2753453578,"fold_8_mse":0.2922179353,"fold_9_mse":0.3175714957,"fold_10_mse":0.2847587027,"fold_11_mse":0.2719394101,"fold_12_mse":0.2597487124,"fold_13_mse":0.3200533439,"fold_14_mse":0.3358612108,"fold_15_mse":0.3004510934},{"problem":1,"candidate":"average_lasso_relaxed","degree":5,"cv_mse":0.3007870549,"fold_mse_sd":0.0252080741,"mean_active_terms":null,"components":2,"spec":"{\\"components\\": [{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.008, \\"relax\\": 0.0}, {\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 5, \\"alpha\\": 0.015, \\"relax\\": 0.5}], \\"weights\\": [1.0, 1.0]}","fold_1_mse":0.2834795946,"fold_2_mse":0.2665209025,"fold_3_mse":0.3183113632,"fold_4_mse":0.3515008302,"fold_5_mse":0.2943383529,"fold_6_mse":0.2999791173,"fold_7_mse":0.2708997958,"fold_8_mse":0.3076598227,"fold_9_mse":0.3252100571,"fold_10_mse":0.292119904,"fold_11_mse":0.2799455723,"fold_12_mse":0.2636049497,"fold_13_mse":0.3219107945,"fold_14_mse":0.3215041775,"fold_15_mse":0.3148205886},{"problem":2,"candidate":"ridge12_legendre","degree":12,"cv_mse":0.3087279403,"fold_mse_sd":0.0543460151,"mean_active_terms":454.0,"components":1,"spec":"{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 12, \\"alpha\\": 0.01, \\"penalty\\": 2.0}","fold_1_mse":0.2655580733,"fold_2_mse":0.33935423,"fold_3_mse":0.2445208154,"fold_4_mse":0.3903298688,"fold_5_mse":0.2781415721,"fold_6_mse":0.252134514,"fold_7_mse":0.3223589048,"fold_8_mse":0.3162046013,"fold_9_mse":0.3265986868,"fold_10_mse":0.264465669,"fold_11_mse":0.3159647496,"fold_12_mse":0.2463245672,"fold_13_mse":0.274070893,"fold_14_mse":0.3870101675,"fold_15_mse":0.4078817914},{"problem":2,"candidate":"ridge14_legendre","degree":14,"cv_mse":0.3105639407,"fold_mse_sd":0.0533516779,"mean_active_terms":679.0,"components":1,"spec":"{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 14, \\"alpha\\": 0.01, \\"penalty\\": 2.0}","fold_1_mse":0.2650433037,"fold_2_mse":0.3413377252,"fold_3_mse":0.2484423993,"fold_4_mse":0.3966312281,"fold_5_mse":0.2836406264,"fold_6_mse":0.2509700354,"fold_7_mse":0.3270917629,"fold_8_mse":0.3297975715,"fold_9_mse":0.3230962956,"fold_10_mse":0.2582145371,"fold_11_mse":0.3250215724,"fold_12_mse":0.2501778306,"fold_13_mse":0.2784473569,"fold_14_mse":0.3990111967,"fold_15_mse":0.3815356686},{"problem":2,"candidate":"ridge11_legendre","degree":11,"cv_mse":0.3068407697,"fold_mse_sd":0.0588617909,"mean_active_terms":363.0,"components":1,"spec":"{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 11, \\"alpha\\": 0.01, \\"penalty\\": 2.0}","fold_1_mse":0.2749794488,"fold_2_mse":0.3345561038,"fold_3_mse":0.2508521496,"fold_4_mse":0.4284439643,"fold_5_mse":0.2649007954,"fold_6_mse":0.2612186508,"fold_7_mse":0.297082583,"fold_8_mse":0.3063751652,"fold_9_mse":0.311441311,"fold_10_mse":0.267947703,"fold_11_mse":0.3099453119,"fold_12_mse":0.2375388311,"fold_13_mse":0.2620019935,"fold_14_mse":0.3953790584,"fold_15_mse":0.3999484758},{"problem":2,"candidate":"ridge8_legendre","degree":8,"cv_mse":0.3121183958,"fold_mse_sd":0.062212387,"mean_active_terms":164.0,"components":1,"spec":"{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 8, \\"alpha\\": 0.31622776601683794, \\"penalty\\": 1.0}","fold_1_mse":0.3513603982,"fold_2_mse":0.317113254,"fold_3_mse":0.2454694092,"fold_4_mse":0.3853351565,"fold_5_mse":0.287026264,"fold_6_mse":0.2552466784,"fold_7_mse":0.3092866,"fold_8_mse":0.328101977,"fold_9_mse":0.3126000063,"fold_10_mse":0.2579720849,"fold_11_mse":0.3055230198,"fold_12_mse":0.224705241,"fold_13_mse":0.2691682203,"fold_14_mse":0.3691822241,"fold_15_mse":0.463685403},{"problem":2,"candidate":"relaxed9_001","degree":9,"cv_mse":0.2910132878,"fold_mse_sd":0.0458319864,"mean_active_terms":118.4,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 9, \\"alpha\\": 0.001, \\"relax\\": 0.5}","fold_1_mse":0.3744329409,"fold_2_mse":0.2955711402,"fold_3_mse":0.2452280455,"fold_4_mse":0.3322672531,"fold_5_mse":0.2959900589,"fold_6_mse":0.2512928109,"fold_7_mse":0.2503124335,"fold_8_mse":0.2710319565,"fold_9_mse":0.365715683,"fold_10_mse":0.2701157558,"fold_11_mse":0.2952221516,"fold_12_mse":0.2128935475,"fold_13_mse":0.2696512237,"fold_14_mse":0.2959464018,"fold_15_mse":0.3395279135},{"problem":2,"candidate":"refit12_003","degree":12,"cv_mse":0.3172990198,"fold_mse_sd":0.0650980614,"mean_active_terms":113.6666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 12, \\"alpha\\": 0.003, \\"relax\\": 1.0}","fold_1_mse":0.3153051395,"fold_2_mse":0.3002280145,"fold_3_mse":0.2447129308,"fold_4_mse":0.4541948664,"fold_5_mse":0.2772486359,"fold_6_mse":0.2487434837,"fold_7_mse":0.3552387598,"fold_8_mse":0.3117604907,"fold_9_mse":0.3801691263,"fold_10_mse":0.2582420054,"fold_11_mse":0.2951551959,"fold_12_mse":0.2465581067,"fold_13_mse":0.2788168174,"fold_14_mse":0.4081925218,"fold_15_mse":0.3849192018},{"problem":2,"candidate":"refit8_legendre","degree":8,"cv_mse":0.3224347181,"fold_mse_sd":0.0424281327,"mean_active_terms":125.4666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"legendre\\", \\"degree\\": 8, \\"alpha\\": 0.02, \\"relax\\": 1.0}","fold_1_mse":0.3011808033,"fold_2_mse":0.3583822879,"fold_3_mse":0.2613117629,"fold_4_mse":0.3510087804,"fold_5_mse":0.3255576065,"fold_6_mse":0.2634636141,"fold_7_mse":0.3686150274,"fold_8_mse":0.3805521929,"fold_9_mse":0.3521480337,"fold_10_mse":0.2846794336,"fold_11_mse":0.3136535667,"fold_12_mse":0.2728395933,"fold_13_mse":0.3178358611,"fold_14_mse":0.2966856968,"fold_15_mse":0.3886065115},{"problem":2,"candidate":"refit9_legendre","degree":9,"cv_mse":0.3161524675,"fold_mse_sd":0.042960612,"mean_active_terms":151.4666666667,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"legendre\\", \\"degree\\": 9, \\"alpha\\": 0.015, \\"relax\\": 1.0}","fold_1_mse":0.3061141571,"fold_2_mse":0.344716019,"fold_3_mse":0.2892561345,"fold_4_mse":0.3235761882,"fold_5_mse":0.3320420934,"fold_6_mse":0.2574625931,"fold_7_mse":0.3889499084,"fold_8_mse":0.3295930184,"fold_9_mse":0.3615008095,"fold_10_mse":0.2781345735,"fold_11_mse":0.3127287475,"fold_12_mse":0.2640181617,"fold_13_mse":0.2760451461,"fold_14_mse":0.2839596921,"fold_15_mse":0.39418977},{"problem":2,"candidate":"lasso9_0005","degree":9,"cv_mse":0.2943632112,"fold_mse_sd":0.0536243775,"mean_active_terms":133.5333333333,"components":1,"spec":"{\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 9, \\"alpha\\": 0.0005, \\"relax\\": 0.0}","fold_1_mse":0.3989550782,"fold_2_mse":0.2855857653,"fold_3_mse":0.239098202,"fold_4_mse":0.3578034928,"fold_5_mse":0.2858673363,"fold_6_mse":0.2438034147,"fold_7_mse":0.2556089525,"fold_8_mse":0.2707859041,"fold_9_mse":0.3654716112,"fold_10_mse":0.2717460252,"fold_11_mse":0.2915494559,"fold_12_mse":0.2117031195,"fold_13_mse":0.2678672884,"fold_14_mse":0.3113990051,"fold_15_mse":0.3582035167},{"problem":2,"candidate":"average_ridge_lasso","degree":12,"cv_mse":0.2781193008,"fold_mse_sd":0.0405011055,"mean_active_terms":null,"components":2,"spec":"{\\"components\\": [{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 12, \\"alpha\\": 0.01, \\"penalty\\": 2.0}, {\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 9, \\"alpha\\": 0.001, \\"relax\\": 0.5}], \\"weights\\": [1.0, 1.0]}","fold_1_mse":0.2934189474,"fold_2_mse":0.2982195845,"fold_3_mse":0.231522369,"fold_4_mse":0.3473083542,"fold_5_mse":0.2617697237,"fold_6_mse":0.2392375873,"fold_7_mse":0.2612596838,"fold_8_mse":0.2663058892,"fold_9_mse":0.3171001875,"fold_10_mse":0.2563166619,"fold_11_mse":0.2911974374,"fold_12_mse":0.2115222621,"fold_13_mse":0.2412373644,"fold_14_mse":0.3156420394,"fold_15_mse":0.3397314202},{"problem":2,"candidate":"average_three","degree":12,"cv_mse":0.2783585195,"fold_mse_sd":0.0358181442,"mean_active_terms":null,"components":3,"spec":"{\\"components\\": [{\\"method\\": \\"ridge\\", \\"basis\\": \\"legendre\\", \\"degree\\": 12, \\"alpha\\": 0.01, \\"penalty\\": 2.0}, {\\"method\\": \\"lasso\\", \\"basis\\": \\"monomial\\", \\"degree\\": 9, \\"alpha\\": 0.001, \\"relax\\": 0.5}, {\\"method\\": \\"lasso\\", \\"basis\\": \\"legendre\\", \\"degree\\": 8, \\"alpha\\": 0.02, \\"relax\\": 1.0}], \\"weights\\": [1.0, 1.0, 1.0]}","fold_1_mse":0.2733903009,"fold_2_mse":0.3012883355,"fold_3_mse":0.2317873222,"fold_4_mse":0.3327950832,"fold_5_mse":0.2743575099,"fold_6_mse":0.2371577511,"fold_7_mse":0.2850019556,"fold_8_mse":0.283673257,"fold_9_mse":0.3045756188,"fold_10_mse":0.2549736474,"fold_11_mse":0.2911747773,"fold_12_mse":0.2153235993,"fold_13_mse":0.2521057998,"fold_14_mse":0.2949436597,"fold_15_mse":0.342829175}]'))
REPO = 'https://github.com/tdixit547/ml-assignment-1-bt2024016'


def make_diagnostics(p, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.read_csv(out_dir/f'var{p}_holdout_predictions.csv')
    y, pred = df.actual_y.to_numpy(), df.predicted_y.to_numpy()
    residual = y-pred
    fig, axes = plt.subplots(1,3,figsize=(10,2.1))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].scatter(y,pred,s=8,alpha=.6,color='#a3343e')
    lo,hi=min(y.min(),pred.min()),max(y.max(),pred.max())
    axes[0].plot([lo,hi],[lo,hi],'--',color='#666666',lw=1)
    axes[0].set(title='Holdout predictions',xlabel='Actual y',ylabel='Predicted y')
    axes[1].scatter(pred,residual,s=8,alpha=.6,color='#a3343e')
    axes[1].axhline(0,color='#666666',ls='--',lw=1)
    axes[1].set(title='Holdout residuals',xlabel='Predicted y',ylabel='Actual - predicted')
    axes[2].hist(residual,bins=20,color='#a3343e',edgecolor='white',lw=.4)
    axes[2].set(title='Residual distribution',xlabel='Residual',ylabel='Rows')
    fig.tight_layout(pad=.7)
    path=out_dir/f'var{p}_table_diagnostics.png'
    fig.savefig(path,dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return path


def write_report(metadata, out_dir):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
    styles.add(ParagraphStyle('Cell',fontName='ReportSans',fontSize=7.4,leading=9.2))
    styles.add(ParagraphStyle('HeadCell',fontName='ReportSans-Bold',fontSize=7.4,leading=9.2,textColor=colors.white))
    styles.add(ParagraphStyle('Note',fontName='ReportSans',fontSize=7.8,leading=10,spaceAfter=5))
    story=[]
    def text(value,style='TextMain'):
        story.append(Paragraph(value,styles[style]))

    def comparison(p, chosen):
        sub=CV[CV.problem==p]
        rows=[['Model','Degree','Basis','Parameters','CV MSE','Fold SD']]
        selected=None
        for _, row in sub.iterrows():
            spec=json.loads(row.spec)
            if 'components' in spec:
                model='Equal average'
                bases=list(dict.fromkeys(c['basis'] for c in spec['components']))
                basis=' / '.join(bases)
                parts=[]
                for c in spec['components']:
                    name={'ridge':'Ridge','ard':'ARD','lasso':'Lasso'}[c['method']]
                    if c.get('relax')==.5: name='relaxed Lasso'
                    if c.get('relax')==1: name='selected-term refit'
                    label=f"{name} d{c['degree']}"
                    if 'alpha' in c: label+=f" alpha={c['alpha']:g}"
                    parts.append(label)
                params=' + '.join(parts)+'; equal weights, component settings above'
            else:
                basis=spec['basis']
                model={'ridge':'Ridge','ard':'ARD','lasso':'Lasso'}[spec['method']]
                if spec.get('relax')==.5: model='Relaxed Lasso'
                if spec.get('relax')==1: model='Selected-term refit'
                if spec['method']=='ard': params='max_iter=500; tol=0.0001'
                else:
                    params=f"alpha={spec['alpha']:g}"
                    if spec.get('penalty'): params+=f"; degree penalty={spec['penalty']:g}"
                    if spec.get('relax'): params+=f"; relax={spec['relax']:g}; refit alpha=0.01"
            if row.candidate==chosen:
                model+=' (selected)'
                selected=len(rows)
            rows.append([model,str(int(row.degree)),basis,params,f'{row.cv_mse:.6f}',f'{row.fold_mse_sd:.6f}'])
            folds=row[[f'fold_{i}_mse' for i in range(1,16)]].astype(float).to_numpy()
            assert np.isclose(folds.mean(),row.cv_mse)
        data=[[Paragraph(escape(c),styles['HeadCell' if i==0 else 'Cell']) for c in r] for i,r in enumerate(rows)]
        width=A4[0]-84
        t=Table(data,colWidths=[78,42,55,width-78-42-55-65-65,65,65],repeatRows=1)
        commands=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#8B1E2D')),
                  ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#cccccc')),
                  ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                  ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
                  ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
        commands.append(('BACKGROUND',(0,selected),(-1,selected),colors.HexColor('#fbe9eb')))
        t.setStyle(TableStyle(commands))
        story.append(t)
        story.append(Spacer(1,5))
        text('All 11 finalists use the same 15 development folds. Fold SD is descriptive dispersion. '
             'Degree is maximum total degree. The degree-penalty value p scales terms of degree k by k^-p. '
             'Relax is the weight on the selected-term Ridge refit; all expansions use fold-fitted standardisation.','Note')

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
    text('The all-feature expanded Ridge search covers degrees 1-6 for var1 and 1-12 for var2. '
         'Further sparse searches test var1 degrees 4, 5, 6, 8, 10 and '
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
    text('GitHub repository','Section')
    text(f'<link href="{REPO}" color="#8B1E2D">{REPO}</link>')

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
             'The following table compares all 11 finalists on the same 15 development folds.')
        comparison(p,result['selected_name'])
        story.append(Image(str(make_diagnostics(p,out_dir)),width=A4[0]-84,height=108))
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
