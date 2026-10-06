"""Train, evaluate and use the selected polynomial regression models."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from polynomial_model import fit_model,predict_model,save_model,load_model
from report import write_report

SEED=42
HOLDOUT_FRACTION=.2
FEATURES={1:[f"x{i}" for i in range(1,7)],2:["x1","x2","x3"]}
PROJECT_DIR=Path(__file__).resolve().parent

def read_dataset(path, expected_features, training=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    # Check before pandas can silently rename duplicate column headers.
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name}: duplicate column names are not supported.")
    frame = pd.read_csv(path)
    required = list(expected_features) + (["y"] if training else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{path.name}: the dataset is empty.")
    for column in required:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{path.name}: {column} must be numeric.")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: {column} contains missing or infinite values.")
    # Test y, if present as a placeholder, and unrelated ID columns are ignored.
    # Test rows are never sorted or dropped.
    return frame


def detect_roll(data_dir, roll=None):
    data_dir = Path(data_dir)
    if roll is None:
        rolls = sorted({p.name[:-len("_train_var1.csv")]
                        for p in data_dir.glob("*_train_var1.csv")})
        if len(rolls) != 1:
            raise ValueError("Place one student's four CSVs in data/ or provide --roll YOUR_ROLL.")
        roll = rolls[0]
    if not re.fullmatch(r"[A-Za-z0-9-]+", roll):
        raise ValueError("The roll number must contain only letters, digits, or hyphens.")
    paths = {problem: {
        split: data_dir / f"{roll}_{split}_var{problem}.csv"
        for split in ("train", "test")
    } for problem in (1, 2)}
    missing = [str(p) for files in paths.values() for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Upload the four personalised datasets. Missing:\n" + "\n".join(missing))
    return roll, paths


def scores(y_true, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError("Model predictions contain non-finite values.")
    # R2 has no meaningful denominator for a constant target.
    r2 = None if np.ptp(np.asarray(y_true, dtype=float)) == 0 else float(r2_score(y_true, prediction))
    return {"mse": float(mean_squared_error(y_true, prediction)), "r2": r2}



def save_predictions(prediction,path,expected_rows):
    prediction=np.asarray(prediction,dtype=float)
    if prediction.shape!=(expected_rows,) or not np.isfinite(prediction).all():
        raise ValueError("Expected one finite prediction per test row.")
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"y":prediction}).to_csv(path,index=False)
    check=pd.read_csv(path)
    if check.columns.tolist()!=["y"] or len(check)!=expected_rows:
        raise RuntimeError("Incorrect prediction CSV format.")
    return path


def diagnostic_plot(result,y,prediction,out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=result["problem"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(9.35,6.),layout="constrained")
    comparisons=result.get("cv_comparison", [{"label":"Selected", "mse":result["repeated_cv_mse"]}])
    labels=[c["label"] for c in comparisons]
    values=[c["mse"] for c in comparisons]
    axes[0,0].barh(labels,values,color=["#276577" if "Average" in label else "#aaaeb3" for label in labels])
    axes[0,0].invert_yaxis()
    axes[0,0].set(title="Candidate comparison: 15 CV folds",xlabel="Mean development CV MSE")
    axes[0,0].tick_params(axis="y",labelsize=8)
    for i,v in enumerate(values):
        axes[0,0].text(v,i,f" {v:.4f}",ha="left",va="center",fontsize=8)
    axes[0,0].set_xlim(0,max(values)*1.3)
    axes[0,1].scatter(y,prediction,s=14,alpha=.7,color="#276577")
    lo=min(y.min(),prediction.min()); hi=max(y.max(),prediction.max())
    axes[0,1].plot([lo,hi],[lo,hi],"--",lw=1,color="#963144")
    axes[0,1].set(title="Selected holdout predictions",xlabel="Actual y",ylabel="Predicted y")
    residual=y-prediction
    axes[1,0].scatter(prediction,residual,s=14,alpha=.7,color="#276577")
    axes[1,0].axhline(0,color="#963144",lw=1,ls="--")
    axes[1,0].set(title="Selected holdout residuals",xlabel="Predicted y",ylabel="Actual - predicted")
    axes[1,1].hist(residual,bins=18,color="#276577",edgecolor="white")
    axes[1,1].set(title="Residual distribution",xlabel="Actual - predicted",ylabel="Rows")
    fig.savefig(Path(out_dir)/f"var{p}_diagnostics.png",dpi=180,facecolor="white")
    plt.close(fig)


def run_assignment(data_dir="data",roll="BT2024016",out_dir="outputs",student_name="Tanmay Dixit",
                   config_path=None):
    # Fail before training if the PDF dependency is unavailable.
    import reportlab
    config_path=Path(config_path) if config_path else PROJECT_DIR/"selected_config.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    roll,paths=detect_roll(data_dir,roll)
    datasets={p:{s:read_dataset(path,FEATURES[p],training=(s=="train"))
                 for s,path in files.items()} for p,files in paths.items()}
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for files in paths.values() for path in files.values()}
    for p in (1,2):
        filename=paths[p]["train"].name
        if hashes[filename]!=config["input_sha256"].get(filename):
            raise ValueError("The training file differs from the data used for model selection. "
                             "Rerun the search for those data before using its validation scores.")
    out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty. Choose a new output directory.")
    out_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    with threadpool_limits(limits=1):
        for p,data in datasets.items():
            train,test=data["train"],data["test"]
            dev,holdout=train_test_split(train,test_size=HOLDOUT_FRACTION,random_state=SEED)
            f=FEATURES[p]; selected=config["selection"][str(p)]
            print(f"Fitting var{p}: {selected['name']}...",flush=True)
            evaluator=fit_model(dev[f],dev.y,f,selected["spec"])
            hp=predict_model(evaluator,holdout[f])
            previous_spec={"method":"ols","basis":"monomial","degree":4 if p==1 else 5,
                           "input_standardize":True}
            previous=fit_model(dev[f],dev.y,f,previous_spec)
            bp=predict_model(previous,holdout[f])
            current_scores=scores(holdout.y,hp); previous_scores=scores(holdout.y,bp)
            # The settings remain fixed. All labelled rows may now be used.
            final_model=fit_model(train[f],train.y,f,selected["spec"])
            predictions=predict_model(final_model,test[f])
            pred_path=save_predictions(predictions,out_dir/f"{roll}_pred_var{p}.csv",len(test))
            model_path=out_dir/f"var{p}_model.json"
            save_model(final_model,model_path)
            np.testing.assert_allclose(predict_model(load_model(model_path),test[f]),predictions,rtol=1e-12,atol=1e-12)
            pd.DataFrame({"row_number_1_based":holdout.index.to_numpy()+1,"actual_y":holdout.y.to_numpy(),
                          "previous_y":bp,"predicted_y":hp,"residual":holdout.y.to_numpy()-hp}).to_csv(
                          out_dir/f"var{p}_holdout_predictions.csv",index=False)
            r={"problem":p,"features":f,"degree":final_model["degree"],"selected_name":selected["name"],
               "spec":selected["spec"],"repeated_cv_mse":selected["repeated_cv_mse"],
               "cv_fold_sd":selected["fold_mse_sd"],"holdout":current_scores,"previous_holdout":previous_scores,
               "train_rows":len(train),"development_rows":len(dev),"holdout_rows":len(holdout),
               "test_rows":len(test),"final_fit_rows":len(train),"prediction_file":pred_path.name,
               "model_file":model_path.name,
               "active_terms_per_component":[m["active_terms"] for m in final_model.get("components",[final_model])]}
            results.append(r)
            r["cv_comparison"]=config.get("comparison_summary",{}).get(str(p),[])
            if not r["cv_comparison"]:
                r.pop("cv_comparison")
            diagnostic_plot(r,holdout.y.to_numpy(),hp,out_dir)
            print(f"var{p}: holdout MSE {current_scores['mse']:.9f}, R2 {current_scores['r2']:.9f}",flush=True)
    metadata={"roll":roll,"student_name":student_name,"seed":SEED,"holdout_fraction":HOLDOUT_FRACTION,
              "selection_rule":config["selection_rule"],"cv_seeds":[42,137,2026],"cv_folds":5,
              "holdout_note":"Original 200-row holdout reused for comparison; excluded from all new tuning.",
              "input_sha256":hashes,"results":results,
              "versions":{p:importlib.metadata.version(p) for p in
                          ["numpy","pandas","scipy","scikit-learn","matplotlib","reportlab","threadpoolctl"]}}
    (out_dir/"metrics.json").write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    # Metrics and predictions survive any report-rendering failure.
    report=write_report(metadata,out_dir)
    print(f"Created both predictions, portable models, diagnostics and {report.name}.",flush=True)
    return metadata


def predict_saved(model_path,test_path,output_path):
    model=load_model(model_path)
    frame=read_dataset(test_path,model["features"])
    return save_predictions(predict_model(model,frame[model["features"]]),output_path,len(frame))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    train=commands.add_parser("train",help="Reproduce the selected model fits and final outputs")
    train.add_argument("--data-dir",default="data")
    train.add_argument("--roll",default="BT2024016")
    train.add_argument("--name",default="Tanmay Dixit")
    train.add_argument("--out",default="outputs")
    train.add_argument("--config",default=None)
    infer=commands.add_parser("predict",help="Infer with a saved JSON polynomial")
    infer.add_argument("--model",required=True); infer.add_argument("--test",required=True)
    infer.add_argument("--output",required=True)
    report=commands.add_parser("report",help="Regenerate only the report from saved metrics and plots")
    report.add_argument("--out",default="outputs")
    args=parser.parse_args()
    if args.action=="train":
        run_assignment(args.data_dir,args.roll,args.out,args.name,args.config)
    elif args.action=="predict":
        path=predict_saved(args.model,args.test,args.output); print(f"Saved {path}")
    else:
        out=Path(args.out); metadata=json.loads((out/"metrics.json").read_text())
        print(write_report(metadata,out))


if __name__=="__main__":
    main()
