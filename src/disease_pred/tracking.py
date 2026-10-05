import json
import os
import subprocess
import yaml
from .config import METRICS_PATH,PARAMS,PROJECT_ROOT,SPEC_PATH,TRAIN_METRICS_PATH

EXPERIMENT='heart-disease'
REGISTERED_NAME= "heart-disease-logreg"

def _git_sha() -> str:
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip()
    except Exception:
        return 'unkown'

def _data_md5() ->str:
    pointer=PROJECT_ROOT/'data'/'raw.dvc'
    if not pointer.exists():
        return 'untracked'
    return yaml.safe_load(pointer.read_text(encoding='utf-8'))['outs'][0]['md5']

def _flatten(d:dict,prefix:str="")->dict:
    """{'model': {'cv_folds': 5}} -> {'model.cv_folds': 5}; lists become JSON text."""
    flat={}
    for k,v in d.items():
        name=f'{prefix}{k}'
        if isinstance(v,dict):
            flat.update(_flatten(v,name + "."))
        else:
            flat[name]=json.dumps(v) if isinstance(v,list) else v
    return flat

def log_run(model,X_example) -> str |None:
    uri=os.getenv('MLFLOW_TRACKING_URI')
    if not uri:
        print("Mlflow tracking url not set --skipping mlflow logging")
        return None
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)

    spec=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    train_m=json.loads(TRAIN_METRICS_PATH.read_text(encoding='utf-8'))
    test_m=json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    sha=_git_sha()

    with mlflow.start_run(run_name=f'logreg-enet-{sha[:7]}') as run:
        mlflow.log_params(_flatten(PARAMS))
        mlflow.log_params({f"best.{k}":v for k,v in spec['best_params'].items()})
        mlflow.log_metrics({f'train.{k}':float(v) for k,v in train_m.items()})
        mlflow.log_metrics({f'test.{k}':float(v) for k,v in test_m.items()})
        mlflow.set_tags({"git_sha":sha,"dvc_data_md5":_data_md5(),'threshold':spec['threshold']})

        mlflow.log_artifact(str(SPEC_PATH))
        mlflow.log_artifact(str(METRICS_PATH.parent / "classification_report.txt"))

        signature=infer_signature(X_example,model.predict_proba(X_example))
        mlflow.sklearn.log_model(sk_model=model,
                                  name="model",
                                  signature=signature,
                                  input_example=X_example.head(3),
                                  registered_model_name=REGISTERED_NAME,
                                  skops_trusted_types=['numpy.dtype'] # dtype descriptors stored by the fitted imputers/encoder
                                  )
    print(f'logged MLFLOW run {run.info.run_id}')
    return run.info.run_id