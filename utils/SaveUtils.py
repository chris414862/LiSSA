import pickle
from pathlib import Path
import pandas as pd


def pickle_results(save_dir:Path, results:pd.DataFrame, full_record:dict):
    results_path = save_dir / "results.pickle"
    full_record_path = save_dir / "full_record.pickle"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    results.to_pickle(results_path)
    with open(full_record_path, 'wb') as f:
        pickle.dump(full_record, f)

def store_experiment_results(save_dir:Path, results:pd.DataFrame, full_record:dict):
    is_valid_dirname = False
    while save_dir.exists() and not is_valid_dirname:
        try:
            save_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            answer = input("\nDirectory: "+str(save_dir)+" exists. Do you want to overwrite? [y/n] ").lower().strip()
            if len(answer) > 1:
                answer = answer[0]
            elif len(answer) == 0:
                print("Must enter y or n. Try again")
                continue
            if answer == 'y':
                break
            elif answer == 'n':
                save_dir = Path(input("Enter new directory: ").strip())
            else:
                print("Answer:", answer, "is invalid. Try again")
                continue

        break

    pickle_results(save_dir, results, full_record)
