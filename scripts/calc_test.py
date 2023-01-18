import numpy as np
import pandas
import sys
import os

if __name__ == "__main__":

    df = pandas.read_csv(
        os.path.join(sys.argv[1], "test_predictions.csv")
    )

    pred_comp = df.filter(["target", "prediction"])

    print("MAE", np.abs(df["target"] - df["prediction"]).mean())
