import numpy as np
import pandas

if __name__ == "__main__":

    df = pandas.read_csv(
        "results/2023-01-12-17-10-24-cgcnn_vn-rv128/test_predictions.csv"
    )

    pred_comp = df.filter(["target", "prediction"])

    print("MAE", np.abs(df["target"] - df["prediction"]).mean())
