import pandas as pd
import re
import os


def tdx_data_preprocess(file_path, fq):
    with open(file_path, "r", errors="ignore") as file:
        lines = file.readlines()
    match = re.search(r"#(\d+)\.", file_path)
    code = match.group(1)
    lines = lines[2:-1]
    out_path = os.path.join("data", fq, f"{code}.csv")
    with open(out_path, "w") as file:
        file.writelines(lines)


def combine():
    df = pd.read_csv("Book1.csv")
    print(df)
    for column_name in df.columns:
        # print(f"列名: {column_name}")
        code = column_name.split(".")[0]
        col = df[column_name]
        col = col.drop(col[(col == 0)].index)

        col.index = pd.to_datetime(col.index)
        print(col)
        base = pd.read_csv(os.path.join("data", fq, f"{code}.csv"))
        print(base)
        base.columns = ["datetime", "open", "high", "low", "close", "volume", "amount"]
        base.set_index("datetime", inplace=True)
        base.index = pd.to_datetime(base.index)
        # print(base)
        print()
        # print(base)
        # df["adj_factor"] = df["close"].shift(1) / df["pre_close"]
        # df["adj_factor"].iloc[0] = 1
        base.assign(pre_close=col).to_csv(
            os.path.join("data", "combine", f"{code}.csv")
        )


if __name__ == "__main__":
    fq = "nfq"
    # for file in os.listdir(os.path.join("tdx_export", fq)):
    #     tdx_data_preprocess(os.path.join("tdx_export", fq, file), fq)

    combine()
