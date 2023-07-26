# third party
import numpy as np
import pandas as pd
import psutil
from scipy.stats import chi2_contingency, f_oneway

from tmp_lucas.loader import DataLoader

if __name__ == "__main__":

    train_model = True
    explain = True

    print("Loading Images")

    print(
        f"GB available before loading data: {psutil.virtual_memory().available/1073741824:.2f}"
    )

    # Use a subprocess to free memory

    DL = DataLoader(
        path_=r"../../data",
        data_src_="PAD-UFES",
        format_="PIL",
    )

    df = DL.load_dataset(raw=True)

    df = df[df.columns.difference(["image"])]

    print(df["label"].value_counts())

    table_order = [1, 4, 2, 0, 3, 5]
    labels_name = ["BCC", "SCC", "MEL", "ACK", "NEV", "SEK"]
    # First Check normality assumption
    continuous_variable = ["age", "diameter_1", "diameter_2"]
    label = "label"
    for variable in continuous_variable:
        # Group the data by 'label'
        df_na = df.dropna(subset=[variable])

        grouped_data = df_na.groupby(label)[variable]
        means = grouped_data.mean().to_numpy()
        Q1 = grouped_data.quantile(0.25)
        Q3 = grouped_data.quantile(0.75)
        IQR = Q3 - Q1
        IQR = IQR.to_numpy()

        # Extract individual groups as separate arrays and apply f_oneway
        F_statistic, p_value = f_oneway(*[group.values for name, group in grouped_data])

        table_output = f"{variable}, "
        for i in range(len(table_order)):
            table_output += f"{labels_name[i]}: {means[table_order[i]]:.2f} ({IQR[table_order[i]]:.2f}), "
        table_output += f"p-value: {p_value}"
        print(table_output)

    categorical_variable = ["elevation", "bleed", "hurt", "itch", "grew", "region"]
    binary = ["elevation", "bleed", "hurt", "itch", "grew"]
    categorical = ["region"]
    for variable in categorical_variable:
        df_na = df.dropna(subset=[variable])
        # Create a 2x5 contingency table (cross-tabulation) of the data
        contingency_table = pd.crosstab(df_na[variable], df_na[label])
        if "UNK" in contingency_table.index:
            contingency_table.drop("UNK", axis=0, inplace=True)

        # Perform Fisher's exact test on the contingency table
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        if variable in binary:
            percentage = (
                contingency_table.loc["True"]
                / (contingency_table.loc["False"] + contingency_table.loc["True"])
                * 100
            ).to_numpy()
            true_value = contingency_table.loc["True"].to_numpy()

            table_output = f"{variable}, "
            for i in range(len(table_order)):
                table_output += f"{labels_name[i]}: {true_value[table_order[i]]:.2f} ({percentage[table_order[i]]:.2f}), "
            table_output += f"p-value: {p_value}"
            print(table_output)

        elif variable in categorical:
            percentage = []
            number = []
            most_present = []
            for col in contingency_table.columns:
                diagnoses = contingency_table[col]
                most_present.append(
                    contingency_table.index[contingency_table[col].argmax()]
                )
                number.append(contingency_table[col].max())
                percentage.append(
                    contingency_table[col].max() / contingency_table[col].sum() * 100
                )
            number = np.array(number)
            percentage = np.array(percentage)
            most_present = np.array(most_present)

            table_output = ""
            for i in range(len(table_order)):
                table_output += f"{variable} {labels_name[i]}: {most_present[table_order[i]]} - {number[table_order[i]]:.0f} ({percentage[table_order[i]]:.2f}), "
            table_output += f"p-value: {p_value}"
            print(table_output)
