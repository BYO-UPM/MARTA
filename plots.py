from matplotlib import pyplot as plt
import numpy as np

# Data from the table
models = ["MARTA", "MARTA-DOS"]
databases = ["NeuroVoz RS", "NeuroVoz Texts", "GITA RS", "GITA Texts"]

# AUC values for 3D and 64D
auc_3d_values = {
    "MARTA": [65, 62, 79, 75],
    "MARTA-DOS": [74, 66, 78, 77],
    # "[15]": [0, 0, 0, 0],
}

auc_3d_std = {
    "MARTA": [23, 19, 18, 8],
    "MARTA-DOS": [22, 10, 16, 12],
    # "[15]": [0, 0, 0, 0],
}

auc_64d_values = {
    "MARTA": [81, 79, 79, 78],
    "MARTA-DOS": [85, 81, 81, 81],
    # "[15]": [80, 86, 62, 72],
}

auc_64d_std = {
    "MARTA": [21, 13, 16, 14],
    "MARTA-DOS": [20, 14, 15, 16],
    # "[15]": [0, 0, 0, 0],
}


# Updated plotting function to change x-axis and group bars by tasks
def plot_auc_grouped_by_task(models, databases, auc_values, auc_std, title):
    x = np.arange(len(databases))  # the label locations for each task
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    offsets = [-1, 0, 1]  # offsets for each model's bar
    # Use a pastel color palette
    colors = ["#FFB6C1", "#FFD700", "#FFA07A"]

    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = [auc_values[model][j] for j in range(len(databases))]
        errors = [auc_std[model][j] for j in range(len(databases))]
        # if model is [15] no std deviation is available
        if model == "[15]":
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                capsize=5,
            )
        else:
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                yerr=errors,
                capsize=5,
            )

    # Labels, title and custom x-axis tick labels
    ax.set_xlabel("Tasks")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()

    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot for 3D grouped by tasks
plot_auc_grouped_by_task(
    models,
    databases,
    auc_3d_values,
    auc_3d_std,
    "AUC Scores for each task and model with std deviation",
)

# Plot for 64D grouped by tasks
plot_auc_grouped_by_task(
    models,
    databases,
    auc_64d_values,
    auc_64d_std,
    "AUC Scores for each task and model with std deviation",
)


#############   CROSS LINGUAL

from matplotlib import pyplot as plt
import numpy as np

# Data from the table
models = ["MARTA", "MARTA-DOS"]
databases = ["NeuroVoz RS", "NeuroVoz Texts", "GITA RS", "GITA Texts"]

# AUC values for 3D and 64D

auc_64d_values = {
    "MARTA": [77, 54, 70, 70],
    "MARTA-DOS": [79, 62, 73, 70],
    # "[15]": [80, 86, 62, 72],
}

auc_64d_std = {
    "MARTA": [20, 6, 16, 13],
    "MARTA-DOS": [16, 12, 18, 16],
    # "[15]": [0, 0, 0, 0],
}


# Updated plotting function to change x-axis and group bars by tasks
def plot_auc_grouped_by_task(models, databases, auc_values, auc_std, title):
    x = np.arange(len(databases))  # the label locations for each task
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    offsets = [-1, 0, 1]  # offsets for each model's bar
    # Use a pastel color palette
    colors = ["#FFB6C1", "#FFD700", "#FFA07A"]

    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = [auc_values[model][j] for j in range(len(databases))]
        errors = [auc_std[model][j] for j in range(len(databases))]
        # if model is [15] no std deviation is available
        if model == "[15]":
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                capsize=5,
            )
        else:
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                yerr=errors,
                capsize=5,
            )

    # Labels, title and custom x-axis tick labels
    ax.set_xlabel("Tasks")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()

    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot for 64D grouped by tasks
plot_auc_grouped_by_task(
    models,
    databases,
    auc_64d_values,
    auc_64d_std,
    "AUC Scores for each task and model with std deviation",
)


################ Whisper vs Manual vs No MULTILINGUAL
from matplotlib import pyplot as plt
import numpy as np

# Data from the table
models = ["Manual Trans.", "Whisper Trans.", "No Trans."]
databases = ["NeuroVoz RS", "NeuroVoz Texts", "GITA RS", "GITA Texts"]

# AUC values for 3D and 64D

auc_64d_values = {
    "Manual Trans.": [85, 81, 81, 81],
    "Whisper Trans.": [82, 80, 80, 73],
    "No Trans.": [82, 83, 80, 80],
    # "[15]": [80, 86, 62, 72],
}

auc_64d_std = {
    "Manual Trans.": [20, 14, 15, 16],
    "Whisper Trans.": [20, 15, 10, 11],
    "No Trans.": [20, 16, 16, 15],
    # "[15]": [80, 86, 62, 72],
}


# Updated plotting function to change x-axis and group bars by tasks
def plot_auc_grouped_by_task(models, databases, auc_values, auc_std, title):
    x = np.arange(len(databases))  # the label locations for each task
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    offsets = [-1, 0, 1]  # offsets for each model's bar
    # Use a pastel color palette
    colors = ["#FFB6C1", "#FFD700", "#FFA07A"]

    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = [auc_values[model][j] for j in range(len(databases))]
        errors = [auc_std[model][j] for j in range(len(databases))]
        # if model is [15] no std deviation is available
        if model == "[15]":
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                capsize=5,
            )
        else:
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                yerr=errors,
                capsize=5,
            )

    # Labels, title and custom x-axis tick labels
    ax.set_xlabel("Tasks")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()

    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot for 64D grouped by tasks
plot_auc_grouped_by_task(
    models,
    databases,
    auc_64d_values,
    auc_64d_std,
    "MULTILINGUAL: AUC Scores for each task and model with std deviation",
)


################ Whisper vs Manual vs No crosslingual
from matplotlib import pyplot as plt
import numpy as np

# Data from the table
models = ["Manual Trans.", "No Trans."]
databases = ["NeuroVoz RS", "NeuroVoz Texts", "GITA RS", "GITA Texts"]

# AUC values for 3D and 64D

auc_64d_values = {
    "Manual Trans.": [79, 62, 73, 70],
    "No Trans.": [74, 60, 70, 69],
    # "[15]": [80, 86, 62, 72],
}

auc_64d_std = {
    "Manual Trans.": [17, 12, 18, 16],
    "No Trans.": [20, 10, 16, 15],
    # "[15]": [80, 86, 62, 72],
}


# Updated plotting function to change x-axis and group bars by tasks
def plot_auc_grouped_by_task(models, databases, auc_values, auc_std, title):
    x = np.arange(len(databases))  # the label locations for each task
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    offsets = [-1, 0, 1]  # offsets for each model's bar
    # Use a pastel color palette
    colors = ["#FFB6C1", "#FFD700", "#FFA07A"]

    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = [auc_values[model][j] for j in range(len(databases))]
        errors = [auc_std[model][j] for j in range(len(databases))]
        # if model is [15] no std deviation is available
        if model == "[15]":
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                capsize=5,
            )
        else:
            ax.bar(
                x + offset * width,
                values,
                width,
                label=model,
                color=colors[i],
                yerr=errors,
                capsize=5,
            )

    # Labels, title and custom x-axis tick labels
    ax.set_xlabel("Tasks")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()

    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot for 64D grouped by tasks
plot_auc_grouped_by_task(
    models,
    databases,
    auc_64d_values,
    auc_64d_std,
    "CROSSLINGUAL: AUC Scores for each task and model with std deviation",
)
