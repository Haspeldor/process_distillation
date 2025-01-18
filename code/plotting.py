import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_shapley(enriched_accuracy_values, modified_accuracy_values, folder_name, file_name):
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    
    data = {
        'Enriched': np.array(enriched_accuracy_values),
        'Modified': np.array(modified_accuracy_values)
    }
    
    # Calculate mean and standard deviation
    stats = {name: (np.mean(values), np.std(values)) for name, values in data.items()}
    print("Accuracy Statistics:")
    for name, (mean, std) in stats.items():
        print(f"{name}: Mean = {mean:.2f}, Std Dev = {std:.3f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define custom colors
    for i, (name, values) in enumerate(data.items()):
        sns.kdeplot(values, label=f'{name} (μ={stats[name][0]:.2f}, σ={stats[name][1]:.3f})',
                    color=colors[i], fill=True, alpha=0.6, linewidth=2)
    
    # Add titles and labels
    plt.title(f'Shapley Score of {folder_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Shapley', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, title='Legend', title_fontsize='13')
    
    # Save the plot
    plt.tight_layout()
    image_path = os.path.join('img', folder_name, file_name)
    plt.savefig(image_path)
    plt.close()

def plot_all_parity(stat_par, folder_name):
    for (feature, event), value in stat_par.items():
        title = f"Statistical Parity of {feature}, {event} for {folder_name}"
        plot_distribution(value["Base"], value["Enriched"], value["Modified"], folder_name, title, "Statistical Parity")

def plot_all_fairness(stat_par, disp_imp, folder_name):
    for (feature, event), value in stat_par.items():
        title = f"Statistical Parity of {feature}, {event} for {folder_name}"
        plot_distribution(value["Base"], value["Enriched"], value["Modified"], folder_name, title, "Statistical Parity")
    for (feature, event), value in disp_imp.items():
        title = f"Disparate Impact of {feature}, {event} for {folder_name}"
        plot_distribution(value["Base"], value["Enriched"], value["Modified"], folder_name, title, "Disparate Impact")

def plot_density(data, title):
    sns.kdeplot(data, fill=True, color="blue", alpha=0.5)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()


def plot_distribution(base_values, enriched_values, modified_values, folder_name, title, measurement):
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    
    data = {
        'Base': np.array(base_values),
        'Enriched': np.array(enriched_values),
        'Modified': np.array(modified_values)
    }
    print(title)
    print(data)
    
    # Calculate mean and standard deviation
    stats = {name: (np.mean(values), np.std(values)) for name, values in data.items()}
    print(f"{title} Statistics:")
    for name, (mean, std) in stats.items():
        print(f"{name}: Mean = {mean:.2f}, Std Dev = {std:.3f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define custom colors
    for i, (name, values) in enumerate(data.items()):
        sns.kdeplot(values, label=f'{name} (μ={stats[name][0]:.3f}, σ={stats[name][1]:.3f})',
                    color=colors[i], fill=True, alpha=0.6, linewidth=2)
    
    # Add titles and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(measurement, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, title='Legend', title_fontsize='13')
    
    # Save the plot
    plt.tight_layout()
    image_path = os.path.join('img', folder_name, title)
    plt.savefig(image_path)
    plt.close()

def plot_accuracy(base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, folder_name):
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    
    data = {
        'Base': np.array(base_accuracy_values),
        'Enriched': np.array(enriched_accuracy_values),
        'Modified': np.array(modified_accuracy_values)
    }
    
    # Calculate mean and standard deviation
    stats = {name: (np.mean(values), np.std(values)) for name, values in data.items()}
    print("Accuracy Statistics:")
    for name, (mean, std) in stats.items():
        print(f"{name}: Mean = {mean:.3f}, Std Dev = {std:.3f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define custom colors
    for i, (name, values) in enumerate(data.items()):
        sns.kdeplot(values, label=f'{name} (μ={stats[name][0]:.2f}, σ={stats[name][1]:.3f})',
                    color=colors[i], fill=True, alpha=0.6, linewidth=2)
    
    # Add titles and labels
    plt.title(f'Accuracy Distribution of {folder_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, title='Legend', title_fontsize='13')
    
    # Save the plot
    plt.tight_layout()
    image_path = os.path.join('img', folder_name, f"accuracy.png")
    plt.savefig(image_path)
    plt.close()

def plot_attributes(df: pd.DataFrame, rules: list, folder_name: str):

    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    # Group by case_id to ensure each case is only counted once
    grouped = df.groupby('case_id')

    # Collect unique attributes and their rules
    attribute_rules = {}
    for rule in rules:
        attribute = rule['attribute']
        if attribute not in attribute_rules:
            attribute_rules[attribute] = []
        attribute_rules[attribute].append(rule)

    for attribute, rules in attribute_rules.items():
        # Combine data for the attribute
        attribute_values = grouped[attribute].first().dropna()

        # Start plotting
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Handle discrete attributes
        if any(rule['distribution']['type'] == 'discrete' for rule in rules):
            # Discrete values and their labels
            discrete_values = []
            for rule in rules:
                if rule['distribution']['type'] == 'discrete':
                    values, _ = zip(*rule['distribution']['values'])
                    discrete_values.extend(values)
            discrete_values = list(set(discrete_values))  # Remove duplicates

            # Calculate percentages
            counts = attribute_values.value_counts(normalize=True).reindex(discrete_values, fill_value=0)
            counts *= 100  # Convert to percentages

            # Bar plot for discrete attributes
            sns.barplot(x=counts.index, y=counts.values, palette="viridis", saturation=0.9)
            plt.title(f"Distribution of {attribute} (Discrete, per Case)", fontsize=16, fontweight="bold")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Percentage (%)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            for i, v in enumerate(counts.values):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10, fontweight="bold")
        
        # Handle continuous attributes
        elif any(rule['distribution']['type'] == 'normal' for rule in rules):
            # Calculate number of bins (using Sturges' rule)
            bins = int(np.ceil(np.log2(len(attribute_values))) + 1)

            # Histogram for continuous attributes
            sns.histplot(attribute_values, bins=bins, kde=True, color='mediumvioletred', alpha=0.7)
            plt.title(f"Distribution of {attribute} (Continuous, per Case)", fontsize=16, fontweight="bold")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Percentage (%)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Display percentages on the histogram bars
            n, bin_edges = np.histogram(attribute_values, bins=bins)
            percentages = (n / n.sum()) * 100
            for i in range(len(n)):
                plt.text(bin_edges[i] + (bin_edges[i+1] - bin_edges[i]) / 2, percentages[i] + 0.5, 
                         f"{percentages[i]:.1f}%", ha='center', fontsize=10, fontweight="bold")

        else:
            print(f"Unsupported distribution type for attribute '{attribute}'. Skipping.")
            continue

        # Add final touches
        plt.tight_layout()
        image_path = os.path.join('img', folder_name, f"{attribute}.png")
        plt.savefig(image_path)
        plt.close()
