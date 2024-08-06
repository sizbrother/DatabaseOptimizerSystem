import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file_path = './query_plans07-15.csv'
df = pd.read_csv(csv_file_path)

# Directory to save the plots
output_dir = 'query_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

grouped = df.groupby('Query Number')

plt.figure(figsize=(12, 6))

data = [group['Cost'] for query_number, group in grouped]

plt.boxplot(data, labels=[f'Query {query_number}' for query_number in grouped.groups.keys()], vert=True)

plt.title('Box Plot of Costs for Each Query')
plt.xlabel('Query Number')
plt.ylabel('Cost')
plt.xticks(rotation=90)

plt.tight_layout()

plot_filename = os.path.join(output_dir, 'all_queries_costs.png')
plt.savefig(plot_filename)
plt.close()

print(f"Plot has been saved to the '{output_dir}' directory.")
