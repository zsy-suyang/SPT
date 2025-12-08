import os
import pandas as pd
import ruamel.yaml

root_path = '/internfs/zsy/proj/AMFormer'
data = pd.read_csv(os.path.join(root_path, 'datasets/O_site.csv'))

mp = data.iloc[:, 1]
leg = mp.drop_duplicates()

des_columns = data.iloc[:, -8:]

# metal: 'OS', 'CN', 'OCN', 'NNRS', 'NNAS', 'MOOD', 'CN2', 'PSGO'
# Oxy: 'OS', 'CN', 'NNRS', 'NNAS', 'NNOSM'
columns_to_process = ['OS', 'CN', 'NNRS', 'NNAS', 'NNOSM', 'CN2', 'PSGO']

des_dict = {}
for idx, col in enumerate(columns_to_process):
    if col != 'NNRS' and col != 'NNAS' and col != 'NNOSM':
        value_counts = des_columns[col].value_counts().to_dict()
        sorted_value_counts = dict(sorted(value_counts.items(), key=lambda item: item[0]))
        value = list(sorted_value_counts.keys())
        cls = [i for i in range(len(value))]
        des_dict[col] = {
            'idx': idx,
            'num': len(sorted_value_counts),
            'value': value,
            'label': cls,
            'mode': 'Cls',
            'cnts': value_counts
        }
    else:
        des_dict[col] = {
            'idx': idx,
            'mode': 'Reg'
        }

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
output_path = os.path.join(root_path, 'config/datasets/O_descriptors.yaml')
with open(output_path, 'w') as file:
    yaml.dump(des_dict, file)

print("YAML File Done")


