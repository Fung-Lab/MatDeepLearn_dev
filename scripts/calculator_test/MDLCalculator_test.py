import yaml
import torch
from matdeeplearn.common.MDLCalculator import MDLCalculator


config_path = './configs/config_calculator.yml'
with open(config_path, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
c = MDLCalculator(config_path)

dta = torch.load('./data/data.pt')

atoms_list = MDLCalculator.data_to_atoms_list(data=dta[0])

for i in range(len(atoms_list)):
    print('Label:', dta[0].y[i])
    c.calculate(atoms_list[i])
    print(c.results)

