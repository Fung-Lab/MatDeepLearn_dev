from mp_api.client import MPRester 
import csv
import numpy as np

mpr = MPRester("7SNHVgZnlVg1F0JzXtMzrU4NZW3wRLvB")
print(mpr.materials.summary.available_fields)
docs = mpr.materials.summary.search(fields=["material_id", "structure", "composition", "formation_energy_per_atom"])
doc_list = [doc for doc in docs]

unique_compositions = []
comp_dict = {}
for doc in doc_list:
    if doc.composition.alphabetical_formula not in unique_compositions:
        #add sorted composition to list
        unique_compositions.append(doc.composition.alphabetical_formula)
        comp_dict[doc.composition.alphabetical_formula] = (doc.formation_energy_per_atom, doc.material_id, doc.structure, doc.composition.alphabetical_formula)
    else:
        if comp_dict[doc.composition.alphabetical_formula][0] > doc.formation_energy_per_atom:
            comp_dict[doc.composition.alphabetical_formula] = (doc.formation_energy_per_atom, doc.material_id, doc.structure, doc.composition.alphabetical_formula)
print(unique_compositions[0])
print(comp_dict[unique_compositions[0]])
print(len(unique_compositions))
