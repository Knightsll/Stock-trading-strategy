import numpy as np

MV = np.arange(0,30)
ORPS = np.ones((30))*10
ORPS[20]=np.nan

if len(MV) > 30:
    temp_OR = np.nan_to_num(ORPS[[29 + i * 30 for i in range(50)]])
    temp_index = np.where(temp_OR == 0)[0]
    temp_OR = np.delete(temp_OR, temp_index)
    temp_MV = np.delete(MV[[29 + i * 30 for i in range(50)]], temp_index)
    temp_PS = temp_MV / temp_OR
    PS_total = np.insert(temp_PS, temp_index, 0)
else:
    temp_OR = np.nan_to_num(ORPS)
    temp_index = np.where(temp_OR == 0)[0]
    temp_OR = np.delete(temp_OR, temp_index)
    temp_MV = np.delete(MV, temp_index)
    temp_PS = temp_MV / temp_OR
    PS_total = np.insert(temp_PS, temp_index, 0)

print(PS_total)
print(len(PS_total))
