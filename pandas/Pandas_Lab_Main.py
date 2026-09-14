import pandas as pd
df = pd.read_csv('Internet Speed 2022.csv')
#print(df.dtypes)
#print(df.sort_values('broadband', ascending=False).head().dropna())

newContry = {'country': "Galactic", "broadband": 1488, "mobile": 293.423}
df1= pd.DataFrame([newContry])
newList = pd.concat([df1, df], ignore_index=True)
newList.drop(0, inplace=True)
print(newList.loc[[10, 11]])
print(newList.iloc[5:8])
print(newList[newList['mobile'] > 100])
print(newList['broadband'].agg(['mean']))
newList.to_csv(r'/home/admin/PycharmProjects/PythonLabs/pandasss.csv')
#print(newList)
