import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("./datasets/insurance.csv")
data["smoker"] = data["smoker"].map({"yes": 1, "no": 0})

# plt.scatter(x,y)
sns.lmplot(data=data, x="age", y="charges", hue="smoker")

plt.title("Linear Regression of Age vs. Medical Charges")
plt.show()